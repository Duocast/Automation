"""Tests for context compaction.

The dangerous failure is orphaning a tool_use (assistant calls a tool, but the matching
tool_result gets summarized away) — that's a 400 on the next call. Several tests assert
the pairing invariant directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forge.compact import (
    _exchange_bounds,
    _has_tool_use,
    compact,
    estimate_tokens,
)
from forge.config import Budget, Config
from forge.llm import Reply, Usage
from forge.loop import Agent


# --------------------------------------------------------------------------- fakes
class SummarizerLLM:
    """Returns a fixed short summary for any call. Records what it was asked to compress."""

    def __init__(self):
        self.calls = []

    def call(self, **kw):
        self.calls.append(kw)
        return Reply(
            content=[{"type": "text", "text": "condensed: cloned repo, confirmed --upper, tests green"}],
            stop_reason="end_turn",
            usage=Usage(50, 20),
        )


def au(id, name="grep", inp=None):
    return {"role": "assistant", "content": [{"type": "tool_use", "id": id, "name": name, "input": inp or {"q": "x"}}]}


def tr(id, text="result"):
    return {"role": "user", "content": [{"type": "tool_result", "tool_use_id": id, "content": text}]}


def build_convo(n_exchanges: int, filler: int = 0) -> list[dict]:
    """Task + n (assistant tool_use, tool_result) exchanges, optionally padded."""
    msgs = [{"role": "user", "content": "the original task"}]
    for i in range(n_exchanges):
        msgs.append(au(f"t{i}", inp={"q": "x" * filler}))
        msgs.append(tr(f"t{i}", text="r" * max(1, filler)))
    return msgs


# --------------------------------------------------------------------------- estimation
def test_estimate_grows_with_content():
    small = estimate_tokens([{"role": "user", "content": "hi"}])
    big = estimate_tokens([{"role": "user", "content": "x" * 10_000}])
    assert big > small > 0


def test_estimate_handles_all_block_types():
    msgs = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "hmm " * 100},
            {"type": "text", "text": "answer"},
            {"type": "tool_use", "id": "t1", "name": "run", "input": {"cmd": "ls"}},
        ]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "files"}]},
    ]
    assert estimate_tokens(msgs) > 0  # no crash on mixed blocks


# --------------------------------------------------------------------------- boundaries
def test_exchange_bounds_pairs_tool_use_with_result():
    msgs = build_convo(3)
    units = _exchange_bounds(msgs)
    assert units[0] == (0, 0)          # task is its own unit
    assert units[1] == (1, 2)          # assistant tool_use + tool_result bound together
    assert units[2] == (3, 4)


def test_exchange_bounds_handles_trailing_toolless_assistant():
    msgs = build_convo(1)
    msgs.append({"role": "assistant", "content": [{"type": "text", "text": "done"}]})
    units = _exchange_bounds(msgs)
    assert units[-1] == (3, 3)         # lone assistant text is its own unit


# --------------------------------------------------------------------------- compaction
def test_no_compaction_under_budget():
    llm = SummarizerLLM()
    msgs = build_convo(3)
    out = compact(llm, "haiku", msgs, None, max_tokens=1_000_000)
    assert out is msgs                 # untouched
    assert llm.calls == []             # summarizer never invoked


def test_no_compaction_when_too_short():
    """Even over budget, refuse to gut a conversation with no summarizable middle."""
    llm = SummarizerLLM()
    msgs = build_convo(3, filler=200_000)
    out = compact(llm, "haiku", msgs, None, max_tokens=10, keep_recent_exchanges=4)
    assert out is msgs                 # 3 exchanges < keep(4)+2, nothing to cut safely


def test_compaction_triggers_and_shrinks():
    llm = SummarizerLLM()
    msgs = build_convo(20, filler=5000)
    before = estimate_tokens(msgs)
    out = compact(llm, "haiku", msgs, None, max_tokens=before // 2, keep_recent_exchanges=4)
    assert out is not msgs
    assert estimate_tokens(out) < before
    assert len(llm.calls) == 1         # exactly one summary produced


def test_compaction_preserves_task_and_recent():
    llm = SummarizerLLM()
    msgs = build_convo(20, filler=5000)
    out = compact(llm, "haiku", msgs, None, max_tokens=estimate_tokens(msgs) // 2,
                  keep_recent_exchanges=4)
    # Task verbatim at the front
    assert out[0]["content"] == "the original task"
    # A single summary message follows
    assert out[1]["content"][0]["text"].startswith("[earlier work, condensed]")
    # The last 4 exchanges (8 messages) are the original tail, verbatim
    assert out[-8:] == msgs[-8:]


def test_compaction_never_orphans_a_tool_use():
    """The load-bearing invariant: every tool_use id in the output has a matching
    tool_result in the output. Orphaning one is a 400 on the next API call."""
    llm = SummarizerLLM()
    msgs = build_convo(30, filler=4000)
    out = compact(llm, "haiku", msgs, None, max_tokens=estimate_tokens(msgs) // 3,
                  keep_recent_exchanges=4)

    used, resulted = set(), set()
    for m in out:
        c = m.get("content")
        if not isinstance(c, list):
            continue
        for b in c:
            if b.get("type") == "tool_use":
                used.add(b["id"])
            elif b.get("type") == "tool_result":
                resulted.add(b["tool_use_id"])
    assert used == resulted, f"orphaned tool_use ids: {used ^ resulted}"


def test_summarizer_gets_the_middle_not_the_edges():
    llm = SummarizerLLM()
    msgs = build_convo(20, filler=3000)
    compact(llm, "haiku", msgs, None, max_tokens=estimate_tokens(msgs) // 2,
            keep_recent_exchanges=4)
    slice_text = llm.calls[0]["messages"][0]["content"]
    assert "the original task" not in slice_text     # task is anchored, not summarized
    # the summarizer runs with thinking off and no streaming (it's a cheap utility call)
    assert llm.calls[0].get("thinking") is False


# --------------------------------------------------------------------------- integration
def _cfg(tmp_path, max_input):
    return Config(
        workspace=tmp_path / "ws",
        skills_dir=tmp_path / "sk",
        budget=Budget(max_turns=12, max_wall_seconds=60, max_input_tokens=max_input),
        max_eval_rounds=1,
    )


class ChattyLLM:
    """Agent model that keeps calling a tool, emitting large outputs, then stops.
    Drives the loop long enough to force compaction."""

    def __init__(self, n_tool_turns: int):
        self.n = n_tool_turns
        self.i = 0
        self.summaries = 0

    def call(self, **kw):
        # Distinguish the summarizer call (no tools) from the agent call (tools present).
        if not kw.get("tools"):
            self.summaries += 1
            return Reply([{"type": "text", "text": "condensed prior work"}], "end_turn", Usage(10, 5))
        self.i += 1
        if self.i <= self.n:
            return Reply(
                [{"type": "tool_use", "id": f"t{self.i}", "name": "list_dir",
                  "input": {"path": "."}}],
                "tool_use", Usage(100, 50),
            )
        return Reply([{"type": "text", "text": "all done"}], "end_turn", Usage(10, 5))


def test_loop_compacts_on_long_run(tmp_path):
    # Tiny budget so compaction must fire; big tool outputs so it grows fast.
    cfg = _cfg(tmp_path, max_input=800)
    (cfg.workspace).mkdir(parents=True, exist_ok=True)
    for i in range(50):
        (cfg.workspace / f"file_{i}.txt").write_text("x" * 500)

    llm = ChattyLLM(n_tool_turns=8)
    events = []
    res = Agent(cfg, llm, on_event=events.append).run("explore the workspace")

    assert res.answer == "all done"
    assert llm.summaries >= 1                       # compaction actually ran
    assert any(e["type"] == "compact" for e in events)
    # Audit trail is complete despite compaction — evaluator sees everything.
    assert len([c for c in res.tool_calls if c["name"] == "list_dir"]) == 8


def test_compaction_leaves_audit_trail_intact(tmp_path):
    cfg = _cfg(tmp_path, max_input=600)
    cfg.workspace.mkdir(parents=True, exist_ok=True)
    llm = ChattyLLM(n_tool_turns=10)
    res = Agent(cfg, llm, on_event=lambda e: None).run("go")
    # Every tool call is in the record even though the conversation was compacted.
    assert len(res.tool_calls) == 10
    assert all("output" in c for c in res.tool_calls)

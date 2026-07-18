"""Tests for the lead/subagent split.

Two tests matter most:
  - `test_subagent_cannot_delegate` — the recursion fuse. Without it, fan-out is
    exponential in depth with real money attached.
  - `test_subagent_evidence_reaches_the_evaluator` — delegated work must leave the same
    evidence as inline work, or the critic (see test_evaluator.py) flags it as fabricated.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forge.config import Budget, Config
from forge.evaluator import TrailTools
from forge.llm import Reply, Usage
from forge.loop import Agent
from forge.orchestrate import Subtask, batches, conflicts, explain, validate
from forge.sandbox import Sandbox, SandboxError
from forge.tools import build_toolset


# --------------------------------------------------------------------------- helpers
def tool_use(name, inp, id="t1"):
    return {"type": "tool_use", "name": name, "input": inp, "id": id}


def text(t):
    return {"type": "text", "text": t}


@pytest.fixture
def cfg(tmp_path):
    return Config(
        workspace=tmp_path / "ws", skills_dir=tmp_path / "sk",
        budget=Budget(max_turns=10, max_wall_seconds=60, max_subagents=8,
                      subagent_max_turns=5, max_subtasks_per_call=6),
        max_eval_rounds=1,
    )


def st(id, writes=(), reads=()):
    return Subtask(id=id, task=f"do {id}", writes=list(writes), reads=list(reads))


# --------------------------------------------------------------------------- hazards
def test_disjoint_subtasks_dont_conflict():
    assert conflicts(st("a", writes=["src/a.py"]), st("b", writes=["src/b.py"])) is None


def test_waw_detected():
    assert conflicts(st("a", writes=["x.py"]), st("b", writes=["x.py"])) == "WAW"


def test_raw_detected():
    assert conflicts(st("a", writes=["x.py"]), st("b", reads=["x.py"])) == "RAW"


def test_war_detected():
    assert conflicts(st("a", reads=["x.py"]), st("b", writes=["x.py"])) == "WAR"


def test_two_readers_dont_conflict():
    assert conflicts(st("a", reads=["x.py"]), st("b", reads=["x.py"])) is None


def test_directory_covers_its_subtree():
    assert conflicts(st("a", writes=["src"]), st("b", writes=["src/deep/x.py"])) == "WAW"


def test_similar_prefix_is_not_a_subtree():
    """'src' must not swallow 'srcfoo' — that would serialize unrelated work."""
    assert conflicts(st("a", writes=["src"]), st("b", writes=["srcfoo"])) is None


def test_whole_workspace_claim_conflicts_with_everything():
    assert conflicts(st("a", writes=["."]), st("b", writes=["anything.py"])) == "WAW"


def test_path_normalisation():
    assert conflicts(st("a", writes=["./src/x.py"]), st("b", writes=["src/x.py/"])) == "WAW"


# --------------------------------------------------------------------------- batching
def test_disjoint_work_batches_together():
    b = batches([st("a", writes=["a.py"]), st("b", writes=["b.py"]), st("c", writes=["c.py"])])
    assert len(b) == 1 and len(b[0]) == 3


def test_conflicting_work_serializes():
    b = batches([st("a", writes=["x.py"]), st("b", writes=["x.py"])])
    assert len(b) == 2


def test_batching_mixes_correctly():
    """a and b conflict; c is free and should ride along with a."""
    b = batches([st("a", writes=["x.py"]), st("b", writes=["x.py"]), st("c", writes=["z.py"])])
    assert [sorted(s.id for s in g) for g in b] == [["a", "c"], ["b"]]


def test_explain_names_the_hazard():
    notes = explain([st("a", writes=["x.py"]), st("b", reads=["x.py"])])
    assert len(notes) == 1 and "RAW" in notes[0] and "a & b" in notes[0]


# --------------------------------------------------------------------------- validation
def test_validate_assigns_default_ids():
    out = validate([{"task": "one"}, {"task": "two"}], 6)
    assert [s.id for s in out] == ["s1", "s2"]


def test_validate_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        validate([], 6)


def test_validate_rejects_too_many():
    with pytest.raises(ValueError, match="exceeds the limit"):
        validate([{"task": f"t{i}"} for i in range(9)], 6)


def test_validate_rejects_blank_task():
    with pytest.raises(ValueError, match="no task text"):
        validate([{"task": "   "}], 6)


def test_validate_rejects_duplicate_ids():
    with pytest.raises(ValueError, match="duplicate"):
        validate([{"id": "x", "task": "a"}, {"id": "x", "task": "b"}], 6)


# --------------------------------------------------------------------------- the fuse
def test_lead_gets_delegate_subagent_does_not(cfg):
    lead, _ = build_toolset(cfg, lead=True)
    sub, _ = build_toolset(cfg, lead=False)
    assert "delegate" in lead
    assert "delegate" not in sub


def test_subagent_cannot_delegate(cfg):
    """The recursion fuse: depth is capped structurally, not by a counter."""
    class Recursive:
        def call(self, **kw):
            if _is_lead(kw):
                if _first_turn(kw):
                    return Reply([tool_use("delegate", {"subtasks": [
                        {"id": "s1", "task": "spawn more", "writes": ["a.txt"]}]}, "d")],
                        "tool_use", Usage(1, 1))
                return Reply([text("done")], "end_turn", Usage(1, 1))
            # A subagent tries to delegate anyway; the tool isn't in its toolset.
            if _first_turn(kw):
                return Reply([tool_use("delegate", {"subtasks": [{"task": "deeper"}]}, "x")],
                             "tool_use", Usage(1, 1))
            return Reply([text("could not delegate")], "end_turn", Usage(1, 1))

    res = Agent(cfg, Recursive(), lead=True).run("split this up")
    sub_calls = [c for c in res.tool_calls if c.get("subagent")]
    assert sub_calls, "subagent should have run"
    assert all(c["is_error"] and "unknown tool" in c["output"] for c in sub_calls)


def test_delegate_unavailable_without_lead(cfg):
    """Even if a tool instance leaks in, ctx.spawn is None for a non-lead."""
    from forge.tools.base import Context
    from forge.tools.delegate import Delegate
    from forge.skillstore import SkillStore
    ctx = Context(cfg=cfg, sandbox=Sandbox(cfg.workspace), skills=SkillStore(cfg.skills_dir))
    with pytest.raises(Exception, match="not available"):
        Delegate().run(ctx, subtasks=[{"task": "x"}])


# --------------------------------------------------------------------------- spawning
def _is_lead(kw) -> bool:
    return "delegate" in {t.get("name") for t in (kw.get("tools") or [])}


def _first_turn(kw) -> bool:
    """Stateless turn detection. Thread-id keying is wrong here: ThreadPoolExecutor
    recycles threads, so a second subtask would inherit the first's counter."""
    return len(kw["messages"]) == 1


class Lead:
    """Delegates once, then reports. Subagents write their declared file."""

    def __init__(self, subtasks, sub_answer="subtask complete"):
        self.subtasks = subtasks
        self.sub_answer = sub_answer
        self.seen_report = None

    def call(self, **kw):
        if _is_lead(kw):
            if _first_turn(kw):
                return Reply([tool_use("delegate", {"subtasks": self.subtasks}, "d")],
                             "tool_use", Usage(10, 5))
            self.seen_report = kw["messages"][-1]["content"][0]["content"]
            return Reply([text("all subtasks done")], "end_turn", Usage(10, 5))

        # subagent: write its declared file, then report
        if _first_turn(kw):
            task = str(kw["messages"][0]["content"])
            fname = task.split("write:")[-1].strip() if "write:" in task else "out.txt"
            return Reply([tool_use("write_file", {"path": fname, "content": "x"}, "w")],
                         "tool_use", Usage(5, 2))
        return Reply([text(self.sub_answer)], "end_turn", Usage(5, 2))


def test_delegate_runs_subagents_and_returns_reports(cfg):
    llm = Lead([
        {"id": "s1", "task": "handle a. write: a.txt", "writes": ["a.txt"]},
        {"id": "s2", "task": "handle b. write: b.txt", "writes": ["b.txt"]},
    ])
    res = Agent(cfg, llm, lead=True).run("do a and b")
    assert res.answer == "all subtasks done"
    assert (cfg.workspace / "a.txt").exists() and (cfg.workspace / "b.txt").exists()
    assert "### s1" in llm.seen_report and "### s2" in llm.seen_report
    assert "subtask complete" in llm.seen_report


def test_disjoint_subtasks_run_in_one_batch(cfg):
    llm = Lead([
        {"id": "s1", "task": "a. write: a.txt", "writes": ["a.txt"]},
        {"id": "s2", "task": "b. write: b.txt", "writes": ["b.txt"]},
    ])
    Agent(cfg, llm, lead=True).run("go")
    assert "1 batch(es)" in llm.seen_report
    assert "parallel" in llm.seen_report


def test_conflicting_subtasks_are_serialized(cfg):
    llm = Lead([
        {"id": "s1", "task": "a. write: shared.txt", "writes": ["shared.txt"]},
        {"id": "s2", "task": "b. write: shared.txt", "writes": ["shared.txt"]},
    ])
    Agent(cfg, llm, lead=True).run("go")
    assert "2 batch(es)" in llm.seen_report
    assert "WAW" in llm.seen_report          # and it says why


def test_subagent_evidence_reaches_the_evaluator(cfg):
    """Delegated work must leave the same evidence trail as inline work."""
    llm = Lead([{"id": "s1", "task": "a. write: a.txt", "writes": ["a.txt"]}])
    res = Agent(cfg, llm, lead=True).run("do a")

    tagged = [c for c in res.tool_calls if c.get("subagent") == "s1"]
    assert tagged, "subagent tool calls must merge into the lead's trail"
    assert any(c["name"] == "write_file" for c in tagged)
    # the evaluator's index attributes it...
    assert "(via s1)" in res.trail_index()
    # ...and its grep reaches into subagent evidence
    assert "a.txt" in res.grep_trail("a.txt")
    # and TrailTools can pull the full output
    out, err = TrailTools(res).dispatch("grep_trail", {"pattern": "a.txt"})
    assert not err and "s1" in res.trail_index()


def test_subagent_usage_is_accounted(cfg):
    llm = Lead([{"id": "s1", "task": "a. write: a.txt", "writes": ["a.txt"]}])
    res = Agent(cfg, llm, lead=True).run("do a")
    # lead turns + subagent turns both billed
    assert res.usage.input_tokens > 10


# --------------------------------------------------------------------------- confinement
def test_write_allowlist_blocks_undeclared_paths(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    sb = Sandbox(ws, write_allowlist=[ws / "mine.txt", ws / ".scratch/s1"])
    assert sb.resolve_write("mine.txt")
    assert sb.resolve_write(".scratch/s1/tmp.json")
    with pytest.raises(SandboxError, match="declared write set"):
        sb.resolve_write("someone_elses.txt")


def test_write_allowlist_covers_declared_directories(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    sb = Sandbox(ws, write_allowlist=[ws / "src"])
    assert sb.resolve_write("src/deep/nested.py")
    with pytest.raises(SandboxError):
        sb.resolve_write("other/x.py")


def test_allowlist_does_not_loosen_workspace_containment(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    sb = Sandbox(ws, write_allowlist=[Path("/etc")])
    with pytest.raises(SandboxError, match="outside the workspace"):
        sb.resolve_write("/etc/passwd")


def test_subagent_confined_to_declared_writes(cfg):
    """A subagent that strays outside its declared set is stopped, not trusted."""
    class Strayer:
        def call(self, **kw):
            if _is_lead(kw):
                if _first_turn(kw):
                    return Reply([tool_use("delegate", {"subtasks": [
                        {"id": "s1", "task": "only touch a.txt", "writes": ["a.txt"]}]}, "d")],
                        "tool_use", Usage(1, 1))
                return Reply([text("done")], "end_turn", Usage(1, 1))
            if _first_turn(kw):
                return Reply([tool_use("write_file", {"path": "b.txt", "content": "sneaky"}, "w")],
                             "tool_use", Usage(1, 1))
            return Reply([text("blocked, understood")], "end_turn", Usage(1, 1))

    res = Agent(cfg, Strayer(), lead=True).run("go")
    assert not (cfg.workspace / "b.txt").exists()
    bad = [c for c in res.tool_calls if c.get("subagent") == "s1"]
    assert bad and bad[0]["is_error"] and "declared write set" in bad[0]["output"]


def test_subagent_always_gets_a_scratch_dir(cfg):
    class Scratcher:
        def call(self, **kw):
            if _is_lead(kw):
                if _first_turn(kw):
                    return Reply([tool_use("delegate", {"subtasks": [
                        {"id": "s1", "task": "use scratch", "writes": ["a.txt"]}]}, "d")],
                        "tool_use", Usage(1, 1))
                return Reply([text("done")], "end_turn", Usage(1, 1))
            if _first_turn(kw):
                return Reply([tool_use("write_file",
                                       {"path": ".scratch/s1/tmp.json", "content": "{}"}, "w")],
                             "tool_use", Usage(1, 1))
            return Reply([text("ok")], "end_turn", Usage(1, 1))

    res = Agent(cfg, Scratcher(), lead=True).run("go")
    call = [c for c in res.tool_calls if c.get("subagent") == "s1"][0]
    assert not call["is_error"], call["output"]


# --------------------------------------------------------------------------- budgets & failure
def test_subagent_budget_enforced(cfg):
    cfg.budget.max_subagents = 2
    llm = Lead([{"id": f"s{i}", "task": f"t{i}. write: f{i}.txt", "writes": [f"f{i}.txt"]}
                for i in range(4)])
    res = Agent(cfg, llm, lead=True).run("go")
    d = [c for c in res.tool_calls if c["name"] == "delegate"]
    assert d and d[0]["is_error"] and "budget" in d[0]["output"]


def test_too_many_subtasks_rejected_recoverably(cfg):
    llm = Lead([{"id": f"s{i}", "task": f"t{i}"} for i in range(9)])
    res = Agent(cfg, llm, lead=True).run("go")
    d = [c for c in res.tool_calls if c["name"] == "delegate"][0]
    assert d["is_error"] and "exceeds the limit" in d["output"]
    assert res.answer == "all subtasks done"   # lead recovered rather than crashing


def test_subagent_crash_does_not_kill_lead(cfg, monkeypatch):
    llm = Lead([{"id": "s1", "task": "a. write: a.txt", "writes": ["a.txt"]}])

    def boom(self, task, effort=None):
        raise RuntimeError("subagent exploded")

    # Patch only the subagent's run by intercepting after the lead has started.
    orig = Agent.run
    calls = {"n": 0}

    def maybe_boom(self, task, effort=None):
        calls["n"] += 1
        if calls["n"] > 1:      # first call is the lead
            raise RuntimeError("subagent exploded")
        return orig(self, task, effort)

    monkeypatch.setattr(Agent, "run", maybe_boom)
    res = Agent(cfg, llm, lead=True).run("go")
    assert res.answer == "all subtasks done"
    assert "subagent crashed" in llm.seen_report


def test_budget_exhausted_subagent_is_flagged_to_lead(cfg):
    """A subagent that ran out of turns must not read as a clean success."""
    cfg.budget.subagent_max_turns = 2

    class Looper:
        def __init__(self):
            self.report = None

        def call(self, **kw):
            if _is_lead(kw):
                if _first_turn(kw):
                    return Reply([tool_use("delegate", {"subtasks": [
                        {"id": "s1", "task": "endless", "writes": ["a.txt"]}]}, "d")],
                        "tool_use", Usage(1, 1))
                self.report = kw["messages"][-1]["content"][0]["content"]
                return Reply([text("done")], "end_turn", Usage(1, 1))
            return Reply([tool_use("list_dir", {"path": "."}, f"l{len(kw['messages'])}")],
                         "tool_use", Usage(1, 1))

    llm = Looper()
    Agent(cfg, llm, lead=True).run("go")
    assert "budget:turns" in llm.report
    assert "may be incomplete" in llm.report


# --------------------------------------------------------------------------- blackboard
from forge.orchestrate import Blackboard, Finding


def test_blackboard_posts_and_renders():
    b = Blackboard()
    assert "1/20" in b.post("s1", "v2 renamed foo to bar")
    assert b.render(b.snapshot()) == "- (s1) v2 renamed foo to bar"


def test_blackboard_truncates_long_findings():
    b = Blackboard(max_len=20)
    assert "truncated" in b.post("s1", "x" * 100)
    assert len(b.snapshot()[0].text) <= 21


def test_blackboard_normalises_whitespace():
    b = Blackboard()
    b.post("s1", "  foo\n\n   bar  ")
    assert b.snapshot()[0].text == "foo bar"


def test_blackboard_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        Blackboard().post("s1", "   ")


def test_blackboard_caps_count():
    b = Blackboard(max_findings=2)
    b.post("s1", "a"); b.post("s1", "b")
    with pytest.raises(ValueError, match="board is full"):
        b.post("s1", "c")
    assert b.rejected == 1


def test_blackboard_is_thread_safe():
    b = Blackboard(max_findings=200)
    def spam(i):
        for j in range(20):
            b.post(f"s{i}", f"finding {i}-{j}")
    ts = [threading.Thread(target=spam, args=(i,)) for i in range(5)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert len(b.snapshot()) == 100      # no lost updates


def test_blackboard_render_empty():
    assert Blackboard.render([]) == ""


# --------------------------------------------------------------------------- tool surfaces
def test_subagent_gets_collab_tools_lead_does_not(cfg):
    lead, _ = build_toolset(cfg, lead=True)
    sub, _ = build_toolset(cfg, subagent=True)
    plain, _ = build_toolset(cfg)
    assert {"ask_lead", "post_finding"} <= set(sub)
    assert "delegate" not in sub
    assert not {"ask_lead", "post_finding"} & set(lead)      # lead has no one to ask
    assert not {"ask_lead", "post_finding", "delegate"} & set(plain)


def test_no_subagent_to_subagent_channel(cfg):
    """Scope guard: escalation only. A direct sibling channel would re-couple the
    contexts that delegation exists to separate."""
    sub, _ = build_toolset(cfg, subagent=True)
    assert not any(n in sub for n in ("send_to_subagent", "message_subagent", "broadcast"))


def test_collab_tools_error_without_hooks(cfg):
    from forge.skillstore import SkillStore
    from forge.tools.base import Context
    from forge.tools.collab import AskLead, PostFinding
    ctx = Context(cfg=cfg, sandbox=Sandbox(cfg.workspace), skills=SkillStore(cfg.skills_dir))
    with pytest.raises(Exception, match="not running as a subagent"):
        AskLead().run(ctx, question="q", why_blocked="w")
    with pytest.raises(Exception, match="not running as a subagent"):
        PostFinding().run(ctx, finding="f")


# --------------------------------------------------------------------------- ask_lead
class Asker:
    """Subagent asks one question, then reports. Lead answers via the one-shot path."""

    def __init__(self, n_questions=1):
        self.n_questions = n_questions
        self.answers_given = 0
        self.answer_prompts = []
        self.report = None

    def call(self, **kw):
        sys = kw.get("system")
        # the one-shot answer path: plain string system prompt, no tools
        if not kw.get("tools") and isinstance(sys, str) and "lead agent" in sys:
            self.answers_given += 1
            self.answer_prompts.append(kw["messages"][0]["content"])
            return Reply([text("Keep backwards compatibility. Ship both v1 and v2 shims.")],
                         "end_turn", Usage(1, 1))
        if _is_lead(kw):
            if _first_turn(kw):
                return Reply([tool_use("delegate", {"subtasks": [
                    {"id": "s1", "task": "port api.py", "writes": ["api.py"]}]}, "d")],
                    "tool_use", Usage(1, 1))
            self.report = kw["messages"][-1]["content"][0]["content"]
            return Reply([text("done")], "end_turn", Usage(1, 1))
        # subagent
        depth = len(kw["messages"])
        if depth <= self.n_questions * 2 - 1:
            return Reply([tool_use("ask_lead", {
                "question": "keep backwards compat?", "why_blocked": "checked the changelog, ambiguous"},
                f"q{depth}")], "tool_use", Usage(1, 1))
        return Reply([text("ported, kept compat per the lead")], "end_turn", Usage(1, 1))


def test_ask_lead_returns_an_answer(cfg):
    llm = Asker()
    res = Agent(cfg, llm, lead=True).run("port the codebase to v2")
    q = [c for c in res.tool_calls if c["name"] == "ask_lead"]
    assert q and not q[0]["is_error"]
    assert "backwards compatibility" in q[0]["output"]
    assert llm.answers_given == 1


def test_ask_lead_sees_the_original_task_not_the_subtask_only(cfg):
    llm = Asker()
    Agent(cfg, llm, lead=True).run("port the codebase to v2")
    prompt = llm.answer_prompts[0]
    assert "port the codebase to v2" in prompt      # lead's original task
    assert "port api.py" in prompt                  # the subtask brief
    assert "checked the changelog" in prompt        # why it's blocked


def test_ask_lead_budget_is_a_fuse(cfg):
    cfg.budget.max_questions_per_subagent = 1
    llm = Asker(n_questions=4)
    res = Agent(cfg, llm, lead=True).run("go")
    qs = [c for c in res.tool_calls if c["name"] == "ask_lead"]
    assert not qs[0]["is_error"]
    assert qs[1]["is_error"] and "budget spent" in qs[1]["output"]
    assert llm.answers_given == 1                   # only the allowed one cost a call


def test_ask_lead_evidence_reaches_the_trail(cfg):
    """A question and its answer are part of the record the evaluator reads."""
    llm = Asker()
    res = Agent(cfg, llm, lead=True).run("port to v2")
    assert "(via s1)" in res.trail_index()
    assert "backwards compat" in res.grep_trail("backwards compat")


# --------------------------------------------------------------------------- findings flow
class Poster:
    """s1 posts a finding; s2 (later batch, RAW dep) should see it in its task text."""

    def __init__(self):
        self.s2_task = None

    def call(self, **kw):
        if _is_lead(kw):
            if _first_turn(kw):
                return Reply([tool_use("delegate", {"subtasks": [
                    {"id": "s1", "task": "probe api. write: api.py", "writes": ["api.py"]},
                    {"id": "s2", "task": "port caller. write: caller.py",
                     "writes": ["caller.py"], "reads": ["api.py"]},
                ]}, "d")], "tool_use", Usage(1, 1))
            return Reply([text("done")], "end_turn", Usage(1, 1))
        task = str(kw["messages"][0]["content"])
        if "port caller" in task:
            self.s2_task = task
            return Reply([text("ported")], "end_turn", Usage(1, 1))
        if _first_turn(kw):
            return Reply([tool_use("post_finding", {"finding": "v2 renamed foo to bar"}, "p")],
                         "tool_use", Usage(1, 1))
        return Reply([text("probed")], "end_turn", Usage(1, 1))


def test_findings_flow_forward_to_later_batches(cfg):
    llm = Poster()
    res = Agent(cfg, llm, lead=True).run("port it")
    assert llm.s2_task is not None
    assert "v2 renamed foo to bar" in llm.s2_task
    assert "(s1)" in llm.s2_task
    assert "findings_from_earlier_subtasks" in llm.s2_task


def test_first_batch_sees_no_findings(cfg):
    """Nothing has run yet — the preamble must not appear at all."""
    seen = {}

    class P:
        def call(self, **kw):
            if _is_lead(kw):
                if _first_turn(kw):
                    return Reply([tool_use("delegate", {"subtasks": [
                        {"id": "s1", "task": "first", "writes": ["a.txt"]}]}, "d")],
                        "tool_use", Usage(1, 1))
                return Reply([text("done")], "end_turn", Usage(1, 1))
            seen["task"] = str(kw["messages"][0]["content"])
            return Reply([text("ok")], "end_turn", Usage(1, 1))

    Agent(cfg, P(), lead=True).run("go")
    assert seen["task"] == "first"
    assert "findings_from_earlier" not in seen["task"]


def test_same_batch_does_not_share_findings(cfg):
    """Scope guard: concurrent subtasks share no paths and must not race on the board.
    A same-batch post is deliberately invisible to its peers."""
    tasks = {}

    class P:
        def call(self, **kw):
            if _is_lead(kw):
                if _first_turn(kw):
                    return Reply([tool_use("delegate", {"subtasks": [
                        {"id": "s1", "task": "alpha. write: a.txt", "writes": ["a.txt"]},
                        {"id": "s2", "task": "beta. write: b.txt", "writes": ["b.txt"]},
                    ]}, "d")], "tool_use", Usage(1, 1))
                return Reply([text("done")], "end_turn", Usage(1, 1))
            t = str(kw["messages"][0]["content"])
            if _first_turn(kw):
                tasks[t.split(".")[0]] = t
                return Reply([tool_use("post_finding", {"finding": f"note from {t.split('.')[0]}"}, "p")],
                             "tool_use", Usage(1, 1))
            return Reply([text("ok")], "end_turn", Usage(1, 1))

    Agent(cfg, P(), lead=True).run("go")
    assert len(tasks) == 2
    for t in tasks.values():
        assert "note from" not in t      # neither saw the other's post


def test_full_board_is_recoverable_not_fatal(cfg):
    cfg.budget.max_findings = 1

    class P:
        def call(self, **kw):
            if _is_lead(kw):
                if _first_turn(kw):
                    return Reply([tool_use("delegate", {"subtasks": [
                        {"id": "s1", "task": "x", "writes": ["a.txt"]}]}, "d")], "tool_use", Usage(1, 1))
                return Reply([text("done")], "end_turn", Usage(1, 1))
            d = len(kw["messages"])
            if d <= 3:
                return Reply([tool_use("post_finding", {"finding": f"note {d}"}, f"p{d}")],
                             "tool_use", Usage(1, 1))
            return Reply([text("ok")], "end_turn", Usage(1, 1))

    res = Agent(cfg, P(), lead=True).run("go")
    posts = [c for c in res.tool_calls if c["name"] == "post_finding"]
    assert not posts[0]["is_error"]
    assert posts[1]["is_error"] and "board is full" in posts[1]["output"]
    assert res.answer == "done"          # lead unaffected

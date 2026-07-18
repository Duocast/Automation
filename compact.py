"""Context window management for long agent runs.

The loop appends an assistant message and a tool-result message every turn. Left alone,
a 40-turn run against a real repo crosses 1M tokens and the API returns a 400 — or you
pay for a 900k-token input on the last turn. This module keeps the *conversation the
model sees* bounded without touching the *audit trail the evaluator judges* (those live
in `calls`/`ctx.audit`, and are never compacted).

Two invariants make this safe:

  1. **Pairing.** An assistant turn containing `tool_use` blocks MUST be followed by a
     user turn whose `tool_result` blocks cover every one of those ids, or the next API
     call is a 400. Compaction therefore operates on whole (assistant, tool_result)
     *exchanges*, never on half of one.

  2. **Anchors.** The first user message (the task) is always kept verbatim — it's the
     ground truth the run is about. Recent exchanges are kept verbatim too, because
     that's the working set. Only the *middle* is summarized.

The summary is produced by a cheap model and inserted as a single user message so the
agent retains the gist ("I already cloned the repo, --upper is confirmed, tests pass")
without carrying every byte of every old tool result.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from .llm import LLM, _battr, _btype

# ~3.5 chars/token is a deliberately conservative estimate for English + code + JSON.
# We would rather compact one exchange too early than blow the window by guessing high.
CHARS_PER_TOKEN = 3.5


def estimate_tokens(messages: list[dict], system: Any = None) -> int:
    """Cheap local estimate. Not exact — we only need to know when to act, and acting
    a little early is free. Avoids a network round-trip to count_tokens every turn."""
    total = _content_chars(system) if system else 0
    for m in messages:
        chars = _content_chars(m.get("content"))
        # Never let a short-but-present message round to zero tokens.
        total += chars + (4 if chars else 0)  # ~role/framing overhead per message
    return int(total / CHARS_PER_TOKEN) or (1 if messages else 0)


def _content_chars(content: Any) -> int:
    if content is None:
        return 0
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        return sum(_block_chars(b) for b in content)
    return len(str(content))


def _block_chars(block: Any) -> int:
    t = _btype(block)
    if t == "text":
        return len(_battr(block, "text") or "")
    if t == "tool_use":
        return len(json.dumps(_battr(block, "input") or {}, default=str)) + 40
    if t == "tool_result":
        c = _battr(block, "content")
        return _content_chars(c) if not isinstance(c, str) else len(c)
    if t == "thinking":
        return len(_battr(block, "thinking") or "")
    # Unknown block: fall back to a serialized guess.
    try:
        return len(json.dumps(block, default=str))
    except TypeError:
        return len(str(block))


def _has_tool_use(msg: dict) -> bool:
    c = msg.get("content")
    return isinstance(c, list) and any(_btype(b) == "tool_use" for b in c)


def _exchange_bounds(messages: list[dict]) -> list[tuple[int, int]]:
    """Group indices into atomic units that must move together.

    Index 0 (the task) is its own unit. After that, an assistant message with tool_use
    binds to the following user (tool_result) message as one exchange. A plain assistant
    turn is its own unit. Returns inclusive (start, end) index pairs.
    """
    units: list[tuple[int, int]] = []
    i = 0
    n = len(messages)
    while i < n:
        if i == 0:
            units.append((0, 0))
            i = 1
            continue
        msg = messages[i]
        if msg.get("role") == "assistant" and _has_tool_use(msg) and i + 1 < n:
            units.append((i, i + 1))  # assistant tool_use + its tool_result
            i += 2
        else:
            units.append((i, i))
            i += 1
    return units


def _summarize_slice(
    llm: LLM,
    model: str,
    messages: list[dict],
    lo: int,
    hi: int,
) -> dict:
    """Compress messages[lo:hi] into one durable user note via a cheap model."""
    transcript_parts: list[str] = []
    for m in messages[lo:hi]:
        role = m.get("role", "?")
        for b in (m.get("content") if isinstance(m.get("content"), list) else [{"type": "text", "text": m.get("content")}]):
            t = _btype(b)
            if t == "text":
                transcript_parts.append(f"[{role}] {(_battr(b,'text') or '')[:1500]}")
            elif t == "tool_use":
                transcript_parts.append(f"[{role} calls {_battr(b,'name')}] {json.dumps(_battr(b,'input') or {}, default=str)[:600]}")
            elif t == "tool_result":
                c = _battr(b, "content")
                s = c if isinstance(c, str) else json.dumps(c, default=str)
                err = " ERROR" if _battr(b, "is_error") else ""
                transcript_parts.append(f"[tool_result{err}] {s[:1200]}")
    transcript = "\n".join(transcript_parts)

    prompt = (
        "Summarize this slice of an agent's working session into a compact note it can "
        "rely on going forward. Preserve concrete, load-bearing facts: files created or "
        "edited, commands run and whether they succeeded, flags confirmed against the "
        "binary, values discovered, errors hit and how they were resolved, and any "
        "decision made. Drop verbose output that has served its purpose. Write in past "
        "tense, first person ('I cloned...', 'I confirmed --upper works'). Be specific; "
        "a vague summary is worse than useless because it reads as progress without "
        "carrying the facts.\n\n"
        f"<slice>\n{transcript}\n</slice>"
    )
    reply = llm.call(
        model=model,
        system="You compress an agent's own working memory. Keep every verified fact; drop the noise.",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000,
        thinking=False,
        stream=False,
    )
    note = reply.text() or "(summary unavailable)"
    return {
        "role": "user",
        "content": [{"type": "text", "text": f"[earlier work, condensed]\n{note}"}],
    }


def compact(
    llm: LLM,
    summarizer_model: str,
    messages: list[dict],
    system: Any,
    max_tokens: int,
    keep_recent_exchanges: int = 4,
    on_event: Callable[[dict], None] | None = None,
) -> list[dict]:
    """Return a shortened message list if over budget, else the original.

    Keeps the task (unit 0) and the last `keep_recent_exchanges` units verbatim;
    replaces the middle with one summary message. Preserves tool_use/tool_result pairing
    by only ever cutting on exchange boundaries.
    """
    before = estimate_tokens(messages, system)
    if before <= max_tokens:
        return messages

    units = _exchange_bounds(messages)
    # Need at least: task + something to summarize + the recent window.
    if len(units) <= keep_recent_exchanges + 2:
        return messages  # too short to compact without gutting the working set

    task_unit = units[0]
    recent = units[-keep_recent_exchanges:]
    middle = units[1:-keep_recent_exchanges]
    if not middle:
        return messages

    lo = middle[0][0]
    hi = middle[-1][1] + 1  # exclusive

    if on_event:
        on_event({"type": "compact", "before_tokens": before, "max": max_tokens,
                  "summarizing_msgs": hi - lo})

    summary_msg = _summarize_slice(llm, summarizer_model, messages, lo, hi)

    kept_head = messages[task_unit[0] : task_unit[1] + 1]
    kept_tail = messages[recent[0][0] :]
    rebuilt = kept_head + [summary_msg] + kept_tail

    after = estimate_tokens(rebuilt, system)
    if on_event:
        on_event({"type": "compact_done", "after_tokens": after,
                  "saved": before - after})
    return rebuilt

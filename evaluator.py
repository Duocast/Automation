"""Evaluator-optimizer loop.

Four things make this a real gate rather than theatre:

1. **Fresh context.** The critic sees the task, the answer, and the evidence — never the
   agent's reasoning. Self-critique grades the reasoning it already committed to.

2. **Evidence on demand, not a truncated dump.** The critic gets an *index* of every tool
   call (bounded, always fits) plus tools to grep the full trail and page through any
   output. A flat truncated trail elides the middle of a build log, which is exactly
   where the error lives — a critic handed that is structurally unable to catch the
   failure it exists to catch.

3. **The critic is itself checked.** Its `unsupported_claims` are adjudicated by an
   independent pass that sees the claim and the evidence but not the critic's reasoning
   or the answer's framing. Claims that don't survive are dropped before they reach the
   agent. A false accusation is expensive: it burns a retry round and tells the agent to
   prove something it already proved, which invites thrash.

4. **Coherence guards.** "revise" with nothing actionable is not a verdict. A reviewer
   who never passes is a broken reviewer, and that is now enforced mechanically rather
   than merely requested in the prompt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .config import Config
from .llm import LLM, _battr
from .loop import Agent, Result
from .prompts import EVALUATOR_SYSTEM

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["pass", "revise", "fail"],
            "description": "pass = shippable. revise = fixable, say exactly how. fail = wrong approach, start over.",
        },
        "score": {"type": "integer", "minimum": 1, "maximum": 10},
        "unsupported_claims": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Claims in the answer contradicted by, or absent from, the evidence. "
                "Quote the claim, then say what the evidence shows instead and which tool "
                "index you checked. Only list a claim after you have actually looked — "
                "grep_trail returning nothing is evidence; not having looked is not."
            ),
        },
        "critique": {"type": "string", "description": "2-5 sentences. What is actually wrong, or why it passes."},
        "required_fixes": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Specific, actionable. Name files, flags, commands. Empty if verdict is pass.",
        },
    },
    "required": ["verdict", "score", "critique", "required_fixes", "unsupported_claims"],
}

SUBMIT_TOOL = {
    "name": "submit_verdict",
    "description": "Submit your final verdict. Call this once, after you have checked the evidence.",
    "input_schema": VERDICT_SCHEMA,
}

EVAL_PROMPT = """\
<task>
{task}
</task>

<agent_answer>
{answer}
</agent_answer>

<evidence_index>
Every tool call the agent made, in order. This is an index, not the contents — outputs
are listed by size and first line only. Use grep_trail and read_tool_output to see what
any of them actually contain.
{index}
</evidence_index>

<run_metadata>
turns: {turns} | stop: {stop} | tool calls: {n_calls} | errored calls: {n_err}
</run_metadata>

The answer above is a *claim* about the evidence. Check it.

Work the evidence before judging. Concretely: for each factual assertion in the answer
("the build passes", "I confirmed --upper", "all tests green"), find the tool call that
would show it and look. grep_trail is the fast path — search for the exit code, the
error string, the flag. Large outputs hide their important lines in the middle, so do
not assume a call succeeded because its first line looks fine.

Then call submit_verdict.
"""


@dataclass
class Verdict:
    verdict: str
    score: int
    critique: str
    required_fixes: list[str]
    unsupported_claims: list[str]
    #: Claims the critic raised that adjudication could not substantiate. Dropped
    #: before reaching the agent; kept here for visibility into critic quality.
    dropped_claims: list[str] = field(default_factory=list)
    #: Set when a coherence guard overrode the critic's stated verdict.
    override: str | None = None
    #: Anomalies from adjudication (bad ids, conflicts, missing rulings). Surfaced
    #: rather than swallowed: a quietly malfunctioning adjudicator looks exactly like
    #: a well-behaved one that found nothing.
    adjudication_notes: list[str] = field(default_factory=list)
    checks: int = 0  # evidence tool calls the critic made

    @property
    def passed(self) -> bool:
        return self.verdict == "pass"


@dataclass
class Outcome:
    result: Result
    verdict: Verdict | None
    rounds: int
    history: list[tuple[Result, Verdict]]


# --------------------------------------------------------------------------- evidence tools
class TrailTools:
    """Read-only access to the full execution record. The critic's only tools."""

    def __init__(self, result: Result):
        self.result = result
        self.calls = 0

    def specs(self) -> list[dict]:
        n = len(self.result.tool_calls)
        return [
            {
                "name": "grep_trail",
                "description": (
                    "Regex search across every tool call's input and FULL output. Nothing is "
                    "elided. This is how you verify a claim: search for the exit code, error "
                    "text, flag, or filename it depends on."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {"pattern": {"type": "string", "description": "Regex, case-insensitive."}},
                    "required": ["pattern"],
                },
            },
            {
                "name": "read_tool_output",
                "description": (
                    f"Read the complete output of one tool call by index (0..{max(0, n - 1)}). "
                    "Paginated; large outputs return a range and tell you how to get the rest."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "offset": {"type": "integer", "default": 0},
                        "limit": {"type": "integer", "default": 8000},
                    },
                    "required": ["index"],
                },
            },
        ]

    def dispatch(self, name: str, inp: dict) -> tuple[str, bool]:
        self.calls += 1
        try:
            if name == "grep_trail":
                return self.result.grep_trail(str(inp.get("pattern", ""))), False
            if name == "read_tool_output":
                return self.result.tool_output(
                    int(inp.get("index", 0)),
                    int(inp.get("offset", 0) or 0),
                    int(inp.get("limit", 8000) or 8000),
                ), False
            return f"unknown tool {name!r}", True
        except (IndexError, ValueError) as e:
            return str(e), True
        except Exception as e:  # noqa: BLE001
            return f"{type(e).__name__}: {e}", True


def _investigate(
    cfg: Config,
    llm: LLM,
    system: str,
    prompt: str,
    submit_tool: dict,
    result: Result,
    max_tokens: int = 8_000,
) -> tuple[dict, int]:
    """Shared loop: let a judge pull evidence, then submit via `submit_tool`.

    Used by both the critic and the adjudicator — they differ only in system prompt,
    question, and submission schema.
    """
    tools = TrailTools(result)
    messages: list[dict] = [{"role": "user", "content": prompt}]
    specs = tools.specs() + [submit_tool]
    name = submit_tool["name"]

    for _ in range(cfg.critic_max_turns):
        reply = llm.call(
            model=cfg.evaluator_model,
            system=system,
            messages=messages,
            tools=specs,
            max_tokens=max_tokens,
            effort="high",
        )
        uses = reply.tool_uses()
        if not uses:
            # Wrote prose instead of calling anything; nudge it to finish.
            messages.append({"role": "assistant", "content": reply.content})
            messages.append({"role": "user", "content": f"Call {name} now."})
            continue

        for u in uses:
            if _battr(u, "name") == name:
                return (_battr(u, "input") or {}), tools.calls

        messages.append({"role": "assistant", "content": reply.content})
        results = []
        for u in uses:
            out, err = tools.dispatch(_battr(u, "name"), _battr(u, "input") or {})
            results.append({
                "type": "tool_result", "tool_use_id": _battr(u, "id"),
                "content": out, "is_error": err,
            })
        messages.append({"role": "user", "content": results})

    # Out of turns without submitting — force it rather than defaulting to a guess.
    forced = llm.call(
        model=cfg.evaluator_model,
        system=system,
        messages=messages + [{"role": "user", "content": f"Submit now with {name}."}],
        tools=[submit_tool],
        tool_choice={"type": "tool", "name": name},
        max_tokens=4_000,
        stream=False,
    )
    return (forced.first_tool_input() or {}), tools.calls


def _run_critic(cfg: Config, llm: LLM, task: str, result: Result) -> tuple[dict, int]:
    n_err = sum(1 for c in result.tool_calls if c.get("is_error"))
    prompt = EVAL_PROMPT.format(
        task=task,
        answer=result.answer or "(the agent produced no final answer)",
        index=result.trail_index(),
        turns=result.turns,
        stop=result.stop,
        n_calls=len(result.tool_calls),
        n_err=n_err,
    )
    return _investigate(cfg, llm, EVALUATOR_SYSTEM, prompt, SUBMIT_TOOL, result)


# --------------------------------------------------------------------------- adjudication
ADJUDICATOR_SYSTEM = """\
You settle one narrow question: does the execution log support a claim, or not?

You are given claims that a reviewer flagged as unsupported. The reviewer may be wrong —
that is why you exist. You cannot see the reviewer's reasoning or the original answer's
framing, only the claim and the log, and that independence is the point.

For each claim, search the log before deciding. Judge only whether evidence exists:
- "supported"   — the log shows it. The reviewer was mistaken; the claim survives.
- "unsupported" — you searched and the log contradicts it, or holds nothing relevant.
- "unclear"     — genuinely ambiguous.

Do not defer to the reviewer, and do not reflexively contradict it either. Absence of
evidence after an honest search IS unsupported; not having looked is not.
"""

ADJUDICATE_SCHEMA = {
    "type": "object",
    "properties": {
        "rulings": {
            "type": "array",
            "description": "Exactly one ruling per numbered claim.",
            "items": {
                "type": "object",
                "properties": {
                    "claim_id": {
                        "type": "integer",
                        "description": "The claim's number from the list you were given (1-based).",
                    },
                    "ruling": {"type": "string", "enum": ["supported", "unsupported", "unclear"]},
                    "evidence": {"type": "string", "description": "Where you looked and what you found."},
                },
                "required": ["claim_id", "ruling", "evidence"],
            },
        },
    },
    "required": ["rulings"],
}

SUBMIT_RULINGS = {
    "name": "submit_rulings",
    "description": "Submit a ruling for every claim. Call once, after searching.",
    "input_schema": ADJUDICATE_SCHEMA,
}

ADJUDICATE_PROMPT = """\
A reviewer flagged these claims as unsupported by the execution log:

{claims}

<evidence_index>
{index}
</evidence_index>

Search the log with grep_trail and read_tool_output, then rule on each claim with
submit_rulings. Check every claim independently; the reviewer flagged them, that does
not make them right.

Refer to each claim by its number above (`claim_id`). Return one ruling per claim — if
you cannot decide on one, rule it "unclear" rather than omitting it.
"""

VALID_RULINGS = {"supported", "unsupported", "unclear"}


def _norm_text(s: str) -> str:
    return " ".join(str(s).lower().split()).strip("'\"`")


def _resolve_claim(r: dict, claims: list[str], notes: list[str]) -> str | None:
    """Map a ruling back to the claim it is about — by id, never by fuzzy text.

    Text round-tripping was a real bug: an adjudicator that echoed a claim with one
    letter's case changed would fall through to substring matching and land its ruling
    on a *different* claim whose text happened to be a prefix. Rulings transposed
    silently, and which claim got dropped depended on dict insertion order. Ids make the
    mapping total and checkable.

    The text path survives only as a fallback for a model that ignores the schema, and
    it is exact-match (modulo case/whitespace/quotes) on purpose. Containment matching is
    what caused the transposition and is never coming back.
    """
    raw_id = r.get("claim_id")
    if raw_id is not None:
        try:
            i = int(raw_id)
        except (TypeError, ValueError):
            notes.append(f"discarded ruling: claim_id {raw_id!r} is not an integer")
            return None
        if 1 <= i <= len(claims):
            return claims[i - 1]
        notes.append(f"discarded ruling: claim_id {i} out of range 1..{len(claims)}")
        return None

    txt = str(r.get("claim") or "")
    if txt:
        norm = _norm_text(txt)
        for c in claims:
            if _norm_text(c) == norm:
                return c
    notes.append(f"discarded ruling: no usable claim_id and text matched nothing: {txt[:60]!r}")
    return None


def _apply_rulings(raw: dict, claims: list[str]) -> tuple[dict[str, str], list[str]]:
    """Turn the adjudicator's output into claim -> ruling, refusing anything ambiguous.

    Every rejection path leaves the claim in place. That asymmetry is deliberate: a
    surviving false accusation costs one retry round, a silently dropped true accusation
    ships broken work.
    """
    notes: list[str] = []
    rulings: dict[str, str] = {}
    conflicting: set[str] = set()

    for r in raw.get("rulings") or []:
        if not isinstance(r, dict):
            notes.append(f"discarded ruling: not an object ({type(r).__name__})")
            continue
        ruling = str(r.get("ruling") or "").strip().lower()
        if ruling not in VALID_RULINGS:
            notes.append(f"discarded ruling: unknown verdict {ruling!r}")
            continue
        c = _resolve_claim(r, claims, notes)
        if c is None:
            continue
        prev = rulings.get(c)
        if prev is not None and prev != ruling:
            conflicting.add(c)
            notes.append(
                f"conflicting rulings for claim {claims.index(c) + 1} ({prev} vs {ruling}) "
                "— distrusting both, claim stands"
            )
            continue
        rulings[c] = ruling

    for c in conflicting:
        rulings.pop(c, None)

    if missing := [i + 1 for i, c in enumerate(claims) if c not in rulings]:
        notes.append(f"no ruling returned for claim(s) {missing} — they stand unchallenged")
    return rulings, notes


def _adjudicate(
    cfg: Config, llm: LLM, claims: list[str], result: Result
) -> tuple[dict[str, str], list[str]]:
    """Independently check each flagged claim. Returns (claim -> ruling, anomaly notes).

    One loop for all claims rather than one per claim: same independence, a fraction
    of the cost.
    """
    numbered = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(claims))
    raw, _ = _investigate(
        cfg, llm, ADJUDICATOR_SYSTEM,
        ADJUDICATE_PROMPT.format(claims=numbered, index=result.trail_index()),
        SUBMIT_RULINGS, result, max_tokens=6_000,
    )
    return _apply_rulings(raw, claims)


# --------------------------------------------------------------------------- public
def evaluate(cfg: Config, llm: LLM, task: str, result: Result) -> Verdict:
    raw, checks = _run_critic(cfg, llm, task, result)

    # The critic's output crosses a trust boundary; a junk score must not crash solve().
    try:
        score = int(raw.get("score", 5) or 5)
    except (TypeError, ValueError):
        score = 5
    v = Verdict(
        verdict=str(raw.get("verdict", "revise") or "revise"),
        score=min(10, max(1, score)),
        critique=raw.get("critique", ""),
        required_fixes=[str(f) for f in (raw.get("required_fixes") or []) if str(f).strip()],
        unsupported_claims=[str(c) for c in (raw.get("unsupported_claims") or []) if str(c).strip()],
        checks=checks,
    )

    # --- spot-check the critic before its accusations reach the agent.
    if v.unsupported_claims and cfg.adjudicate:
        rulings, v.adjudication_notes = _adjudicate(cfg, llm, v.unsupported_claims, result)
        kept, dropped = [], []
        for c in v.unsupported_claims:
            # Only an explicit "supported" clears a claim. "unclear" and missing rulings
            # keep it: the critic looked and the adjudicator did not overturn it, so the
            # benefit of the doubt goes to raising the concern, not burying it.
            (dropped if rulings.get(c) == "supported" else kept).append(c)
        v.unsupported_claims, v.dropped_claims = kept, dropped

    # --- coherence guards. A verdict must carry substance to send work back.
    if v.verdict in {"revise", "fail"} and not v.required_fixes and not v.unsupported_claims:
        v.override = (
            "every flagged claim failed adjudication and no other fixes were given"
            if v.dropped_claims
            else "critic returned 'revise' with no actionable fixes and no claims"
        )
        v.verdict = "pass"
        v.score = max(v.score, cfg.pass_score)

    return v


REVISION = """\
Your previous attempt was reviewed and sent back. The reviewer saw your final answer
and the log of what you actually ran — not your reasoning.

Reviewer score: {score}/10
Critique: {critique}

{unsupported}Required fixes:
{fixes}

Address each fix by doing the work, not by rewording the answer. If a fix rests on a
misreading, say so plainly and show the evidence. Your workspace is unchanged from
your last attempt, so build on it rather than starting over.
"""

UNSUPPORTED = """\
Claims the reviewer could not find support for in your execution log. Each was
independently double-checked against the full log before being put to you, so treat
them as real. Either prove each one by running something that demonstrates it, or
retract it:
{items}

"""


def solve(
    cfg: Config,
    llm: LLM,
    task: str,
    on_event: Callable[[dict], None] | None = None,
) -> Outcome:
    """Run the agent, critique, retry with the critique. Return the best attempt."""
    agent = Agent(cfg, llm, on_event=on_event)
    history: list[tuple[Result, Verdict]] = []
    prompt = task

    for rnd in range(1, cfg.max_eval_rounds + 1):
        effort = cfg.effort if rnd == 1 else cfg.retry_effort
        result = agent.run(prompt, effort=effort)

        if cfg.max_eval_rounds <= 1:
            return Outcome(result, None, rnd, history)

        v = evaluate(cfg, llm, task, result)
        history.append((result, v))
        if on_event:
            on_event({"type": "verdict", "round": rnd, "verdict": v.verdict,
                      "score": v.score, "critique": v.critique, "checks": v.checks,
                      "dropped": len(v.dropped_claims), "override": v.override,
                      "adjudication_notes": v.adjudication_notes})

        if v.passed or v.score >= cfg.pass_score:
            return Outcome(result, v, rnd, history)
        if rnd == cfg.max_eval_rounds:
            break

        unsup = ""
        if v.unsupported_claims:
            unsup = UNSUPPORTED.format(items="\n".join(f"  - {c}" for c in v.unsupported_claims))
        prompt = (
            f"{task}\n\n---\n"
            + REVISION.format(
                score=v.score,
                critique=v.critique,
                unsupported=unsup,
                fixes="\n".join(f"  {i}. {f}" for i, f in enumerate(v.required_fixes, 1)) or "  (none given)",
            )
        )

    best_result, best_v = max(history, key=lambda h: h[1].score)
    return Outcome(best_result, best_v, len(history), history)

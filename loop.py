"""The agentic loop.

Written by hand rather than using the SDK's tool_runner because we need the seams the
runner hides: approval gating on hard-to-reverse calls, per-attempt budgets, an audit
trail the evaluator can read, and pause_turn handling for server-side tools.
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace as dc_replace
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .compact import compact, estimate_tokens
from .config import Config
from .llm import LLM, Reply, Usage, _battr, _btype
from .orchestrate import Blackboard, Finding, Subtask, batches, explain, validate
from .prompts import LEAD_ANSWER_PROMPT, LEAD_ANSWER_SYSTEM, agent_system
from .sandbox import Sandbox
from .skillstore import SkillStore
from .tools import Context, Tool, ToolError, build_toolset

# Executed by Anthropic's servers; never dispatch these locally.
SERVER_TOOLS = {"web_search", "web_fetch"}


@dataclass
class Result:
    answer: str
    turns: int
    usage: Usage
    audit: list[dict]
    stop: str
    tool_calls: list[dict] = field(default_factory=list)

    def trail(self, limit: int = 12_000) -> str:
        """A truncated narrative of what the agent did.

        Lossy by construction. Fine for priming a drafting step; NOT fine as the sole
        evidence for judging claims — the elided middle is exactly where compiler errors
        and non-zero exits live. The evaluator uses trail_index + grep_trail + tool_output
        instead, which reach the full record.
        """
        if not self.tool_calls:
            return "(no tools were called)"
        lines = []
        for c in self.tool_calls:
            head = f"[{c['name']}] {json.dumps(c['input'], default=str)[:400]}"
            out = str(c["output"]).strip()
            if len(out) > 900:
                out = out[:600] + "\n  ...[elided]...\n" + out[-250:]
            flag = " !ERROR" if c.get("is_error") else ""
            lines.append(f"{head}{flag}\n  -> {out}")
        t = "\n\n".join(lines)
        if len(t) > limit:
            t = t[:limit] + "\n... [trail truncated]"
        return t

    # --- full-fidelity access, for the evaluator ---------------------------
    def trail_index(self) -> str:
        """One line per tool call. Bounded, so it always fits in a prompt.

        This is a map, not the territory: it tells the critic what exists and how big
        it is, so it can decide what to actually pull.
        """
        if not self.tool_calls:
            return "(no tools were called)"
        lines = []
        for i, c in enumerate(self.tool_calls):
            inp = json.dumps(c["input"], default=str)
            if len(inp) > 150:
                inp = inp[:150] + "…"
            out = str(c["output"])
            first = next((ln for ln in out.strip().splitlines() if ln.strip()), "(empty)")
            flag = "  !ERROR" if c.get("is_error") else ""
            who = f" (via {c['subagent']})" if c.get("subagent") else ""
            lines.append(f"[{i}] {c['name']}{who}{flag} {inp}\n     {len(out):,} chars | starts: {first[:110]}")
        return "\n".join(lines)

    def tool_output(self, index: int, offset: int = 0, limit: int = 8000) -> str:
        """Full, untruncated output of one tool call, paginated."""
        if not 0 <= index < len(self.tool_calls):
            raise IndexError(f"no tool call at index {index}; valid range 0..{len(self.tool_calls) - 1}")
        c = self.tool_calls[index]
        out = str(c["output"])
        chunk = out[offset : offset + limit]
        header = (f"[{index}] {c['name']}({json.dumps(c['input'], default=str)[:200]})"
                  f"{'  !ERROR' if c.get('is_error') else ''}\n"
                  f"chars {offset}-{min(offset + limit, len(out))} of {len(out):,}\n---\n")
        more = ""
        if offset + limit < len(out):
            more = f"\n---\n[{len(out) - offset - limit:,} more chars; re-read with offset={offset + limit}]"
        return header + chunk + more

    def grep_trail(self, pattern: str, max_results: int = 40) -> str:
        """Regex across every tool call's input and full output. Nothing is elided here."""
        try:
            rx = re.compile(pattern, re.I)
        except re.error as e:
            raise ValueError(f"bad regex: {e}") from e
        hits: list[str] = []
        for i, c in enumerate(self.tool_calls):
            body = f"{json.dumps(c['input'], default=str)}\n{c['output']}"
            for ln_no, line in enumerate(str(body).splitlines(), 1):
                if rx.search(line):
                    hits.append(f"[{i}] {c['name']} line {ln_no}: {line.strip()[:180]}")
                    if len(hits) >= max_results:
                        return "\n".join(hits) + f"\n... [stopped at {max_results} matches]"
        return "\n".join(hits) if hits else f"no matches for {pattern!r} anywhere in the trail"


class Denied(Exception):
    pass


def approve_always(_name: str, _inp: dict) -> bool:
    return True


def approve_never(_name: str, _inp: dict) -> bool:
    return False


def approve_interactive(name: str, inp: dict) -> bool:
    print(f"\n  [approval] {name}({json.dumps(inp, default=str)[:300]})")
    return input("  allow? [y/N] ").strip().lower() in {"y", "yes"}


class Agent:
    def __init__(
        self,
        cfg: Config,
        llm: LLM,
        on_event: Callable[[dict], None] | None = None,
        lead: bool = False,
        write_allowlist: list[Path] | None = None,
        subagent: bool = False,
        ask_lead: Callable[[str, str], str] | None = None,
        post_finding: Callable[[str], str] | None = None,
    ):
        self.cfg = cfg
        self.llm = llm
        self.on_event = on_event or (lambda e: None)
        self.is_lead = lead
        self.is_subagent = subagent
        self._ask_lead = ask_lead
        self._post_finding = post_finding
        self._spawned = 0

        read_roots = [p for p in (cfg.uploads_dir, cfg.skills_dir) if p]
        if cfg.target_exe:
            read_roots.append(cfg.target_exe.parent)
        self.sandbox = Sandbox(cfg.workspace, read_roots, cfg.allow_network_in_shell,
                               cfg.cmd_timeout, write_allowlist=write_allowlist)
        self.skills = SkillStore(cfg.skills_dir)
        self.tools, self.specs = build_toolset(cfg, lead=lead, subagent=subagent)
        self.approver = {
            "auto": approve_always,
            "deny": approve_never,
            "prompt": approve_interactive,
        }[cfg.approval]

    # --- public ------------------------------------------------------------
    def run(self, task: str, effort: str | None = None) -> Result:
        ctx = Context(cfg=self.cfg, sandbox=self.sandbox, skills=self.skills)
        messages: list[dict] = [{"role": "user", "content": task}]
        system = [{
            "type": "text",
            "text": agent_system(self.cfg, self.skills.index(), lead=self.is_lead,
                                 subagent=self.is_subagent),
            # The system prompt + tool defs are stable across every turn; cache them.
            "cache_control": {"type": "ephemeral"},
        }]

        usage = Usage()
        calls: list[dict] = []
        # Bind the fan-out hook to this run's call list so subagent evidence lands in
        # the same trail the evaluator reads.
        if self.is_lead:
            ctx.spawn = lambda subtasks: self._spawn(subtasks, calls, usage, task)
        ctx.ask_lead = self._ask_lead
        ctx.post_finding = self._post_finding
        deadline = time.time() + self.cfg.budget.max_wall_seconds
        stop = "end_turn"
        reply: Reply | None = None
        turns = 0

        while True:
            if turns >= self.cfg.budget.max_turns:
                stop = "budget:turns"
                break
            if time.time() > deadline:
                stop = "budget:time"
                break

            turns += 1

            # Keep the window the model sees under budget. The audit trail (`calls`,
            # `ctx.audit`) is untouched — the evaluator still judges the full record.
            messages = compact(
                self.llm,
                self.cfg.utility_model,
                messages,
                system,
                self.cfg.budget.max_input_tokens,
                on_event=self._log,
            )

            reply = self.llm.call(
                model=self.cfg.agent_model,
                system=system,
                messages=messages,
                tools=self.specs,
                max_tokens=self.cfg.max_tokens,
                effort=effort or self.cfg.effort,
            )
            usage.add(reply.usage)
            self._log({"type": "turn", "n": turns, "stop_reason": reply.stop_reason,
                       "in": reply.usage.input_tokens, "out": reply.usage.output_tokens})

            if reply.stop_reason == "refusal":
                stop = "refusal"
                break

            messages.append({"role": "assistant", "content": reply.content})

            # A server-side tool hit its internal iteration cap. Re-send to continue;
            # no local work to do.
            if reply.stop_reason == "pause_turn":
                continue

            if reply.stop_reason == "max_tokens":
                stop = "max_tokens"
                break

            uses = [b for b in reply.tool_uses() if _battr(b, "name") not in SERVER_TOOLS]
            if not uses:
                stop = reply.stop_reason or "end_turn"
                break

            results = self._dispatch(ctx, uses, calls)
            messages.append({"role": "user", "content": results})

        answer = reply.text() if reply else ""
        if stop.startswith("budget:"):
            answer = (answer or "") + f"\n\n[halted: {stop}]"
        return Result(answer, turns, usage, ctx.audit, stop, calls)

    # --- delegation --------------------------------------------------------
    def _spawn(self, raw: list, calls: list[dict], usage: Usage, lead_task: str) -> str:
        """Fan work out to subagents, merge their evidence back, return their reports.

        Two things here are load-bearing beyond the obvious:

        - **Trail merge.** Every subagent tool call is appended to the lead's `calls`,
          tagged with the subtask id. Without this the evaluator would read the lead's
          answer ("I ported all nine files"), find no tool call that did it, and flag
          the whole thing as fabricated. Delegated work has to leave the same evidence
          as work done in-line.
        - **Batching.** Subtasks that declare overlapping paths never run concurrently.
          See orchestrate.py for the hazard rules.
        """
        subtasks = validate(raw, self.cfg.budget.max_subtasks_per_call)

        remaining = self.cfg.budget.max_subagents - self._spawned
        if len(subtasks) > remaining:
            raise ValueError(
                f"budget: {len(subtasks)} subtasks requested but only {remaining} subagent(s) "
                f"left of {self.cfg.budget.max_subagents}. Do the rest yourself or ask for fewer."
            )

        groups = batches(subtasks)
        notes = explain(subtasks)
        self._log({"type": "delegate", "subtasks": len(subtasks), "batches": len(groups),
                   "serialized": len(notes)})

        board = Blackboard(self.cfg.budget.max_findings)
        reports: list[str] = []
        for gi, group in enumerate(groups):
            # Snapshot before the batch starts: everyone in it sees the same findings,
            # from completed batches only. Same-batch posts would be a race, and
            # same-batch subtasks share no paths anyway.
            visible = board.snapshot()
            if len(group) > 1:
                with ThreadPoolExecutor(max_workers=min(6, len(group))) as ex:
                    done = list(ex.map(lambda st: self._run_sub(st, visible, board, lead_task), group))
            else:
                done = [self._run_sub(group[0], visible, board, lead_task)]

            # Merge sequentially in the parent thread — no concurrent mutation of `calls`.
            for st, res in done:
                self._spawned += 1
                usage.add(res.usage)
                for c in res.tool_calls:
                    calls.append({**c, "subagent": st.id})
                reports.append(_render_report(st, res, gi))
        return _render_scheduling(groups, notes) + "\n\n" + "\n\n".join(reports)

    def _run_sub(
        self, st: Subtask, visible: list[Finding], board: Blackboard, lead_task: str
    ) -> tuple[Subtask, Result]:
        """One subagent: fresh context, confined writes, its own smaller budget."""
        scratch = self.cfg.workspace / ".scratch" / st.id
        scratch.mkdir(parents=True, exist_ok=True)
        # Declared writes, plus a private scratch dir so it always has somewhere to put
        # intermediates without having to declare them.
        allow = [self.cfg.workspace / w for w in st.writes] + [scratch]

        # Question budget is per-subagent and only ever touched by this thread, so it
        # needs no lock. The board does; it has its own.
        asked = 0

        def ask(question: str, why: str) -> str:
            nonlocal asked
            if asked >= self.cfg.budget.max_questions_per_subagent:
                raise ValueError(
                    f"question budget spent ({self.cfg.budget.max_questions_per_subagent}). "
                    "Decide it yourself, and flag the assumption in your report."
                )
            asked += 1
            return self._answer_question(lead_task, st, question, why)

        sub_cfg = dc_replace(
            self.cfg,
            budget=dc_replace(self.cfg.budget, max_turns=self.cfg.budget.subagent_max_turns),
        )
        sub = Agent(sub_cfg, self.llm, on_event=self.on_event, lead=False,
                    write_allowlist=allow, subagent=True,
                    ask_lead=ask, post_finding=lambda t: board.post(st.id, t))
        self._log({"type": "subagent_start", "id": st.id, "task": st.task[:160]})
        try:
            res = sub.run(_with_findings(st.task, visible))
        except Exception as e:  # noqa: BLE001 - a failed subagent must not kill the lead
            res = Result(answer=f"[subagent crashed: {type(e).__name__}: {e}]",
                         turns=0, usage=Usage(), audit=[], stop="error", tool_calls=[])
        self._log({"type": "subagent_done", "id": st.id, "stop": res.stop,
                   "calls": len(res.tool_calls), "asked": asked})
        return st, res

    def _answer_question(self, lead_task: str, st: Subtask, question: str, why: str) -> str:
        """Answer a blocked subagent in one shot.

        Deliberately NOT a re-entry into the lead's own loop. The lead is suspended
        inside its `delegate` tool call; resuming its conversation from in here would be
        re-entrant, would need locking against concurrent askers, and could recurse. A
        single stateless call over the original task answers scope questions — which is
        what these are — without any of that.
        """
        self._log({"type": "question", "id": st.id, "question": question[:200]})
        reply = self.llm.call(
            model=self.cfg.agent_model,
            system=LEAD_ANSWER_SYSTEM,
            messages=[{"role": "user", "content": LEAD_ANSWER_PROMPT.format(
                task=lead_task, sid=st.id, subtask=st.task, question=question, why=why)}],
            max_tokens=1_000,
            stream=False,
        )
        answer = reply.text() or "(the lead gave no answer — use your judgement and flag it)"
        self._log({"type": "answer", "id": st.id, "answer": answer[:200]})
        return f"The lead answered:\n\n{answer}"

    # --- internals ---------------------------------------------------------
    def _dispatch(self, ctx: Context, uses: list[Any], calls: list[dict]) -> list[dict]:
        """Execute one turn's tool calls. Fan out read-only ones; serialise the rest.

        The harness can only know a call is parallel-safe because it's a dedicated,
        typed tool. This is the concrete payoff of not routing everything through bash.
        """
        all_safe = all(self._safe(_battr(u, "name")) for u in uses)
        if all_safe and len(uses) > 1:
            with ThreadPoolExecutor(max_workers=min(8, len(uses))) as ex:
                outs = list(ex.map(lambda u: self._one(ctx, u, calls), uses))
        else:
            outs = [self._one(ctx, u, calls) for u in uses]
        return outs

    def _safe(self, name: str) -> bool:
        t = self.tools.get(name)
        return bool(t and t.parallel_safe)

    def _one(self, ctx: Context, use: Any, calls: list[dict]) -> dict:
        name = _battr(use, "name")
        inp = _battr(use, "input") or {}
        uid = _battr(use, "id")
        tool: Tool | None = self.tools.get(name)

        if tool is None:
            out, err = f"unknown tool {name!r}", True
        else:
            try:
                if tool.gated and not self.approver(name, inp):
                    raise Denied(f"{name} was denied by the approval policy. Do not retry it; "
                                 "find another approach or explain what you need.")
                out, err = tool.run(ctx, **inp), False
            except Denied as e:
                out, err = str(e), True
            except ToolError as e:
                # Expected, recoverable — hand it back so the model can self-correct.
                out, err = str(e), True
            except TypeError as e:
                out, err = f"bad arguments for {name}: {e}", True
            except Exception as e:  # noqa: BLE001 - never kill the loop on a tool bug
                out, err = f"{type(e).__name__}: {e}", True

        cap = self.cfg.budget.max_tool_output_chars
        if len(out) > cap:
            out = out[:cap] + f"\n... [truncated at {cap} chars]"

        calls.append({"name": name, "input": inp, "output": out, "is_error": err})
        self._log({"type": "tool", "name": name, "is_error": err,
                   "input": inp, "output": out[:2000]})
        return {"type": "tool_result", "tool_use_id": uid, "content": out, "is_error": err}

    def _log(self, event: dict) -> None:
        event["t"] = time.time()
        self.on_event(event)
        p: Path | None = self.cfg.transcript_path
        if p:
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, default=str) + "\n")


FINDINGS_PREAMBLE = """\
<findings_from_earlier_subtasks>
Facts established by subtasks that already finished. Treat them as true; they were
established by an agent with tools, not guessed. If one contradicts what you observe,
trust your own observation and say so in your report.
{findings}
</findings_from_earlier_subtasks>

"""


def _with_findings(task: str, visible: list[Finding]) -> str:
    if not visible:
        return task
    return FINDINGS_PREAMBLE.format(findings=Blackboard.render(visible)) + task


def _render_scheduling(groups: list[list[Subtask]], notes: list[str]) -> str:
    lines = [f"scheduling: {sum(len(g) for g in groups)} subtask(s) in {len(groups)} batch(es)"]
    for i, g in enumerate(groups):
        how = "parallel" if len(g) > 1 else "alone"
        lines.append(f"  batch {i + 1} ({how}): {', '.join(s.id for s in g)}")
    lines.extend(f"  {n}" for n in notes)
    return "\n".join(lines)


def _render_report(st: Subtask, res: Result, batch: int) -> str:
    """Compact. The lead gets conclusions, not transcripts — that's the whole point."""
    n_err = sum(1 for c in res.tool_calls if c.get("is_error"))
    health = "ok" if res.stop in {"end_turn", "tool_use"} else f"**{res.stop}**"
    head = (f"### {st.id} — batch {batch + 1} — {health}\n"
            f"turns: {res.turns} | tool calls: {len(res.tool_calls)} ({n_err} errored) | "
            f"declared writes: {', '.join(st.writes) or '(none)'}")
    body = (res.answer or "(no report returned)").strip()
    if len(body) > 4000:
        body = body[:4000] + "\n... [subagent report truncated]"
    warn = ""
    if res.stop.startswith("budget:"):
        warn = ("\n\n> This subagent hit its budget and may be incomplete. Verify before "
                "relying on it.")
    return f"{head}\n\n{body}{warn}"

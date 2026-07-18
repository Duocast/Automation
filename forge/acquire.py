"""Learn to drive an unknown binary from its source repository.

The naive version of this is "read the README, write notes". That produces confident
fiction: READMEs document intent and drift from the shipped binary, and a model asked
to summarise one will happily invent a plausible `--output` flag that does not exist.
A future agent then trusts it.

So the pipeline closes the loop against ground truth:

    RECON   explore the repo AND probe the binary (--help, subcommands, errors)
    DRAFT   emit a structured skill, every example a real executable command
    VERIFY  actually run every example; compare exit codes to what was claimed
    REPAIR  hand the failures back; the model corrects or retracts
    COMMIT  write SKILL.md, stamped with what passed

An example that fails verification twice is deleted, not shipped. The skill records
its own verification rate in frontmatter so a future reader can calibrate trust.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Callable

from .config import Config
from .llm import LLM, structured
from .loop import Agent, Result
from .prompts import SKILL_AUTHOR_SYSTEM
from .sandbox import Sandbox
from .skillstore import SkillStore

MAX_REPAIRS = 2

RECON_TASK = """\
Learn how to actually use the binary at {exe}, well enough to write an operating
manual another engineer could follow without touching the repo.

Do all of this:
1. fetch_repo {repo} — then read the README and, importantly, the argument-parser
   source (grep for argparse/clap/cobra/commander/getopt). The parser is what the
   binary really accepts; prose drifts.
2. probe_exe with '--help', then '-h', then 'help' — one of them will work. Then probe
   each subcommand's help.
3. Probe the interesting flags for real. Confirm they parse. Note the exit code.
4. Deliberately get it wrong once: pass a bad flag and a missing argument. Record the
   error text — a future agent will hit those errors and needs to recognise them.
5. If it processes files, create a small fixture in the workspace and run it end to
   end. A manual whose examples were never executed is worthless.

Report: what the binary does, its real invocation grammar, the flags you CONFIRMED by
running them, any place the repo and the binary disagree, and the gotchas.
Be explicit about what you verified versus what you only read.
"""

SKILL_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "lowercase-hyphenated, e.g. 'jq-cli'"},
        "description": {
            "type": "string",
            "description": (
                "One or two sentences covering WHAT the tool does and WHEN to reach for it. "
                "This is the trigger text a future agent matches against, so name the concrete "
                "nouns and tasks it applies to."
            ),
        },
        "summary": {"type": "string", "description": "2-4 sentences: what it is, what it's for."},
        "grammar": {"type": "string", "description": "The real invocation grammar, e.g. 'tool <subcmd> [opts] <file>'."},
        "flags": {
            "type": "array",
            "description": "Only flags you confirmed by running them.",
            "items": {
                "type": "object",
                "properties": {
                    "flag": {"type": "string"},
                    "meaning": {"type": "string"},
                    "verified": {"type": "boolean", "description": "true only if you ran it and saw it accepted"},
                },
                "required": ["flag", "meaning", "verified"],
            },
        },
        "examples": {
            "type": "array",
            "description": "Real commands. Each one WILL be executed and checked.",
            "items": {
                "type": "object",
                "properties": {
                    "setup": {"type": "string", "description": "Optional shell to create fixtures first, e.g. printf 'a,b\\n1,2\\n' > t.csv"},
                    "args": {"type": "string", "description": "Args only, no binary name."},
                    "purpose": {"type": "string"},
                    "expect_success": {"type": "boolean", "description": "true if exit 0 is expected"},
                },
                "required": ["args", "purpose", "expect_success"],
            },
        },
        "gotchas": {"type": "array", "items": {"type": "string"}},
        "discrepancies": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Where the repo's docs disagree with the binary's actual behaviour.",
        },
    },
    "required": ["name", "description", "summary", "grammar", "flags", "examples", "gotchas", "discrepancies"],
}

DRAFT_PROMPT = """\
An agent investigated the binary at `{exe}` (repo: {repo}) and produced these findings.

<findings>
{report}
</findings>

<execution_log>
Everything it actually ran, with real output. Only what appears here counts as verified.
{trail}
</execution_log>

Write the skill. Every example you emit will be executed against the real binary and
checked, so emit only commands you have evidence for. Include 3-6 examples spanning
the common cases plus at least one expected-failure case if you saw one.
"""

REPAIR_PROMPT = """\
Your draft skill was verified by running every example. Some failed.

<failures>
{failures}
</failures>

<previous_draft>
{draft}
</previous_draft>

Fix the failures. For each one: correct the command if you can see the right form in
the execution log, otherwise delete the example. Do not invent a new flag to patch a
broken example — that is how the fiction gets in. Re-emit the complete skill.
"""


@dataclass
class ExampleCheck:
    args: str
    purpose: str
    expected: bool
    actual_exit: int
    passed: bool
    output: str


@dataclass
class Acquisition:
    skill_name: str
    path: str
    checks: list[ExampleCheck]
    repairs: int
    recon: Result

    @property
    def verified(self) -> int:
        return sum(1 for c in self.checks if c.passed)


def _verify(sandbox: Sandbox, exe: str, examples: list[dict], timeout: int = 30) -> list[ExampleCheck]:
    """Run each example for real. This is the whole point of the module."""
    out: list[ExampleCheck] = []
    for ex in examples:
        args = str(ex.get("args", "")).strip()
        expect = bool(ex.get("expect_success", True))
        if not args:
            continue
        if setup := ex.get("setup"):
            try:
                sandbox.run(setup, timeout=timeout)
            except Exception as e:  # noqa: BLE001
                out.append(ExampleCheck(args, str(ex.get("purpose", "")), expect, -1, False, f"setup failed: {e}"))
                continue
        try:
            res = sandbox.run(f"'{exe}' {args}", timeout=timeout, max_chars=4000)
            passed = res.ok() == expect
            out.append(ExampleCheck(args, str(ex.get("purpose", "")), expect, res.exit_code, passed, res.render(2000)))
        except Exception as e:  # noqa: BLE001
            out.append(ExampleCheck(args, str(ex.get("purpose", "")), expect, -1, False, f"blocked: {e}"))
    return out


def _render(skill: dict, checks: list[ExampleCheck], exe: str, repo: str | None) -> str:
    by_args = {c.args: c for c in checks}
    L: list[str] = [skill.get("summary", "").strip(), ""]
    L += [f"**Binary:** `{exe}`"]
    if repo:
        L += [f"**Source:** {repo}"]
    L += ["", "## Invocation", "", f"```\n{skill.get('grammar', '').strip()}\n```", ""]

    if flags := skill.get("flags"):
        L += ["## Flags", "", "| Flag | Meaning | Verified |", "|---|---|---|"]
        for f in flags:
            mark = "yes" if f.get("verified") else "**no — unconfirmed**"
            L.append(f"| `{f.get('flag','')}` | {f.get('meaning','')} | {mark} |")
        L.append("")

    L += ["## Examples", "", "_Every example below was executed against the binary. "
          "The status is the observed result, not a claim._", ""]
    for ex in skill.get("examples", []):
        c = by_args.get(str(ex.get("args", "")).strip())
        if c and not c.passed:
            continue  # never ship an example that failed verification
        status = f"verified, exit {c.actual_exit}" if c else "not run"
        L.append(f"**{ex.get('purpose','')}** ({status})")
        if s := ex.get("setup"):
            L.append(f"```sh\n{s}\n{_base(exe)} {ex.get('args','')}\n```")
        else:
            L.append(f"```sh\n{_base(exe)} {ex.get('args','')}\n```")
        L.append("")

    if g := skill.get("gotchas"):
        L += ["## Gotchas", ""] + [f"- {x}" for x in g] + [""]
    if d := skill.get("discrepancies"):
        L += ["## Where the docs lie", "",
              "_Repo documentation that does not match the shipped binary. Trust the binary._", ""]
        L += [f"- {x}" for x in d] + [""]

    failed = [c for c in checks if not c.passed]
    L += ["## Verification", "",
          f"{sum(1 for c in checks if c.passed)}/{len(checks)} examples passed when executed."]
    if failed:
        L += ["", "Dropped during verification (claimed but did not work):"]
        L += [f"- `{c.args}` — expected {'success' if c.expected else 'failure'}, got exit {c.actual_exit}"
              for c in failed]
    return "\n".join(L)


def _base(exe: str) -> str:
    return exe.rsplit("/", 1)[-1]


def acquire_skill(
    cfg: Config,
    llm: LLM,
    exe: str | None = None,
    repo: str | None = None,
    on_event: Callable[[dict], None] | None = None,
) -> Acquisition:
    exe = str(exe or cfg.target_exe or "")
    repo = repo or cfg.target_repo
    if not exe:
        raise ValueError("no target binary: pass exe= or set cfg.target_exe")

    emit = on_event or (lambda e: None)

    # --- RECON: a full agent run, tools and all.
    emit({"type": "phase", "phase": "recon"})
    agent = Agent(cfg, llm, on_event=on_event)
    recon = agent.run(RECON_TASK.format(exe=exe, repo=repo or "(none provided)"))

    # --- DRAFT
    emit({"type": "phase", "phase": "draft"})
    draft = structured(
        llm,
        model=cfg.skill_author_model,
        system=SKILL_AUTHOR_SYSTEM,
        prompt=DRAFT_PROMPT.format(exe=exe, repo=repo or "(none)", report=recon.answer, trail=recon.trail()),
        schema=SKILL_SCHEMA,
        tool_name="emit_skill",
        effort="high",
        max_tokens=12_000,
    )

    # --- VERIFY / REPAIR
    sandbox = agent.sandbox
    checks = _verify(sandbox, exe, draft.get("examples", []), cfg.cmd_timeout)
    repairs = 0
    while repairs < MAX_REPAIRS and any(not c.passed for c in checks):
        emit({"type": "phase", "phase": "repair", "round": repairs + 1,
              "failing": sum(1 for c in checks if not c.passed)})
        fails = "\n\n".join(
            f"- `{c.args}` ({c.purpose})\n  expected: {'success' if c.expected else 'failure'}\n"
            f"  actual exit: {c.actual_exit}\n  output:\n{_indent(c.output)}"
            for c in checks if not c.passed
        )
        draft = structured(
            llm,
            model=cfg.skill_author_model,
            system=SKILL_AUTHOR_SYSTEM,
            prompt=REPAIR_PROMPT.format(failures=fails, draft=_pretty(draft)),
            schema=SKILL_SCHEMA,
            tool_name="emit_skill",
            effort="high",
            max_tokens=12_000,
        )
        checks = _verify(sandbox, exe, draft.get("examples", []), cfg.cmd_timeout)
        repairs += 1

    # --- COMMIT
    emit({"type": "phase", "phase": "commit"})
    store = SkillStore(cfg.skills_dir)
    n_ok = sum(1 for c in checks if c.passed)
    name = draft.get("name") or _base(exe).replace("_", "-").lower()
    path = store.save(
        name=name,
        description=draft.get("description", ""),
        body=_render(draft, checks, exe, repo),
        meta={
            "verified_examples": f"{n_ok}/{len(checks)}",
            "verified_on": date.today().isoformat(),
            "source_repo": repo or "",
            "binary": exe,
        },
    )
    emit({"type": "acquired", "skill": name, "verified": f"{n_ok}/{len(checks)}"})
    return Acquisition(name, str(path), checks, repairs, recon)


def _indent(s: str, n: int = 4) -> str:
    pad = " " * n
    return "\n".join(pad + ln for ln in s.splitlines()[:20])


def _pretty(d: dict) -> str:
    import json

    return json.dumps(d, indent=2)[:8000]

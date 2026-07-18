"""Command-line interface.

    forge learn   --exe ./bin/widget --repo /path/to/checkout
    forge solve   "task..." --exe ./bin/widget
    forge enhance --exe ./bin/widget --src /path/to/checkout
    forge skills

Every command takes --provider local --model <name> [--base-url URL] to run
against a local OpenAI-compatible server (Ollama, LM Studio, llama.cpp server,
vLLM) instead of the hosted Anthropic API. The local provider is offline by
default: no web_search tool, and fetch_repo accepts local checkouts only.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from .acquire import acquire_skill
from .config import Budget, Config
from .evaluator import solve
from .llm import make_llm
from .skillstore import SkillStore

DEFAULT_GOAL = (
    "identify the highest-value improvements — bugs and crash paths first, then "
    "error handling, performance, usability, and code health — and implement the "
    "top few properly rather than many superficially."
)

ENHANCE_TASK = """\
You are testing and enhancing the program whose executable is at {exe}.
A working copy of its source tree is in ./src — edit that copy freely; it is
yours. The original is untouched elsewhere.

Work in this order:

1. UNDERSTAND. Explore ./src: entry points, build system, test layout. Probe
   the binary (probe_exe: --help, subcommands, a few real invocations) so you
   know its observed behaviour, not just its documented behaviour. Where the
   docs and the binary disagree, trust the binary and note the discrepancy.

2. TEST. Run the project's own test suite if it has one; otherwise exercise
   the binary end-to-end with fixtures you create in the workspace. Record
   every failure, crash, or surprising behaviour with the exact command that
   reproduces it.

3. ENHANCE. Goal: {goal}
   Implement the changes directly in ./src. Keep each change minimal and in
   the codebase's existing style. After each change, rebuild if needed and
   re-run the relevant tests or probes to prove it works and nothing regressed.

4. REPORT. Write ENHANCEMENTS.md at the workspace root covering: what you
   tested and how, what you found, what you changed file by file and why, and
   the command output that proves each change is verified. Unfixed findings go
   in a "Known issues" section with repro commands.

Only claim what you verified by running something in this session.
"""


def _add_common(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("model")
    g.add_argument("--provider", choices=["anthropic", "local"], default="anthropic",
                   help="anthropic = hosted API; local = OpenAI-compatible server on localhost")
    g.add_argument("--model", help="model name. Required for --provider local "
                                   "(e.g. qwen3-coder:30b, qwen2.5-coder:14b, llama3.1:70b)")
    g.add_argument("--base-url", default="http://localhost:11434/v1",
                   help="local server URL (default: Ollama's)")
    g.add_argument("--api-key", help="API key (hosted) or bearer token if your local server wants one")
    g.add_argument("--online", action="store_true",
                   help="allow web_search and git-URL fetches even with --provider local")
    g = p.add_argument_group("run")
    g.add_argument("--workspace", type=Path, default=Path("forge-work"))
    g.add_argument("--skills-dir", type=Path, default=Path("skills"))
    g.add_argument("--approval", choices=["auto", "prompt", "deny"], default="auto",
                   help="gate hard-to-reverse tool calls (writes, runs, delegation)")
    g.add_argument("--max-eval-rounds", type=int, help="1 disables the reviewer entirely")
    g.add_argument("--max-turns", type=int, help="per-attempt turn budget")
    g.add_argument("-v", "--verbose", action="store_true")


def _config(args: argparse.Namespace, **over) -> Config:
    if args.provider == "local" and not args.model:
        sys.exit("error: --model is required with --provider local "
                 "(the name your server serves, e.g. `ollama list`)")
    kw: dict = {
        "workspace": args.workspace,
        "skills_dir": args.skills_dir,
        "provider": args.provider,
        "base_url": args.base_url,
        "api_key": args.api_key,
        "offline": args.provider == "local" and not args.online,
        "approval": args.approval,
    }
    if args.model:
        kw["agent_model"] = args.model
    if args.max_eval_rounds:
        kw["max_eval_rounds"] = args.max_eval_rounds
    kw.update(over)
    cfg = Config(**kw)
    if args.max_turns:
        cfg.budget.max_turns = args.max_turns
    return cfg


def _printer(verbose: bool):
    if not verbose:
        return lambda e: None

    def on_event(e: dict) -> None:
        t = e.get("type")
        if t == "turn":
            print(f"  turn {e['n']} [{e.get('stop_reason')}] in={e['in']} out={e['out']}")
        elif t == "tool":
            mark = "!" if e.get("is_error") else " "
            print(f"  {mark} {e['name']} {json.dumps(e.get('input', {}), default=str)[:140]}")
        elif t == "verdict":
            print(f"  review r{e['round']}: {e['verdict']} score={e['score']} — {str(e.get('critique', ''))[:180]}")
        elif t == "phase":
            print(f"== {e['phase']} ==")
        elif t == "delegate":
            print(f"  delegate: {e['subtasks']} subtask(s) in {e['batches']} batch(es)")
        elif t in {"subagent_start", "subagent_done"}:
            print(f"  {t.replace('_', ' ')}: {e.get('id')}")
        elif t == "compact":
            print(f"  compacting: {e['before_tokens']:,} tokens > {e['max']:,}")
        elif t == "acquired":
            print(f"  learned skill {e['skill']} ({e['verified']} verified)")
    return on_event


def cmd_learn(args: argparse.Namespace) -> int:
    cfg = _config(args, target_exe=args.exe, target_repo=args.repo)
    acq = acquire_skill(cfg, make_llm(cfg), on_event=_printer(args.verbose))
    ok = acq.verified
    print(f"learned: {acq.skill_name}")
    print(f"  verified examples: {ok}/{len(acq.checks)} ({acq.repairs} repair round(s))")
    print(f"  skill file: {acq.path}")
    return 0 if ok else 1


def cmd_solve(args: argparse.Namespace) -> int:
    cfg = _config(args, target_exe=args.exe, target_repo=args.repo, uploads_dir=args.uploads)
    out = solve(cfg, make_llm(cfg), args.task, on_event=_printer(args.verbose),
                lead=args.subagents)
    print(out.result.answer)
    if out.verdict:
        print(f"\n[review: {out.verdict.verdict}, score {out.verdict.score}/10, "
              f"{out.rounds} round(s)]")
    return 0


def cmd_enhance(args: argparse.Namespace) -> int:
    src = args.src.resolve()
    if not src.is_dir():
        sys.exit(f"error: --src {src} is not a directory")
    exe = args.exe.resolve()
    if not exe.exists():
        sys.exit(f"error: --exe {exe} does not exist")

    cfg = _config(args, target_exe=exe, target_repo=str(src))
    work_src = cfg.workspace / "src"
    work_src.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        src, work_src, dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(".git", "node_modules", "__pycache__", ".venv", "venv"),
    )
    print(f"source copied to {work_src} (original untouched)")

    task = ENHANCE_TASK.format(exe=exe, goal=args.goal or DEFAULT_GOAL)
    out = solve(cfg, make_llm(cfg), task, on_event=_printer(args.verbose),
                lead=not args.no_subagents)
    print(out.result.answer)
    if out.verdict:
        print(f"\n[review: {out.verdict.verdict}, score {out.verdict.score}/10]")
    report = cfg.workspace / "ENHANCEMENTS.md"
    if report.exists():
        print(f"\nreport: {report}\nmodified source: {work_src}")
    else:
        print(f"\n(no ENHANCEMENTS.md was produced — see the answer above; "
              f"modified source, if any, is in {work_src})")
    return 0


def cmd_skills(args: argparse.Namespace) -> int:
    print(SkillStore(args.skills_dir).index())
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="forge", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("learn", help="learn a binary: recon, draft a skill, verify every example")
    p.add_argument("--exe", type=Path, required=True)
    p.add_argument("--repo", help="git URL, or a local checkout path (required form when offline)")
    _add_common(p)
    p.set_defaults(fn=cmd_learn)

    p = sub.add_parser("solve", help="do a task with the agent (+reviewer), loading learned skills")
    p.add_argument("task")
    p.add_argument("--exe", type=Path)
    p.add_argument("--repo")
    p.add_argument("--uploads", type=Path, help="extra read-only directory for input data")
    p.add_argument("--subagents", action="store_true", help="allow the agent to delegate to subagents")
    _add_common(p)
    p.set_defaults(fn=cmd_solve)

    p = sub.add_parser("enhance", help="test an executable against its source, then implement "
                                       "and verify code enhancements")
    p.add_argument("--exe", type=Path, required=True, help="the built executable to test")
    p.add_argument("--src", type=Path, required=True, help="its source tree (copied into the workspace)")
    p.add_argument("--goal", help="what to optimise for; default: " + DEFAULT_GOAL)
    p.add_argument("--no-subagents", action="store_true", help="single agent, no delegation")
    _add_common(p)
    p.set_defaults(fn=cmd_enhance)

    p = sub.add_parser("skills", help="list learned skills and their verification rates")
    p.add_argument("--skills-dir", type=Path, default=Path("skills"))
    p.set_defaults(fn=cmd_skills)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())

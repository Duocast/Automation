# Forge

An agentic loop that learns to drive unknown binaries by reading their source, probing
them, and writing itself **verified** skills.

```
                    ┌──────────────────────────────────────┐
                    │  RECON    repo + binary               │
                    │  DRAFT    structured skill            │
   learn ──────────▶│  VERIFY   execute every example  ◀─┐  │
                    │  REPAIR   hand back the failures ──┘  │
                    │  COMMIT   SKILL.md + pass rate        │
                    └──────────────────────────────────────┘
                                     │
                                     ▼  skills/*/SKILL.md
                    ┌──────────────────────────────────────┐
   solve ──────────▶│  AGENT ⇄ tools ──▶ EVALUATOR ─┐      │
                    │    ▲                          │      │
                    │    └──── critique + retry ◀───┘      │
                    └──────────────────────────────────────┘
```

## The idea

Point it at a binary and the GitHub repo that documents it. It reads the repo, probes
the binary, and writes a skill — an operating manual a future run loads instead of
relearning from scratch.

The part that matters is that **it doesn't trust the README**. READMEs document intent
and drift from what shipped. A model asked to summarise one will confidently invent a
plausible `--output` flag, and every future agent inherits the lie.

So the acquisition pipeline closes the loop against ground truth: every example the
model writes gets **executed against the real binary**. Wrong ones come back for repair.
Ones that fail twice are deleted rather than shipped, and the skill records its own
pass rate in frontmatter so a future reader can calibrate trust:

```yaml
---
name: widget-cli
description: Drives the widget binary. Use for text transformation tasks.
verified_examples: 5/6
verified_on: 2026-07-17
---
```

When the repo and the binary disagree, the skill gets a **"Where the docs lie"** section.
That's usually the most valuable thing in the file.

## Quick start

```bash
pip install -e .            # core has zero dependencies (local provider is stdlib-only)
pip install -e .[anthropic] # only if using the hosted API
export ANTHROPIC_API_KEY=sk-...

# Learn a binary from its repo. Writes skills/<name>/SKILL.md
forge learn --exe ./target/release/widget \
            --repo https://github.com/acme/widget -v

forge skills          # what it knows, and how well-verified

# Do work. The learned skill is loaded automatically.
forge solve "convert every csv under uploads/ to parquet, verify row counts match" \
      --exe ./target/release/widget --uploads ./data

# Test an executable against its source, then implement + verify enhancements.
forge enhance --exe ./target/release/widget --src ./widget-checkout \
      --goal "tighten error handling; make --output actually work" -v
```

## Fully local / offline

Every command runs against a local model instead of the hosted API. Any
OpenAI-compatible server works — Ollama, LM Studio, llama.cpp server, vLLM —
and the client is stdlib-only, so nothing needs network access beyond loopback:

```bash
ollama pull qwen3-coder:30b          # once; any tool-calling-capable model

forge learn   --provider local --model qwen3-coder:30b \
              --exe ./bin/widget --repo /path/to/widget-checkout -v

forge enhance --provider local --model qwen3-coder:30b \
              --exe ./bin/widget --src /path/to/widget-checkout -v
```

What `--provider local` changes:

- **Transport**: `LocalLLM` speaks the OpenAI chat/tools wire format over
  `--base-url` (default `http://localhost:11434/v1`, Ollama's). Anthropic-only
  knobs (adaptive thinking, effort, prompt caching) are accepted and ignored.
- **Offline by default**: the `web_search` server tool is not advertised, and
  `fetch_repo` accepts local checkout paths only (pass `--online` to relax).
- **One model everywhere**: agent, evaluator, adjudicator, and summarizer all
  default to `--model`; set `evaluator_model`/`utility_model` in `Config` to
  split them across differently sized local models.
- **Robust structured output**: skill drafting forces a tool call, and falls
  back to parsing JSON out of prose for local models that ignore forced
  tool choice.

Pick a model that can call tools (qwen3 / qwen2.5-coder / llama3.1+ / mistral
class); the whole loop is tool-driven. The local path is integration-tested
end-to-end over real HTTP in `tests/test_local_llm.py`.

Library use:

```python
from forge import Config, LocalLLM, make_llm, solve, acquire_skill

cfg = Config(
    workspace="work", target_exe="./bin/widget", target_repo="/path/to/checkout",
    provider="local", agent_model="qwen3-coder:30b", offline=True,
)
llm = make_llm(cfg)                          # LocalLLM here; AnthropicLLM for provider="anthropic"

acquire_skill(cfg, llm)                      # learn it once
out = solve(cfg, llm, "process the inbox")   # then use it
print(out.result.answer, out.verdict.score)
```

## Architecture

| Module | Does |
|---|---|
| `loop.py` | The agent loop. Tool dispatch, approval gating, budgets, `pause_turn`, audit trail, delegation. |
| `orchestrate.py` | Subtask validation, read/write hazard scheduling, findings board. |
| `acquire.py` | Recon → draft → **verify** → repair → commit. The skill-learning pipeline. |
| `evaluator.py` | Fresh-context critic + adjudication + retry orchestration. |
| `sandbox.py` | Path confinement, write-set enforcement, command denylist, timeouts, truncation. |
| `skillstore.py` | SKILL.md read/write, progressive-disclosure index. |
| `tools/` | `read_file` `edit_file` `grep` `list_dir` `write_file` `run` `probe_exe` `fetch_repo` `skill_read` `delegate` `ask_lead` `post_finding` |
| `llm.py` | `AnthropicLLM` (Messages API: adaptive thinking, effort, streaming, retries) and `LocalLLM` (OpenAI-compatible local servers, stdlib-only), behind one protocol. |
| `config.py` | The one dataclass: provider, models, budgets, evaluation knobs, offline mode. |
| `prompts.py` | Every system prompt in one place. |
| `cli.py` | `forge learn / solve / enhance / skills`. |

### Why a hand-written loop

The SDK's `tool_runner` handles the loop for you and hides exactly the seams this needs:
approval gating on hard-to-reverse calls, per-attempt budgets, and an audit trail the
evaluator reads back. Rule of thumb from Anthropic's agent-design guidance — start with
`bash` for breadth, promote to a dedicated tool when the harness needs to **gate, render,
audit, or parallelise**. Three places that pays off here:

- **`probe_exe` is separate from `run`.** It pins argv[0] to the configured binary, so
  the model can't drift onto something else, and every invocation lands in the audit
  trail the verifier reads.
- **`edit_file` enforces read-before-write.** It records mtime at read and rejects the
  edit if the file changed since. A bash harness cannot enforce that invariant — it only
  sees an opaque command string.
- **Read-only tools are marked `parallel_safe`** and fan out concurrently in one turn.
  Through bash, the harness can't tell a parallel-safe `grep` from a `git push`, so it
  must serialise everything.

### Why the evaluator works

Four choices, all load-bearing:

1. **Fresh context.** The critic sees the task, the answer, and the evidence — never the
   agent's reasoning. Self-critique grades the reasoning it already committed to.
2. **Evidence on demand, not a truncated dump.** The critic gets a bounded *index* of
   every tool call plus `grep_trail` and `read_tool_output` to reach the full record.
   This matters more than it sounds: a 24,755-char build log flattened into a trail
   shows the critic 902 chars, and the `error[E0433]` in the middle is invisible. A
   critic handed that is structurally unable to catch the one failure it exists for.
   Pinned as a regression test in `test_trail_still_hides_the_error`.
3. **The critic is itself checked.** Its `unsupported_claims` are adjudicated by an
   independent pass that sees the claim and the log but not the critic's reasoning or
   the answer's framing. Claims that don't survive are dropped before the agent hears
   them — a false accusation burns a retry and tells the agent to prove what it already
   proved, which invites thrash. Rulings map back to claims **by id, never by matching
   text**: substring matching transposed rulings between claims whose text overlapped,
   and could drop a true accusation. Every rejection path (bad id, conflicting rulings,
   no ruling, unparseable) leaves the claim standing — a surviving false accusation
   costs one retry, a silently dropped true one ships broken work. Anomalies land in
   `verdict.adjudication_notes` rather than being swallowed, because a malfunctioning
   adjudicator otherwise looks exactly like a well-behaved one that found nothing.
4. **Coherence guards.** "revise" with no fixes and no surviving claims isn't a verdict —
   it's promoted to pass with `verdict.override` set. A reviewer who never passes is a
   broken reviewer, now enforced mechanically rather than requested in the prompt.

Retries escalate `effort`. Attempts aren't monotonic, so if round 2 scores worse than
round 1, round 1 is what you get back.

## Orchestration: lead and subagents

`Agent(cfg, llm, lead=True)` gets a `delegate` tool and can fan work out to subagents,
each with its own context window.

**Parallelism is the lesser reason.** The real win is *context isolation*: a subagent
spends its own window figuring out file 7 and hands back a paragraph. The lead never
carries twelve files' worth of tool output, so it stays coherent on long work instead of
compacting away the detail it needs.

**The hazard.** Subagents share one workspace, and two of them writing the same file
concurrently is a data race with a language model holding the pen — nondeterministic, and
the loser's work vanishes silently. So each subtask declares the paths it `reads` and
`writes`, and the scheduler batches them so conflicting work never runs at once:

```
WAW  both write the same path    -> last writer wins, first is lost
RAW  B reads what A writes       -> B may see a half-written file
WAR  B writes what A reads       -> A's view changes under it
```

Disjoint subtasks run together; anything sharing a path is serialized. Conservative on
purpose — a false serialization costs latency, a missed hazard costs correctness, and
only one of those is recoverable. Declared write-sets are *enforced*, not advisory: a
subagent's sandbox hard-blocks writes outside its declaration (plus a private
`.scratch/<id>/`), because the hazard analysis is only sound if the declarations are true.

### Escalation, not conversation

Subagents get exactly two channels back, and nothing else:

- **`ask_lead`** — one bounded question, one answer, budget of 2. For decisions only the
  lead can make (scope, tradeoffs, what it meant), not facts the subagent could look up.
  The lead answers via a **one-shot stateless call** over the original task, not by
  re-entering its own loop: it's suspended inside its `delegate` call, and resuming its
  conversation from in there would be re-entrant, need locking against concurrent askers,
  and could recurse. Scope questions don't need any of that.
- **`post_finding`** — a capped, one-sentence note that flows **forward across batches**.
  The batch structure already *is* the dependency graph, so batch → batch flow matches
  the real dependency order for free and deterministically.

**What's deliberately absent: any subagent-to-subagent channel, and same-batch
visibility.** Delegation exists so a subagent burns its own context and returns a
paragraph; letting them stream detail at each other re-couples the contexts we split
apart and turns a scheduler into a distributed system with deadlocks in it. Same-batch
subtasks share no paths by construction, so a shared board there would buy a race and pay
for it in reproducibility. Both refusals have tests named after them.

The lead could always relay findings by delegating in stages — that still works, and is
the right move when it knows in advance. The board covers the case it *couldn't*
anticipate.

Two details that are easy to miss and load-bearing:

- **Subagents never get `delegate`.** Depth is capped at 1 structurally, not by a counter
  someone forgets to check. Fan-out is exponential in depth with real money attached.
- **Subagent tool calls merge into the lead's trail**, tagged `(via s1)`. Without this the
  evaluator would read "I ported all nine files", find no tool call that did it, and flag
  the whole thing as fabricated. Delegated work has to leave the same evidence as inline
  work.

## Long-run context management

The loop keeps the window the model sees under `budget.max_input_tokens` (default 700k,
under the 1M ceiling). When it would overflow, `compact.py` summarizes the *middle* of
the conversation with a cheap model, keeping the original task and the recent working set
verbatim. Two invariants hold:

- **tool_use/tool_result pairing is never broken** — compaction only cuts on whole
  exchange boundaries, so no `tool_use` is ever orphaned (that would be a 400).
- **the audit trail is untouched** — `calls`/`ctx.audit` carry the full record, so the
  evaluator still judges everything the agent did, not the compacted view.

Verified in `test_compact.py`: `test_compaction_never_orphans_a_tool_use` and
`test_loop_compacts_on_long_run` (window climbs, then holds flat instead of growing).

## Known gaps (honest backlog, by priority)

1. **`run` is a denylist, and `python3 -c` bypasses it.** The real boundary must be the
   container. Next step: gate interpreters behind approval, or make `run` opt-in.
2. **Skill verification checks exit codes, not output correctness**, and examples share a
   workspace, so one example's fixtures can leak into another's "pass". Next: per-example
   isolation + optional output assertions.
3. **Skills are overwrite-only snapshots** — no merge, no staleness signal when the
   binary changes under a skill.
4. **Hazard declarations are only as good as the lead's honesty.** The sandbox enforces
   `writes`, so an under-declaration fails loudly — but an *over*-declaration (claiming
   `.`) silently serializes everything and costs the parallelism. Nothing detects that.
5. **Nothing checks the adjudicator's *reasoning*, only its bookkeeping.** An adjudicator
   that searches badly and rules "supported" on a real fabrication will drop it, and the
   only defence is that it's a separate call with separate evidence. Turtles stop here.
6. **The lead answers questions without seeing the workspace.** `_answer_question` is a
   stateless call over the original task — right for scope decisions, wrong if a subagent
   asks something the lead would need to investigate. It's prompted to push back, not
   prevented.
7. **Findings are trusted on assertion.** A subagent posting something false propagates it
   to every later batch. They're capped and attributed, not verified.

## Config worth knowing

| Setting | Default | Note |
|---|---|---|
| `provider` | `anthropic` | `local` = any OpenAI-compatible server, stdlib HTTP only |
| `base_url` | `http://localhost:11434/v1` | local provider; Ollama's default |
| `offline` | `False` | drops `web_search`, restricts `fetch_repo` to local paths; CLI forces it on for `local` |
| `agent_model` | `claude-opus-4-8` | `claude-sonnet-5` is cheaper and close; for local, your served model |
| `evaluator_model` / `utility_model` | `agent_model` | split across differently sized local models if you like |
| `effort` / `retry_effort` | `medium` / `high` | escalate on review failure |
| `max_eval_rounds` | 3 | `1` disables review entirely |
| `critic_max_turns` | 8 | evidence queries the critic/adjudicator may make |
| `adjudicate` | `True` | independently spot-check the critic's accusations |
| `pass_score` | 8 | 1–10 |
| `budget.max_turns` | 40 | per attempt |
| `budget.max_subagents` | 8 | total fan-out across a run |
| `budget.subagent_max_turns` | 20 | per subagent; smaller than the lead's |
| `budget.max_subtasks_per_call` | 6 | per `delegate()` call |
| `budget.max_questions_per_subagent` | 2 | `ask_lead` escalations; a fuse, not an allowance |
| `budget.max_findings` | 20 | shared notes per `delegate()` call |
| `approval` | `auto` | `auto` \| `prompt` \| `deny` |

## Tests

```bash
pytest tests/ -q     # 154 passed
```

No API key needed — the LLM is scripted, but the sandbox and the binary are real, so
the verification path is exercised for real. `test_acquire_end_to_end_repairs_hallucination`
is the one to read: the scripted model invents a `--lowercase` flag, verification runs it
against a binary that exits 2, the repair round corrects it, and the assertion is that
the lie never reaches disk.

## Notes on the API

Verified against `anthropics/skills @ skills/claude-api`, not recalled:

- `thinking={"type": "adaptive"}` — `budget_tokens` is a **400** on Opus 4.7+/Sonnet 5/Fable 5.
- Never send `thinking={"type": "disabled"}` to Fable 5 — also a 400. Omit the param.
- Sampling params (`temperature`, `top_p`) are rejected on Opus 4.7+. Never sent.
- Effort lives in `output_config`, not top level.
- Web search is `web_search_20260209` on Opus 4.6+/Sonnet 4.6+; older models need
  `web_search_20250305`. `config.web_search_type()` picks per model.
- System prompt + tool defs are marked `cache_control: ephemeral` — they're identical
  every turn, and a 40-turn run re-sends them 40 times otherwise.

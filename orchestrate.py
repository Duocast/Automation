"""Subtask scheduling for the lead/subagent split.

Pure functions — no Agent import, so the hazard logic is testable on its own and there's
no import cycle with loop.py.

**Why subagents at all.** Parallelism is the obvious answer and the less important one.
The real payoff is *context isolation*: a subagent spends its own 200k-token window
figuring out file 7 and hands back a paragraph. The lead never carries twelve files'
worth of tool output, so it stays coherent on long work instead of compacting away the
detail it needs. Parallelism is a bonus that falls out of the same structure.

**The hazard.** Subagents share one workspace. Two of them writing the same file
concurrently is a data race with a language model holding the pen — nondeterministic,
and the loser's work vanishes silently. So each subtask declares the paths it reads and
writes, and this module batches them so no two conflicting subtasks ever run at once.
The classic three hazards all apply:

    WAW  both write the same path       -> last writer wins, first is lost
    RAW  B reads what A writes          -> B may see a half-written file
    WAR  B writes what A reads          -> A's view changes under it

Subtasks that share nothing run together. Anything else is serialized. This is
conservative on purpose: a false serialization costs latency, a missed hazard costs
correctness, and only one of those is recoverable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePosixPath


@dataclass
class Subtask:
    id: str
    task: str
    writes: list[str] = field(default_factory=list)
    reads: list[str] = field(default_factory=list)

    def norm_writes(self) -> set[str]:
        return {_norm(p) for p in self.writes}

    def norm_reads(self) -> set[str]:
        return {_norm(p) for p in self.reads}


def _norm(p: str) -> str:
    """Normalise a declared path for comparison. Relative, no ./, no trailing slash."""
    s = str(p).strip().strip("/")
    if s.startswith("./"):
        s = s[2:]
    return str(PurePosixPath(s)) if s else "."


def _overlaps(a: set[str], b: set[str]) -> bool:
    """True if any path in `a` touches any in `b`, treating a dir as covering its subtree.

    'src' and 'src/foo.py' overlap; 'src' and 'srcfoo' do not.
    """
    for x in a:
        for y in b:
            if x == y:
                return True
            if x == "." or y == ".":
                return True  # a whole-workspace claim conflicts with everything
            if y.startswith(x + "/") or x.startswith(y + "/"):
                return True
    return False


def conflicts(a: Subtask, b: Subtask) -> str | None:
    """Return the hazard name if these two cannot run concurrently, else None."""
    aw, ar = a.norm_writes(), a.norm_reads()
    bw, br = b.norm_writes(), b.norm_reads()
    if _overlaps(aw, bw):
        return "WAW"
    if _overlaps(aw, br):
        return "RAW"
    if _overlaps(ar, bw):
        return "WAR"
    return None


def validate(raw: list[dict], max_subtasks: int) -> list[Subtask]:
    """Coerce model-supplied subtasks into specs, or raise ValueError with a fixable message."""
    if not raw:
        raise ValueError("subtasks is empty — give at least one subtask, or do the work yourself")
    if len(raw) > max_subtasks:
        raise ValueError(
            f"{len(raw)} subtasks exceeds the limit of {max_subtasks}. "
            "Group related work into fewer, larger subtasks."
        )
    out: list[Subtask] = []
    seen: set[str] = set()
    for i, r in enumerate(raw):
        if not isinstance(r, dict):
            raise ValueError(f"subtask {i} is not an object")
        task = str(r.get("task", "")).strip()
        if not task:
            raise ValueError(f"subtask {i} has no task text")
        sid = str(r.get("id") or f"s{i + 1}").strip()
        if sid in seen:
            raise ValueError(f"duplicate subtask id {sid!r}")
        seen.add(sid)
        out.append(Subtask(
            id=sid,
            task=task,
            writes=[str(w) for w in (r.get("writes") or [])],
            reads=[str(x) for x in (r.get("reads") or [])],
        ))
    return out


def batches(subtasks: list[Subtask]) -> list[list[Subtask]]:
    """Greedily pack subtasks into batches that can run concurrently.

    Order is preserved: a subtask never lands in an earlier batch than one it conflicts
    with, so declaring a dependency via reads/writes also sequences it.
    """
    out: list[list[Subtask]] = []
    for st in subtasks:
        placed = False
        for batch in out:
            if all(conflicts(st, other) is None for other in batch):
                batch.append(st)
                placed = True
                break
        if not placed:
            out.append([st])
    return out


def explain(subtasks: list[Subtask]) -> list[str]:
    """Human-readable notes on why anything got serialized. Surfaced to the lead."""
    notes: list[str] = []
    for i, a in enumerate(subtasks):
        for b in subtasks[i + 1 :]:
            if h := conflicts(a, b):
                notes.append(f"{a.id} & {b.id}: {h} hazard on shared paths — serialized")
    return notes


# --------------------------------------------------------------------------- blackboard
@dataclass
class Finding:
    subtask_id: str
    text: str


class Blackboard:
    """Append-only notes that flow *forward* across batches. Deliberately not a chat.

    The batch structure computed above already is the dependency graph: subtasks in one
    batch share no paths (that's why they're concurrent), and later batches were pushed
    later precisely because they depend on earlier ones. So findings flowing batch ->
    batch matches the dependency order for free, and does it deterministically.

    Within a batch, nothing is shared. Concurrent subtasks would see each other's posts
    in a nondeterministic order, and by construction they don't interact anyway. Making
    same-batch posts visible would buy a race and pay for it with reproducibility.

    Capped hard on both count and length. A finding is a sentence another subtask would
    be wrong without — not a progress log, and not a transcript. Letting subagents stream
    detail at each other would re-couple the contexts that delegation exists to separate.
    """

    def __init__(self, max_findings: int = 20, max_len: int = 400):
        self._items: list[Finding] = []
        self._lock = __import__("threading").Lock()
        self.max_findings = max_findings
        self.max_len = max_len
        self.rejected = 0

    def post(self, subtask_id: str, text: str) -> str:
        text = " ".join(str(text).split())
        if not text:
            raise ValueError("finding is empty")
        truncated = False
        if len(text) > self.max_len:
            text = text[: self.max_len] + "…"
            truncated = True
        with self._lock:
            if len(self._items) >= self.max_findings:
                self.rejected += 1
                raise ValueError(
                    f"the board is full ({self.max_findings} findings). Yours was not posted — "
                    "put it in your final report instead."
                )
            self._items.append(Finding(subtask_id, text))
            n = len(self._items)
        return (f"posted finding {n}/{self.max_findings}"
                + (f" (truncated to {self.max_len} chars)" if truncated else ""))

    def snapshot(self) -> list[Finding]:
        with self._lock:
            return list(self._items)

    @staticmethod
    def render(items: list[Finding]) -> str:
        if not items:
            return ""
        return "\n".join(f"- ({f.subtask_id}) {f.text}" for f in items)

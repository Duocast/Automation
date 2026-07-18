"""Tool plumbing: the base class, the per-run Context, and small shared helpers.

A Tool is a typed, named capability. The loop only knows three things about
one: its spec (for the API), whether it is `gated` (approval required before
running), and whether it is `parallel_safe` (read-only, so a turn's worth of
them can fan out concurrently). That metadata is the whole reason these are
dedicated tools rather than strings through bash — a harness cannot gate or
parallelise what it cannot see.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:  # avoid import cycles at runtime; these are type-only
    from ..config import Config
    from ..sandbox import Sandbox
    from ..skillstore import SkillStore


class ToolError(Exception):
    """Expected, recoverable tool failure. Returned to the model as an error
    tool_result so it can self-correct; never crashes the loop."""


@dataclass
class Context:
    """Everything a tool may touch, threaded through every call in a run."""

    cfg: "Config"
    sandbox: "Sandbox"
    skills: "SkillStore"
    #: path -> mtime_ns at last read; how edit_file detects stale writes.
    read_versions: dict[str, int] = field(default_factory=dict)
    #: free-form audit entries tools may append (kept alongside the call trail).
    audit: list[dict] = field(default_factory=list)
    #: lead-only hook: fan a list of subtasks out to subagents.
    spawn: Callable[[list], str] | None = None
    #: subagent-only hooks: escalate a question / post a forward finding.
    ask_lead: Callable[[str, str], str] | None = None
    post_finding: Callable[[str], str] | None = None


class Tool:
    name: str = ""
    description: str = ""
    input_schema: dict[str, Any] = {"type": "object", "properties": {}}
    #: gated tools go through the approval policy before running.
    gated: bool = False
    #: parallel_safe tools are read-only and may fan out within one turn.
    parallel_safe: bool = False

    def run(self, ctx: Context, **kwargs: Any) -> str:
        raise NotImplementedError

    def spec(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


def rel(ctx: Context, p: Path) -> str:
    """Render a path relative to the workspace where possible; keeps tool
    output short and keeps absolute host paths out of the model's context."""
    try:
        return str(Path(p).resolve().relative_to(ctx.sandbox.workspace))
    except ValueError:
        return str(p)

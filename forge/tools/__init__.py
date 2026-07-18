"""Tool registry. build_toolset decides which agent tier sees which tools.

The tiers are structural, not advisory:
- only a LEAD gets `delegate` (the recursion fuse: depth is capped at 1 by
  construction, not by a counter someone forgets to check);
- only a SUBAGENT gets `ask_lead`/`post_finding` (the lead has no one to ask
  and nothing downstream to inform);
- `probe_exe` exists only when a target binary is configured;
- the web_search server tool is advertised only when not offline. It executes
  on Anthropic's servers, so the local/offline path simply omits it.
"""

from __future__ import annotations

from .base import Context, Tool, ToolError, rel
from .collab import AskLead, PostFinding
from .delegate import Delegate
from .fs import EditFile, Grep, ListDir, ReadFile, WriteFile
from .repo import FetchRepo
from .run import ProbeExe, RunCmd
from .skills import SkillRead

__all__ = ["Context", "Tool", "ToolError", "build_toolset", "rel"]


def build_toolset(cfg, lead: bool = False, subagent: bool = False) -> tuple[dict[str, Tool], list[dict]]:
    """Return (name -> Tool, API tool specs) for one agent."""
    tools: list[Tool] = [
        ListDir(), ReadFile(), Grep(), WriteFile(), EditFile(),
        RunCmd(), FetchRepo(), SkillRead(),
    ]
    if cfg.target_exe:
        tools.append(ProbeExe())
    if lead:
        tools.append(Delegate())
    if subagent:
        tools.extend([AskLead(), PostFinding()])

    specs = [t.spec() for t in tools]
    if not cfg.offline:
        # Server-side tool: executed by Anthropic, never dispatched locally.
        specs.append({"type": cfg.web_search_type(), "name": "web_search", "max_uses": 5})
    return {t.name: t for t in tools}, specs

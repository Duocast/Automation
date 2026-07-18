"""The lead's fan-out primitive.

Only a lead agent gets this tool. Subagents never do — see `build_toolset`. That's not
tidiness, it's a cost fuse: a subagent that can delegate can spawn subagents that can
delegate, and fan-out is exponential in depth with real money attached.
"""

from __future__ import annotations

from .base import Context, Tool, ToolError


class Delegate(Tool):
    name = "delegate"
    description = (
        "Split independent work across subagents, each with its own fresh context, and "
        "get back their reports.\n\n"
        "Use this when the task has parts that don't need to see each other's details — "
        "'port each of these 9 files', 'analyse each CSV', 'investigate these 4 "
        "subsystems'. Each subagent burns its own context window and hands you a summary, "
        "so you stay coherent instead of drowning in twelve files' worth of tool output.\n\n"
        "Do NOT use it for work that is small, sequential, or needs one continuous train "
        "of thought. Two subagents on a two-file change is slower and worse than doing it "
        "yourself. Delegation costs a full agent run per subtask.\n\n"
        "Declare `writes` and `reads` honestly for every subtask: they are how the "
        "scheduler decides what can run concurrently. Subtasks that share a path are run "
        "one after another; subtasks that share nothing run at the same time. A subagent "
        "is HARD-BLOCKED from writing outside its declared `writes` (plus its own scratch "
        "dir), so an incomplete list will make it fail."
    )
    gated = True
    input_schema = {
        "type": "object",
        "properties": {
            "subtasks": {
                "type": "array",
                "description": "Independent units of work. Each gets one subagent.",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "Short handle, e.g. 's1'."},
                        "task": {
                            "type": "string",
                            "description": (
                                "Self-contained instructions. The subagent sees NOTHING of this "
                                "conversation — no task text, no prior findings, no context. "
                                "Spell out what to do, which files, and what to report back. "
                                "A vague subtask returns a vague report and you cannot fix it "
                                "after the fact."
                            ),
                        },
                        "writes": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Every path this subtask may modify. Enforced.",
                        },
                        "reads": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Paths it depends on but won't modify. Used for hazard detection.",
                        },
                    },
                    "required": ["task"],
                },
            },
        },
        "required": ["subtasks"],
    }

    def run(self, ctx: Context, subtasks: list) -> str:
        if ctx.spawn is None:
            raise ToolError("delegation is not available to this agent")
        try:
            return ctx.spawn(subtasks)
        except ValueError as e:
            raise ToolError(str(e)) from e

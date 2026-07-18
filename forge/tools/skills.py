"""Progressive disclosure for the skill library: the system prompt carries only
name + description per skill; this tool pulls a full manual on demand."""

from __future__ import annotations

from .base import Context, Tool, ToolError


class SkillRead(Tool):
    name = "skill_read"
    description = (
        "Load the full text of a learned skill by name. The index of available "
        "skills (with verification rates) is in your system prompt. Read the "
        "skill before driving its binary — it was verified against the real "
        "binary and is more reliable than the repo's docs."
    )
    parallel_safe = True
    input_schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }

    def run(self, ctx: Context, name: str) -> str:
        s = ctx.skills.get(str(name))
        if s is None:
            have = ", ".join(x.name for x in ctx.skills.list()) or "(none learned yet)"
            raise ToolError(f"no skill named {name!r}. Available: {have}")
        return s.render()

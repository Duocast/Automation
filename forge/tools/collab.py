"""Subagent -> lead escalation, and forward-flowing findings.

Only subagents get these. The lead has no one to ask, and nothing downstream to inform.

**What this deliberately is not.** There is no subagent-to-subagent channel, no
same-batch visibility, and no multi-turn negotiation. Delegation exists so a subagent can
burn its own context window and hand back a paragraph; letting them stream detail at each
other re-couples exactly the contexts we split apart, and turns a scheduler into a
distributed system with deadlocks in it. `ask_lead` is a bounded question with a one-shot
answer, and `post_finding` is a capped note that only later batches read. Both are
escalation, not conversation.
"""

from __future__ import annotations

from .base import Context, Tool, ToolError


class AskLead(Tool):
    name = "ask_lead"
    description = (
        "Ask the lead agent one bounded question when you are blocked by a decision only "
        "it can make — scope, priorities, a tradeoff, what it meant by an instruction.\n\n"
        "Investigate first. This is not for facts: if the answer is in a file, the repo, "
        "or the binary, go read it — you have the tools and the lead does not have the "
        "answer either. Use this when you would otherwise have to *guess*, because a guess "
        "baked into your work is expensive to unpick and the lead only discovers it at the "
        "end.\n\n"
        "You get one answer, not a conversation. Ask a specific question with the options "
        "you are choosing between. Your question budget is small; spend it on decisions "
        "that change what you build."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "Specific and answerable. Name the options you're deciding between.",
            },
            "why_blocked": {
                "type": "string",
                "description": "What you already checked, and what you'd have to guess without an answer.",
            },
        },
        "required": ["question", "why_blocked"],
    }

    def run(self, ctx: Context, question: str, why_blocked: str) -> str:
        if ctx.ask_lead is None:
            raise ToolError("no lead to ask — you are not running as a subagent")
        try:
            return ctx.ask_lead(question, why_blocked)
        except ValueError as e:
            raise ToolError(str(e)) from e


class PostFinding(Tool):
    name = "post_finding"
    description = (
        "Record one durable discovery for subtasks that run AFTER yours.\n\n"
        "Post only what a later subtask would be WRONG without: a renamed API, a format "
        "quirk, a flag that doesn't work, a convention the code actually follows. One "
        "sentence, stated as fact.\n\n"
        "Not for progress ('finished reading the file'), not for things already visible in "
        "the files you wrote — later subtasks can read those. Findings are capped and "
        "shared with strangers who lack all your context, so a vague note is worse than "
        "none: it reads as knowledge and carries nothing. Anything that only matters to "
        "the lead belongs in your final report instead."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "finding": {"type": "string", "description": "One factual sentence."},
        },
        "required": ["finding"],
    }

    def run(self, ctx: Context, finding: str) -> str:
        if ctx.post_finding is None:
            raise ToolError("no findings board — you are not running as a subagent")
        try:
            return ctx.post_finding(finding)
        except ValueError as e:
            raise ToolError(str(e)) from e

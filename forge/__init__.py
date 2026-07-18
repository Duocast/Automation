"""Forge: an agentic loop that learns verified skills for driving binaries,
then uses them — against the hosted Anthropic API or a fully local LLM."""

from .acquire import Acquisition, acquire_skill
from .config import Budget, Config
from .evaluator import Outcome, Verdict, evaluate, solve
from .llm import LLM, AnthropicLLM, LocalLLM, Reply, Usage, make_llm
from .loop import Agent, Result
from .sandbox import Sandbox, SandboxError
from .skillstore import Skill, SkillStore

__all__ = [
    "Acquisition", "acquire_skill",
    "Budget", "Config",
    "Outcome", "Verdict", "evaluate", "solve",
    "LLM", "AnthropicLLM", "LocalLLM", "Reply", "Usage", "make_llm",
    "Agent", "Result",
    "Sandbox", "SandboxError",
    "Skill", "SkillStore",
]

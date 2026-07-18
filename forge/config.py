"""Configuration. One dataclass, no magic.

Two providers are supported:

- ``anthropic``: the hosted Messages API. Needs the ``anthropic`` package,
  an API key, and network access.
- ``local``: any OpenAI-compatible chat server — Ollama, LM Studio,
  llama.cpp server, vLLM. Stdlib HTTP only, so the whole system runs
  offline against a model on localhost. Set ``base_url`` and point
  ``agent_model`` at a model the server actually serves; the other
  ``*_model`` fields default to ``agent_model`` unless set.

``offline=True`` removes the web_search server tool from the toolset and
restricts fetch_repo to local checkouts. The CLI forces it on for the local
provider unless ``--online`` is passed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Budget:
    max_turns: int = 40
    max_wall_seconds: int = 1800
    max_input_tokens: int = 700_000
    max_tool_output_chars: int = 30_000
    max_subagents: int = 8
    subagent_max_turns: int = 20
    max_subtasks_per_call: int = 6
    max_questions_per_subagent: int = 2
    max_findings: int = 20


@dataclass
class Config:
    workspace: Path
    skills_dir: Path | None = None
    uploads_dir: Path | None = None
    target_exe: Path | None = None
    target_repo: str | None = None

    # --- provider -----------------------------------------------------------
    provider: str = "anthropic"                    # "anthropic" | "local"
    base_url: str = "http://localhost:11434/v1"    # local provider only
    api_key: str | None = None
    offline: bool = False

    # --- models -------------------------------------------------------------
    agent_model: str = "claude-opus-4-8"
    evaluator_model: str = ""      # empty -> agent_model
    skill_author_model: str = ""   # empty -> agent_model
    utility_model: str = ""        # cheap summarizer; empty -> sensible default

    effort: str = "medium"
    retry_effort: str = "high"
    max_tokens: int = 16_000

    # --- evaluation ---------------------------------------------------------
    max_eval_rounds: int = 3
    critic_max_turns: int = 8
    adjudicate: bool = True
    pass_score: int = 8

    # --- execution ----------------------------------------------------------
    approval: str = "auto"                         # auto | prompt | deny
    allow_network_in_shell: bool = False
    cmd_timeout: int = 60
    transcript_path: Path | None = None

    budget: Budget = field(default_factory=Budget)

    def __post_init__(self) -> None:
        self.workspace = Path(self.workspace)
        self.skills_dir = Path(self.skills_dir) if self.skills_dir else self.workspace / "skills"
        self.uploads_dir = Path(self.uploads_dir) if self.uploads_dir else None
        self.target_exe = Path(self.target_exe) if self.target_exe else None
        self.transcript_path = Path(self.transcript_path) if self.transcript_path else None
        self.evaluator_model = self.evaluator_model or self.agent_model
        self.skill_author_model = self.skill_author_model or self.agent_model
        if not self.utility_model:
            self.utility_model = (
                "claude-haiku-4-5-20251001" if self.provider == "anthropic" else self.agent_model
            )

    def web_search_type(self) -> str:
        """Pick the web_search tool version the configured agent model accepts.

        web_search_20260209 exists on Opus 4.6+/Sonnet 4.6+ and the Claude 5
        family; everything older takes web_search_20250305. Unknown (e.g. local)
        model names get the old version, which is harmless: offline mode omits
        the tool entirely.
        """
        n = self.agent_model
        if re.match(r"claude-[123]\b", n):
            return "web_search_20250305"
        if re.match(r"claude-(fable|mythos)-", n):
            return "web_search_20260209"
        m = re.match(r"claude-(opus|sonnet|haiku)-(\d+)(?:-(\d+))?", n)
        if m:
            fam, major, minor = m.group(1), int(m.group(2)), int(m.group(3) or 0)
            if major >= 5 or (major == 4 and minor >= 6 and fam in ("opus", "sonnet")):
                return "web_search_20260209"
        return "web_search_20250305"

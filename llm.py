"""Thin wrapper over the Messages API.

Deliberately thin: the agent loop needs to see raw content blocks and stop_reason
to do approval gating, budget accounting, and pause_turn handling. The SDK's
tool_runner hides exactly those seams, so we drive the loop ourselves.

API notes that differ from older patterns (verified against anthropic/skills):
  - thinking={"type": "adaptive"}. budget_tokens is a 400 on Opus 4.8 / Sonnet 5 / Fable 5.
  - Never send thinking={"type": "disabled"} to Fable 5 -> 400. Omit instead.
  - Sampling params (temperature/top_p) are rejected on Opus 4.7+. We never send them.
  - Effort goes in output_config, not top level.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

MAX_RETRIES = 5
RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 529}


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_tokens += other.cache_read_tokens


@dataclass
class Reply:
    """Normalised response so the loop never touches SDK types."""

    content: list[Any]
    stop_reason: str | None
    usage: Usage = field(default_factory=Usage)

    def text(self) -> str:
        parts = []
        for b in self.content:
            if _btype(b) == "text":
                parts.append(_battr(b, "text"))
        return "\n".join(parts).strip()

    def tool_uses(self) -> list[Any]:
        return [b for b in self.content if _btype(b) == "tool_use"]

    def first_tool_input(self) -> dict | None:
        for b in self.content:
            if _btype(b) == "tool_use":
                return _battr(b, "input")
        return None


def _btype(block: Any) -> str:
    return block.get("type") if isinstance(block, dict) else getattr(block, "type", "")


def _battr(block: Any, name: str) -> Any:
    return block.get(name) if isinstance(block, dict) else getattr(block, name, None)


class LLM(Protocol):
    def call(self, **kwargs: Any) -> Reply: ...


class AnthropicLLM:
    """Real client. Retries on transient failures with jittered backoff."""

    def __init__(self, api_key: str | None = None, adaptive_thinking: bool = True):
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("pip install anthropic") from e
        self._anthropic = anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.adaptive_thinking = adaptive_thinking

    def call(
        self,
        *,
        model: str,
        messages: list[dict],
        system: str | list | None = None,
        tools: list[dict] | None = None,
        tool_choice: dict | None = None,
        max_tokens: int = 16_000,
        effort: str | None = None,
        thinking: bool | None = None,
        stream: bool | None = None,
    ) -> Reply:
        params: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system is not None:
            params["system"] = system
        if tools:
            params["tools"] = tools
        if tool_choice:
            params["tool_choice"] = tool_choice
        if effort:
            params["output_config"] = {"effort": effort}

        want_thinking = self.adaptive_thinking if thinking is None else thinking
        # Omit entirely when off — sending {"type":"disabled"} is a 400 on Fable 5.
        if want_thinking:
            params["thinking"] = {"type": "adaptive"}

        # Stream long generations so we don't trip the non-streaming request timeout.
        if stream is None:
            stream = max_tokens > 8_000

        last_err: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                if stream:
                    with self.client.messages.stream(**params) as s:
                        msg = s.get_final_message()
                else:
                    msg = self.client.messages.create(**params)
                return Reply(
                    content=list(msg.content),
                    stop_reason=msg.stop_reason,
                    usage=Usage(
                        input_tokens=getattr(msg.usage, "input_tokens", 0) or 0,
                        output_tokens=getattr(msg.usage, "output_tokens", 0) or 0,
                        cache_read_tokens=getattr(msg.usage, "cache_read_input_tokens", 0) or 0,
                    ),
                )
            except Exception as e:  # noqa: BLE001 - we re-raise below
                last_err = e
                status = getattr(e, "status_code", None)
                if status is not None and status not in RETRYABLE_STATUS:
                    raise
                if attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(min(2**attempt + random.random(), 30))
        raise last_err  # type: ignore[misc]


def structured(
    llm: LLM,
    *,
    model: str,
    system: str,
    prompt: str,
    schema: dict,
    tool_name: str,
    max_tokens: int = 8_000,
    effort: str | None = None,
) -> dict:
    """Get JSON out of the model reliably by forcing a single tool call.

    More robust than 'reply only in JSON' + parsing: the API validates the shape.
    """
    tool = {
        "name": tool_name,
        "description": f"Emit the {tool_name} result. Call this exactly once.",
        "input_schema": schema,
    }
    reply = llm.call(
        model=model,
        system=system,
        messages=[{"role": "user", "content": prompt}],
        tools=[tool],
        tool_choice={"type": "tool", "name": tool_name},
        max_tokens=max_tokens,
        effort=effort,
        stream=False,
    )
    out = reply.first_tool_input()
    if out is None:
        raise RuntimeError(f"{tool_name}: model returned no tool call")
    return out

"""LLM clients behind one tiny protocol: `call(**kw) -> Reply`.

Two implementations:

- AnthropicLLM — the hosted Messages API, via the anthropic SDK.
- LocalLLM — any OpenAI-compatible chat server (Ollama, LM Studio, llama.cpp
  server, vLLM). Stdlib urllib only, so the whole system runs offline against
  a model on localhost with zero cloud dependencies installed.

Both are deliberately thin: the agent loop needs raw content blocks and
stop_reason to do approval gating, budget accounting, and pause_turn handling.

Anthropic API notes that differ from older patterns (verified against anthropic/skills):
  - thinking={"type": "adaptive"}. budget_tokens is a 400 on Opus 4.8 / Sonnet 5 / Fable 5.
  - Never send thinking={"type": "disabled"} to Fable 5 -> 400. Omit instead.
  - Sampling params (temperature/top_p) are rejected on Opus 4.7+. We never send them.
  - Effort goes in output_config, not top level.

LocalLLM accepts the same keyword arguments and simply ignores the ones a
local server has no equivalent for (effort, thinking, cache_control).
"""

from __future__ import annotations

import json
import random
import re
import time
import urllib.error
import urllib.request
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
                parts.append(_battr(b, "text") or "")
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


class LocalLLM:
    """OpenAI-compatible chat client for local servers: Ollama, LM Studio,
    llama.cpp server, vLLM. No SDK, no network beyond the configured base_url.

    Tool calling uses the OpenAI functions wire format, which all of the above
    support. Pick a tool-calling-capable model (e.g. qwen3, qwen2.5-coder,
    llama3.1+, mistral) — the agent loop is tool-driven. `structured()` has a
    JSON-from-text fallback for models that ignore a forced tool choice.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        api_key: str | None = None,
        timeout: int = 600,
        max_retries: int = MAX_RETRIES,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max(1, max_retries)

    def call(
        self,
        *,
        model: str,
        messages: list[dict],
        system: str | list | None = None,
        tools: list[dict] | None = None,
        tool_choice: dict | None = None,
        max_tokens: int = 16_000,
        effort: str | None = None,      # accepted, ignored
        thinking: bool | None = None,   # accepted, ignored
        stream: bool | None = None,     # accepted, ignored (single HTTP call)
    ) -> Reply:
        payload: dict[str, Any] = {
            "model": model,
            "messages": self._to_openai(system, messages),
            "max_tokens": max_tokens,
        }
        # Server tools (web_search etc.) have no input_schema and no local
        # equivalent — they are filtered out rather than mistranslated.
        fn_tools = [self._tool(t) for t in (tools or []) if isinstance(t, dict) and "input_schema" in t]
        if fn_tools:
            payload["tools"] = fn_tools
            if tool_choice:
                payload["tool_choice"] = self._choice(tool_choice)
        data = self._post("/chat/completions", payload)
        return self._reply(data)

    # --- translation --------------------------------------------------------
    @staticmethod
    def _system_text(system: str | list | None) -> str:
        if system is None:
            return ""
        if isinstance(system, str):
            return system
        return "\n".join((_battr(b, "text") or "") for b in system if _btype(b) == "text")

    def _to_openai(self, system: str | list | None, messages: list[dict]) -> list[dict]:
        out: list[dict] = []
        if s := self._system_text(system):
            out.append({"role": "system", "content": s})
        for m in messages:
            role, content = m.get("role", "user"), m.get("content")
            if isinstance(content, str):
                out.append({"role": role, "content": content})
                continue
            if role == "assistant":
                texts, calls = [], []
                for b in content or []:
                    t = _btype(b)
                    if t == "text":
                        texts.append(_battr(b, "text") or "")
                    elif t == "tool_use":
                        calls.append({
                            "id": _battr(b, "id") or f"call_{len(calls)}",
                            "type": "function",
                            "function": {
                                "name": _battr(b, "name") or "",
                                "arguments": json.dumps(_battr(b, "input") or {}, default=str),
                            },
                        })
                    # thinking blocks are dropped: local servers have no slot for them
                msg: dict[str, Any] = {"role": "assistant", "content": "\n".join(t for t in texts if t) or None}
                if calls:
                    msg["tool_calls"] = calls
                out.append(msg)
            else:
                texts = []
                for b in content or []:
                    t = _btype(b)
                    if t == "tool_result":
                        c = _battr(b, "content")
                        s = c if isinstance(c, str) else json.dumps(c, default=str)
                        if _battr(b, "is_error"):
                            s = f"[tool errored]\n{s}"
                        out.append({
                            "role": "tool",
                            "tool_call_id": _battr(b, "tool_use_id") or "",
                            "content": s,
                        })
                    elif t == "text":
                        texts.append(_battr(b, "text") or "")
                if texts:
                    out.append({"role": role, "content": "\n".join(texts)})
        return out

    @staticmethod
    def _tool(t: dict) -> dict:
        return {
            "type": "function",
            "function": {
                "name": t.get("name", ""),
                "description": t.get("description", ""),
                "parameters": t.get("input_schema") or {"type": "object", "properties": {}},
            },
        }

    @staticmethod
    def _choice(tc: dict) -> Any:
        if tc.get("type") == "tool" and tc.get("name"):
            return {"type": "function", "function": {"name": tc["name"]}}
        if tc.get("type") == "any":
            return "required"
        return "auto"

    @staticmethod
    def _reply(data: dict) -> Reply:
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        blocks: list[dict] = []
        if msg.get("content"):
            blocks.append({"type": "text", "text": msg["content"]})
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            if not isinstance(args, dict):
                args = {}
            blocks.append({
                "type": "tool_use",
                "id": tc.get("id") or f"call_{len(blocks)}",
                "name": fn.get("name", ""),
                "input": args,
            })
        finish = choice.get("finish_reason")
        stop = {
            "tool_calls": "tool_use",
            "function_call": "tool_use",
            "stop": "end_turn",
            "length": "max_tokens",
        }.get(finish, finish or "end_turn")
        # Some servers report finish_reason "stop" even when tool calls came back.
        if stop == "end_turn" and any(b.get("type") == "tool_use" for b in blocks):
            stop = "tool_use"
        u = data.get("usage") or {}
        return Reply(
            content=blocks,
            stop_reason=stop,
            usage=Usage(
                input_tokens=u.get("prompt_tokens", 0) or 0,
                output_tokens=u.get("completion_tokens", 0) or 0,
            ),
        )

    # --- transport ----------------------------------------------------------
    def _post(self, path: str, payload: dict) -> dict:
        url = self.base_url + path
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                detail = ""
                try:
                    detail = e.read().decode("utf-8", "replace")[:400]
                except Exception:  # noqa: BLE001
                    pass
                last_err = RuntimeError(f"local LLM HTTP {e.code} from {url}: {detail}")
                if e.code not in RETRYABLE_STATUS:
                    raise last_err from e
            except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as e:
                last_err = RuntimeError(
                    f"cannot reach local LLM at {self.base_url} ({e}). Is the server "
                    "running? (e.g. `ollama serve`, LM Studio's local server, llama-server)"
                )
            if attempt == self.max_retries - 1:
                raise last_err  # type: ignore[misc]
            time.sleep(min(2**attempt + random.random(), 30))
        raise last_err  # type: ignore[misc]  # pragma: no cover


def make_llm(cfg) -> LLM:
    """Build the right client for a Config. `cfg.provider`: "anthropic" | "local"."""
    if cfg.provider == "local":
        return LocalLLM(base_url=cfg.base_url, api_key=cfg.api_key)
    return AnthropicLLM(api_key=cfg.api_key)


def _json_from_text(s: str) -> dict | None:
    """Best-effort JSON object out of prose — the fallback for local models that
    answer a forced tool call with text. Fenced block first, then outermost braces."""
    s = (s or "").strip()
    if not s:
        return None
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, re.S)
    candidates = [m.group(1)] if m else []
    lo, hi = s.find("{"), s.rfind("}")
    if lo != -1 and hi > lo:
        candidates.append(s[lo : hi + 1])
    for c in candidates:
        try:
            out = json.loads(c)
        except json.JSONDecodeError:
            continue
        if isinstance(out, dict):
            return out
    return None


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
        # Local models sometimes answer a forced tool call with prose containing
        # the JSON. Salvage it rather than failing the whole pipeline.
        out = _json_from_text(reply.text())
    if out is None:
        raise RuntimeError(f"{tool_name}: model returned no tool call and no parseable JSON")
    return out

"""Tests for the local-provider path: LocalLLM's OpenAI wire translation, and
the full agent loop driven over real HTTP against a stub local server.

The stub is a genuine http.server on 127.0.0.1 speaking /chat/completions, so
these tests exercise the exact seam a real Ollama / LM Studio / llama.cpp /
vLLM deployment sits behind — request encoding, tool-call translation both
ways, and the retry/transport code — with no network beyond loopback.
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forge.config import Budget, Config
from forge.llm import LocalLLM, _json_from_text, make_llm, structured
from forge.loop import Agent


# --------------------------------------------------------------------------- stub server
class StubHandler(BaseHTTPRequestHandler):
    """Replays a scripted queue of OpenAI-style responses; records requests."""

    script: list[dict] = []
    requests: list[dict] = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append({"path": self.path, "body": body})
        msg = type(self).script.pop(0) if type(self).script else {"content": "done"}
        payload = {
            "choices": [{
                "message": {"role": "assistant", **msg},
                "finish_reason": "tool_calls" if msg.get("tool_calls") else "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        out = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):  # keep test output clean
        pass


@pytest.fixture
def stub_server():
    StubHandler.script = []
    StubHandler.requests = []
    srv = HTTPServer(("127.0.0.1", 0), StubHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/v1", StubHandler
    srv.shutdown()


def fn_call(name, args, id="call_1"):
    return {"id": id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


# --------------------------------------------------------------------------- translation
def test_messages_translate_to_openai_format():
    llm = LocalLLM()
    system = [{"type": "text", "text": "be helpful", "cache_control": {"type": "ephemeral"}}]
    messages = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "hmm"},
            {"type": "text", "text": "on it"},
            {"type": "tool_use", "id": "t1", "name": "grep", "input": {"pattern": "x"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "hit", "is_error": False},
        ]},
    ]
    out = llm._to_openai(system, messages)
    assert out[0] == {"role": "system", "content": "be helpful"}
    assert out[1] == {"role": "user", "content": "task"}
    asst = out[2]
    assert asst["content"] == "on it"                       # thinking dropped
    assert asst["tool_calls"][0]["function"]["name"] == "grep"
    assert json.loads(asst["tool_calls"][0]["function"]["arguments"]) == {"pattern": "x"}
    assert out[3] == {"role": "tool", "tool_call_id": "t1", "content": "hit"}


def test_server_tools_filtered_from_local_payload(stub_server):
    """web_search has no input_schema and no local equivalent; it must not be
    mistranslated into a function tool the local model would try to call."""
    base_url, handler = stub_server
    handler.script = [{"content": "hi"}]
    llm = LocalLLM(base_url=base_url)
    llm.call(model="m", messages=[{"role": "user", "content": "x"}], tools=[
        {"name": "grep", "description": "d", "input_schema": {"type": "object", "properties": {}}},
        {"type": "web_search_20260209", "name": "web_search", "max_uses": 5},
    ])
    sent = handler.requests[0]["body"]["tools"]
    assert [t["function"]["name"] for t in sent] == ["grep"]


def test_reply_parses_tool_calls_and_stop_reason(stub_server):
    base_url, handler = stub_server
    handler.script = [{"content": None, "tool_calls": [fn_call("write_file", {"path": "a", "content": "b"})]}]
    r = LocalLLM(base_url=base_url).call(model="m", messages=[{"role": "user", "content": "x"}])
    assert r.stop_reason == "tool_use"
    (u,) = r.tool_uses()
    assert u["name"] == "write_file" and u["input"] == {"path": "a", "content": "b"}


def test_forced_tool_choice_translates(stub_server):
    base_url, handler = stub_server
    handler.script = [{"content": None, "tool_calls": [fn_call("emit", {"k": 1})]}]
    LocalLLM(base_url=base_url).call(
        model="m", messages=[{"role": "user", "content": "x"}],
        tools=[{"name": "emit", "description": "d", "input_schema": {"type": "object"}}],
        tool_choice={"type": "tool", "name": "emit"},
    )
    assert handler.requests[0]["body"]["tool_choice"] == {"type": "function", "function": {"name": "emit"}}


def test_unreachable_server_raises_helpfully():
    llm = LocalLLM(base_url="http://127.0.0.1:9", timeout=1, max_retries=1)  # port 9: nothing listens
    with pytest.raises(RuntimeError, match="cannot reach local LLM"):
        llm.call(model="m", messages=[{"role": "user", "content": "x"}], max_tokens=10)


# --------------------------------------------------------------------------- json fallback
def test_json_from_text_fenced_and_bare():
    assert _json_from_text('here:\n```json\n{"a": 1}\n```') == {"a": 1}
    assert _json_from_text('prefix {"a": {"b": 2}} suffix') == {"a": {"b": 2}}
    assert _json_from_text("no json here") is None


def test_structured_salvages_prose_json(stub_server):
    """A local model that answers a forced tool call with prose-wrapped JSON
    must not sink the acquisition pipeline."""
    base_url, handler = stub_server
    handler.script = [{"content": 'Sure! ```json\n{"name": "x", "flags": []}\n```'}]
    out = structured(LocalLLM(base_url=base_url), model="m", system="s", prompt="p",
                     schema={"type": "object"}, tool_name="emit_skill")
    assert out == {"name": "x", "flags": []}


# --------------------------------------------------------------------------- end to end
def test_agent_loop_end_to_end_over_local_http(tmp_path, stub_server):
    """The whole offline path: Agent -> LocalLLM -> HTTP -> stub server, with a
    real tool dispatch (write_file hits the real sandbox) in the middle."""
    base_url, handler = stub_server
    handler.script = [
        {"content": None, "tool_calls": [fn_call("write_file", {"path": "out.txt", "content": "hello"})]},
        {"content": "wrote the file"},
    ]
    cfg = Config(
        workspace=tmp_path / "ws", skills_dir=tmp_path / "sk",
        provider="local", base_url=base_url, offline=True,
        agent_model="test-model",
        budget=Budget(max_turns=5, max_wall_seconds=30), max_eval_rounds=1,
    )
    res = Agent(cfg, make_llm(cfg)).run("make out.txt")

    assert (cfg.workspace / "out.txt").read_text() == "hello"
    assert res.answer == "wrote the file"
    assert res.stop == "end_turn" and res.turns == 2

    # offline: no web_search advertised to the local model
    first = handler.requests[0]["body"]
    assert all(t["function"]["name"] != "web_search" for t in first["tools"])
    # second call carried the tool result back in OpenAI form
    roles = [m["role"] for m in handler.requests[1]["body"]["messages"]]
    assert "tool" in roles
    tool_msg = next(m for m in handler.requests[1]["body"]["messages"] if m["role"] == "tool")
    assert "wrote" in tool_msg["content"]


def test_local_provider_defaults_all_models(tmp_path):
    cfg = Config(workspace=tmp_path, provider="local", agent_model="qwen3-coder:30b")
    assert cfg.evaluator_model == "qwen3-coder:30b"
    assert cfg.utility_model == "qwen3-coder:30b"
    assert cfg.skill_author_model == "qwen3-coder:30b"

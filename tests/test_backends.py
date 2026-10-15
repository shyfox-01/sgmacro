"""LLM backends with fake transports: request shapes, history handling, cache, fallbacks."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests as real_requests

from sgmacro.llm import backends
from sgmacro.llm.backends import (AnthropicBackend, BackendUnavailable, OllamaBackend, OpenAIBackend,
                                  ProviderError, ResponseCache, ToolCall, ToolResult, ToolSpec, request_key,
                                  resolve_backend)
from sgmacro.llm.summary import SCHEMA, summarise
from sgmacro.pipeline import RunConfig, analyse, fetch_stage
from sgmacro.registry import load_registry
from tests.conftest import REGISTRY_PATH, STORE_ROOT


class FakeMessages:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def block(**kw):
    return SimpleNamespace(**kw)


def anthropic_client(responses):
    messages = FakeMessages(responses)
    return SimpleNamespace(beta=SimpleNamespace(messages=messages)), messages


def test_anthropic_json_request_shape():
    response = SimpleNamespace(stop_reason="end_turn", stop_details=None, model="claude-opus-5-5",
                               content=[block(type="thinking", thinking=""), block(type="text", text='{"a": 1}')])
    client, messages = anthropic_client([response])
    backend = AnthropicBackend(client=client)
    assert backend.complete_json("sys", "prompt", {"type": "object"}) == '{"a": 1}'
    call = messages.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["fallbacks"] == "default" and call["betas"] == ["server-side-fallback-2026-07-01"]
    assert call["output_config"]["format"] == {"type": "json_schema", "schema": {"type": "object"}}
    assert "temperature" not in call and "thinking" not in call and "tool_choice" not in call


def test_anthropic_refusal_is_unavailable():
    client, _ = anthropic_client([SimpleNamespace(stop_reason="refusal", stop_details={"category": "x"}, content=[])])
    with pytest.raises(BackendUnavailable, match="declined"):
        AnthropicBackend(client=client).complete_json("s", "p", {})


def test_anthropic_session_is_append_only_with_strict_tools():
    first = SimpleNamespace(stop_reason="tool_use", model="m", content=[
        block(type="thinking", thinking="", signature="sig"),
        block(type="tool_use", id="tu_1", name="list_registry", input={})])
    second = SimpleNamespace(stop_reason="end_turn", model="m", content=[block(type="text", text="done")])
    client, messages = anthropic_client([first, second])
    session = AnthropicBackend(client=client).session("sys", [ToolSpec("list_registry", "d", {"type": "object"})])
    turn = session.send_user("go")
    assert turn.tool_calls[0].name == "list_registry"
    turn = session.send_tool_results([ToolResult(turn.tool_calls[0], "ok")])
    assert turn.text == "done"
    second_call = messages.calls[1]
    assert second_call["tools"][0]["strict"] is True
    # the assistant turn (thinking block included) is echoed back verbatim
    assert second_call["messages"][1]["content"] is first.content
    assert second_call["messages"][2]["content"][0]["tool_use_id"] == "tu_1"


def test_anthropic_without_credentials_is_unavailable(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    with pytest.raises(BackendUnavailable):
        AnthropicBackend()


def response(body, status=200):
    return SimpleNamespace(status_code=status, raise_for_status=lambda: None, json=lambda: body, text=str(body))


class FakeRequests:
    RequestException, Timeout, ConnectionError = (real_requests.RequestException, real_requests.Timeout,
                                                  real_requests.ConnectionError)

    def __init__(self, models, replies, capabilities=None):
        self.models = models
        self.replies = list(replies)
        self.capabilities = capabilities or {}
        self.posts = []
        self.shown = []

    def get(self, url, timeout=None):
        return response({"models": [{"name": m} for m in self.models]})

    def post(self, url, json=None, timeout=None):
        if url.endswith("/api/show"):
            self.shown.append(json["model"])
            return response({"capabilities": self.capabilities.get(json["model"], [])})
        self.posts.append(json)
        return response(self.replies.pop(0))


def test_ollama_tool_session_and_schema(monkeypatch):
    fake = FakeRequests(["qwen3.5:9b"], [
        {"message": {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "search_singstat", "arguments": '{"keyword": "rent"}'}}]}, "done_reason": "stop"},
        {"message": {"role": "assistant", "content": "finished"}, "done_reason": "stop",
         "prompt_eval_count": 1200, "eval_count": 34},
        {"message": {"content": '{"ok": true}'}},
    ])
    monkeypatch.setattr(backends, "requests", fake)
    backend = OllamaBackend(model="qwen3.5:9b")
    session = backend.session("sys", [ToolSpec("search_singstat", "d", {"type": "object"})])
    turn = session.send_user("go")
    assert turn.tool_calls == [ToolCall("call_1", "search_singstat", {"keyword": "rent"})]
    assert turn.context_used is None                              # no counts reported
    after = session.send_tool_results([ToolResult(turn.tool_calls[0], "M1 | x")])
    assert after.text == "finished" and after.context_used == 1234
    assert backend.context_tokens == 32768
    # the payload holds the live history list, which has since gained the reply
    assert fake.posts[1]["messages"][-2] == {"role": "tool", "tool_name": "search_singstat", "content": "M1 | x"}
    assert backend.complete_json("s", "p", {"type": "object"}) == '{"ok": true}'
    assert fake.posts[2]["format"] == {"type": "object"} and fake.posts[2]["options"]["temperature"] == 0


def test_a_long_ollama_session_sends_older_results_shortened(monkeypatch):
    """Live, the conversation filled the 32k window by step 16. Older tool results
    and assistant texts are shortened in what is sent; recent ones stay whole."""
    replies = [{"message": {"role": "assistant", "content": "thinking out loud " * 40,
                            "tool_calls": [{"function": {"name": "search_singstat", "arguments": {"keyword": str(i)}}}]},
                "done_reason": "stop"} for i in range(12)]
    fake = FakeRequests(["qwen3.5:9b"], replies)
    monkeypatch.setattr(backends, "requests", fake)
    session = OllamaBackend(model="qwen3.5:9b").session("sys", [ToolSpec("search_singstat", "d", {"type": "object"})])
    turn = session.send_user("go")
    for i in range(10):
        turn = session.send_tool_results([ToolResult(turn.tool_calls[0], f"result {i} " + "x" * 2000)])
    sent = fake.posts[-1]["messages"]
    tools = [m["content"] for m in sent if m["role"] == "tool"]
    assert len(tools) == 10 and all(len(t) > 2000 for t in tools[-6:])          # the latest six whole
    assert all(len(t) < 600 and "left out to save context" in t for t in tools[:4])
    assert tools[0].startswith("result 0 ")                                       # the start is kept
    texts = [m["content"] for m in sent if m["role"] == "assistant"]
    assert len(texts[0]) < 400 and len(texts[-1]) > 600
    assert sent[0] == {"role": "system", "content": "sys"}


def test_ollama_missing_model_is_unavailable(monkeypatch):
    monkeypatch.setattr(backends, "requests", FakeRequests(["llama"], []))
    with pytest.raises(BackendUnavailable, match="not installed in Ollama"):
        OllamaBackend(model="qwen3.5:9b")


def test_ollama_picks_an_installed_tool_model_when_the_default_is_missing(monkeypatch):
    monkeypatch.delenv("SGMACRO_OLLAMA_MODEL", raising=False)
    installed = ["gemma:2b", "llama3.1:8b", "nomic-embed-text:latest", "qwen3:4b"]
    caps = {"gemma:2b": ["completion"], "llama3.1:8b": ["completion", "tools"], "qwen3:4b": ["completion", "tools"],
            "nomic-embed-text:latest": ["embedding"]}
    fake = FakeRequests(installed, [], caps)
    monkeypatch.setattr(backends, "requests", fake)
    backend = OllamaBackend()
    assert backend.model == "llama3.1:8b" and backend.picked          # first by name with tool support
    assert fake.shown == ["gemma:2b", "llama3.1:8b"]                 # stop at the first usable model
    monkeypatch.setenv("SGMACRO_OLLAMA_MODEL", "gemma")                  # a name without a tag means :latest
    with pytest.raises(BackendUnavailable, match="gemma"):
        OllamaBackend()
    monkeypatch.setenv("SGMACRO_OLLAMA_MODEL", "gemma:2b")
    assert OllamaBackend().model == "gemma:2b" and not OllamaBackend().picked   # an explicit choice is honoured
    monkeypatch.delenv("SGMACRO_OLLAMA_MODEL")
    monkeypatch.setattr(backends, "requests", FakeRequests(["gemma:2b"], [], caps))
    with pytest.raises(BackendUnavailable, match="ollama pull qwen3.5:9b"):
        OllamaBackend()


def unavailable(reason):
    def construct(*args, **kwargs):
        raise BackendUnavailable(reason)
    return construct


def test_an_unreachable_local_ollama_says_whether_it_is_installed(monkeypatch, tmp_path):
    class Down(FakeRequests):
        def get(self, url, timeout=None):
            raise real_requests.ConnectionError("refused")

    monkeypatch.setattr(backends, "requests", Down([], []))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setattr(backends.shutil, "which", lambda name: r"C:\Ollama\ollama.exe")
    with pytest.raises(BackendUnavailable, match="installed but nothing answered at that address: start the Ollama"):
        OllamaBackend()
    monkeypatch.setattr(backends.shutil, "which", lambda name: None)
    with pytest.raises(BackendUnavailable, match="does not appear to be installed"):
        OllamaBackend()
    with pytest.raises(BackendUnavailable) as remote:
        OllamaBackend(base_url="http://gpu-box.example:11434")
    assert "installed" not in str(remote.value)                    # no local hint for a remote server


def test_resolution_modes(monkeypatch):
    assert resolve_backend("cache")[0] is None and resolve_backend("off")[0] is None
    with pytest.raises(ValueError):
        resolve_backend("gpt")
    monkeypatch.setattr(backends, "AnthropicBackend", unavailable("no key"))
    monkeypatch.setattr(backends, "OpenAIBackend", unavailable("not configured"))
    monkeypatch.setattr(backends, "OllamaBackend", unavailable("no server"))
    backend, reasons = resolve_backend("auto")
    assert backend is None and [r.split(":")[0] for r in reasons] == ["anthropic", "openai", "ollama"]
    with pytest.raises(BackendUnavailable):
        resolve_backend("ollama")
    monkeypatch.setattr(backends, "OpenAIBackend", lambda **kw: SimpleNamespace(name="openai", model="m"))
    assert resolve_backend("auto")[0].name == "openai"                  # first available, in auto order


def test_offline_never_constructs_the_claude_backend(monkeypatch):
    def remote():
        raise AssertionError("an offline run must not reach the Claude API")

    monkeypatch.setattr(backends, "AnthropicBackend", remote)
    monkeypatch.setattr(backends, "OpenAIBackend", unavailable("not configured"))
    monkeypatch.setattr(backends, "OllamaBackend", unavailable("no server"))
    backend, reasons = resolve_backend("auto", offline=True)
    assert backend is None and any("offline" in r for r in reasons)
    with pytest.raises(BackendUnavailable, match="offline"):
        resolve_backend("anthropic", offline=True)


@pytest.mark.parametrize("preference", ["ollama", "auto", "cache", "off"])
def test_offline_remote_ollama_sends_no_requests(monkeypatch, preference):
    def unexpected(*args, **kwargs):
        pytest.fail("offline resolution must reject a remote URL before any request")

    monkeypatch.setenv("SGMACRO_OLLAMA_URL", "https://remote-model.example")
    monkeypatch.delenv("SGMACRO_OPENAI_BASE_URL", raising=False)
    monkeypatch.setattr(backends.requests, "get", unexpected)
    monkeypatch.setattr(backends.requests, "post", unexpected)
    if preference == "ollama":
        with pytest.raises(BackendUnavailable, match="remote service.*offline"):
            resolve_backend(preference, offline=True)
    else:
        backend, _ = resolve_backend(preference, offline=True)
        assert backend is None


@pytest.mark.parametrize("url", ["http://localhost:11434", "http://127.0.0.1:11434", "http://[::1]:11434"])
def test_offline_local_ollama_remains_usable(monkeypatch, url):
    monkeypatch.setenv("SGMACRO_OLLAMA_URL", url)
    monkeypatch.delenv("SGMACRO_OLLAMA_MODEL", raising=False)
    fake = FakeRequests([backends.OLLAMA_MODEL], [{"message": {"content": '{"ok": true}'}}])
    monkeypatch.setattr(backends, "requests", fake)
    backend, _ = resolve_backend("ollama", offline=True)
    assert backend.complete_json("s", "p", {}) == '{"ok": true}'


# ---------------------------------------------------------------------------
# OpenAI-compatible services
# ---------------------------------------------------------------------------

KEY = "sk-test-0123456789abcdefghijklmnop"


class FakeHTTP:
    """requests stand-in for an OpenAI-compatible server: records every request."""

    RequestException, Timeout, ConnectionError = (real_requests.RequestException, real_requests.Timeout,
                                                  real_requests.ConnectionError)

    def __init__(self, models=("m1", "m2"), replies=(), models_status=200):
        self.models, self.models_status = list(models), models_status
        self.replies = list(replies)
        self.requests = []

    def get(self, url, headers=None, timeout=None):
        self.requests.append(("GET", url, headers or {}, None))
        return response({"data": [{"id": m} for m in self.models]}, self.models_status)

    def post(self, url, json=None, headers=None, timeout=None):
        self.requests.append(("POST", url, headers or {}, deepcopy(json)))
        body, status = self.replies.pop(0)
        return response(body, status)


def reply(message, finish="stop"):
    return {"choices": [{"message": message, "finish_reason": finish}]}, 200


@pytest.fixture
def openai_env(monkeypatch):
    for var in ("SGMACRO_OPENAI_BASE_URL", "SGMACRO_OPENAI_API_KEY", "SGMACRO_OPENAI_MODEL", "OPENAI_API_KEY",
                "GEMINI_API_KEY", "DEEPSEEK_API_KEY", "DASHSCOPE_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def test_openai_json_request_shape_and_schema_fallback(openai_env):
    openai_env.setenv("SGMACRO_OPENAI_BASE_URL", "deepseek")
    openai_env.setenv("DEEPSEEK_API_KEY", KEY)
    openai_env.setenv("SGMACRO_OPENAI_MODEL", "m1")
    http = FakeHTTP(replies=[({"error": {"message": "response_format json_schema is unavailable"}}, 400),
                             reply({"role": "assistant", "content": '```json\n{"a": 1}\n```'})])
    openai_env.setattr(backends, "requests", http)
    backend = OpenAIBackend()
    assert backend.model == "m1" and backend.host == "api.deepseek.com"
    assert backend.complete_json("sys", "prompt", {"type": "object"}) == '{"a": 1}'
    (_, url, headers, _), first, second = http.requests
    assert url == "https://api.deepseek.com/models" and headers == {"Authorization": f"Bearer {KEY}"}
    assert first[3]["model"] == "m1" and first[3]["response_format"]["type"] == "json_schema"
    assert first[3]["response_format"]["json_schema"]["schema"] == {"type": "object"}
    # the fallback asks for a JSON object and states the schema in the instructions
    assert second[3]["response_format"] == {"type": "json_object"}
    assert '{"type": "object"}' in second[3]["messages"][0]["content"]
    assert "temperature" not in first[3] and "max_tokens" not in first[3]    # only what every service accepts


def test_openai_tool_session_without_reasoning_extension(openai_env):
    openai_env.setenv("SGMACRO_OPENAI_BASE_URL", "http://localhost:1234/v1")
    http = FakeHTTP(models=["local-model"], replies=[
        reply({"role": "assistant", "content": None,
               "tool_calls": [{"id": "c1", "type": "function",
                               "function": {"name": "search_singstat", "arguments": '{"keyword": "rent"}'}}]},
              "tool_calls"),
        reply({"role": "assistant", "content": "<think>done?</think>finished"})])
    openai_env.setattr(backends, "requests", http)
    backend = OpenAIBackend()
    assert backend.model == "local-model"                 # the only model a local server offers
    session = backend.session("sys", [ToolSpec("search_singstat", "d", {"type": "object"})])
    turn = session.send_user("go")
    assert turn.tool_calls == [ToolCall("c1", "search_singstat", {"keyword": "rent"})]
    assert turn.stop_reason == "tool_calls"
    assert session.send_tool_results([ToolResult(turn.tool_calls[0], "M1 | x")]).text == "finished"
    history = http.requests[-1][3]["messages"]
    assert history[2] == {"role": "assistant", "content": None, "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "search_singstat", "arguments": '{"keyword": "rent"}'}}]}
    assert history[3] == {"role": "tool", "tool_call_id": "c1", "content": "M1 | x"}
    assert all(h == {} for _, _, h, _ in http.requests)   # a local server without a key gets no header


def test_deepseek_tool_history_retains_reasoning_across_rounds(openai_env):
    openai_env.setenv("SGMACRO_OPENAI_BASE_URL", "deepseek")
    openai_env.setenv("DEEPSEEK_API_KEY", KEY)
    openai_env.setenv("SGMACRO_OPENAI_MODEL", "m1")

    class ThinkingServer(FakeHTTP):
        def post(self, url, json=None, **kwargs):
            previous = [m for m in json["messages"] if m["role"] == "assistant"]
            # The mock enforces the protocol at the receiving boundary, including
            # old tool rounds and an earlier final answer on a new user turn.
            for index, message in enumerate(previous, 1):
                assert message["reasoning_content"] == f"reasoning-{index}"
                assert "provider_diagnostic" not in message
            index = len(previous) + 1
            message = {"role": "assistant", "content": "done" if index >= 3 else None,
                       "reasoning_content": f"reasoning-{index}", "provider_diagnostic": "not history"}
            if index <= 2:
                message["tool_calls"] = [{"id": f"c{index}", "type": "function", "function": {
                    "name": "list_registry", "arguments": "{}"}}]
            self.replies.append(reply(message, "tool_calls" if index <= 2 else "stop"))
            return super().post(url, json=json, **kwargs)

    http = ThinkingServer()
    openai_env.setattr(backends, "requests", http)
    session = OpenAIBackend().session("sys", [ToolSpec("list_registry", "d", {"type": "object"})])
    turn = session.send_user("go")
    for index in (1, 2):
        assert turn.tool_calls[0].id == f"c{index}"
        turn = session.send_tool_results([ToolResult(turn.tool_calls[0], "ok")])
    assert turn.text == "done" and not turn.tool_calls
    assert session.send_user("follow up").text == "done"
    assert len([r for r in http.requests if r[0] == "POST"]) == 4


@pytest.mark.parametrize("status", [401, 403, 400, 429, 500, 503])
def test_failed_model_probe_is_unavailable_and_auto_skips_it(openai_env, status):
    openai_env.setenv("SGMACRO_OPENAI_BASE_URL", "deepseek")
    openai_env.setenv("DEEPSEEK_API_KEY", KEY)
    openai_env.setenv("SGMACRO_OPENAI_MODEL", "m1")
    http = FakeHTTP(models_status=status)
    openai_env.setattr(backends, "requests", http)
    openai_env.setattr(backends, "AnthropicBackend", unavailable("not configured"))
    openai_env.setattr(backends, "OllamaBackend", unavailable("no server"))
    row = next(row for row in backends.backend_status() if row[0] == "openai")
    assert row[1] is False and f"HTTP {status}" in row[2]
    assert "key accepted" not in row[2]
    backend, reasons = resolve_backend("auto")
    assert backend is None and any(f"HTTP {status}" in r for r in reasons)
    assert all(r[0] == "GET" for r in http.requests)


@pytest.mark.parametrize("status", [404, 405, 501])
def test_unsupported_model_listing_requires_explicit_model_and_reports_unverified(openai_env, status):
    openai_env.setenv("SGMACRO_OPENAI_BASE_URL", "http://localhost:1234/v1")
    openai_env.setattr(backends, "requests", FakeHTTP(models_status=status))
    with pytest.raises(BackendUnavailable, match="SGMACRO_OPENAI_MODEL"):
        OpenAIBackend()
    openai_env.setenv("SGMACRO_OPENAI_MODEL", "m1")
    openai_env.setattr(backends, "AnthropicBackend", unavailable("not configured"))
    openai_env.setattr(backends, "OllamaBackend", unavailable("no server"))
    row = next(row for row in backends.backend_status() if row[0] == "openai")
    assert row[1] is True and "model and key unverified" in row[2]
    assert "key accepted" not in row[2]


def test_openai_configuration_and_key_safety(openai_env):
    http = FakeHTTP()
    openai_env.setattr(backends, "requests", http)
    with pytest.raises(BackendUnavailable, match="not configured"):
        OpenAIBackend()
    openai_env.setenv("SGMACRO_OPENAI_BASE_URL", "deepseek")
    with pytest.raises(BackendUnavailable, match="DEEPSEEK_API_KEY"):
        OpenAIBackend()
    # a provider's own key is never sent to another address
    openai_env.setenv("DEEPSEEK_API_KEY", KEY)
    openai_env.setenv("SGMACRO_OPENAI_BASE_URL", "https://proxy.example.com/v1")
    with pytest.raises(BackendUnavailable, match="SGMACRO_OPENAI_API_KEY"):
        OpenAIBackend()
    # nor over plain HTTP to a remote host
    openai_env.setenv("SGMACRO_OPENAI_API_KEY", KEY)
    openai_env.setenv("SGMACRO_OPENAI_BASE_URL", "http://proxy.example.com/v1")
    with pytest.raises(BackendUnavailable, match="plain HTTP"):
        OpenAIBackend()
    assert http.requests == []                            # nothing was sent in any of these cases
    openai_env.setenv("SGMACRO_OPENAI_BASE_URL", "https://proxy.example.com/v1")
    with pytest.raises(BackendUnavailable, match="offline"):
        OpenAIBackend(offline=True)
    with pytest.raises(BackendUnavailable, match="SGMACRO_OPENAI_MODEL"):
        OpenAIBackend()                                   # two models listed, none chosen
    openai_env.setattr(backends, "requests", FakeHTTP(models_status=401))
    with pytest.raises(BackendUnavailable, match="rejected the API key") as rejected:
        OpenAIBackend()
    assert KEY not in str(rejected.value)


def test_an_empty_or_cut_off_reply_is_a_failure_not_an_answer(openai_env):
    """Seen live: a local server with a 4k context returned empty content once the
    agent's history outgrew it. That must fail loudly, not read as 'agent stopped'."""
    openai_env.setenv("SGMACRO_OPENAI_BASE_URL", "http://localhost:11434/v1")
    openai_env.setenv("SGMACRO_OPENAI_MODEL", "m1")
    openai_env.setattr(backends, "requests", FakeHTTP(replies=[
        reply({"role": "assistant", "content": "", "reasoning": "thinking..."}),
        reply({"role": "assistant", "content": ""}, "length")]))
    backend = OpenAIBackend()
    with pytest.raises(ProviderError, match="no content"):
        backend.complete_json("s", "p", {})
    with pytest.raises(ProviderError, match="context window"):
        backend.session("s", []).send_user("go")


@pytest.mark.parametrize("provider", ["openai", "ollama"])
@pytest.mark.parametrize("kind", ["empty", "thinking_only", "truncated_text", "truncated_tools"])
def test_agent_rejects_empty_and_truncated_replies(openai_env, provider, kind):
    message = {"role": "assistant", "content": ""}
    finish = "stop"
    if kind == "thinking_only":
        message["content"] = "<think>not a final answer</think>"
    elif kind.startswith("truncated"):
        finish = "length"
        message["content"] = "unfinished"
        if kind == "truncated_tools":
            message["tool_calls"] = [{"id": "c1", "function": {"name": "list_registry", "arguments": "{"}}]
    if provider == "openai":
        openai_env.setenv("SGMACRO_OPENAI_BASE_URL", "http://localhost:1234/v1")
        openai_env.setenv("SGMACRO_OPENAI_MODEL", "m1")
        openai_env.setattr(backends, "requests", FakeHTTP(replies=[reply(message, finish)]))
        backend = OpenAIBackend()
    else:
        openai_env.setattr(backends, "requests", FakeRequests([backends.OLLAMA_MODEL], [
            {"message": message, "done_reason": finish}]))
        backend = OllamaBackend(model=backends.OLLAMA_MODEL)
    expected = "finish_reason=length" if finish == "length" else "no content"
    with pytest.raises(ProviderError, match=expected):
        backend.session("s", []).send_user("go")


@pytest.mark.parametrize("provider", ["openai", "ollama", "anthropic"])
def test_truncated_summary_is_rejected_even_if_its_json_looks_complete(openai_env, provider):
    if provider == "openai":
        openai_env.setenv("SGMACRO_OPENAI_BASE_URL", "http://localhost:1234/v1")
        openai_env.setenv("SGMACRO_OPENAI_MODEL", "m1")
        openai_env.setattr(backends, "requests", FakeHTTP(replies=[reply({"content": "{}"}, "length")]))
        backend = OpenAIBackend()
    elif provider == "ollama":
        openai_env.setattr(backends, "requests", FakeRequests([backends.OLLAMA_MODEL], [
            {"message": {"content": "{}"}, "done_reason": "length"}]))
        backend = OllamaBackend(model=backends.OLLAMA_MODEL)
    else:
        client, _ = anthropic_client([SimpleNamespace(stop_reason="max_tokens",
                                                      content=[block(type="text", text="{}")])])
        backend = AnthropicBackend(client=client)
    with pytest.raises(ProviderError, match="cut off"):
        backend.complete_json("s", "p", {})


def test_an_empty_summary_response_is_not_cached(analysed, tmp_path, monkeypatch):
    monkeypatch.setattr("sgmacro.llm.summary.resolve_backend", lambda p, **kw: (FixedBackend("  "), []))
    summary = summarise(analysed, backend_preference="auto", cache_dir=tmp_path, log=lambda m: None)
    assert summary.source == "deterministic" and any("empty response" in n for n in summary.execution_notes)
    assert not list(tmp_path.iterdir())                  # a later run asks again instead of replaying nothing


def test_provider_errors_never_carry_the_key(openai_env):
    openai_env.setenv("SGMACRO_OPENAI_BASE_URL", "openai")
    openai_env.setenv("OPENAI_API_KEY", KEY)
    openai_env.setenv("SGMACRO_OPENAI_MODEL", "m1")
    openai_env.setattr(backends, "requests", FakeHTTP(replies=[
        ({"error": {"message": f"Incorrect API key provided: {KEY}. Header was Bearer {KEY}"}}, 500)]))
    with pytest.raises(ProviderError) as failed:
        OpenAIBackend().complete_json("s", "p", {})
    assert KEY not in str(failed.value) and "[key redacted]" in str(failed.value) and failed.value.status == 500


def test_nothing_containing_a_key_is_written(openai_env):
    from sgmacro.store import assert_no_credentials

    assert_no_credentials("a report about rents, risk-adjusted")
    for leaked in (f"error: {KEY}", "AIza" + "x" * 35, "Authorization: Bearer abcdefgh12345678"):
        with pytest.raises(ValueError, match="API key"):
            assert_no_credentials(leaked)
    openai_env.setenv("DEEPSEEK_API_KEY", "plain-looking-secret-value")    # no known shape, still caught
    with pytest.raises(ValueError, match="API key"):
        assert_no_credentials("echoed plain-looking-secret-value")


def test_models_command_reports_backends_without_showing_keys(openai_env):
    from typer.testing import CliRunner

    from sgmacro.cli import app

    class FakeClaude:
        name, model = "anthropic", "claude-opus-5-5"

    openai_env.setenv("ANTHROPIC_API_KEY", KEY)
    openai_env.setenv("GEMINI_API_KEY", "AIza" + "y" * 35)
    openai_env.setattr(backends, "AnthropicBackend", FakeClaude)
    openai_env.setattr(backends, "OllamaBackend", unavailable("no Ollama server at http://localhost:11434"))
    result = CliRunner().invoke(app, ["models"])
    assert result.exit_code == 0, result.output
    assert "anthropic  ready" in result.output and "openai     not ready  not configured" in result.output
    assert "`auto` would use: anthropic (claude-opus-5-5)" in result.output
    assert "GEMINI_API_KEY is set; to use it, set SGMACRO_OPENAI_BASE_URL=gemini" in result.output
    assert KEY not in result.output and "AIza" not in result.output


def test_unavailable_explicit_backend_falls_back_with_a_note(analysed, tmp_path, monkeypatch):
    monkeypatch.setattr(backends, "OllamaBackend", unavailable("no server"))
    lines = []
    summary = summarise(analysed, backend_preference="ollama", cache_dir=tmp_path, log=lines.append)
    assert summary.source == "deterministic" and summary.response_sha256 is None
    assert any("unavailable" in n for n in summary.execution_notes) and any("warning" in m for m in lines)
    assert summary.notes == []          # nothing about the content: the template simply wrote it


def test_cache_key_ignores_model_and_round_trips(tmp_path):
    cache = ResponseCache(tmp_path)
    key = request_key("s", "p", SCHEMA)
    assert key == request_key("s", "p", SCHEMA) != request_key("s", "p2", SCHEMA)
    cache.put(key, backend="ollama", model="m", system="s", prompt="p", response="{}")
    assert cache.get(key)["model"] == "m"


@pytest.mark.parametrize("field", ["response", "prompt", "model"])
def test_cache_rejects_credentials_before_creating_files(tmp_path, field):
    cache = ResponseCache(tmp_path / "cache")
    entry = dict(backend="fake", model="m", system="s", prompt="p", response="{}")
    entry[field] = KEY
    with pytest.raises(ValueError, match="API key"):
        cache.put("request", **entry)
    assert not cache.directory.exists()


@pytest.fixture(scope="module")
def analysed():
    registry = load_registry(REGISTRY_PATH)
    config = RunConfig(as_of=date(2026, 10, 2), store_root=STORE_ROOT, offline=True)
    return analyse(config, registry, fetch_stage(config, registry, lambda m: None))


class FixedBackend:
    name, model = "fake", "fake-1"

    def __init__(self, text):
        self.text = text

    def complete_json(self, system, prompt, schema):
        return self.text


@pytest.mark.parametrize("credential", [KEY, "ordinary-looking-test-secret", "Bearer abcdefgh12345678"])
def test_credential_response_never_reaches_cache_or_summary(analysed, tmp_path, monkeypatch, credential):
    from sgmacro.store import assert_no_credentials

    monkeypatch.setenv("SGMACRO_OPENAI_API_KEY", "ordinary-looking-test-secret")
    text = json.dumps({"sales": {"lead": {"text": credential, "facts": []}, "points": []}})
    lines = []
    summary = summarise(analysed, cache_dir=tmp_path, backend=FixedBackend(text), log=lines.append)
    assert summary.source == "deterministic" and summary.model is None
    assert summary.response_sha256 is None
    assert any("API key" in n for n in summary.execution_notes)
    assert not list(tmp_path.iterdir())
    assert_no_credentials(json.dumps(summary.to_json()) + "\n".join(lines + summary.execution_notes))
    replay = summarise(analysed, cache_dir=tmp_path, backend_preference="cache", log=lines.append)
    assert replay.to_json() == summary.to_json()


def test_unsafe_existing_cache_is_ignored(analysed, tmp_path):
    from sgmacro.llm.summary import SYSTEM, build_prompt, build_sheet
    from sgmacro.store import assert_no_credentials

    key = request_key(SYSTEM, build_prompt(build_sheet(analysed)), SCHEMA)
    (tmp_path / f"{key}.json").write_text(json.dumps({"response": KEY, "model": "fake", "backend": "fake"}))
    summary = summarise(analysed, cache_dir=tmp_path, backend_preference="cache", log=lambda m: None)
    assert summary.source == "deterministic" and not summary.from_cache
    assert any("cached response ignored" in n for n in summary.execution_notes)
    assert_no_credentials(json.dumps(summary.to_json()) + "\n".join(summary.execution_notes))


def test_summary_live_backend_failure_and_garbage_fall_back(analysed, tmp_path, monkeypatch):
    monkeypatch.setattr("sgmacro.llm.summary.resolve_backend", lambda p, **kw: (FixedBackend("not json"), []))
    summary = summarise(analysed, backend_preference="auto", cache_dir=tmp_path, log=lambda m: None)
    assert summary.source == "deterministic"
    assert any("not valid JSON" in n for n in summary.notes)


def test_summary_rejects_wrong_claims_and_uses_template_for_that_market(analysed, tmp_path, monkeypatch):
    bad = {"sales": {"lead": {"text": "Private home prices rose 9.9%.", "facts": ["F1"]}, "points": []},
           "rents": {"lead": {"text": "Rents fell sharply.", "facts": []}, "points": []}}
    monkeypatch.setattr("sgmacro.llm.summary.resolve_backend", lambda p, **kw: (FixedBackend(json.dumps(bad)), []))
    summary = summarise(analysed, backend_preference="auto", cache_dir=tmp_path, log=lambda m: None)
    assert all(m.source == "deterministic" and m.fallback_text for m in summary.markets.values())
    rejected = [c for m in summary.markets.values() for c in m.rejected]
    assert len(rejected) == 2
    # the live response was cached, so a re-run reproduces the same outcome offline
    again = summarise(analysed, backend_preference="cache", cache_dir=tmp_path, log=lambda m: None)
    assert again.source == "deterministic" and again.prompt_sha256 == summary.prompt_sha256

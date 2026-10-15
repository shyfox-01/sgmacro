"""LLM backends: one interface over the Claude API, any OpenAI-compatible service
and a local Ollama model.

Two call shapes are needed:

``complete_json``   one request, output constrained to a JSON schema (the
                    executive summary). Responses are cached on disk keyed by a
                    hash of the full request, so a committed cache reproduces the
                    report with no key, no model and no network.
``session``         a multi-turn tool-use conversation (the discovery agent). The
                    provider-native history is kept append-only inside the
                    session object, which is what the Claude API's preserved
                    thinking requires (assistant turns are echoed back verbatim).

Resolution for ``auto``: cache, then the Claude API if a key is set, then an
OpenAI-compatible service if one is configured, then a local Ollama model; if none
is available the caller falls back to deterministic text and records why. No
backend is contacted in a way that costs money until a request is actually made:
checking availability reads the environment and, for the HTTP backends, lists the
server's models.

Keys are handled as :mod:`sgmacro.credentials` describes.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit

import requests

from sgmacro.credentials import redact
from sgmacro.store import assert_no_credentials

ANTHROPIC_MODEL = "claude-opus-5-5"
OLLAMA_MODEL = "qwen3.5:9b"
OLLAMA_URL = "http://localhost:11434"
#: Bumped when prompts change in a way that should invalidate cached responses.
CONTRACT_VERSION = 2

BACKENDS = ("anthropic", "openai", "ollama")


class BackendUnavailable(RuntimeError):
    pass


class ProviderError(RuntimeError):
    """A request reached the provider and was refused or failed (message redacted)."""

    def __init__(self, status: int | None, message: str) -> None:
        super().__init__(redact(message))
        self.status = status


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Turn:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = ""
    model: str = ""
    #: Tokens in the conversation after this reply, when the server reports them.
    context_used: int | None = None


@dataclass
class ToolResult:
    call: ToolCall
    content: str
    is_error: bool = False


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]


class Session(Protocol):
    def send_user(self, text: str) -> Turn: ...
    def send_tool_results(self, results: list[ToolResult]) -> Turn: ...


class Backend(Protocol):
    name: str
    model: str

    def complete_json(self, system: str, prompt: str, schema: dict[str, Any]) -> str: ...
    def session(self, system: str, tools: list[ToolSpec]) -> Session: ...


def _unreachable(exc: requests.RequestException) -> str:
    """A short reason instead of urllib3's nested repr."""
    if isinstance(exc, requests.Timeout):
        return "timed out"
    if isinstance(exc, requests.ConnectionError):
        return "connection refused or host unreachable"
    return type(exc).__name__


def _error_text(response: Any) -> str:
    try:
        body = response.json()
        error = body.get("error", body) if isinstance(body, dict) else body
        text = error.get("message", error) if isinstance(error, dict) else error
    except ValueError:
        text = response.text
    return redact(str(text))[:300]


def _parse_arguments(args: Any) -> dict[str, Any]:
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError:
            return {"_unparseable": args}
    return args if isinstance(args, dict) else {}


def _empty_reply(host: str, finish: str) -> str:
    if finish in ("length", "max_tokens"):
        return (f"{host}: the reply was cut off at the model's output or context limit (finish_reason={finish}). "
                "A long session can outgrow the context window, often because the model repeated itself; "
                "fewer steps (--max-steps) or a larger model help, and a local server reached through the "
                "OpenAI-compatible API needs a context window of about 32k tokens (the ollama backend already "
                "requests that)")
    return f"{host}: the reply had no content (finish_reason={finish or 'none'})"


def _check_reply(host: str, text: str, finish: str, *, has_tools: bool = False) -> None:
    # Truncated tool calls are incomplete too: never execute their partial arguments.
    if finish in ("length", "max_tokens") or (not text.strip() and not has_tools):
        raise ProviderError(None, _empty_reply(host, finish))


_THINK = re.compile(r"^\s*<think>.*?</think>\s*", re.S)
_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.S)


def _clean_json_text(text: str) -> str:
    """Drop a leading <think> block and a Markdown fence, which some models add
    even when asked for JSON; the claim checker validates what remains."""
    text = _THINK.sub("", text)
    fenced = _FENCE.match(text)
    return fenced.group(1) if fenced else text


# ---------------------------------------------------------------------------
# Claude API
# ---------------------------------------------------------------------------

class AnthropicBackend:
    """Claude via the official SDK.

    Claude Opus 5.5 always thinks (depth set by ``effort``), rejects forced
    ``tool_choice``, and runs a server-side fallback model if a request is
    declined by a safety classifier, which is enabled here as recommended.
    """

    name = "anthropic"

    def __init__(self, model: str = ANTHROPIC_MODEL, effort: str = "medium", client: Any = None) -> None:
        if client is None:
            if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
                raise BackendUnavailable("no ANTHROPIC_API_KEY in the environment")
            try:
                import anthropic  # noqa: PLC0415 - optional dependency
            except ImportError as exc:
                raise BackendUnavailable('ANTHROPIC_API_KEY is set but the anthropic package is not installed '
                                         '(pip install -e ".[anthropic]" -c constraints.txt)') from exc
            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.effort = effort

    def _create(self, **kwargs: Any) -> Any:
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            **kwargs,
        )
        if response.stop_reason == "refusal":
            raise BackendUnavailable(f"request declined (stop_details={response.stop_details})")
        return response

    def complete_json(self, system: str, prompt: str, schema: dict[str, Any]) -> str:
        response = self._create(
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
        )
        text = "".join(b.text for b in response.content if b.type == "text")
        _check_reply("Anthropic", text, response.stop_reason)
        return text

    def session(self, system: str, tools: list[ToolSpec]) -> Session:
        return _AnthropicSession(self, system, tools)


class _AnthropicSession:
    def __init__(self, backend: AnthropicBackend, system: str, tools: list[ToolSpec]) -> None:
        self.backend = backend
        self.system = system
        self.tools = [
            {"name": t.name, "description": t.description, "input_schema": t.parameters, "strict": True}
            for t in tools
        ]
        self.messages: list[dict[str, Any]] = []

    def _step(self) -> Turn:
        response = self.backend._create(
            system=self.system, tools=self.tools, messages=self.messages,
            output_config={"effort": self.backend.effort},
        )
        # Append the full content, thinking blocks included: append-only history.
        self.messages.append({"role": "assistant", "content": response.content})
        calls = [ToolCall(b.id, b.name, dict(b.input)) for b in response.content if b.type == "tool_use"]
        text = "".join(b.text for b in response.content if b.type == "text")
        _check_reply("Anthropic", text, response.stop_reason, has_tools=bool(calls))
        return Turn(text, calls, response.stop_reason, getattr(response, "model", self.backend.model))

    def send_user(self, text: str) -> Turn:
        self.messages.append({"role": "user", "content": text})
        return self._step()

    def send_tool_results(self, results: list[ToolResult]) -> Turn:
        self.messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": r.call.id, "content": r.content, "is_error": r.is_error}
            for r in results
        ]})
        return self._step()


# ---------------------------------------------------------------------------
# OpenAI-compatible services: OpenAI, Gemini, DeepSeek, Qwen, OpenRouter, and
# local servers such as LM Studio, vLLM or llama.cpp
# ---------------------------------------------------------------------------

#: Short names accepted in SGMACRO_OPENAI_BASE_URL: the address, and the variable
#: that provider's key is read from. Any other compatible URL works too, with its
#: key in SGMACRO_OPENAI_API_KEY.
OPENAI_PRESETS: dict[str, tuple[str, str]] = {
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY"),
    "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY"),
    "qwen": ("https://dashscope-intl.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
}


def _ollama_hint(base_url: str) -> str:
    """What to do when a local Ollama server does not answer."""
    if not is_local(base_url):
        return ""
    app = Path(os.environ["LOCALAPPDATA"]) / "Programs" / "Ollama" / "ollama.exe" if os.environ.get(
        "LOCALAPPDATA") else None
    if shutil.which("ollama") or (app is not None and app.exists()):
        return ("; Ollama is installed but nothing answered at that address: start the Ollama app (or run "
                "`ollama serve`), or check SGMACRO_OLLAMA_URL")
    return "; Ollama does not appear to be installed (https://ollama.com)"


def is_local(url: str) -> bool:
    host = (urlsplit(url).hostname or "").lower()
    return host in ("localhost", "127.0.0.1", "::1") or host.endswith(".localhost")


@dataclass(frozen=True)
class OpenAIConfig:
    """Where to send requests and which variable holds the key (never the key)."""

    base_url: str
    key_vars: tuple[str, ...]
    model: str | None

    @property
    def host(self) -> str:
        return urlsplit(self.base_url).hostname or self.base_url


def openai_config() -> OpenAIConfig | None:
    raw = os.environ.get("SGMACRO_OPENAI_BASE_URL", "").strip()
    if not raw:
        return None
    preset = OPENAI_PRESETS.get(raw.lower())
    if preset:
        base_url, key_vars = preset[0], ("SGMACRO_OPENAI_API_KEY", preset[1])
    else:
        # A provider's own key goes only to that provider's preset address.
        base_url, key_vars = raw.rstrip("/"), ("SGMACRO_OPENAI_API_KEY",)
    return OpenAIConfig(base_url, key_vars, os.environ.get("SGMACRO_OPENAI_MODEL", "").strip() or None)


class OpenAIBackend:
    """Any service that implements the OpenAI Chat Completions API.

    Requests share a small payload (model, messages, tools, response format);
    tool history also preserves reasoning_content when a service returns it. A
    service without JSON-schema output gets a JSON-object request with the schema
    in the instructions instead; the claim checker validates the result either way.
    """

    name = "openai"

    def __init__(self, config: OpenAIConfig | None = None, *, offline: bool = False, timeout: float = 900) -> None:
        config = config or openai_config()
        if config is None:
            raise BackendUnavailable("not configured (set SGMACRO_OPENAI_BASE_URL)")
        scheme = urlsplit(config.base_url).scheme
        if scheme not in ("http", "https"):
            raise BackendUnavailable(f"SGMACRO_OPENAI_BASE_URL must be a preset name "
                                     f"({', '.join(OPENAI_PRESETS)}) or an http(s) URL")
        local = is_local(config.base_url)
        if offline and not local:
            raise BackendUnavailable(f"{config.host} is a remote service, not called in an offline run")
        key_var = next((v for v in config.key_vars if os.environ.get(v)), None)
        self._key = os.environ.get(key_var) if key_var else None
        if self._key is None and not local:
            raise BackendUnavailable(f"no API key for {config.host} (set {' or '.join(config.key_vars[::-1])})")
        if self._key is not None and scheme != "https" and not local:
            raise BackendUnavailable(f"refusing to send an API key over plain HTTP to {config.host}; use https://")
        self.base_url, self.host, self.timeout = config.base_url, config.host, timeout
        self.models_listed = False
        models = self._list_models()
        self.model = config.model or (models[0] if len(models) == 1 else "")
        if not self.model:
            listed = f"; it lists {len(models)}, e.g. {', '.join(models[:5])}" if models else ""
            raise BackendUnavailable(f"set SGMACRO_OPENAI_MODEL to the model to use at {self.host}{listed}")

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._key}"} if self._key else {}

    def _list_models(self) -> list[str]:
        """Probe the model-list endpoint without making a generation request."""
        try:
            response = requests.get(f"{self.base_url}/models", headers=self._headers(), timeout=10)
        except requests.RequestException as exc:
            raise BackendUnavailable(f"cannot reach {self.host} ({_unreachable(exc)})") from None
        if response.status_code in (401, 403):
            raise BackendUnavailable(f"{self.host} rejected the API key (HTTP {response.status_code})")
        if response.status_code in (404, 405, 501):
            return []                     # listing unsupported; generation is still unverified
        if response.status_code >= 400:
            raise BackendUnavailable(f"{self.host} model listing failed (HTTP {response.status_code}): "
                                     f"{_error_text(response)}")
        try:
            data = response.json().get("data")
        except (ValueError, AttributeError):
            raise BackendUnavailable(f"{self.host} returned an invalid model list") from None
        if not isinstance(data, list):
            raise BackendUnavailable(f"{self.host} returned an invalid model list")
        self.models_listed = True
        return sorted(str(m.get("id")) for m in data if isinstance(m, dict) and m.get("id"))

    def chat(self, payload: dict[str, Any]) -> tuple[dict[str, Any], str]:
        try:
            response = requests.post(f"{self.base_url}/chat/completions", json={"model": self.model, **payload},
                                     headers=self._headers(), timeout=self.timeout)
        except requests.RequestException as exc:
            raise ProviderError(None, f"{self.host}: {_unreachable(exc)}") from None
        if response.status_code >= 400:
            raise ProviderError(response.status_code,
                                f"{self.host} HTTP {response.status_code}: {_error_text(response)}")
        choices = response.json().get("choices") or []
        if not choices:
            raise ProviderError(None, f"{self.host}: the response has no choices")
        return choices[0].get("message") or {}, str(choices[0].get("finish_reason") or "")

    def complete_json(self, system: str, prompt: str, schema: dict[str, Any]) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        try:
            message, finish = self.chat({"messages": messages, "response_format": {
                "type": "json_schema", "json_schema": {"name": "response", "schema": schema, "strict": True}}})
        except ProviderError as exc:
            if exc.status not in (400, 422):
                raise
            # The service does not take this schema; ask for a JSON object that follows it.
            messages[0] = {"role": "system", "content": f"{system}\n\nReply with one JSON object that follows "
                                                        f"this JSON Schema:\n{json.dumps(schema)}"}
            message, finish = self.chat({"messages": messages, "response_format": {"type": "json_object"}})
        text = _clean_json_text(message.get("content") or "")
        _check_reply(self.host, text, finish)
        return text

    def session(self, system: str, tools: list[ToolSpec]) -> Session:
        return _OpenAISession(self, system, tools)


class _OpenAISession:
    def __init__(self, backend: OpenAIBackend, system: str, tools: list[ToolSpec]) -> None:
        self.backend = backend
        self.tools = [{"type": "function", "function": {"name": t.name, "description": t.description,
                                                        "parameters": t.parameters}} for t in tools]
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        self._counter = 0

    def _step(self) -> Turn:
        message, finish = self.backend.chat({"messages": self.messages, "tools": self.tools})
        text = _THINK.sub("", message.get("content") or "")
        _check_reply(self.backend.host, text, finish, has_tools=bool(message.get("tool_calls")))
        calls, echoed = [], []
        for raw in message.get("tool_calls") or []:
            self._counter += 1
            raw = {**raw, "id": raw.get("id") or f"call_{self._counter}"}
            fn = raw.get("function") or {}
            calls.append(ToolCall(raw["id"], fn.get("name", ""), _parse_arguments(fn.get("arguments"))))
            echoed.append(raw)
        # DeepSeek thinking/tool conversations require this field in every later
        # request. Preserve it when supplied, without echoing unrelated extras.
        turn: dict[str, Any] = {"role": "assistant", "content": message.get("content")}
        if "reasoning_content" in message:
            turn["reasoning_content"] = message["reasoning_content"]
        if echoed:
            turn["tool_calls"] = echoed
        self.messages.append(turn)
        return Turn(text, calls, finish, self.backend.model)

    def send_user(self, text: str) -> Turn:
        self.messages.append({"role": "user", "content": text})
        return self._step()

    def send_tool_results(self, results: list[ToolResult]) -> Turn:
        for r in results:
            self.messages.append({"role": "tool", "tool_call_id": r.call.id, "content": r.content})
        return self._step()


# ---------------------------------------------------------------------------
# Ollama (local, keyless)
# ---------------------------------------------------------------------------

class OllamaBackend:
    """An Ollama server (local by default). The model is ``model``, else ``SGMACRO_OLLAMA_MODEL``,
    else the default; if the default is not installed, the first installed model
    (by name) that supports tool calls is used, and ``picked`` says so."""

    name = "ollama"

    def __init__(self, model: str | None = None, base_url: str | None = None, think: bool = False,
                 num_ctx: int = 32768, num_predict: int = 8192, *, offline: bool = False) -> None:
        #: Hard cap on generated tokens (thinking included): an uncapped thinking
        #: model once reasoned for over 25 minutes on the summary prompt.
        self.num_predict = num_predict
        self.base_url = base_url or os.environ.get("SGMACRO_OLLAMA_URL", "").strip() or OLLAMA_URL
        if offline and not is_local(self.base_url):
            raise BackendUnavailable("Ollama is a remote service, not called in an offline run")
        self.think = think
        self.num_ctx = num_ctx
        self.picked = False
        try:
            tags = requests.get(f"{self.base_url}/api/tags", timeout=4)
            tags.raise_for_status()
            available = sorted(m.get("name") for m in tags.json().get("models", []) if m.get("name"))
        except requests.RequestException as exc:
            raise BackendUnavailable(f"no Ollama server at {self.base_url} ({_unreachable(exc)})"
                                     + _ollama_hint(self.base_url)) from None
        requested = model or os.environ.get("SGMACRO_OLLAMA_MODEL", "").strip()
        if requested:
            name = requested if requested in available or ":" in requested else f"{requested}:latest"
            if name not in available:
                raise BackendUnavailable(f"model {requested!r} is not installed in Ollama (have "
                                         f"{', '.join(available) or 'none'}); `ollama pull {requested}` or set "
                                         f"SGMACRO_OLLAMA_MODEL to one of those")
            self.model = name
        elif OLLAMA_MODEL in available:
            self.model = OLLAMA_MODEL
        else:
            capable = next((m for m in available if "tools" in self._capabilities(m)), None)
            if capable is None:
                raise BackendUnavailable(f"the default model {OLLAMA_MODEL} is not installed and no installed "
                                         f"model reports tool support (have {', '.join(available) or 'none'}); "
                                         f"`ollama pull {OLLAMA_MODEL}`")
            self.model, self.picked = capable, True

    @property
    def context_tokens(self) -> int:
        """The context window this backend requests, so the agent can stop in time."""
        return self.num_ctx

    def _capabilities(self, model: str) -> list[str]:
        try:
            shown = requests.post(f"{self.base_url}/api/show", json={"model": model}, timeout=4)
            shown.raise_for_status()
            return list(shown.json().get("capabilities") or [])
        except (requests.RequestException, ValueError):
            return []

    def chat(self, messages: list[dict[str, Any]], *, tools: list[dict] | None = None,
             schema: dict[str, Any] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model, "messages": messages, "stream": False, "think": self.think,
            "options": {"temperature": 0, "num_ctx": self.num_ctx, "seed": 7, "num_predict": self.num_predict},
        }
        if tools:
            payload["tools"] = tools
        if schema:
            payload["format"] = schema
        try:
            response = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=900)
        except requests.RequestException as exc:
            raise ProviderError(None, f"Ollama: {_unreachable(exc)}") from None
        if response.status_code >= 400:
            raise ProviderError(response.status_code, f"Ollama HTTP {response.status_code}: {_error_text(response)}")
        return response.json()

    def complete_json(self, system: str, prompt: str, schema: dict[str, Any]) -> str:
        reply = self.chat([{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                          schema=schema)
        text = _clean_json_text((reply.get("message") or {}).get("content") or "")
        _check_reply("Ollama", text, reply.get("done_reason", ""))
        return text

    def session(self, system: str, tools: list[ToolSpec]) -> Session:
        return _OllamaSession(self, system, tools)


class _OllamaSession:
    #: Tool results and assistant texts kept whole; older ones are shortened.
    KEEP_RECENT = 6
    #: Characters an older tool result keeps; older assistant texts keep half.
    SHORT_CHARS = 400

    def __init__(self, backend: OllamaBackend, system: str, tools: list[ToolSpec]) -> None:
        self.backend = backend
        self.tools = [{"type": "function", "function": {"name": t.name, "description": t.description,
                                                        "parameters": t.parameters}} for t in tools]
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        self._counter = 0
        self._shortened: set[int] = set()

    def _compact(self) -> None:
        """Keep a long session inside a local context window: live, a 9B model's
        conversation grew about 1,500 tokens a step and filled 32k by step 16, and
        Ollama then silently drops the start of the prompt (the instructions). Only
        what is sent is shortened; the agent's record keeps every result whole, and
        a repeated call gets its earlier result back from that record."""
        for role, keep in (("tool", self.SHORT_CHARS), ("assistant", self.SHORT_CHARS // 2)):
            indices = [i for i, m in enumerate(self.messages) if m.get("role") == role]
            for i in indices[:-self.KEEP_RECENT]:
                content = self.messages[i].get("content") or ""
                if i in self._shortened or len(content) <= keep:
                    continue
                self.messages[i] = {**self.messages[i], "content": content[:keep] + (
                    f"\n[... {len(content) - keep} more characters shown earlier, left out to save context; "
                    "call the tool again if you need them]")}
                self._shortened.add(i)

    def _step(self) -> Turn:
        reply = self.backend.chat(self.messages, tools=self.tools)
        message = reply.get("message") or {}
        text = _THINK.sub("", message.get("content") or "")
        _check_reply("Ollama", text, reply.get("done_reason", ""), has_tools=bool(message.get("tool_calls")))
        self.messages.append(message)
        calls = []
        for raw in message.get("tool_calls") or []:
            fn = raw.get("function", {})
            self._counter += 1
            calls.append(ToolCall(raw.get("id") or f"call_{self._counter}", fn.get("name", ""),
                                  _parse_arguments(fn.get("arguments"))))
        # Ollama counts the whole prompt even when its prefix is cached.
        used = (reply.get("prompt_eval_count") or 0) + (reply.get("eval_count") or 0)
        return Turn(text, calls, reply.get("done_reason", ""), self.backend.model, used or None)

    def send_user(self, text: str) -> Turn:
        self.messages.append({"role": "user", "content": text})
        return self._step()

    def send_tool_results(self, results: list[ToolResult]) -> Turn:
        for r in results:
            self.messages.append({"role": "tool", "tool_name": r.call.name, "content": r.content})
        self._compact()
        return self._step()


# ---------------------------------------------------------------------------
# response cache and resolution
# ---------------------------------------------------------------------------

def request_key(system: str, prompt: str, schema: dict[str, Any] | None) -> str:
    blob = json.dumps({"contract": CONTRACT_VERSION, "system": system, "prompt": prompt, "schema": schema},
                      sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


@dataclass
class ResponseCache:
    """On-disk cache of model responses, committed so runs reproduce offline.

    The key covers the request contract only. The model that produced an entry is
    recorded inside it but kept out of the key, so a reviewer without that model
    still hits the cache.
    """

    directory: Path

    def get(self, key: str) -> dict[str, Any] | None:
        path = self.directory / f"{key}.json"
        if not path.exists():
            return None
        text = path.read_text(encoding="utf-8")
        assert_no_credentials(text)
        return json.loads(text)

    def put(self, key: str, *, backend: str, model: str, system: str, prompt: str, response: str) -> None:
        entry = {"key": key, "backend": backend, "model": model,
                 "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                 "system": system, "prompt": prompt, "response": response}
        text = json.dumps(entry, indent=2, ensure_ascii=False) + "\n"
        assert_no_credentials(text)
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / f"{key}.json").write_text(text, encoding="utf-8", newline="\n")


def _construct(name: str, *, offline: bool) -> Backend:
    if name == "anthropic":
        if offline:
            raise BackendUnavailable("not called in an offline run")
        return AnthropicBackend()
    if name == "openai":
        return OpenAIBackend(offline=offline)
    return OllamaBackend(offline=offline)


def resolve_backend(preference: str = "auto", *, offline: bool = False) -> tuple[Backend | None, list[str]]:
    """A live backend (or None) plus the reasons the others were unavailable.

    ``cache``/``off`` never touch a model. ``auto`` tries the Claude API, an
    OpenAI-compatible service, then Ollama. ``offline`` rules out remote services;
    a local server is still allowed. An explicitly chosen backend that is
    unavailable raises.
    """
    if preference in ("cache", "off"):
        return None, [f"live models disabled (--llm {preference})"]
    if preference != "auto" and preference not in BACKENDS:
        raise ValueError(f"unknown LLM backend {preference!r}")
    reasons: list[str] = []
    for name in (BACKENDS if preference == "auto" else (preference,)):
        try:
            return _construct(name, offline=offline), reasons
        except BackendUnavailable as exc:
            reasons.append(f"{name}: {redact(str(exc))}")
    if preference != "auto":
        raise BackendUnavailable("; ".join(reasons))
    return None, reasons


SETUP_HELP = """\
To connect a model (any one is enough):
  Claude API      pip install -e ".[anthropic]" -c constraints.txt, then set ANTHROPIC_API_KEY
  OpenAI-compatible service
                  set SGMACRO_OPENAI_BASE_URL to openai, gemini, deepseek, qwen or openrouter
                  (or any compatible URL, e.g. http://localhost:1234/v1 for LM Studio),
                  that provider's key (OPENAI_API_KEY, GEMINI_API_KEY, DEEPSEEK_API_KEY,
                  DASHSCOPE_API_KEY, OPENROUTER_API_KEY; SGMACRO_OPENAI_API_KEY for any URL)
                  and SGMACRO_OPENAI_MODEL
  Local Ollama    install it from https://ollama.com, then `ollama pull qwen3.5:9b`
                  (or set SGMACRO_OLLAMA_MODEL to an installed model with tool support)
Keys are read from environment variables only and never written to disk.
`sgmacro models` shows what this machine has and what `auto` would use."""


def backend_status() -> list[tuple[str, bool, str]]:
    """``(name, available, detail)`` per backend, in ``auto`` order. Nothing is
    sent that costs money: the Claude API is not contacted at all."""
    rows: list[tuple[str, bool, str]] = []
    for name in BACKENDS:
        try:
            backend = _construct(name, offline=False)
        except BackendUnavailable as exc:
            rows.append((name, False, redact(str(exc))))
            continue
        if name == "anthropic":
            detail = f"{backend.model} (key set; not contacted, so the key is not checked)"
        elif name == "openai":
            probe = ("model listing checked; generation not tested" if backend.models_listed else
                     "model listing unsupported; model and key unverified")
            detail = f"{backend.model} at {backend.host} ({probe})"
        else:
            detail = f"{backend.model} at {backend.base_url}" + (
                f" (picked automatically: {OLLAMA_MODEL} is not installed)" if backend.picked else "")
        rows.append((name, True, detail))
    return rows


def unused_keys() -> list[str]:
    """Provider keys present in the environment that no backend is set up to use."""
    if openai_config() is not None:
        return []
    return [f"{var} is set; to use it, set SGMACRO_OPENAI_BASE_URL={name} and SGMACRO_OPENAI_MODEL"
            for name, (_, var) in OPENAI_PRESETS.items() if os.environ.get(var)]

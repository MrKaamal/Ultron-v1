"""LLM providers against a local fake HTTP server: wire format, retries, errors, secret hygiene."""

from __future__ import annotations

import json
import socket
from collections.abc import Callable
from typing import Any

import pytest

from helpers import FakeServer, Recorder
from ultron.config import Config
from ultron.errors import ConfigError, ProviderError
from ultron.messages import LLMResponse, Message, ToolCall, ToolResult, ToolSpec
from ultron.providers import (
    AnthropicProvider,
    MockProvider,
    OpenAICompatProvider,
    make_provider,
    resolve_provider_name,
)
from ultron.runtime import build_runtime

KEY = "sk-ant-test-0123456789abcdef"
CALC = ToolSpec("calc", "Do arithmetic.", {"type": "object", "properties": {"expression": {"type": "string"}}})


def lower(headers: dict[str, str]) -> dict[str, str]:
    return {k.lower(): v for k, v in headers.items()}


def anthropic_message(content: list[dict[str, Any]], stop: str = "end_turn", inp: int = 5, out: int = 2) -> dict:
    return {
        "id": "msg_01",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-5-5",
        "content": content,
        "stop_reason": stop,
        "usage": {"input_tokens": inp, "output_tokens": out},
    }


def tool_use(call_id: str, name: str, **args: Any) -> dict[str, Any]:
    return {"type": "tool_use", "id": call_id, "name": name, "input": args}


def chat_completion(message: dict[str, Any], finish: str = "stop") -> dict:
    return {"choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3}}


@pytest.fixture
def server() -> Callable[[], FakeServer]:
    started: list[FakeServer] = []

    def make() -> FakeServer:
        srv = FakeServer()
        srv.__enter__()
        started.append(srv)
        return srv

    yield make
    for srv in started:
        srv.__exit__()


def anthropic(srv: FakeServer, **kwargs: Any) -> AnthropicProvider:
    sleeps: list[float] = kwargs.pop("sleeps", [])
    kwargs.setdefault("backoff", 0.5)
    return AnthropicProvider(KEY, "claude-sonnet-5-5", base_url=srv.url, sleep=sleeps.append, **kwargs)


# ============================================================================ Anthropic: requests
def test_anthropic_request_matches_the_messages_api(server: Callable[[], FakeServer]) -> None:
    srv = server()
    srv.push(200, anthropic_message([{"type": "text", "text": "hi"}]))
    anthropic(srv).complete("be brief", [Message.user("hello")], [CALC])
    req = srv.requests[0]
    headers = lower(req["headers"])
    assert req["method"] == "POST" and req["path"] == "/v1/messages"
    assert headers["x-api-key"] == KEY
    assert headers["anthropic-version"] == "2026-10-21"
    assert headers["content-type"] == "application/json"
    body = req["body"]
    assert body["model"] == "claude-sonnet-5-5" and body["max_tokens"] == 16000
    assert body["system"] == "be brief"
    assert body["messages"] == [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
    assert body["tools"] == [{"name": "calc", "description": "Do arithmetic.", "input_schema": CALC.parameters}]


def test_anthropic_never_sends_parameters_current_models_reject(server: Callable[[], FakeServer]) -> None:
    srv = server()
    srv.push(200, anthropic_message([{"type": "text", "text": "ok"}]))
    anthropic(srv).complete("s", [Message.user("x")], [CALC])
    assert not {"temperature", "top_p", "top_k", "tool_choice", "thinking"} & set(srv.requests[0]["body"])


def test_anthropic_optional_settings(server: Callable[[], FakeServer]) -> None:
    srv = server()
    srv.push(200, anthropic_message([{"type": "text", "text": "ok"}]))
    provider = AnthropicProvider(
        KEY, "claude-opus-5-5", base_url=srv.url + "/v1", max_tokens=1234, effort="high", workspace_id="wrkspc_1",
        version="2023-06-01",
    )
    provider.complete("s", [Message.user("x")], [])
    req = srv.requests[0]
    assert req["path"] == "/v1/messages"  # a base URL that already ends in /v1 is not doubled
    assert lower(req["headers"])["anthropic-workspace-id"] == "wrkspc_1"
    assert lower(req["headers"])["anthropic-version"] == "2023-06-01"
    assert req["body"]["max_tokens"] == 1234 and req["body"]["output_config"] == {"effort": "high"}
    assert req["body"]["model"] == "claude-opus-5-5" and "tools" not in req["body"]


# ============================================================================ Anthropic: history conversion
def test_anthropic_replays_thinking_blocks_verbatim_in_a_tool_loop(server: Callable[[], FakeServer]) -> None:
    srv = server()
    first_content = [
        {"type": "thinking", "thinking": "I should add.", "signature": "sig-abc"},
        {"type": "text", "text": "Calculating."},
        tool_use("toolu_1", "calc", expression="2+2"),
    ]
    srv.push(200, anthropic_message(first_content, "tool_use"))
    srv.push(200, anthropic_message([{"type": "text", "text": "It is 4."}]))
    provider = anthropic(srv)

    history = [Message.user("2+2?")]
    response = provider.complete("s", history, [CALC])
    assert response.text == "Calculating." and response.stop_reason == "tool_use"
    assert response.tool_calls == [ToolCall("toolu_1", "calc", {"expression": "2+2"})]
    assert (response.input_tokens, response.output_tokens) == (5, 2)

    history.append(Message.assistant(response.text, response.tool_calls, raw=response.raw))
    history.append(Message.tool([ToolResult("toolu_1", "calc", "4")]))
    provider.complete("s", history, [CALC])
    sent = srv.requests[1]["body"]["messages"]
    assert sent[1] == {"role": "assistant", "content": first_content}  # thinking block untouched, in order
    assert sent[2] == {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "4"}]}


def test_anthropic_merges_parallel_results_and_puts_tool_results_first(server: Callable[[], FakeServer]) -> None:
    srv = server()
    srv.push(200, anthropic_message([{"type": "text", "text": "ok"}]))
    history = [
        Message.user("go"),
        Message.assistant("", [ToolCall("a", "calc", {"expression": "1"}), ToolCall("b", "calc", {"expression": "2"})]),
        Message.tool([ToolResult("a", "calc", "1"), ToolResult("b", "calc", "boom", is_error=True)]),
        Message.user("and then?"),  # a follow-up turn right after tool results
    ]
    anthropic(srv).complete("s", history, [CALC])
    sent = srv.requests[0]["body"]["messages"]
    assert [m["role"] for m in sent] == ["user", "assistant", "user"]
    assert sent[1]["content"] == [
        {"type": "tool_use", "id": "a", "name": "calc", "input": {"expression": "1"}},
        {"type": "tool_use", "id": "b", "name": "calc", "input": {"expression": "2"}},
    ]
    blocks = sent[2]["content"]
    assert [b["type"] for b in blocks] == ["tool_result", "tool_result", "text"]
    assert blocks[1] == {"type": "tool_result", "tool_use_id": "b", "content": "boom", "is_error": True}
    assert "is_error" not in blocks[0]


def test_anthropic_fills_in_empty_content(server: Callable[[], FakeServer]) -> None:
    srv = server()
    srv.push(200, anthropic_message([{"type": "text", "text": "ok"}]))
    history = [Message.user(""), Message.assistant(""), Message.user("x")]
    history.insert(2, Message.tool([ToolResult("z", "t", "")]))  # unanswered id is the caller's problem
    anthropic(srv).complete("s", history, [])
    sent = srv.requests[0]["body"]["messages"]
    assert sent[0]["content"][0]["text"] == "(empty)"
    assert sent[1]["content"] == [{"type": "text", "text": "(no output)"}]
    assert sent[2]["content"][0]["content"] == "(no output)"


def test_anthropic_response_parsing_edge_cases() -> None:
    parsed = AnthropicProvider.parse_response(
        anthropic_message([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}, tool_use("t", "x")], "")
    )
    assert parsed.text == "a\nb" and parsed.stop_reason == "tool_use" and len(parsed.tool_calls) == 1
    bad = AnthropicProvider.parse_response(
        anthropic_message([{"type": "tool_use", "id": "t", "name": "x", "input": "not-an-object"}], "tool_use")
    )
    assert bad.tool_calls[0].parse_error and bad.tool_calls[0].arguments == {}
    assert AnthropicProvider.parse_response(anthropic_message([], "refusal")).stop_reason == "refusal"
    with pytest.raises(ProviderError, match="no content list"):
        AnthropicProvider.parse_response({"type": "error"})


def test_anthropic_declares_itself_append_only() -> None:
    assert AnthropicProvider(KEY).append_only is True  # replayed thinking blocks bind to the exact prefix


# ============================================================================ retries and errors
def test_transient_errors_are_retried_with_exponential_backoff(server: Callable[[], FakeServer]) -> None:
    srv = server()
    srv.push(529, {"type": "error", "error": {"type": "overloaded_error", "message": "busy"}})
    srv.push(500, {"type": "error", "error": {"type": "api_error", "message": "oops"}})
    srv.push(200, anthropic_message([{"type": "text", "text": "ok"}]))
    sleeps: list[float] = []
    response = anthropic(srv, sleeps=sleeps).complete("s", [Message.user("x")], [])
    assert response.text == "ok" and len(srv.requests) == 3
    assert sleeps == [0.5, 1.0]


def test_retry_after_is_honoured_but_capped(server: Callable[[], FakeServer]) -> None:
    srv = server()
    srv.push(429, {"error": {"message": "slow down"}}, retry_after="2")
    srv.push(429, {"error": {"message": "slow down"}}, retry_after="9999")
    srv.push(200, anthropic_message([{"type": "text", "text": "ok"}]))
    sleeps: list[float] = []
    anthropic(srv, sleeps=sleeps).complete("s", [Message.user("x")], [])
    assert sleeps == [2.0, 30.0]


def test_giving_up_after_the_retry_budget(server: Callable[[], FakeServer]) -> None:
    srv = server()
    for _ in range(5):
        srv.push(529, {"error": {"message": "overloaded"}})
    with pytest.raises(ProviderError, match="HTTP 529: overloaded"):
        anthropic(srv, retries=2).complete("s", [Message.user("x")], [])
    assert len(srv.requests) == 3


def test_authentication_errors_are_not_retried_and_never_leak_the_key(server: Callable[[], FakeServer]) -> None:
    srv = server()
    srv.push(401, {"type": "error", "error": {"type": "authentication_error", "message": f"bad key {KEY}"}})
    with pytest.raises(ProviderError) as info:
        anthropic(srv).complete("s", [Message.user("x")], [])
    assert len(srv.requests) == 1
    text = str(info.value)
    assert "HTTP 401" in text and "check your API key" in text
    assert KEY not in text and "[REDACTED]" in text


def test_bad_requests_surface_the_providers_explanation(server: Callable[[], FakeServer]) -> None:
    srv = server()
    srv.push(400, {"type": "error", "error": {"type": "invalid_request_error", "message": "max_tokens too big"},
                   "request_id": "req_123"})
    with pytest.raises(ProviderError, match=r"HTTP 400: max_tokens too big \(request_id=req_123\)"):
        anthropic(srv).complete("s", [Message.user("x")], [])
    assert len(srv.requests) == 1


@pytest.mark.parametrize("payload,message", [
    ("<html>gateway</html>", "not valid JSON"),
    ([1, 2], "not an object"),
    ({"type": "message"}, "no content list"),
])
def test_malformed_success_responses_are_provider_errors(
    server: Callable[[], FakeServer], payload: Any, message: str
) -> None:
    srv = server()
    srv.push(200, payload)
    with pytest.raises(ProviderError, match=message):
        anthropic(srv).complete("s", [Message.user("x")], [])


def test_an_unreachable_server_is_a_clear_error() -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    provider = AnthropicProvider(KEY, base_url=f"http://127.0.0.1:{port}", retries=1, backoff=0, sleep=lambda s: None)
    with pytest.raises(ProviderError, match="cannot reach 127.0.0.1"):
        provider.complete("s", [Message.user("x")], [])


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com/x", "gopher://x"])
def test_only_http_urls_are_accepted(url: str) -> None:
    with pytest.raises(ProviderError, match="http"):
        AnthropicProvider(KEY, base_url=url).complete("s", [Message.user("x")], [])


# ============================================================================ OpenAI-compatible
def compat(srv: FakeServer, **kwargs: Any) -> OpenAICompatProvider:
    return OpenAICompatProvider(srv.url + "/v1", "some-model", sleep=lambda s: None, backoff=0, **kwargs)


def test_openai_request_shape(server: Callable[[], FakeServer]) -> None:
    srv = server()
    srv.push(200, chat_completion({"role": "assistant", "content": "hi"}))
    compat(srv, api_key="sk-openai-test-key-0123456789").complete("sys", [Message.user("hello")], [CALC])
    req = srv.requests[0]
    assert req["path"] == "/v1/chat/completions"
    assert lower(req["headers"])["authorization"] == "Bearer sk-openai-test-key-0123456789"
    body = req["body"]
    assert body["model"] == "some-model"
    assert body["messages"] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "hello"}]
    assert body["tools"] == [{"type": "function", "function": {
        "name": "calc", "description": "Do arithmetic.", "parameters": CALC.parameters}}]
    assert body["tool_choice"] == "auto"
    assert "temperature" not in body and "max_tokens" not in body


def test_openai_optional_settings_are_sent_only_when_set(server: Callable[[], FakeServer]) -> None:
    srv = server()
    srv.push(200, chat_completion({"role": "assistant", "content": "hi"}))
    srv.push(200, chat_completion({"role": "assistant", "content": "hi"}))
    compat(srv).complete("s", [Message.user("x")], [])
    compat(srv, temperature=0.2, max_tokens=99).complete("s", [Message.user("x")], [])
    plain, tuned = (r["body"] for r in srv.requests)
    assert "temperature" not in plain and "max_tokens" not in plain and "tools" not in plain
    assert tuned["temperature"] == 0.2 and tuned["max_tokens"] == 99
    assert "authorization" not in lower(srv.requests[0]["headers"])  # no key, no header (Ollama)


def test_openai_converts_tool_history(server: Callable[[], FakeServer]) -> None:
    srv = server()
    srv.push(200, chat_completion({"role": "assistant", "content": "done"}))
    history = [
        Message.user("go"),
        Message.assistant("", [ToolCall("c1", "calc", {"expression": "1+1"})]),
        Message.tool([ToolResult("c1", "calc", "2"), ToolResult("c2", "calc", "bad input", is_error=True)]),
    ]
    compat(srv).complete("s", history, [CALC])
    sent = srv.requests[0]["body"]["messages"]
    assert sent[2] == {"role": "assistant", "content": None, "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "calc", "arguments": '{"expression": "1+1"}'}}]}
    assert sent[3] == {"role": "tool", "tool_call_id": "c1", "content": "2"}
    assert sent[4] == {"role": "tool", "tool_call_id": "c2", "content": "ERROR: bad input"}


def test_openai_response_parsing() -> None:
    message = {"role": "assistant", "content": "<think>hmm</think>Calling.", "tool_calls": [
        {"id": "x1", "type": "function", "function": {"name": "calc", "arguments": '{"expression": "1+1"}'}},
        {"type": "function", "function": {"name": "calc", "arguments": "{broken"}},
        {"id": "x3", "type": "function", "function": {"name": "calc", "arguments": {"expression": "2"}}},
    ]}
    parsed = OpenAICompatProvider.parse_response(chat_completion(message, "stop"))
    assert parsed.text == "Calling." and parsed.stop_reason == "tool_use"  # "stop" with calls still means tool_use
    first, second, third = parsed.tool_calls
    assert first.arguments == {"expression": "1+1"} and first.parse_error is None
    assert second.id and second.arguments == {} and "not valid JSON" in (second.parse_error or "")
    assert third.arguments == {"expression": "2"}
    assert (parsed.input_tokens, parsed.output_tokens) == (7, 3)


@pytest.mark.parametrize("finish,expected", [("stop", "end_turn"), ("length", "max_tokens"),
                                             ("content_filter", "refusal"), ("tool_calls", "tool_use")])
def test_openai_finish_reasons_are_normalised(finish: str, expected: str) -> None:
    parsed = OpenAICompatProvider.parse_response(chat_completion({"content": "x"}, finish))
    assert parsed.stop_reason == expected


def test_openai_rejects_a_response_without_choices() -> None:
    with pytest.raises(ProviderError, match="choices"):
        OpenAICompatProvider.parse_response({"choices": []})


# ============================================================================ factory
def cfg_for(make_cfg: Callable[..., Config], **overrides: Any) -> Config:
    return make_cfg(**overrides)


def test_auto_selection_prefers_a_real_provider_when_configured(make_cfg: Callable[..., Config]) -> None:
    auto = make_cfg(provider="auto")
    assert resolve_provider_name(auto, {}) == "mock"
    assert resolve_provider_name(auto, {"ANTHROPIC_API_KEY": "k"}) == "anthropic"
    assert resolve_provider_name(auto, {"OPENAI_API_KEY": "k"}) == "mock"  # an OpenAI model must be named
    assert resolve_provider_name(make_cfg(provider="auto", model="gpt-x"), {"OPENAI_API_KEY": "k"}) == "openai"
    assert resolve_provider_name(make_cfg(provider="ollama"), {"ANTHROPIC_API_KEY": "k"}) == "ollama"


def test_the_mock_provider_needs_nothing(make_cfg: Callable[..., Config]) -> None:
    assert isinstance(make_provider(make_cfg(provider="mock"), {}), MockProvider)


def test_anthropic_factory(make_cfg: Callable[..., Config]) -> None:
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        make_provider(make_cfg(provider="anthropic"), {})
    provider = make_provider(make_cfg(provider="anthropic"), {"ANTHROPIC_API_KEY": KEY})
    assert isinstance(provider, AnthropicProvider)
    assert provider.model == "claude-sonnet-5-5" and provider.version == "2026-10-21"
    assert provider.url == "https://api.anthropic.com/v1/messages" and provider.max_tokens == 4096
    env = {"ANTHROPIC_API_KEY": KEY, "ANTHROPIC_BASE_URL": "http://gateway.local", "ANTHROPIC_WORKSPACE_ID": "wrkspc_9"}
    custom = make_provider(make_cfg(provider="anthropic", model="claude-opus-5-5", anthropic_version="2023-06-01",
                                    effort="low"), env)
    assert isinstance(custom, AnthropicProvider) and custom.model == "claude-opus-5-5"
    custom = make_provider(make_cfg(provider="anthropic"), env)
    assert custom.url == "http://gateway.local/v1/messages" and custom.headers["anthropic-workspace-id"] == "wrkspc_9"


def test_openai_and_compatible_factories(make_cfg: Callable[..., Config]) -> None:
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        make_provider(make_cfg(provider="openai", model="m"), {})
    with pytest.raises(ConfigError, match="needs a model"):
        make_provider(make_cfg(provider="openai"), {"OPENAI_API_KEY": "k"})
    openai = make_provider(make_cfg(provider="openai", model="m"), {"OPENAI_API_KEY": "k"})
    assert openai.name == "openai" and openai.url == "https://api.openai.com/v1/chat/completions"

    with pytest.raises(ConfigError, match="base_url"):
        make_provider(make_cfg(provider="openai-compat", model="m"), {})
    with pytest.raises(ConfigError, match="needs a model"):
        make_provider(make_cfg(provider="openai-compat", base_url="http://x/v1"), {})
    compatible = make_provider(make_cfg(provider="openai-compat", model="m", base_url="http://x/v1"),
                               {"ULTRON_API_KEY": "secret-key-123"})
    assert compatible.name == "openai-compat" and compatible.api_key == "secret-key-123"


def test_ollama_needs_no_key_and_has_defaults(make_cfg: Callable[..., Config]) -> None:
    provider = make_provider(make_cfg(provider="ollama"), {})
    assert provider.name == "ollama" and provider.model == "llama3.2:3b"
    assert provider.url == "http://localhost:11434/v1/chat/completions" and provider.api_key == ""
    assert "ollama serve" in provider.unreachable_hint


def test_ollama_end_to_end_and_localhost_bypasses_proxies(
    server: Callable[[], FakeServer], make_cfg: Callable[..., Config], monkeypatch: pytest.MonkeyPatch
) -> None:
    srv = server()
    srv.push(200, chat_completion({"role": "assistant", "content": "local answer"}))
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")  # a dead proxy: a local model must not use it
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")
    provider = make_provider(make_cfg(provider="ollama", base_url=srv.url + "/v1"), {})
    assert provider.complete("s", [Message.user("x")], []).text == "local answer"


def test_unreachable_ollama_explains_how_to_start_it(make_cfg: Callable[..., Config]) -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    provider = make_provider(make_cfg(provider="ollama", base_url=f"http://127.0.0.1:{port}/v1"), {})
    provider.retries, provider.backoff, provider.sleep = 0, 0, lambda s: None
    with pytest.raises(ProviderError, match="ollama serve"):
        provider.complete("s", [Message.user("x")], [])


# ============================================================================ the real loop over HTTP
def test_agent_runs_a_tool_loop_over_the_anthropic_wire_format(
    server: Callable[[], FakeServer], make_cfg: Callable[..., Config]
) -> None:
    srv = server()
    thinking = {"type": "thinking", "thinking": "add them", "signature": "sig-1"}
    srv.push(200, anthropic_message([thinking, tool_use("toolu_1", "calc", expression="19*3")], "tool_use", 40, 9))
    srv.push(200, anthropic_message([tool_use("toolu_2", "finish", summary="19*3 is 57", status="success")],
                                    "tool_use", 60, 8))
    rt = build_runtime(make_cfg(), approver=Recorder(False), provider=anthropic(srv), skills=False, env={})
    result = rt.agent.run("What is 19 times 3?")
    assert result.status == "success" and result.summary == "19*3 is 57"
    assert (result.input_tokens, result.output_tokens) == (100, 17)

    second = srv.requests[1]["body"]["messages"]
    assert second[1]["content"][0] == thinking  # the signed thinking block went back untouched
    assert second[2]["content"][0]["type"] == "tool_result"
    assert second[2]["content"][0]["tool_use_id"] == "toolu_1" and "57" in second[2]["content"][0]["content"]
    assert srv.requests[0]["body"]["system"] == srv.requests[1]["body"]["system"]
    assert KEY not in json.dumps(srv.requests[1]["body"])  # the key travels in a header only


def test_llm_response_defaults() -> None:
    empty = LLMResponse()
    assert empty.text == "" and empty.tool_calls == [] and empty.raw is None

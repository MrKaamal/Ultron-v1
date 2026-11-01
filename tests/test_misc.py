"""Small shared helpers: redaction, utilities, network helpers, message sizes."""

from __future__ import annotations

import re

import pytest

from helpers import FakeServer
from ultron.errors import ProviderError
from ultron.messages import Message, ToolCall, ToolResult
from ultron.netutil import is_loopback_host, post_json
from ultron.redact import REPLACEMENT, Redactor, env_secrets, redact
from ultron.util import clip, human_size, new_run_id, utc_iso


@pytest.mark.parametrize("secret", [
    "sk-ant-api03-abcdefghijklmnopqrstuvwxyz",
    "sk-abcdefghijklmnopqrstuvwxyz0123",
    "Bearer abcdefghijklmnop0123456789",
    "AKIAABCDEFGHIJKLMNOP",
    "ghp_" + "a" * 36,
    "xoxb-1234567890-abcdefghij",
    "-----BEGIN RSA PRIVATE KEY-----\nMIIBOgIBAAJBAK\n-----END RSA PRIVATE KEY-----",
])
def test_known_secret_shapes_are_redacted(secret: str) -> None:
    out = redact(f"before {secret} after", secrets=[])
    assert secret not in out and out == f"before {REPLACEMENT} after"


def test_ordinary_text_is_left_alone() -> None:
    text = "The sky is blue. sk- is a prefix. TODO: ship it."
    assert redact(text, secrets=[]) == text and redact("", secrets=[]) == ""


def test_exact_secret_values_are_redacted_longest_first() -> None:
    assert redact("a hunter2hunter2 b hunter2", secrets=["hunter2", "hunter2hunter2"]) == (
        f"a {REPLACEMENT} b {REPLACEMENT}"
    )


def test_environment_secrets_are_found_by_variable_name() -> None:
    env = {"MY_API_KEY": "abcdefgh1234", "DB_PASSWORD": "pw-0123456789", "SHORT_TOKEN": "tiny", "PATH": "/usr/bin"}
    assert sorted(env_secrets(env)) == ["abcdefgh1234", "pw-0123456789"]
    assert Redactor(env_secrets(env))("login with abcdefgh1234 now") == f"login with {REPLACEMENT} now"


def test_redaction_picks_up_the_live_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOME_SECRET", "live-value-12345")
    assert "live-value-12345" not in redact("the value is live-value-12345")


def test_clip_and_human_size() -> None:
    assert clip("hello", 10) == "hello" and clip("hello world", 8) == "hello w…" and clip("x", 0) == ""
    assert [human_size(n) for n in (0, 1023, 1536, 5 * 1024**2)] == ["0 B", "1023 B", "1.5 KB", "5.0 MB"]


def test_timestamps_and_run_ids() -> None:
    assert utc_iso(0) == "1970-01-01T00:00:00Z"
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", utc_iso())
    first, second = new_run_id(), new_run_id()
    assert re.fullmatch(r"\d{8}-\d{6}-[0-9a-f]{6}", first) and first != second


@pytest.mark.parametrize("host,expected", [
    ("localhost", True), ("127.0.0.1", True), ("[::1]", True), ("LOCALHOST", True),
    ("example.com", False), ("10.0.0.1", False), ("", False),
])
def test_loopback_detection(host: str, expected: bool) -> None:
    assert is_loopback_host(host) is expected


def test_post_json_sends_json_and_a_user_agent() -> None:
    with FakeServer() as srv:
        srv.push(200, {"ok": True})
        assert post_json(srv.url + "/x", {"X-Test": "1"}, {"a": 1}) == {"ok": True}
        headers = {k.lower(): v for k, v in srv.requests[0]["headers"].items()}
        assert srv.requests[0]["body"] == {"a": 1}
        assert headers["x-test"] == "1" and headers["user-agent"].startswith("ultron/")
        assert headers["accept"] == "application/json"


def test_post_json_refuses_non_http_schemes() -> None:
    with pytest.raises(ProviderError, match="http"):
        post_json("file:///etc/passwd", {}, {})


def test_message_size_counts_text_calls_and_results() -> None:
    message = Message.assistant("abc", [ToolCall("1", "calc", {"expression": "1+1"})])
    assert message.size() > len("abc") + len("calc")
    assert Message.tool([ToolResult("1", "calc", "x" * 10)]).size() == 10

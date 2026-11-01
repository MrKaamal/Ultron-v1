"""Test doubles shared by the suite."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from ultron.messages import LLMResponse, Message, ToolCall, ToolSpec


def call(name: str, call_id: str | None = None, **arguments: Any) -> ToolCall:
    digest = abs(hash(json.dumps(arguments, sort_keys=True))) % 10000
    return ToolCall(call_id or f"call_{name}_{digest}", name, arguments)


def reply(*calls: ToolCall, text: str = "", stop: str | None = None) -> LLMResponse:
    return LLMResponse(text=text, tool_calls=list(calls), stop_reason=stop or ("tool_use" if calls else "end_turn"))


class ScriptedProvider:
    """Returns pre-written responses (or calls a callable) and records what it was sent."""

    name = "scripted"
    model = "scripted-1"

    def __init__(self, *responses: LLMResponse | Callable[[], LLMResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[list[Message]] = []
        self.systems: list[str] = []
        self.tools: list[list[ToolSpec]] = []

    def complete(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> LLMResponse:
        self.calls.append(list(messages))
        self.systems.append(system)
        self.tools.append(list(tools))
        if not self.responses:
            raise AssertionError("the provider was asked for more turns than the test scripted")
        item = self.responses.pop(0)
        return item() if callable(item) else item


class Recorder:
    """An approver that records requests and answers from a fixed policy."""

    def __init__(self, answer: bool | Callable[[Any], bool] = False) -> None:
        self.answer = answer
        self.requests: list[Any] = []

    def __call__(self, req: Any) -> bool:
        self.requests.append(req)
        return self.answer(req) if callable(self.answer) else bool(self.answer)


class FakeServer:
    """A tiny local HTTP server that replays queued responses and records every request."""

    def __init__(self) -> None:
        self.queue: list[tuple[int, Any, dict[str, str]]] = []
        self.requests: list[dict[str, Any]] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def _serve(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                try:
                    body = json.loads(raw) if raw else None
                except ValueError:
                    body = raw
                outer.requests.append(
                    {"method": self.command, "path": self.path, "headers": dict(self.headers), "body": body}
                )
                status, payload, headers = outer.queue.pop(0) if outer.queue else (500, {"error": "empty"}, {})
                if isinstance(payload, (dict, list)):
                    data, ctype = json.dumps(payload).encode(), "application/json"
                elif isinstance(payload, bytes):
                    data, ctype = payload, headers.pop("Content-Type", "application/octet-stream")
                else:
                    data, ctype = str(payload).encode(), headers.pop("Content-Type", "text/plain")
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                for key, value in headers.items():
                    self.send_header(key, value)
                self.end_headers()
                self.wfile.write(data)

            do_GET = do_POST = _serve  # noqa: N815

            def log_message(self, *args: Any) -> None:  # silence the test output
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def push(self, status: int, payload: Any, **headers: str) -> None:
        self.queue.append((status, payload, {k.replace("_", "-"): v for k, v in headers.items()}))

    def __enter__(self) -> FakeServer:
        self.thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

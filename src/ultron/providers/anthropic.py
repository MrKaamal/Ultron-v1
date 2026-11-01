"""Anthropic Messages API over plain HTTPS (no SDK).

Notes verified against the official docs (platform.claude.com/docs, October 2026):

* ``POST {base}/v1/messages`` with ``x-api-key`` and ``anthropic-version`` headers. Accounts whose
  key is not scoped to a workspace also need ``anthropic-workspace-id``.
* Tools are ``{name, description, input_schema}``; calls arrive as ``tool_use`` blocks and
  results go back as ``tool_result`` blocks that must come *first* in a user message.
* Current models can return ``thinking`` blocks. In a tool loop they must be sent back
  unmodified, so the full response content is stored in ``Message.raw`` and replayed as-is.
  Replayed blocks are bound to the conversation prefix, so this provider is ``append_only``:
  the agent never rewrites earlier messages while talking to it.
* ``temperature`` / ``top_p`` / ``top_k`` are rejected from API version 2026-10-21, and forced
  ``tool_choice`` is rejected by current models, so none of them are ever sent.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from ultron.errors import ProviderError
from ultron.messages import LLMResponse, Message, ToolCall, ToolSpec
from ultron.netutil import post_json

DEFAULT_BASE_URL = "https://api.anthropic.com"
DEFAULT_MODEL = "claude-sonnet-5-5"
DEFAULT_VERSION = "2026-10-21"
#: Adaptive thinking also spends output tokens, so leave generous headroom.
DEFAULT_MAX_TOKENS = 16000


class AnthropicProvider:
    name = "anthropic"
    #: See module docstring: earlier messages must never be rewritten for this provider.
    append_only = True

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        *,
        base_url: str = DEFAULT_BASE_URL,
        version: str = DEFAULT_VERSION,
        max_tokens: int = 0,
        effort: str = "",
        workspace_id: str = "",
        timeout: float = 120.0,
        retries: int = 3,
        backoff: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.api_key = api_key
        self.model = model or DEFAULT_MODEL
        self.base_url = base_url.rstrip("/")
        self.version = version
        self.max_tokens = max_tokens or DEFAULT_MAX_TOKENS
        self.effort = effort
        self.workspace_id = workspace_id
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self.sleep = sleep

    # -- request ----------------------------------------------------------------------
    @property
    def url(self) -> str:
        return self.base_url + ("/messages" if self.base_url.endswith("/v1") else "/v1/messages")

    @property
    def headers(self) -> dict[str, str]:
        headers = {"x-api-key": self.api_key, "anthropic-version": self.version}
        if self.workspace_id:
            headers["anthropic-workspace-id"] = self.workspace_id
        return headers

    @staticmethod
    def _convert(messages: list[Message]) -> list[dict[str, Any]]:
        """Canonical history -> Anthropic messages (consecutive same-role turns are merged)."""
        out: list[dict[str, Any]] = []

        def push(role: str, blocks: list[dict[str, Any]]) -> None:
            if out and out[-1]["role"] == role:
                out[-1]["content"].extend(blocks)
            else:
                out.append({"role": role, "content": list(blocks)})

        for msg in messages:
            if msg.role == "user":
                push("user", [{"type": "text", "text": msg.text or "(empty)"}])
            elif msg.role == "assistant":
                raw = msg.raw
                if isinstance(raw, dict) and raw.get("provider") == "anthropic":
                    blocks = list(raw["content"])  # replay verbatim, thinking blocks included
                else:
                    blocks = []
                    if msg.text.strip():
                        blocks.append({"type": "text", "text": msg.text})
                    for call in msg.tool_calls:
                        blocks.append({"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments})
                    if not blocks:
                        blocks.append({"type": "text", "text": "(no output)"})
                push("assistant", blocks)
            elif msg.role == "tool":
                results = []
                for res in msg.tool_results:
                    block: dict[str, Any] = {
                        "type": "tool_result",
                        "tool_use_id": res.call_id,
                        "content": res.content or "(no output)",
                    }
                    if res.is_error:
                        block["is_error"] = True
                    results.append(block)
                push("user", results)  # tool_result blocks first, as the API requires
        return out

    def build_payload(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": self._convert(messages),
        }
        if tools:
            payload["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters} for t in tools
            ]
        if self.effort:
            payload["output_config"] = {"effort": self.effort}
        return payload

    # -- response ---------------------------------------------------------------------
    @staticmethod
    def parse_response(data: dict[str, Any]) -> LLMResponse:
        content = data.get("content")
        if not isinstance(content, list):
            raise ProviderError("Anthropic response has no content list")
        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for block in content:
            kind = block.get("type") if isinstance(block, dict) else None
            if kind == "text":
                text_parts.append(str(block.get("text", "")))
            elif kind == "tool_use":
                arguments = block.get("input")
                calls.append(
                    ToolCall(
                        id=str(block.get("id", "")),
                        name=str(block.get("name", "")),
                        arguments=arguments if isinstance(arguments, dict) else {},
                        parse_error=None if isinstance(arguments, dict) else "tool input was not an object",
                    )
                )
        usage = data.get("usage") or {}
        return LLMResponse(
            text="\n".join(part for part in text_parts if part).strip(),
            tool_calls=calls,
            stop_reason=str(data.get("stop_reason") or ("tool_use" if calls else "end_turn")),
            input_tokens=int(usage.get("input_tokens", 0) or 0),
            output_tokens=int(usage.get("output_tokens", 0) or 0),
            raw={"provider": "anthropic", "content": content},
        )

    def complete(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> LLMResponse:
        data = post_json(
            self.url,
            self.headers,
            self.build_payload(system, messages, tools),
            timeout=self.timeout,
            retries=self.retries,
            backoff=self.backoff,
            sleep=self.sleep,
            secrets=[self.api_key],
        )
        return self.parse_response(data)

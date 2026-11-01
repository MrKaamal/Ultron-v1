"""OpenAI-compatible Chat Completions client.

Works with OpenAI, Ollama (``http://localhost:11434/v1``), LM Studio, vLLM, OpenRouter,
Groq, Together and anything else that speaks ``POST {base}/chat/completions`` with tools.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from collections.abc import Callable
from typing import Any

from ultron.errors import ProviderError
from ultron.messages import LLMResponse, Message, ToolCall, ToolSpec
from ultron.netutil import post_json

_FINISH_REASONS = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "length": "max_tokens",
    "content_filter": "refusal",
}
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


def parse_arguments(raw: Any) -> tuple[dict[str, Any], str | None]:
    """Decode tool arguments that may arrive as a JSON string, a dict, or garbage."""
    if isinstance(raw, dict):
        return raw, None
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return {}, None
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except ValueError:
            return {}, f"arguments were not valid JSON: {raw[:120]!r}"
        if isinstance(parsed, dict):
            return parsed, None
        return {}, "arguments JSON was not an object"
    return {}, "arguments had an unexpected type"


class OpenAICompatProvider:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str = "",
        name: str = "openai-compat",
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float = 120.0,
        retries: int = 3,
        backoff: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
        unreachable_hint: str = "",
    ) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self.sleep = sleep
        self.unreachable_hint = unreachable_hint

    @property
    def url(self) -> str:
        return f"{self.base_url}/chat/completions"

    @staticmethod
    def _convert(system: str, messages: list[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for msg in messages:
            if msg.role == "user":
                out.append({"role": "user", "content": msg.text})
            elif msg.role == "assistant":
                content = msg.text or (None if msg.tool_calls else "")
                entry: dict[str, Any] = {"role": "assistant", "content": content}
                if msg.tool_calls:
                    entry["tool_calls"] = [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
                        }
                        for call in msg.tool_calls
                    ]
                out.append(entry)
            elif msg.role == "tool":
                for res in msg.tool_results:
                    content = res.content or "(no output)"
                    if res.is_error and not content.upper().startswith("ERROR"):
                        content = f"ERROR: {content}"
                    out.append({"role": "tool", "tool_call_id": res.call_id, "content": content})
        return out

    def build_payload(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": self.model, "messages": self._convert(system, messages)}
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
                }
                for t in tools
            ]
            payload["tool_choice"] = "auto"
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens
        return payload

    @staticmethod
    def parse_response(data: dict[str, Any]) -> LLMResponse:
        try:
            choice = data["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError):
            raise ProviderError("response has no choices[0].message") from None
        text = _THINK_BLOCK.sub("", message.get("content") or "").strip()
        calls: list[ToolCall] = []
        for index, item in enumerate(message.get("tool_calls") or []):
            function = item.get("function") or {}
            arguments, error = parse_arguments(function.get("arguments"))
            calls.append(
                ToolCall(
                    id=str(item.get("id") or f"call_{index}_{uuid.uuid4().hex[:8]}"),
                    name=str(function.get("name") or ""),
                    arguments=arguments,
                    parse_error=error,
                )
            )
        reason = str(choice.get("finish_reason") or "")
        stop_reason = _FINISH_REASONS.get(reason, reason or ("tool_use" if calls else "end_turn"))
        if calls and stop_reason == "end_turn":
            stop_reason = "tool_use"  # some servers report "stop" even when calling tools
        usage = data.get("usage") or {}
        return LLMResponse(
            text=text,
            tool_calls=calls,
            stop_reason=stop_reason,
            input_tokens=int(usage.get("prompt_tokens", 0) or 0),
            output_tokens=int(usage.get("completion_tokens", 0) or 0),
        )

    def complete(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> LLMResponse:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        data = post_json(
            self.url,
            headers,
            self.build_payload(system, messages, tools),
            timeout=self.timeout,
            retries=self.retries,
            backoff=self.backoff,
            sleep=self.sleep,
            secrets=[self.api_key] if self.api_key else [],
            unreachable_hint=self.unreachable_hint,
        )
        return self.parse_response(data)

"""Provider-agnostic message types."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

#: Separates the context block (time, recalled notes) from the operator's goal in the first
#: user message. The offline demo brain splits on it; real models just read it.
GOAL_HEADER = "OPERATOR GOAL:"


@dataclass(frozen=True)
class ToolSpec:
    """What the model is told about a tool."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    #: Set when the provider could not decode the arguments (empty or None means "fine").
    parse_error: str | None = None


@dataclass
class ToolResult:
    call_id: str
    name: str
    content: str
    is_error: bool = False


@dataclass
class Message:
    """One conversation turn.

    ``role`` is ``user`` (text), ``assistant`` (text + tool_calls) or ``tool`` (tool_results
    answering the previous assistant turn). ``raw`` carries opaque provider content that must be
    replayed verbatim (for example signed thinking blocks).
    """

    role: str
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    raw: Any = None

    @classmethod
    def user(cls, text: str) -> Message:
        return cls("user", text)

    @classmethod
    def assistant(
        cls, text: str = "", tool_calls: Sequence[ToolCall] | None = None, raw: Any = None
    ) -> Message:
        return cls("assistant", text, list(tool_calls or []), raw=raw)

    @classmethod
    def tool(cls, results: Sequence[ToolResult]) -> Message:
        return cls("tool", tool_results=list(results))

    def size(self) -> int:
        """Rough size in characters, used to keep the transcript inside the context budget."""
        total = len(self.text)
        for call in self.tool_calls:
            total += len(call.name) + len(json.dumps(call.arguments, default=str))
        for result in self.tool_results:
            total += len(result.content)
        return total


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    raw: Any = None

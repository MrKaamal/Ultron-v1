"""The one interface every LLM backend implements."""

from __future__ import annotations

from typing import Protocol

from ultron.messages import LLMResponse, Message, ToolSpec


class Provider(Protocol):
    """A model backend.

    Optional attribute ``append_only`` (default False): set it to True when earlier messages
    must never be rewritten between calls. The agent then skips context compaction, which
    trims old tool output. The Anthropic provider needs this because the API binds replayed
    ``thinking`` blocks to the exact conversation prefix.
    """

    #: Short identifier shown in logs, e.g. "anthropic".
    name: str
    #: Model identifier sent to the backend (or a label for offline backends).
    model: str

    def complete(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> LLMResponse:
        """Run one model turn. Must raise ``ProviderError`` (never leak secrets) on failure."""
        ...

"""Tool registry and the built-in tool set."""

from __future__ import annotations

from collections.abc import Iterator

from ultron.errors import ConfigError
from ultron.messages import ToolSpec
from ultron.tools import calc, fs, notes, shell, web
from ultron.tools.base import Tool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            existing = self._tools[tool.name]
            raise ConfigError(f"tool '{tool.name}' is already registered by {existing.source}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self) -> list[ToolSpec]:
        return [self._tools[name].spec() for name in self.names()]

    def __iter__(self) -> Iterator[Tool]:
        return (self._tools[name] for name in self.names())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools


def builtin_tools() -> list[Tool]:
    return [*fs.TOOLS, *shell.TOOLS, *web.TOOLS, *calc.TOOLS, *notes.TOOLS]


def build_registry() -> ToolRegistry:
    """A registry containing only the built-in tools (skills are added separately)."""
    registry = ToolRegistry()
    for tool in builtin_tools():
        registry.register(tool)
    return registry

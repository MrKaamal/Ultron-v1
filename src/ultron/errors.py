"""Exception hierarchy."""

from __future__ import annotations


class UltronError(Exception):
    """Base class for all ULTRON errors."""


class ConfigError(UltronError):
    """Invalid or unsafe configuration."""


class ProviderError(UltronError):
    """The LLM provider failed (network, auth, malformed response)."""


class SandboxViolation(UltronError):
    """A path or resource lies outside what the agent may touch."""


class ToolError(UltronError):
    """A tool failed in an expected way; the message is shown to the model."""


class MemoryFull(ToolError):
    """Long-term memory reached its note limit."""


class KillSwitchEngaged(UltronError):
    """The kill switch is engaged; all agent activity must stop."""


class AuditError(UltronError):
    """The audit log is unreadable or inconsistent."""

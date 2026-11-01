"""Best-effort secret scrubbing for logs, tool output and error messages."""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping

REPLACEMENT = "[REDACTED]"

_PATTERNS = [
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{10,}"),
    re.compile(r"sk-[A-Za-z0-9_\-]{20,}"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{16,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9\-]{10,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]+?-----END [A-Z ]*PRIVATE KEY-----"),
]
_SECRET_NAME = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL)", re.I)


def env_secrets(env: Mapping[str, str] | None = None) -> list[str]:
    """Values of environment variables that look like secrets (length >= 8)."""
    env = os.environ if env is None else env
    return [v for k, v in env.items() if _SECRET_NAME.search(k) and len(v) >= 8]


class Redactor:
    """``redact`` with the environment's secret values captured once, for hot paths."""

    def __init__(self, secrets: Iterable[str] | None = None) -> None:
        self._secrets = list(env_secrets() if secrets is None else secrets)

    def __call__(self, text: str) -> str:
        return redact(text, self._secrets)


def redact(text: str, secrets: Iterable[str] | None = None) -> str:
    """Replace known secret shapes and exact secret values in ``text``."""
    if not text:
        return text
    for value in sorted(env_secrets() if secrets is None else secrets, key=len, reverse=True):
        if value:
            text = text.replace(value, REPLACEMENT)
    for pattern in _PATTERNS:
        text = pattern.sub(REPLACEMENT, text)
    return text

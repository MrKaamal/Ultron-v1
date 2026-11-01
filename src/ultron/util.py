"""Tiny shared helpers."""

from __future__ import annotations

import secrets
import time
from datetime import UTC, datetime


def utc_iso(ts: float | None = None) -> str:
    """UTC timestamp like 2026-10-27T05:00:00Z."""
    dt = datetime.fromtimestamp(ts, tz=UTC) if ts is not None else datetime.now(UTC)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def new_run_id() -> str:
    """Sortable, collision-resistant run identifier."""
    return time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + "-" + secrets.token_hex(3)


def clip(text: str, limit: int, marker: str = "…") -> str:
    """Shorten ``text`` to at most ``limit`` characters (marker included)."""
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - len(marker))] + marker


def human_size(n: int) -> str:
    """1536 -> '1.5 KB'."""
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{int(size)} B" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{n} B"  # pragma: no cover - loop always returns

"""Append-only, hash-chained audit log (JSON Lines).

Each record stores the SHA-256 of the previous record, so editing or deleting
any past line is detectable with :meth:`AuditLog.verify`. This is
tamper-*evidence*, not tamper-proofing: someone with write access can rebuild the
whole chain, so note the head hash somewhere you trust if that matters.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from ultron.errors import AuditError
from ultron.redact import redact

try:  # POSIX only; on Windows appends are unlocked (single-writer assumption)
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

GENESIS = "0" * 64
_MAX_STR = 2000
_MAX_ITEMS = 50


@dataclass
class VerifyResult:
    ok: bool
    count: int
    head: str
    error: str = ""
    line: int = 0


@contextmanager
def _locked(f: IO[bytes]) -> Iterator[None]:
    if fcntl is None:  # pragma: no cover
        yield
        return
    fcntl.flock(f, fcntl.LOCK_EX)
    try:
        yield
    finally:
        fcntl.flock(f, fcntl.LOCK_UN)


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _digest(prev: str, body: dict[str, Any]) -> str:
    return hashlib.sha256((prev + _canonical(body)).encode("utf-8")).hexdigest()


def _scrub(obj: Any) -> Any:
    if isinstance(obj, str):
        text = redact(obj)
        if len(text) > _MAX_STR:
            text = f"{text[:_MAX_STR]}...[+{len(text) - _MAX_STR} chars]"
        return text
    if isinstance(obj, dict):
        return {str(k): _scrub(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        items = [_scrub(v) for v in obj[:_MAX_ITEMS]]
        if len(obj) > _MAX_ITEMS:
            items.append(f"...[+{len(obj) - _MAX_ITEMS} items]")
        return items
    if obj is None or isinstance(obj, (bool, int, float)):
        return obj
    return _scrub(str(obj))


class AuditLog:
    def __init__(self, path: Path):
        self.path = Path(path)

    def _tail(self, f: IO[bytes]) -> tuple[int, str]:
        f.seek(0, 2)
        size = f.tell()
        if size == 0:
            return 0, GENESIS
        f.seek(max(0, size - 65536))
        lines = [ln for ln in f.read().splitlines() if ln.strip()]
        try:
            last = json.loads(lines[-1])
            return int(last["seq"]), str(last["hash"])
        except (IndexError, ValueError, KeyError) as exc:
            raise AuditError(
                f"audit log {self.path} is corrupted at its tail; run `ultron audit --verify`"
            ) from exc

    def append(self, event: str, run_id: str | None = None, **data: Any) -> dict[str, Any]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "ab+") as f, _locked(f):
            seq, prev = self._tail(f)
            body = {
                "seq": seq + 1,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "run_id": run_id,
                "event": event,
                "data": _scrub(data),
            }
            record = {**body, "prev": prev, "hash": _digest(prev, body)}
            f.seek(0, 2)
            f.write(_canonical(record).encode("utf-8") + b"\n")
            f.flush()
        return record

    def records(self) -> Iterator[dict[str, Any]]:
        if not self.path.exists():
            return
        with open(self.path, "rb") as f:
            for raw in f:
                if raw.strip():
                    yield json.loads(raw)

    def tail(self, n: int = 20) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for rec in self.records():
            out.append(rec)
            del out[:-n]
        return out

    def verify(self) -> VerifyResult:
        prev, count = GENESIS, 0
        if not self.path.exists():
            return VerifyResult(True, 0, prev)
        with open(self.path, "rb") as f:
            for lineno, raw in enumerate(f, 1):
                if not raw.strip():
                    continue
                try:
                    rec = json.loads(raw)
                    body = {k: rec[k] for k in ("seq", "ts", "run_id", "event", "data")}
                    rec_prev, rec_hash = rec["prev"], rec["hash"]
                except (ValueError, KeyError, TypeError):
                    return VerifyResult(False, count, prev, "malformed record", lineno)
                if body["seq"] != count + 1:
                    return VerifyResult(False, count, prev, "sequence gap or reorder", lineno)
                if rec_prev != prev:
                    return VerifyResult(False, count, prev, "broken chain link", lineno)
                if _digest(prev, body) != rec_hash:
                    return VerifyResult(False, count, prev, "record was modified", lineno)
                prev, count = rec_hash, count + 1
        return VerifyResult(True, count, prev)

"""Persistent memory: notes the agent can recall, plus a history of runs (SQLite)."""

from __future__ import annotations

import re
import sqlite3
import time
from collections.abc import Iterator, Sequence
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ultron.errors import MemoryFull, ToolError

MAX_NOTE_CHARS = 2000

_STOP = frozenset(
    {"the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are", "was", "were", "my", "me", "i",
     "you", "do", "does", "about", "what", "that", "this", "it", "with", "at", "by", "be"}
)
_SCHEMA = """
CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    text TEXT NOT NULL,
    tags TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT 'agent'
);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    started_at REAL NOT NULL,
    source TEXT NOT NULL,
    goal TEXT NOT NULL,
    status TEXT NOT NULL,
    summary TEXT NOT NULL,
    steps INTEGER NOT NULL,
    tool_calls INTEGER NOT NULL,
    elapsed REAL NOT NULL
);
"""


@dataclass
class Note:
    id: int
    created_at: float
    text: str
    tags: list[str]
    source: str
    score: float = 0.0


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP}


class Memory:
    def __init__(self, path: Path, max_notes: int = 5000, max_note_chars: int = MAX_NOTE_CHARS):
        self.path = Path(path)
        self.max_notes = max_notes
        self.max_note_chars = max_note_chars
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        with closing(sqlite3.connect(self.path, timeout=10)) as conn:
            conn.row_factory = sqlite3.Row
            with conn:
                yield conn

    # -- notes ---------------------------------------------------------
    def add_note(self, text: str, tags: Sequence[str] = (), source: str = "agent") -> int:
        text = text.strip()
        if not text:
            raise ToolError("note text is empty")
        if len(text) > self.max_note_chars:
            raise ToolError(f"note too long ({len(text)} > {self.max_note_chars} chars)")
        with self._conn() as conn:
            (count,) = conn.execute("SELECT COUNT(*) FROM notes").fetchone()
            if count >= self.max_notes:
                raise MemoryFull(f"memory is full ({self.max_notes} notes); ask the user to prune it")
            cur = conn.execute(
                "INSERT INTO notes (created_at, text, tags, source) VALUES (?, ?, ?, ?)",
                (time.time(), text, ",".join(t.strip() for t in tags if t.strip()), source),
            )
            return int(cur.lastrowid or 0)

    @staticmethod
    def _note(row: sqlite3.Row, score: float = 0.0) -> Note:
        tags = [t for t in row["tags"].split(",") if t]
        return Note(row["id"], row["created_at"], row["text"], tags, row["source"], score)

    def search(self, query: str, limit: int = 5) -> list[Note]:
        q = _tokens(query)
        if not q:
            return []
        needle = query.strip().lower()
        scored: list[Note] = []
        with self._conn() as conn:
            for row in conn.execute("SELECT * FROM notes"):
                score = len(q & _tokens(row["text"])) + 0.5 * len(q & _tokens(row["tags"]))
                if needle and needle in row["text"].lower():
                    score += 1
                if score > 0:
                    scored.append(self._note(row, score))
        scored.sort(key=lambda n: (n.score, n.created_at), reverse=True)
        return scored[:limit]

    def recent(self, limit: int = 20) -> list[Note]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM notes ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [self._note(r) for r in rows]

    def forget(self, note_id: int) -> bool:
        with self._conn() as conn:
            return conn.execute("DELETE FROM notes WHERE id = ?", (note_id,)).rowcount > 0

    def count(self) -> int:
        with self._conn() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0])

    # -- runs ----------------------------------------------------------
    def record_run(self, run_id: str, started_at: float, source: str, goal: str, result: dict[str, Any]) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    started_at,
                    source,
                    goal[:2000],
                    result["status"],
                    result["summary"][:2000],
                    result["steps"],
                    result["tool_calls"],
                    result["elapsed"],
                ),
            )

    def recent_runs(self, limit: int = 10) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def runs_since(self, ts: float, source: str | None = None) -> int:
        sql, args = "SELECT COUNT(*) FROM runs WHERE started_at >= ?", [ts]
        if source:
            sql += " AND source = ?"
            args.append(source)
        with self._conn() as conn:
            return int(conn.execute(sql, args).fetchone()[0])

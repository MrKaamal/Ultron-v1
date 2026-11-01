"""Long-term notes and run history (SQLite)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from ultron.errors import MemoryFull, ToolError
from ultron.memory import Memory


@pytest.fixture
def memory(tmp_path: Path) -> Memory:
    return Memory(tmp_path / "state" / "memory.db", max_notes=50)


def test_notes_round_trip_with_tags_and_source(memory: Memory) -> None:
    note_id = memory.add_note("  my favorite language is Python  ", tags=["prefs", " ", "lang"], source="user")
    (note,) = memory.recent()
    assert note.id == note_id and note.text == "my favorite language is Python"
    assert note.tags == ["prefs", "lang"] and note.source == "user" and note.created_at <= time.time()
    assert memory.count() == 1


def test_search_ranks_by_overlap_and_phrase(memory: Memory) -> None:
    memory.add_note("the build uses make and ninja")
    memory.add_note("python packaging notes")
    memory.add_note("my favorite language is python")
    hits = memory.search("favorite language python")
    assert [h.text for h in hits][0] == "my favorite language is python"
    assert len(hits) == 2 and hits[0].score > hits[1].score
    assert memory.search("kubernetes") == []


def test_search_matches_tags_and_ignores_stopwords(memory: Memory) -> None:
    memory.add_note("remember to renew the cert", tags=["deploy"])
    assert [n.text for n in memory.search("deploy")] == ["remember to renew the cert"]
    assert memory.search("what is the") == []  # nothing but stopwords
    assert memory.search("   ") == []


def test_search_limit_and_recency_tie_break(memory: Memory) -> None:
    for i in range(5):
        memory.add_note(f"deploy step {i}")
    hits = memory.search("deploy", limit=3)
    assert len(hits) == 3 and hits[0].text == "deploy step 4"  # same score: newest first


def test_recent_is_newest_first(memory: Memory) -> None:
    for text in ("one", "two", "three"):
        memory.add_note(text)
    assert [n.text for n in memory.recent(2)] == ["three", "two"]


def test_invalid_notes_are_rejected(memory: Memory) -> None:
    with pytest.raises(ToolError, match="empty"):
        memory.add_note("   ")
    with pytest.raises(ToolError, match="too long"):
        memory.add_note("x" * 3000)
    assert memory.count() == 0


def test_the_note_limit_is_enforced(tmp_path: Path) -> None:
    small = Memory(tmp_path / "m.db", max_notes=2)
    small.add_note("a")
    small.add_note("b")
    with pytest.raises(MemoryFull, match="ask the user to prune"):
        small.add_note("c")
    assert small.forget(1) is True
    small.add_note("c")  # space was freed


def test_forget(memory: Memory) -> None:
    note_id = memory.add_note("temporary")
    assert memory.forget(note_id) is True and memory.forget(note_id) is False
    assert memory.count() == 0


def test_memory_persists_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "state" / "memory.db"
    Memory(path).add_note("survives a restart")
    assert [n.text for n in Memory(path).recent()] == ["survives a restart"]


def test_run_history(memory: Memory) -> None:
    def result(status: str) -> dict:
        return {"status": status, "summary": "s" * 3000, "steps": 2, "tool_calls": 3, "elapsed": 1.5}

    memory.record_run("run-1", 1000.0, "cli", "first goal", result("success"))
    memory.record_run("run-2", 2000.0, "sentinel", "second goal", result("failed"))
    memory.record_run("run-3", time.time(), "sentinel", "third goal", result("success"))
    runs = memory.recent_runs(2)
    assert [r["run_id"] for r in runs] == ["run-3", "run-2"]
    assert len(runs[1]["summary"]) == 2000  # long summaries are clipped
    assert memory.runs_since(1500.0) == 2
    assert memory.runs_since(0.0, "cli") == 1
    memory.record_run("run-1", 1000.0, "cli", "first goal", result("failed"))  # same id replaces
    assert memory.runs_since(0.0, "cli") == 1 and len(memory.recent_runs(10)) == 3

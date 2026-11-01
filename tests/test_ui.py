"""Terminal output, the approval prompt and the guided demo."""

from __future__ import annotations

import io
import re
from pathlib import Path

import pytest

from ultron.agent import RunResult
from ultron.demo import run_demo
from ultron.policy import ApprovalRequest
from ultron.ui import LiveReporter, Printer, deny_all, format_args, make_approver, print_result


class Utf8(io.StringIO):
    encoding = "utf-8"


def printer(color: bool = False, utf: bool = False) -> tuple[Printer, io.StringIO]:
    stream = Utf8() if utf else io.StringIO()
    return Printer(stream, color=color), stream


def request(tool: str = "write_file") -> ApprovalRequest:
    return ApprovalRequest("run-1", tool, {"path": "a.txt", "content": "hi"}, "create a.txt (2 B)")


def test_colour_is_only_used_when_asked_or_on_a_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    plain, _ = printer(color=False)
    assert plain.style("x", "bold") == "x"
    coloured, _ = printer(color=True)
    assert coloured.style("x", "bold", "red") == "\033[1;31mx\033[0m" and coloured.style("x") == "x"
    assert Printer(io.StringIO(), env={}).color is False  # not a tty
    monkeypatch.setenv("NO_COLOR", "1")
    assert Printer(io.StringIO()).color is False


def test_symbols_fall_back_to_ascii_without_utf8() -> None:
    assert printer(utf=False)[0].sym["ok"] == "+" and printer(utf=True)[0].sym["ok"] == "✓"


def test_format_args_summarises_long_content() -> None:
    assert format_args({"path": "a.txt", "content": "x" * 100}) == "path=a.txt content=<100 chars>"
    assert format_args({"q": "a\n  b"}) == "q=a b" and len(format_args({"q": "y" * 500}, width=20)) < 30


def test_print_result_shows_status_and_stats() -> None:
    p, stream = printer()
    result = RunResult("r", "goal", "blocked", "line one\nline two", steps=2, tool_calls=3, denials=1, elapsed=0.5)
    print_result(p, result)
    out = stream.getvalue()
    assert "x blocked" in out and "2 step(s)" in out and "1 denied" in out and "  line two" in out


def test_the_live_reporter_narrates_a_run() -> None:
    p, stream = printer()
    live = LiveReporter(p, verbose=True)
    live("run_start", {"run_id": "r1", "provider": "mock", "model": "m"})
    live("step", {"n": 1})
    live("llm", {"text": "thinking about it"})
    live("tool_start", {"name": "read_file", "args": {"path": "a.txt"}})
    live("decision", {"verdict": "deny", "reason": "escapes the workspace"})
    live("approval", {"approved": False})
    live("tool_end", {"is_error": True, "preview": "no such file\nmore", "denied": False})
    live("warning", {"message": "careful"})
    out = stream.getvalue()
    for expected in ("run r1", "step 1", "thinking about it", "read_file path=a.txt", "blocked: escapes",
                     "declined", "x no such file", "careful"):
        assert expected in out


# ------------------------------------------------------------------ approvals
def approver_with(answers: list[str], interactive: bool = True):  # type: ignore[no-untyped-def]
    p, stream = printer()
    queue = iter(answers)
    return make_approver(p, ask=lambda prompt: next(queue), interactive=interactive), stream


@pytest.mark.parametrize("answer,expected", [("y", True), ("YES", True), ("n", False), ("", False), ("no", False)])
def test_the_approver_follows_the_humans_answer(answer: str, expected: bool) -> None:
    approve, stream = approver_with([answer])
    assert approve(request()) is expected
    assert "create a.txt" in stream.getvalue() and "content: hi" in stream.getvalue()


def test_the_approver_reprompts_on_unclear_answers() -> None:
    approve, _ = approver_with(["what?", "maybe", "y"])
    assert approve(request()) is True


def test_always_applies_to_the_tool_for_the_session_only() -> None:
    approve, _ = approver_with(["a"])
    assert approve(request("write_file")) is True
    assert approve(request("write_file")) is True  # no second prompt (the iterator would be exhausted)


@pytest.mark.parametrize("tool", ["run_command", "http_get"])
def test_risky_tools_never_offer_always(tool: str) -> None:
    approve, stream = approver_with(["a", "n"])
    assert approve(request(tool)) is False  # "a" is not accepted for these, so it asks again
    assert "[a]lways" not in stream.getvalue()


def test_without_a_terminal_everything_is_declined() -> None:
    approve, stream = approver_with([], interactive=False)
    assert approve(request()) is False and "no terminal" in stream.getvalue()


def test_end_of_input_declines() -> None:
    p, _ = printer()

    def eof(prompt: str) -> str:
        raise EOFError

    assert make_approver(p, ask=eof, interactive=True)(request()) is False
    assert deny_all(request()) is False


# ------------------------------------------------------------------ the guided demo
def test_the_demo_runs_clean_and_removes_its_files() -> None:
    p, stream = printer()
    assert run_demo(p) == 0
    out = stream.getvalue()
    for section in ("1. Autonomy", "2. Memory", "3. Guardrails", "4. Kill switch", "5. Sentinel", "6. Tamper"):
        assert section in out
    workspace = Path(re.search(r"workspace: (\S+)", out).group(1))  # type: ignore[union-attr]
    assert not workspace.parent.exists()
    assert "do-not-leak-me" not in out


def test_the_demo_can_keep_its_files() -> None:
    p, stream = printer()
    assert run_demo(p, keep=True) == 0
    kept = Path(re.search(r"kept demo files in (\S+)", stream.getvalue()).group(1))  # type: ignore[union-attr]
    try:
        assert (kept / "state" / "audit.jsonl").is_file()
    finally:
        import shutil

        shutil.rmtree(kept, ignore_errors=True)

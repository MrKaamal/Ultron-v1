"""The command line, end to end, using the offline brain."""

from __future__ import annotations

import io
import json
from collections.abc import Callable
from pathlib import Path

import pytest

from ultron.cli import EXIT_BLOCKED, EXIT_KILLED, EXIT_OTHER, EXIT_USAGE, exit_code, load_env_file, main

Cli = Callable[..., tuple[int, str, str]]


@pytest.fixture
def cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> Cli:
    monkeypatch.chdir(tmp_path)
    flags = ["--workspace", str(tmp_path / "ws"), "--state-dir", str(tmp_path / "state"), "--no-color"]

    def run(*argv: str, settings: bool = True) -> tuple[int, str, str]:
        rc = main([*argv, *(flags if settings else [])])
        captured = capsys.readouterr()
        return rc, captured.out, captured.err

    return run


def test_exit_codes() -> None:
    assert [exit_code(s) for s in ("success", "failed", "blocked", "killed")] == [0, 1, 3, 4]
    assert exit_code("budget") == exit_code("stuck") == exit_code("error") == EXIT_OTHER


def test_version_and_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as info:
        main(["--version"])
    assert info.value.code == 0 and capsys.readouterr().out.startswith("ultron ")
    assert main([]) == 0 and "usage: ultron" in capsys.readouterr().out


def test_package_exposes_version() -> None:
    from ultron import __version__

    assert __version__


def test_run_answers_a_question(cli: Cli) -> None:
    rc, out, _ = cli("run", "What is 17 * 23?")
    assert rc == 0 and "391" in out and "success" in out and "offline demo brain" in out


def test_run_json_keeps_stdout_machine_readable(cli: Cli) -> None:
    rc, out, err = cli("run", "What is 6 * 7?", "--json")
    data = json.loads(out)
    assert rc == 0 and data["status"] == "success" and "42" in data["summary"] and data["tool_calls"] == 2
    assert err == ""  # nothing but the JSON document is printed


def test_quiet_prints_only_the_result(cli: Cli) -> None:
    rc, out, _ = cli("run", "What is 2 + 2?", "-q")
    assert rc == 0 and "run 2" not in out and "workspace:" not in out and "4" in out


def test_the_goal_can_come_from_stdin(cli: Cli, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("What is 9 * 9?"))
    rc, out, _ = cli("run", "-")
    assert rc == 0 and "81" in out


def test_a_missing_goal_is_a_usage_error(cli: Cli) -> None:
    rc, _, err = cli("run")
    assert rc == EXIT_USAGE and "no goal given" in err


def test_writes_are_declined_without_a_terminal(cli: Cli, tmp_path: Path) -> None:
    rc, out, _ = cli("run", "Create a file called hello.txt with hello world")
    assert rc == EXIT_BLOCKED and "no terminal" in out
    assert not (tmp_path / "ws" / "hello.txt").exists()


def test_approval_auto_allows_the_write_and_deny_blocks_it(cli: Cli, tmp_path: Path) -> None:
    rc, out, _ = cli("run", "Create a file called hello.txt with hello world", "--approval", "auto")
    assert rc == 0 and "approval=auto" in out
    assert (tmp_path / "ws" / "hello.txt").read_text() == "hello world\n"
    rc, _, _ = cli("run", "Create a file called other.txt with x", "--approval", "deny")
    assert rc == EXIT_BLOCKED and not (tmp_path / "ws" / "other.txt").exists()


def test_guardrails_block_secrets_and_escapes(cli: Cli, tmp_path: Path) -> None:
    (tmp_path / "ws").mkdir()
    (tmp_path / "ws" / ".env").write_text("KEY=1")
    for goal in ("Read .env", "Read ../../etc/passwd", "Run command rm -rf /"):
        rc, out, _ = cli("run", goal, "-q")
        assert rc == EXIT_BLOCKED, goal
        assert "KEY=1" not in out


def test_kill_stops_runs_until_revived(cli: Cli) -> None:
    rc, out, _ = cli("kill", "testing", "the", "switch")
    assert rc == 0 and "testing the switch" in out
    rc, out, _ = cli("status")
    assert "ENGAGED" in out
    rc, out, _ = cli("run", "What is 1 + 1?", "-q")
    assert rc == EXIT_KILLED and "kill switch" in out
    assert cli("revive")[1].startswith("kill switch released")
    assert cli("revive")[1].startswith("kill switch was not engaged")
    assert cli("run", "What is 1 + 1?", "-q")[0] == 0


def test_tools_lists_policy(cli: Cli) -> None:
    rc, out, _ = cli("tools")
    lines = {line.split()[0]: line for line in out.splitlines() if line.strip() and not line.startswith("TOOL")}
    assert rc == 0 and "runs" in lines["read_file"] and "asks" in lines["write_file"] and "finish" in lines


def test_memory_commands(cli: Cli) -> None:
    assert cli("memory", "list")[1].strip() == "(no notes)"
    assert "saved note #1" in cli("memory", "add", "my", "editor", "is", "vim")[1]
    assert "my editor is vim" in cli("memory", "list")[1]
    assert "vim" in cli("memory", "search", "editor")[1]
    assert cli("memory", "add")[0] == EXIT_USAGE
    assert cli("memory", "forget", "abc")[0] == EXIT_USAGE
    assert cli("memory", "forget", "99")[0] == 1
    assert "forgot note #1" in cli("memory", "forget", "1")[1]


def test_the_agent_can_use_memory_across_runs(cli: Cli) -> None:
    assert cli("run", "Remember that my favorite language is Python", "-q")[0] == 0
    rc, out, _ = cli("run", "What do you remember about language?", "-q")
    assert rc == 0 and "favorite language is Python" in out


def test_history_and_audit(cli: Cli, tmp_path: Path) -> None:
    assert "no runs yet" in cli("history")[1]
    cli("run", "What is 3 * 3?", "-q")
    assert "What is 3 * 3?" in cli("history")[1]
    rc, out, _ = cli("audit")
    assert rc == 0 and "run_start" in out and "run_end" in out
    rc, out, _ = cli("audit", "--verify")
    assert rc == 0 and "chain intact" in out


def test_a_tampered_audit_log_fails_verification(cli: Cli, tmp_path: Path) -> None:
    cli("run", "What is 3 * 3?", "-q")
    log = tmp_path / "state" / "audit.jsonl"
    log.write_text(log.read_text().replace('"status":"success"', '"status":"failed"', 1))
    rc, out, _ = cli("audit", "--verify")
    assert rc == 1 and "TAMPERED" in out
    rc, out, _ = cli("doctor")
    assert rc == 1 and "FAIL" in out
    assert "BROKEN" in cli("status")[1]


def test_status_on_a_fresh_install(cli: Cli) -> None:
    rc, out, _ = cli("status")
    assert rc == 0 and "kill switch: off" in out and "0 notes" in out and "chain intact" in out


def test_doctor_on_a_fresh_install(cli: Cli) -> None:
    rc, out, _ = cli("doctor")
    assert rc == 0 and "all good" in out and "offline demo brain" in out and "http_get disabled" in out


def test_doctor_reports_bad_configuration(cli: Cli) -> None:
    rc, out, _ = cli("doctor", "--max-steps", "0")
    assert rc == EXIT_USAGE and "FAIL" in out


def test_a_missing_config_file_is_a_usage_error(cli: Cli) -> None:
    rc, _, err = cli("run", "x", "--config", "nope.toml")
    assert rc == EXIT_USAGE and "configuration:" in err


def test_init_creates_a_working_project(cli: Cli, tmp_path: Path) -> None:
    rc, out, _ = cli("init", settings=False)
    assert rc == 0 and "wrote ultron.toml" in out
    assert (tmp_path / "workspace" / "WELCOME.md").is_file() and (tmp_path / "skills").is_dir()
    assert "already exists" in cli("init", settings=False)[1]
    assert "wrote ultron.toml" in cli("init", "--force", settings=False)[1]
    rc, out, _ = main(["doctor", "--no-color"]), "", ""
    assert rc == 0
    rc, out, _ = cli("run", "Summarize the files in the workspace", settings=False)
    assert rc == 0 and "WELCOME.md" in out


def test_sentinel_once(cli: Cli, tmp_path: Path) -> None:
    assert "inbox is empty" in cli("sentinel", "--once")[1]
    inbox = tmp_path / "state" / "inbox"
    (inbox / "q.task").write_text("What is 6 * 7?")
    rc, out, _ = cli("sentinel", "--once")
    assert rc == 0 and "q.task: success" in out
    assert any((tmp_path / "state" / "outbox").glob("q-*.md"))


def test_the_demo_passes_all_its_own_checks(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["demo", "--no-color"])
    out = capsys.readouterr().out
    assert rc == 0 and "every guardrail behaved as expected" in out
    assert "TAMPERED" not in out and "intact=False" in out


# ------------------------------------------------------------------ .env handling
def test_only_credentials_are_read_from_dotenv(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text(
        "# comment\nANTHROPIC_API_KEY='sk-ant-quoted'\nexport OPENAI_API_KEY=abc\n"
        "ULTRON_APPROVAL=auto\nANTHROPIC_BASE_URL=http://evil.example\nBROKEN LINE\nULTRON_API_KEY=\n",
        encoding="utf-8",
    )
    env: dict[str, str] = {"OPENAI_API_KEY": "from-shell"}
    assert load_env_file(path, env) == ["ANTHROPIC_API_KEY"]
    assert env == {"OPENAI_API_KEY": "from-shell", "ANTHROPIC_API_KEY": "sk-ant-quoted"}
    assert load_env_file(tmp_path / "missing", env) == []

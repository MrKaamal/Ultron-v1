"""The hash-chained audit log and the file-based kill switch."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from ultron.audit import GENESIS, AuditLog
from ultron.errors import AuditError, KillSwitchEngaged
from ultron.killswitch import KillSwitch
from ultron.policy import Policy
from ultron.runtime import Runtime
from ultron.tools.base import ToolContext, Verdict


@pytest.fixture
def log(tmp_path: Path) -> AuditLog:
    return AuditLog(tmp_path / "state" / "audit.jsonl")


def fill(log: AuditLog, n: int = 3) -> None:
    for i in range(n):
        log.append(f"event{i}", f"run-{i}", index=i, note="x")


def rewrite(log: AuditLog, edit) -> None:  # type: ignore[no-untyped-def]
    lines = log.path.read_text(encoding="utf-8").splitlines()
    log.path.write_text("\n".join(edit(lines)) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ the chain
def test_records_are_chained_from_a_genesis_hash(log: AuditLog) -> None:
    first = log.append("start", "r1", goal="hello")
    second = log.append("end", "r1")
    assert first["seq"] == 1 and first["prev"] == GENESIS and first["run_id"] == "r1"
    assert second["seq"] == 2 and second["prev"] == first["hash"]
    assert first["data"] == {"goal": "hello"} and first["ts"]
    result = log.verify()
    assert result.ok and result.count == 2 and result.head == second["hash"]


def test_an_empty_or_missing_log_is_valid(log: AuditLog) -> None:
    result = log.verify()
    assert result.ok and result.count == 0 and result.head == GENESIS
    assert log.tail(5) == [] and list(log.records()) == []


def test_a_second_instance_continues_the_same_chain(log: AuditLog) -> None:
    fill(log, 2)
    AuditLog(log.path).append("later", None)
    assert log.verify().count == 3


def test_tail_returns_the_newest_records(log: AuditLog) -> None:
    fill(log, 5)
    assert [r["seq"] for r in log.tail(2)] == [4, 5]


# ------------------------------------------------------------------ tamper evidence
def test_editing_a_record_is_detected(log: AuditLog) -> None:
    fill(log)
    rewrite(log, lambda lines: [lines[0], lines[1].replace('"index":1', '"index":9'), lines[2]])
    result = log.verify()
    assert not result.ok and result.line == 2 and "modified" in result.error


def test_deleting_a_record_is_detected(log: AuditLog) -> None:
    fill(log)
    rewrite(log, lambda lines: [lines[0], lines[2]])
    result = log.verify()
    assert not result.ok and result.line == 2 and "sequence gap" in result.error


def test_reordering_records_is_detected(log: AuditLog) -> None:
    fill(log)
    rewrite(log, lambda lines: [lines[1], lines[0], lines[2]])
    assert not log.verify().ok


def test_inserting_a_forged_record_is_detected(log: AuditLog) -> None:
    fill(log)
    forged = json.loads(log.path.read_text(encoding="utf-8").splitlines()[0])
    forged["data"] = {"forged": True}
    rewrite(log, lambda lines: [lines[0], json.dumps(forged), lines[1], lines[2]])
    assert not log.verify().ok


def test_garbage_lines_are_reported_as_malformed(log: AuditLog) -> None:
    fill(log, 1)
    with log.path.open("a", encoding="utf-8") as fh:
        fh.write("not json\n")
    result = log.verify()
    assert not result.ok and result.error == "malformed record" and result.line == 2


def test_cutting_off_the_tail_changes_the_head_hash(log: AuditLog) -> None:
    """A chain cannot prove nothing was removed at the end; pinning the head hash elsewhere can."""
    fill(log, 4)
    pinned = log.verify().head
    rewrite(log, lambda lines: lines[:3])
    result = log.verify()
    assert result.ok and result.count == 3 and result.head != pinned


def test_appending_to_a_corrupted_log_fails_closed(log: AuditLog) -> None:
    fill(log, 1)
    with log.path.open("a", encoding="utf-8") as fh:
        fh.write("garbage\n")
    with pytest.raises(AuditError, match="corrupted"):
        log.append("more", None)


# ------------------------------------------------------------------ what gets written
def test_secrets_are_redacted_before_they_are_written(log: AuditLog) -> None:
    log.append("tool", "r", args={"header": "Bearer abcdef0123456789abcdef", "keys": ["sk-ant-secretsecretsecret"]})
    text = log.path.read_text(encoding="utf-8")
    assert "abcdef0123456789abcdef" not in text and "secretsecretsecret" not in text
    assert text.count("[REDACTED]") == 2


def test_oversized_values_are_truncated(log: AuditLog) -> None:
    record = log.append("big", None, text="a" * 5000, items=list(range(80)), obj=object())
    assert len(record["data"]["text"]) < 2100 and "+3000 chars" in record["data"]["text"]
    assert len(record["data"]["items"]) == 51 and "+30 items" in record["data"]["items"][-1]
    assert isinstance(record["data"]["obj"], str)
    assert log.verify().ok


def test_concurrent_writers_keep_the_chain_intact(log: AuditLog) -> None:
    def worker(tag: int) -> None:
        for i in range(25):
            AuditLog(log.path).append("tick", f"w{tag}", i=i)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    result = log.verify()
    assert result.ok and result.count == 200


# ------------------------------------------------------------------ the kill switch
def test_kill_switch_lifecycle(tmp_path: Path) -> None:
    kill = KillSwitch(tmp_path / "deep" / "KILL")
    assert not kill.engaged and kill.info() is None
    kill.check()  # does nothing while released
    kill.engage("maintenance window")
    assert kill.engaged
    info = kill.info()
    assert info is not None and info["reason"] == "maintenance window" and info["pid"] > 0
    with pytest.raises(KillSwitchEngaged, match="maintenance window"):
        kill.check()
    assert kill.release() is True and kill.release() is False
    kill.check()


def test_an_unreadable_kill_file_still_stops_everything(tmp_path: Path) -> None:
    path = tmp_path / "KILL"
    path.write_text("{not json", encoding="utf-8")
    kill = KillSwitch(path)
    assert kill.info() == {"reason": "unknown (kill file unreadable)"}
    with pytest.raises(KillSwitchEngaged):
        kill.check()


@pytest.mark.parametrize("target", ["../state/KILL", "../state/audit.jsonl", "../state/memory.db",
                                    "../state/inbox/evil.task"])
def test_the_agent_cannot_reach_its_own_guardrails(rt: Runtime, target: str) -> None:
    ctx = ToolContext(rt.cfg, rt.sandbox, rt.memory)
    for tool, args in (("write_file", {"path": target, "content": "x", "mode": "overwrite"}),
                       ("read_file", {"path": target})):
        decision = Policy(rt.cfg).evaluate(rt.registry.get(tool), args, ctx)  # type: ignore[arg-type]
        assert decision.verdict is Verdict.DENY, (tool, target)
    command = Policy(rt.cfg).evaluate(rt.registry.get("run_command"), {"command": f"cat {target}"}, ctx)  # type: ignore[arg-type]
    assert command.verdict is Verdict.DENY

"""Sentinel mode: inbox in, outbox out, with limits and the kill switch."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from helpers import ScriptedProvider, call, reply
from ultron.messages import LLMResponse
from ultron.runtime import Runtime
from ultron.sentinel import MAX_TASK_BYTES, Sentinel

MakeRt = Callable[..., Runtime]


def finish(summary: str = "done", status: str = "success") -> LLMResponse:
    return reply(call("finish", None, summary=summary, status=status))


def drop(rt: Runtime, name: str, text: str = "do the thing", age: float = 0.0) -> Path:
    path = rt.cfg.inbox_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if age:
        stamp = path.stat().st_mtime - age
        os.utime(path, (stamp, stamp))
    return path


def test_a_task_runs_and_leaves_a_result_file(make_rt: MakeRt) -> None:
    rt = make_rt(provider=ScriptedProvider(finish("Everything is fine.")))
    sentinel = Sentinel(rt)
    drop(rt, "report.task", "Write the weekly report")
    (outcome,) = sentinel.run_once()
    assert outcome.name == "report.task" and outcome.status == "success"
    assert outcome.result_path is not None and outcome.result_path.parent == rt.cfg.outbox_dir
    body = outcome.result_path.read_text(encoding="utf-8")
    assert "status: **success**" in body and "Write the weekly report" in body and "Everything is fine." in body
    assert (rt.cfg.inbox_dir / "done" / "report.task").exists()
    assert not (rt.cfg.inbox_dir / "report.task").exists()
    assert rt.memory.recent_runs(1)[0]["source"] == "sentinel"


def test_failed_tasks_are_filed_under_failed(make_rt: MakeRt) -> None:
    rt = make_rt(provider=ScriptedProvider(finish("could not", "failed")))
    drop(rt, "bad.md")
    (outcome,) = Sentinel(rt).run_once()
    assert outcome.status == "failed" and (rt.cfg.inbox_dir / "failed" / "bad.md").exists()


def test_tasks_run_oldest_first(make_rt: MakeRt) -> None:
    rt = make_rt(provider=ScriptedProvider(finish(), finish(), finish()))
    drop(rt, "c.txt", age=0)
    drop(rt, "a.txt", age=100)
    drop(rt, "b.txt", age=50)
    assert [o.name for o in Sentinel(rt).run_once()] == ["a.txt", "b.txt", "c.txt"]


def test_unattended_runs_cannot_get_human_approval(make_rt: MakeRt, workspace: Path) -> None:
    provider = ScriptedProvider(reply(call("write_file", "w", path="x.txt", content="x")), finish("no", "blocked"))
    rt = make_rt(provider=provider)  # the default test approver answers "no", like deny_all
    drop(rt, "write.task")
    (outcome,) = Sentinel(rt).run_once()
    assert outcome.status == "blocked" and not (workspace / "x.txt").exists()
    assert "denied actions: 1" in outcome.result_path.read_text(encoding="utf-8")  # type: ignore[union-attr]


def test_oversized_empty_and_unreadable_tasks_are_rejected_without_running(make_rt: MakeRt) -> None:
    provider = ScriptedProvider()  # any model call would fail the test
    rt = make_rt(provider=provider)
    drop(rt, "huge.task", "x" * (MAX_TASK_BYTES + 1))
    drop(rt, "empty.task", "  \n")
    outcomes = {o.name: o for o in Sentinel(rt).run_once()}
    assert outcomes["huge.task"].status == "rejected" and "larger than" in outcomes["huge.task"].summary
    assert outcomes["empty.task"].status == "rejected" and "empty" in outcomes["empty.task"].summary
    assert len(list((rt.cfg.inbox_dir / "failed").iterdir())) == 2 and provider.calls == []


def test_only_plain_task_files_are_picked_up(make_rt: MakeRt, tmp_path: Path) -> None:
    rt = make_rt(provider=ScriptedProvider(finish()))
    sentinel = Sentinel(rt)
    drop(rt, "real.task")
    drop(rt, "notes.pdf")
    drop(rt, ".hidden.task")
    secret = tmp_path / "secret.txt"
    secret.write_text("read me")
    (rt.cfg.inbox_dir / "link.task").symlink_to(secret)
    assert [p.name for p in sentinel.pending()] == ["real.task"]


def test_a_task_that_was_already_claimed_is_skipped(make_rt: MakeRt) -> None:
    rt = make_rt(provider=ScriptedProvider())
    outcome = Sentinel(rt).process(rt.cfg.inbox_dir / "ghost.task")
    assert outcome.status == "skipped" and outcome.summary == "already claimed"


def test_repeated_names_do_not_overwrite_the_archive(make_rt: MakeRt) -> None:
    rt = make_rt(provider=ScriptedProvider(finish("one"), finish("two")))
    sentinel = Sentinel(rt)
    drop(rt, "same.task", "first")
    sentinel.run_once()
    drop(rt, "same.task", "second")
    sentinel.run_once()
    names = sorted(p.name for p in (rt.cfg.inbox_dir / "done").iterdir())
    assert len(names) == 2 and "same.task" in names


def test_the_hourly_rate_limit_leaves_extra_tasks_queued(make_rt: MakeRt) -> None:
    rt = make_rt(provider=ScriptedProvider(finish(), finish()), max_runs_per_hour=1)
    sentinel = Sentinel(rt)
    drop(rt, "one.task", age=10)
    drop(rt, "two.task")
    first, second = sentinel.run_once()
    assert first.status == "success" and second.status == "skipped" and "hourly limit" in second.summary
    assert (rt.cfg.inbox_dir / "two.task").exists()
    assert sentinel.runs_left() == 0


def test_the_kill_switch_stops_processing_and_leaves_tasks_queued(make_rt: MakeRt) -> None:
    provider = ScriptedProvider()
    rt = make_rt(provider=provider)
    sentinel = Sentinel(rt)
    drop(rt, "wait.task")
    rt.killswitch.engage("stop")
    assert sentinel.run_once() == []
    assert sentinel.watch() == "killed"
    assert (rt.cfg.inbox_dir / "wait.task").exists() and provider.calls == []


def test_a_task_killed_mid_run_is_not_silently_retried(make_rt: MakeRt) -> None:
    holder: dict[str, Runtime] = {}

    def approver(req: object) -> bool:
        holder["rt"].killswitch.engage("halt")
        return True

    provider = ScriptedProvider(reply(call("write_file", "w", path="x.txt", content="x")))
    rt = make_rt(provider=provider, approver=approver)
    holder["rt"] = rt
    drop(rt, "risky.task")
    (outcome,) = Sentinel(rt).run_once()
    assert outcome.status == "killed" and (rt.cfg.inbox_dir / "failed" / "risky.task").exists()


def test_watch_polls_until_the_cycle_limit(make_rt: MakeRt) -> None:
    sleeps: list[float] = []
    rt = make_rt(provider=ScriptedProvider(finish()))
    sentinel = Sentinel(rt, sleep=sleeps.append)
    drop(rt, "later.task")
    seen: list[str] = []
    assert sentinel.watch(on_outcome=lambda o: seen.append(o.name), max_cycles=3) == "finished"
    assert seen == ["later.task"] and sleeps == [rt.cfg.poll_interval] * 2


def test_the_inbox_and_outbox_are_outside_the_workspace(rt: Runtime) -> None:
    Sentinel(rt)
    for folder in (rt.cfg.inbox_dir, rt.cfg.outbox_dir):
        assert rt.cfg.workspace not in folder.parents and folder != rt.cfg.workspace

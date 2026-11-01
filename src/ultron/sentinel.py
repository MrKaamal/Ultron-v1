"""Sentinel mode: drop a goal into the inbox folder, get a result file in the outbox.

``Sentinel`` watches ``cfg.inbox_dir`` for ``*.task``, ``*.txt`` and ``*.md`` files. Each file is
one goal. The agent runs it unattended (nothing that needs a human can run, so risky tools are
declined unless the operator configured ``approval = "auto"``), the result is written to
``cfg.outbox_dir`` and the task file moves to ``done/`` or ``failed/`` inside the inbox.

Safety properties:

* The inbox and outbox live in the state directory, *outside* the agent's workspace, so the agent
  cannot queue tasks for itself or edit results.
* Whoever can write to the inbox can steer the agent, so protect it like a login. Symlinks and
  oversized files are ignored.
* A rate limit (``max_runs_per_hour``) and the kill switch stop a runaway queue.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ultron.agent import RunResult
from ultron.runtime import Runtime
from ultron.util import clip, utc_iso

SUFFIXES = (".task", ".txt", ".md")
MAX_TASK_BYTES = 8000
SOURCE = "sentinel"


@dataclass
class TaskOutcome:
    name: str
    status: str  # a RunResult status, or "rejected" / "skipped"
    summary: str
    result_path: Path | None = None


class Sentinel:
    def __init__(
        self,
        runtime: Runtime,
        *,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.rt = runtime
        self.cfg = runtime.cfg
        self.clock = clock
        self.sleep = sleep
        self.inbox = self.cfg.inbox_dir
        self.outbox = self.cfg.outbox_dir
        for sub in ("done", "failed", "processing"):
            (self.inbox / sub).mkdir(parents=True, exist_ok=True)
        self.outbox.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ queue
    def pending(self) -> list[Path]:
        """Task files waiting in the inbox, oldest first. Symlinks and odd files are ignored."""
        found = []
        for path in self.inbox.iterdir():
            if path.is_symlink() or not path.is_file() or path.name.startswith("."):
                continue
            if path.suffix.lower() in SUFFIXES:
                found.append(path)
        return sorted(found, key=lambda p: (p.stat().st_mtime, p.name))

    def runs_left(self) -> int:
        used = self.rt.memory.runs_since(self.clock() - 3600, SOURCE)
        return max(0, self.cfg.max_runs_per_hour - used)

    # ------------------------------------------------------------------ work
    def run_once(self) -> list[TaskOutcome]:
        """Process everything currently queued; stops early at the rate limit or kill switch."""
        outcomes: list[TaskOutcome] = []
        for path in self.pending():
            if self.rt.killswitch.engaged:
                break
            if self.runs_left() <= 0:
                outcomes.append(
                    TaskOutcome(
                        path.name,
                        "skipped",
                        f"hourly limit of {self.cfg.max_runs_per_hour} runs reached; task left in the inbox",
                    )
                )
                break
            outcomes.append(self.process(path))
        return outcomes

    def process(self, path: Path) -> TaskOutcome:
        claimed = self._claim(path)
        if claimed is None:  # another sentinel took it
            return TaskOutcome(path.name, "skipped", "already claimed")
        try:
            raw = claimed.read_bytes()
        except OSError as exc:
            return self._finish_unrun(claimed, "rejected", f"could not read the task: {exc}")
        if len(raw) > MAX_TASK_BYTES:
            return self._finish_unrun(claimed, "rejected", f"task file is larger than {MAX_TASK_BYTES} bytes")
        goal = raw.decode("utf-8", errors="replace").strip()
        if not goal:
            return self._finish_unrun(claimed, "rejected", "task file is empty")

        result = self.rt.agent.run(goal, source=SOURCE)
        result_path = self._write_result(path.name, result)
        self._archive(claimed, "done" if result.ok else "failed", result.run_id)
        return TaskOutcome(path.name, result.status, result.summary, result_path)

    def watch(
        self,
        *,
        on_outcome: Callable[[TaskOutcome], None] | None = None,
        max_cycles: int | None = None,
    ) -> str:
        """Poll until the kill switch is engaged (returns ``"killed"``) or ``max_cycles`` ends."""
        cycles = 0
        while True:
            if self.rt.killswitch.engaged:
                return "killed"
            for outcome in self.run_once():
                if on_outcome:
                    on_outcome(outcome)
            cycles += 1
            if max_cycles is not None and cycles >= max_cycles:
                return "finished"
            self.sleep(self.cfg.poll_interval)

    # ------------------------------------------------------------------ files
    def _claim(self, path: Path) -> Path | None:
        target = self.inbox / "processing" / path.name
        try:
            os.replace(path, target)
        except FileNotFoundError:
            return None
        return target

    def _archive(self, claimed: Path, folder: str, run_id: str) -> None:
        dest = self.inbox / folder / claimed.name
        if dest.exists():
            dest = dest.with_name(f"{claimed.stem}-{run_id}{claimed.suffix}")
        os.replace(claimed, dest)

    def _finish_unrun(self, claimed: Path, status: str, why: str) -> TaskOutcome:
        self._archive(claimed, "failed", f"rej-{int(self.clock())}")
        return TaskOutcome(claimed.name, status, why)

    def _write_result(self, task_name: str, result: RunResult) -> Path:
        stem = Path(task_name).stem
        path = self.outbox / f"{stem}-{result.run_id}.md"
        body = (
            f"# ULTRON result: {task_name}\n\n"
            f"- status: **{result.status}**\n"
            f"- run id: `{result.run_id}`\n"
            f"- finished: {utc_iso()}\n"
            f"- steps: {result.steps}, tool calls: {result.tool_calls}, denied actions: {result.denials}\n\n"
            f"## Goal\n\n{clip(result.goal, 2000)}\n\n"
            f"## Outcome\n\n{result.summary}\n"
        )
        path.write_text(body, encoding="utf-8")
        return path

"""File-based kill switch.

The kill file lives in the state directory, which is outside the agent's
workspace, so the agent cannot remove it. Every agent step and every tool call
checks it.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from ultron.errors import KillSwitchEngaged


class KillSwitch:
    def __init__(self, path: Path):
        self.path = Path(path)

    @property
    def engaged(self) -> bool:
        return self.path.exists()

    def info(self) -> dict[str, Any] | None:
        if not self.engaged:
            return None
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"reason": "unknown (kill file unreadable)"}

    def engage(self, reason: str = "manual") -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"reason": reason, "ts": time.time(), "pid": os.getpid()}
        self.path.write_text(json.dumps(payload), encoding="utf-8")

    def release(self) -> bool:
        try:
            self.path.unlink()
            return True
        except FileNotFoundError:
            return False

    def check(self) -> None:
        if self.engaged:
            info = self.info() or {}
            raise KillSwitchEngaged(str(info.get("reason", "manual")))

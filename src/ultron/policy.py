"""The policy engine: every tool call passes through ``Policy.evaluate`` before it runs.

Order of evaluation:

1. FORBIDDEN tools never run.
2. The tool's own ``check`` hook may DENY (e.g. path escape, command not allowed), ASK, or ALLOW.
3. Otherwise the tool's base risk decides: SAFE -> allow, CONFIRM -> ask.
4. An ASK is then softened or hardened by configuration:
   ``auto_approve`` (per tool) and ``approval = "auto" | "deny"`` (global).
   DENY is final - no configuration turns a denial into an allow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ultron.config import Config
from ultron.errors import SandboxViolation
from ultron.tools.base import Decision, Risk, Tool, ToolContext, Verdict

_BASE = {Risk.SAFE: Verdict.ALLOW, Risk.CONFIRM: Verdict.ASK, Risk.FORBIDDEN: Verdict.DENY}


@dataclass(frozen=True)
class ApprovalRequest:
    run_id: str
    tool: str
    args: dict[str, Any]
    reason: str


class Policy:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def evaluate(self, tool: Tool, args: dict[str, Any], ctx: ToolContext) -> Decision:
        """Classify one call. ToolError from a check hook (bad input) propagates to the caller."""
        if tool.risk is Risk.FORBIDDEN:
            return Decision.deny(f"tool '{tool.name}' is forbidden")

        decision: Decision | None = None
        if tool.check is not None:
            try:
                decision = tool.check(ctx, args)
            except SandboxViolation as exc:
                return Decision.deny(str(exc))
        if decision is None:
            decision = Decision(_BASE[tool.risk], f"{tool.risk.value} tool")

        if decision.verdict is Verdict.DENY:
            return decision
        if decision.verdict is Verdict.ASK:
            if tool.name in self.cfg.auto_approve:
                return Decision.allow(f"auto-approved by config ({decision.reason})")
            if self.cfg.approval == "deny":
                return Decision.deny(f"approval mode is 'deny' ({decision.reason})")
            if self.cfg.approval == "auto":
                return Decision.allow(f"auto-approve mode ({decision.reason})")
        return decision

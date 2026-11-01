"""The agent loop.

One run is: repeat { model turn -> for each tool call: validate -> policy -> (approval) ->
execute } until the model calls ``finish`` or a budget, the kill switch or a safety rule stops
it. Every decision is appended to the audit log, and nothing the model says can bypass the
policy engine: it only ever *requests* actions.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ultron.audit import AuditLog
from ultron.config import Config
from ultron.errors import AuditError, KillSwitchEngaged, ProviderError, SandboxViolation, ToolError, UltronError
from ultron.killswitch import KillSwitch
from ultron.memory import Memory
from ultron.messages import LLMResponse, Message, ToolCall, ToolResult
from ultron.policy import ApprovalRequest, Policy
from ultron.prompts import build_system_prompt, build_user_message
from ultron.providers.base import Provider
from ultron.redact import Redactor
from ultron.sandbox import Sandbox
from ultron.tools import ToolRegistry
from ultron.tools.base import ToolContext, Verdict, coerce_args
from ultron.util import clip, new_run_id

Approver = Callable[[ApprovalRequest], bool]
Reporter = Callable[[str, dict[str, Any]], None]

#: Identical consecutive calls: warn the model at the first, abort the run at the second.
REPEAT_WARN = 3
REPEAT_ABORT = 5
#: Stop reasons meaning the model's reply was cut off, so a tool call in it may be incomplete.
CUT_OFF = frozenset({"max_tokens", "model_context_window_exceeded"})
TRIM_MARK = " …[older output trimmed to save context]"
#: Events written to the audit log (the rest only go to the live reporter).
AUDITED = frozenset({"run_start", "llm", "decision", "approval", "tool_end", "warning", "run_end"})

DENIAL_ADVICE = (
    "Do not retry this action or look for a workaround. Choose a safe alternative, or call finish "
    "with status 'blocked' and explain what you need."
)


@dataclass
class RunResult:
    run_id: str
    goal: str
    status: str  # success | failed | blocked | budget | killed | stuck | interrupted | error
    summary: str
    steps: int = 0
    tool_calls: int = 0
    denials: int = 0
    elapsed: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    messages: list[Message] = field(default_factory=list, repr=False)

    @property
    def ok(self) -> bool:
        return self.status == "success"

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "goal": self.goal,
            "status": self.status,
            "summary": self.summary,
            "steps": self.steps,
            "tool_calls": self.tool_calls,
            "denials": self.denials,
            "elapsed": round(self.elapsed, 3),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


@dataclass
class _State:
    run_id: str
    goal: str
    source: str
    started: float  # on the agent's (monotonic) clock
    wall_start: float
    steps: int = 0
    tool_calls: int = 0
    denials: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    last_signature: tuple[str, str] | None = None
    repeat: int = 0


def compact_messages(messages: list[Message], max_chars: int, keep_recent: int = 4) -> int:
    """Trim the oldest bulky tool results in place until the transcript fits. Returns the count.

    Only called for providers that tolerate history rewrites (see ``Provider.append_only``).
    """
    total = sum(m.size() for m in messages)
    if total <= max_chars:
        return 0
    trimmed = 0
    candidates = messages[:-keep_recent] if keep_recent else messages
    for message in candidates:
        if message.role != "tool":
            continue
        for result in message.tool_results:
            if len(result.content) > 240 and not result.content.endswith(TRIM_MARK):
                total -= len(result.content) - (120 + len(TRIM_MARK))
                result.content = result.content[:120] + TRIM_MARK
                trimmed += 1
                if total <= max_chars:
                    return trimmed
    return trimmed


class Agent:
    def __init__(
        self,
        *,
        cfg: Config,
        provider: Provider,
        registry: ToolRegistry,
        policy: Policy,
        sandbox: Sandbox,
        memory: Memory,
        audit: AuditLog,
        killswitch: KillSwitch,
        approver: Approver,
        reporter: Reporter | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cfg = cfg
        self.provider = provider
        self.registry = registry
        self.policy = policy
        self.sandbox = sandbox
        self.memory = memory
        self.audit = audit
        self.killswitch = killswitch
        self.approver = approver
        self.reporter = reporter
        self.clock = clock
        self._redact = Redactor()
        self._append_only = bool(getattr(provider, "append_only", False))

    # ------------------------------------------------------------------ public API
    def run(self, goal: str, *, history: list[Message] | None = None, source: str = "cli") -> RunResult:
        """Run one goal to completion. ``history`` continues an earlier conversation."""
        goal = goal.strip()
        if not goal:
            raise UltronError("the goal is empty")
        state = _State(new_run_id(), goal, source, self.clock(), time.time())
        ctx = ToolContext(self.cfg, self.sandbox, self.memory, state.run_id)
        messages = list(history or [])
        self._emit(
            "run_start",
            state,
            goal=goal,
            provider=self.provider.name,
            model=self.provider.model,
            source=source,
        )
        try:
            notes = self.memory.search(goal, limit=self.cfg.memory_recall_k)
            messages.append(Message.user(build_user_message(goal, notes)))
            return self._loop(state, ctx, messages)
        except KillSwitchEngaged as exc:
            return self._end(state, "killed", f"Stopped by the kill switch ({exc}).", messages)
        except KeyboardInterrupt:
            return self._end(state, "interrupted", "Interrupted by the operator.", messages)
        except ProviderError as exc:
            return self._end(state, "error", f"Model provider error: {exc}", messages)
        except AuditError:
            raise  # fail closed: without an audit trail the run must not continue
        except Exception as exc:  # noqa: BLE001 - a bug must end the run cleanly, not crash a daemon
            return self._end(state, "error", f"Internal error: {type(exc).__name__}: {exc}", messages)

    # ------------------------------------------------------------------ the loop
    def _loop(self, state: _State, ctx: ToolContext, messages: list[Message]) -> RunResult:
        cfg = self.cfg
        system = build_system_prompt(cfg)
        specs = self.registry.specs()
        for step in range(1, cfg.max_steps + 1):
            self.killswitch.check()
            if self.clock() - state.started > cfg.max_seconds:
                return self._end(state, "budget", f"Time budget of {cfg.max_seconds}s exhausted.", messages)
            state.steps = step
            self._emit("step", state, audit=False, n=step)
            if not self._append_only:
                compact_messages(messages, cfg.max_context_chars)

            response = self.provider.complete(system, messages, specs)
            state.input_tokens += response.input_tokens
            state.output_tokens += response.output_tokens
            messages.append(Message.assistant(response.text, response.tool_calls, raw=response.raw))
            self._emit(
                "llm",
                state,
                text=response.text,
                tools=[c.name for c in response.tool_calls],
                stop=response.stop_reason,
                tokens_in=response.input_tokens,
                tokens_out=response.output_tokens,
            )

            if not response.tool_calls:
                return self._answer_without_tools(state, response, messages)

            results: list[ToolResult] = []
            try:
                outcome = self._run_calls(state, ctx, response, results)
            finally:
                # Keep the transcript valid even if we are stopped mid-step: every tool call
                # in an assistant message must be answered before the history is reused.
                for call in response.tool_calls[len(results) :]:
                    results.append(ToolResult(call.id, call.name, "cancelled: the run was stopped", True))
                messages.append(Message.tool(results))

            if outcome is not None:
                return self._end(state, outcome[0], outcome[1], messages)
            if state.denials >= cfg.max_denials:
                return self._end(
                    state,
                    "blocked",
                    f"Stopped after {state.denials} denied actions: "
                    "the goal needs something the policy does not allow.",
                    messages,
                )
            if state.repeat >= REPEAT_ABORT:
                return self._end(
                    state,
                    "stuck",
                    f"Stopped: the same tool call was repeated {state.repeat} times in a row.",
                    messages,
                )
            if state.tool_calls >= cfg.max_tool_calls:
                return self._end(state, "budget", f"Tool-call budget of {cfg.max_tool_calls} exhausted.", messages)
        return self._end(
            state,
            "budget",
            f"Step budget of {cfg.max_steps} exhausted before the model called finish.",
            messages,
        )

    def _answer_without_tools(self, state: _State, response: LLMResponse, messages: list[Message]) -> RunResult:
        """The model replied with plain text instead of calling finish."""
        text = response.text.strip()
        if response.stop_reason == "refusal":
            return self._end(state, "blocked", text or "The model declined to continue.", messages)
        if response.stop_reason in CUT_OFF:
            note = f"[the model's reply was cut off ({response.stop_reason}); raise max_output_tokens]"
            return self._end(state, "failed", f"{text}\n{note}".strip(), messages)
        if not text:
            return self._end(state, "failed", "The model returned an empty reply.", messages)
        return self._end(state, "success", text, messages)

    def _run_calls(
        self, state: _State, ctx: ToolContext, response: LLMResponse, results: list[ToolResult]
    ) -> tuple[str, str] | None:
        """Execute the calls of one assistant turn, appending to ``results`` as it goes."""
        cut_off = response.stop_reason in CUT_OFF
        outcome: tuple[str, str] | None = None
        for call in response.tool_calls:
            if cut_off:
                note = "Your reply was cut off before this call was complete, so it was NOT executed. Retry shorter."
                results.append(self._reject(state, call, note))
            elif outcome is not None:
                results.append(self._reject(state, call, "skipped: the run already finished"))
            elif state.tool_calls >= self.cfg.max_tool_calls:
                note = f"tool-call budget of {self.cfg.max_tool_calls} is used up; call finish now"
                results.append(self._reject(state, call, note))
            else:
                result, finished = self._execute(state, ctx, call)
                results.append(result)
                outcome = finished
        return outcome

    # ------------------------------------------------------------------ one tool call
    def _execute(self, state: _State, ctx: ToolContext, call: ToolCall) -> tuple[ToolResult, tuple[str, str] | None]:
        state.tool_calls += 1
        self._emit("tool_start", state, audit=False, name=call.name, args=call.arguments, call_id=call.id)
        self._track_repeats(state, call)

        tool = self.registry.get(call.name)
        if call.parse_error:
            note = f"Could not read the arguments for '{call.name}': {call.parse_error}. Send a JSON object."
            return self._reject(state, call, note), None
        if tool is None:
            note = f"Unknown tool '{call.name}'. Available tools: {', '.join(self.registry.names())}."
            return self._reject(state, call, note), None
        args, problems = coerce_args(tool.parameters, call.arguments)
        if problems:
            return self._reject(state, call, f"Invalid arguments for '{call.name}': {'; '.join(problems)}."), None

        self.killswitch.check()
        try:
            decision = self.policy.evaluate(tool, args, ctx)
        except ToolError as exc:  # a usage mistake (for example a shell operator), not a policy denial
            return self._reject(state, call, str(exc)), None
        self._emit(
            "decision", state, tool=tool.name, verdict=decision.verdict.value, reason=decision.reason, args=args
        )

        if decision.verdict is Verdict.DENY:
            return self._deny(state, call, f"DENIED by policy: {decision.reason}. {DENIAL_ADVICE}"), None
        if decision.verdict is Verdict.ASK:
            approved = bool(self.approver(ApprovalRequest(state.run_id, tool.name, args, decision.reason)))
            self._emit("approval", state, tool=tool.name, approved=approved, reason=decision.reason)
            if not approved:
                note = f"DENIED: the operator did not approve this action. {DENIAL_ADVICE}"
                return self._deny(state, call, note), None

        self.killswitch.check()
        try:
            output = tool.func(ctx, args)
        except SandboxViolation as exc:  # second line of defence: the tool itself refused
            return self._deny(state, call, f"DENIED by sandbox: {exc}. {DENIAL_ADVICE}"), None
        except ToolError as exc:
            return self._reject(state, call, str(exc)), None
        except KillSwitchEngaged:
            raise
        except Exception as exc:  # noqa: BLE001 - a crashing tool must not take the agent down
            self._emit("warning", state, message=f"tool {tool.name} crashed: {type(exc).__name__}: {exc}")
            return self._reject(state, call, f"{tool.name} crashed: {type(exc).__name__}: {exc}"), None

        output = self._redact(str(output))
        limit = self.cfg.max_tool_output_chars
        if len(output) > limit:
            output = output[:limit] + f"\n[output truncated: {len(output)} characters in total]"
        if state.repeat == REPEAT_WARN:
            output += (
                f"\n[ULTRON: you have made this exact call {state.repeat} times in a row. "
                "Try something different or call finish.]"
            )
        result = ToolResult(call.id, call.name, output)
        self._emit("tool_end", state, tool=call.name, is_error=False, preview=clip(output, 300), call_id=call.id)

        finished = None
        if tool.name == "finish":
            finished = (str(args.get("status", "success")), str(args["summary"]).strip())
        return result, finished

    def _track_repeats(self, state: _State, call: ToolCall) -> None:
        signature = (call.name, json.dumps(call.arguments, sort_keys=True, default=str))
        if signature == state.last_signature:
            state.repeat += 1
        else:
            state.last_signature, state.repeat = signature, 1

    def _reject(self, state: _State, call: ToolCall, message: str) -> ToolResult:
        """An error result for the model (bad arguments, tool failure) - not a policy denial."""
        message = self._redact(message)
        self._emit("tool_end", state, tool=call.name, is_error=True, preview=clip(message, 300), call_id=call.id)
        return ToolResult(call.id, call.name, message, is_error=True)

    def _deny(self, state: _State, call: ToolCall, message: str) -> ToolResult:
        state.denials += 1
        message = self._redact(message)
        self._emit(
            "tool_end", state, tool=call.name, is_error=True, denied=True, preview=clip(message, 300), call_id=call.id
        )
        return ToolResult(call.id, call.name, message, is_error=True)

    # ------------------------------------------------------------------ bookkeeping
    def _end(self, state: _State, status: str, summary: str, messages: list[Message]) -> RunResult:
        summary = summary.strip() or "(no summary)"
        if not messages or messages[-1].role != "assistant":
            messages.append(Message.assistant(summary))  # leaves the history ready for a follow-up turn
        result = RunResult(
            run_id=state.run_id,
            goal=state.goal,
            status=status,
            summary=summary,
            steps=state.steps,
            tool_calls=state.tool_calls,
            denials=state.denials,
            elapsed=self.clock() - state.started,
            input_tokens=state.input_tokens,
            output_tokens=state.output_tokens,
            messages=messages,
        )
        try:
            self.memory.record_run(result.run_id, state.wall_start, state.source, state.goal, result.to_dict())
        except Exception as exc:  # noqa: BLE001 - history is a convenience, never a reason to fail a run
            self._emit("warning", state, message=f"could not record the run in memory: {exc}")
        self._emit("run_end", state, **{k: v for k, v in result.to_dict().items() if k not in {"run_id", "goal"}})
        return result

    def _emit(self, kind: str, state: _State, *, audit: bool = True, **data: Any) -> None:
        if audit and kind in AUDITED:
            self.audit.append(kind, state.run_id, **data)
        if self.reporter is not None:
            self.reporter(kind, {"run_id": state.run_id, **data})

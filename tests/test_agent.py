"""The agent loop: budgets, approvals, denials, kill switch, error handling and transcript hygiene."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from helpers import Recorder, ScriptedProvider, call, reply
from ultron.agent import TRIM_MARK, RunResult, compact_messages
from ultron.errors import ProviderError, SandboxViolation, UltronError
from ultron.messages import GOAL_HEADER, LLMResponse, Message, ToolCall, ToolResult
from ultron.runtime import Runtime
from ultron.tools.base import Risk, Tool

MakeRt = Callable[..., Runtime]


def finish(summary: str = "done", status: str = "success", cid: str | None = None) -> LLMResponse:
    return reply(call("finish", cid, summary=summary, status=status))


def results_of(result: RunResult) -> list[ToolResult]:
    return [res for msg in result.messages if msg.role == "tool" for res in msg.tool_results]


def extra_tool(rt: Runtime, name: str, func: Callable[[Any, dict], str], risk: Risk = Risk.SAFE) -> None:
    rt.registry.register(Tool(name, "test tool", {"type": "object", "properties": {}}, func, risk=risk))


# ------------------------------------------------------------------ how a run ends
def test_run_ends_when_the_model_calls_finish(make_rt: MakeRt) -> None:
    rt = make_rt(provider=ScriptedProvider(finish("All done.")))
    result = rt.agent.run("do the thing")
    assert result.ok and result.status == "success"
    assert result.summary == "All done."
    assert (result.steps, result.tool_calls, result.denials) == (1, 1, 0)
    assert result.messages[-1].role == "assistant" and result.messages[-1].text == "All done."


@pytest.mark.parametrize("status", ["failed", "blocked"])
def test_the_status_given_to_finish_is_reported(make_rt: MakeRt, status: str) -> None:
    rt = make_rt(provider=ScriptedProvider(finish("could not", status)))
    result = rt.agent.run("goal")
    assert result.status == status and not result.ok


def test_a_plain_text_reply_counts_as_the_answer(make_rt: MakeRt) -> None:
    result = make_rt(provider=ScriptedProvider(reply(text="The answer is 4."))).agent.run("2+2?")
    assert result.status == "success" and result.summary == "The answer is 4."


def test_an_empty_reply_is_a_failure(make_rt: MakeRt) -> None:
    result = make_rt(provider=ScriptedProvider(reply(text="  "))).agent.run("goal")
    assert result.status == "failed" and "empty reply" in result.summary


def test_a_refusal_is_reported_as_blocked(make_rt: MakeRt) -> None:
    result = make_rt(provider=ScriptedProvider(reply(text="I won't.", stop="refusal"))).agent.run("goal")
    assert result.status == "blocked" and result.summary == "I won't."


def test_a_reply_cut_off_by_the_token_limit_fails_with_a_hint(make_rt: MakeRt) -> None:
    result = make_rt(provider=ScriptedProvider(reply(text="partial", stop="max_tokens"))).agent.run("goal")
    assert result.status == "failed"
    assert "cut off" in result.summary and "partial" in result.summary


def test_an_empty_goal_is_rejected(rt: Runtime) -> None:
    with pytest.raises(UltronError, match="empty"):
        rt.agent.run("   ")


# ------------------------------------------------------------------ the conversation
def test_tool_results_are_fed_back_to_the_model(make_rt: MakeRt) -> None:
    provider = ScriptedProvider(reply(call("calc", "c1", expression="6*7")), finish("42"))
    result = make_rt(provider=provider).agent.run("compute")
    assert result.status == "success"
    last = provider.calls[1][-1]
    assert last.role == "tool"
    assert last.tool_results[0].call_id == "c1" and "42" in last.tool_results[0].content


def test_the_first_message_carries_the_goal_and_recalled_notes(make_rt: MakeRt) -> None:
    provider = ScriptedProvider(finish())
    rt = make_rt(provider=provider)
    rt.memory.add_note("my favorite language is Python", source="user")
    rt.agent.run("what is my favorite language?")
    text = provider.calls[0][0].text
    assert GOAL_HEADER in text and "what is my favorite language?" in text
    assert "favorite language is Python" in text and "untrusted" in text


def test_the_system_prompt_never_changes_between_turns(make_rt: MakeRt) -> None:
    provider = ScriptedProvider(
        reply(call("calc", "a", expression="1+1")), reply(call("calc", "b", expression="2+2")), finish()
    )
    make_rt(provider=provider).agent.run("goal")
    assert len(set(provider.systems)) == 1
    assert "Current time" not in provider.systems[0]  # volatile context belongs in the user message


def test_all_tools_are_offered_on_every_turn(make_rt: MakeRt) -> None:
    provider = ScriptedProvider(finish())
    rt = make_rt(provider=provider)
    rt.agent.run("goal")
    assert [t.name for t in provider.tools[0]] == rt.registry.names()


def test_a_conversation_can_continue_from_earlier_messages(make_rt: MakeRt) -> None:
    provider = ScriptedProvider(finish("first"), finish("second"))
    rt = make_rt(provider=provider)
    first = rt.agent.run("first goal")
    second = rt.agent.run("second goal", history=first.messages)
    assert second.summary == "second"
    sent = provider.calls[1]
    assert "first goal" in sent[0].text and "second goal" in sent[-1].text
    assert len(sent) == len(first.messages) + 1


def test_each_run_is_recorded_in_memory_and_the_audit_log(make_rt: MakeRt) -> None:
    rt = make_rt(provider=ScriptedProvider(reply(call("calc", "c", expression="1+1")), finish()))
    result = rt.agent.run("goal")
    row = rt.memory.recent_runs(1)[0]
    assert row["run_id"] == result.run_id and row["status"] == "success" and row["steps"] == 2
    events = [r["event"] for r in rt.audit.records() if r["run_id"] == result.run_id]
    assert events[0] == "run_start" and events[-1] == "run_end"
    assert {"llm", "decision", "tool_end"} <= set(events)
    assert rt.audit.verify().ok


def test_the_reporter_sees_events_live(make_rt: MakeRt) -> None:
    seen: list[str] = []
    rt = make_rt(provider=ScriptedProvider(reply(call("calc", "c", expression="1+1")), finish()))
    rt.agent.reporter = lambda kind, data: seen.append(kind)
    rt.agent.run("goal")
    assert seen[0] == "run_start" and seen[-1] == "run_end"
    assert {"step", "llm", "tool_start", "decision", "tool_end"} <= set(seen)


def test_token_usage_is_summed(make_rt: MakeRt) -> None:
    a = reply(call("calc", "c", expression="1+1"))
    a.input_tokens, a.output_tokens = 10, 3
    b = finish()
    b.input_tokens, b.output_tokens = 20, 4
    result = make_rt(provider=ScriptedProvider(a, b)).agent.run("goal")
    assert (result.input_tokens, result.output_tokens) == (30, 7)


def test_finish_makes_later_calls_in_the_same_turn_skip(make_rt: MakeRt) -> None:
    provider = ScriptedProvider(reply(call("finish", "f", summary="ok"), call("calc", "c", expression="1+1")))
    result = make_rt(provider=provider).agent.run("goal")
    assert result.status == "success"
    assert "skipped" in results_of(result)[1].content


# ------------------------------------------------------------------ approvals and denials
def test_a_write_waits_for_approval_and_nothing_happens_when_declined(make_rt: MakeRt, workspace: Path) -> None:
    rec = Recorder(False)
    write = reply(call("write_file", "w", path="out.txt", content="hi"))
    provider = ScriptedProvider(write, finish("gave up", "blocked"))
    result = make_rt(provider=provider, approver=rec).agent.run("write it")
    assert not (workspace / "out.txt").exists()
    assert [r.tool for r in rec.requests] == ["write_file"]
    assert "out.txt" in rec.requests[0].reason and rec.requests[0].args["content"] == "hi"
    res = results_of(result)[0]
    assert res.is_error and "did not approve" in res.content
    assert result.denials == 1 and result.status == "blocked"


def test_an_approved_write_happens(make_rt: MakeRt, workspace: Path) -> None:
    rec = Recorder(True)
    provider = ScriptedProvider(reply(call("write_file", "w", path="out.txt", content="hi")), finish("written"))
    result = make_rt(provider=provider, approver=rec).agent.run("write it")
    assert (workspace / "out.txt").read_text() == "hi"
    assert result.status == "success" and result.denials == 0
    approvals = [r for r in rec.requests]
    assert len(approvals) == 1


def test_policy_denials_are_explained_counted_and_audited(make_rt: MakeRt) -> None:
    provider = ScriptedProvider(reply(call("read_file", "r", path="../../etc/passwd")), finish("blocked", "blocked"))
    rt = make_rt(provider=provider)
    result = rt.agent.run("read it")
    res = results_of(result)[0]
    assert res.is_error and "DENIED by policy" in res.content and "workaround" in res.content
    assert result.denials == 1
    decisions = [r["data"] for r in rt.audit.records() if r["event"] == "decision"]
    assert decisions[0]["verdict"] == "deny"


def test_the_run_stops_after_too_many_denials(make_rt: MakeRt) -> None:
    provider = ScriptedProvider(
        reply(call("read_file", "a", path="../a"), call("read_file", "b", path="../b")),
    )
    result = make_rt(provider=provider, max_denials=2).agent.run("goal")
    assert result.status == "blocked" and "2 denied actions" in result.summary


def test_a_tool_that_raises_sandbox_violation_is_a_denial(make_rt: MakeRt) -> None:
    rt = make_rt(provider=ScriptedProvider(reply(call("sneaky", "s")), finish("x", "blocked")))

    def sneaky(ctx: Any, args: dict) -> str:
        raise SandboxViolation("outside the workspace")

    extra_tool(rt, "sneaky", sneaky)
    result = rt.agent.run("goal")
    assert "DENIED by sandbox" in results_of(result)[0].content and result.denials == 1


def test_approval_deny_mode_never_reaches_the_approver(make_rt: MakeRt) -> None:
    rec = Recorder(True)
    provider = ScriptedProvider(reply(call("write_file", "w", path="x.txt", content="x")), finish("no", "blocked"))
    result = make_rt(provider=provider, approver=rec, approval="deny").agent.run("goal")
    assert rec.requests == [] and result.denials == 1


# ------------------------------------------------------------------ malformed calls and failing tools
def test_unknown_tools_get_an_error_listing_the_real_ones(make_rt: MakeRt) -> None:
    result = make_rt(provider=ScriptedProvider(reply(call("nope", "u")), finish())).agent.run("goal")
    res = results_of(result)[0]
    assert res.is_error and "Unknown tool 'nope'" in res.content and "read_file" in res.content
    assert result.denials == 0


def test_missing_and_mistyped_arguments_are_reported(make_rt: MakeRt) -> None:
    provider = ScriptedProvider(
        reply(call("read_file", "m"), call("list_dir", "t", path=["x"])), finish()
    )
    result = make_rt(provider=provider).agent.run("goal")
    first, second = results_of(result)[:2]
    assert "missing required argument 'path'" in first.content
    assert "expected a string" in second.content


def test_undecodable_arguments_are_reported(make_rt: MakeRt) -> None:
    bad = ToolCall("p", "calc", {}, parse_error="arguments were not valid JSON")
    result = make_rt(provider=ScriptedProvider(reply(bad), finish())).agent.run("goal")
    assert "not valid JSON" in results_of(result)[0].content


def test_expected_tool_errors_do_not_count_as_denials(make_rt: MakeRt) -> None:
    provider = ScriptedProvider(reply(call("read_file", "r", path="missing.txt")), finish("none", "failed"))
    result = make_rt(provider=provider).agent.run("goal")
    assert "no such file" in results_of(result)[0].content and result.denials == 0


def test_a_crashing_tool_does_not_crash_the_agent(make_rt: MakeRt) -> None:
    rt = make_rt(provider=ScriptedProvider(reply(call("boom", "b")), finish("ok")))

    def boom(ctx: Any, args: dict) -> str:
        raise RuntimeError("kaput")

    extra_tool(rt, "boom", boom)
    result = rt.agent.run("goal")
    assert "boom crashed: RuntimeError: kaput" in results_of(result)[0].content
    assert result.status == "success"
    assert any(r["event"] == "warning" for r in rt.audit.records())


def test_secrets_in_tool_output_never_reach_the_model(make_rt: MakeRt, workspace: Path) -> None:
    (workspace / "notes.txt").write_text("deploy token sk-ant-api03-abcdefghijklmnop done")
    provider = ScriptedProvider(reply(call("read_file", "r", path="notes.txt")), finish())
    result = make_rt(provider=provider).agent.run("goal")
    content = results_of(result)[0].content
    assert "abcdefghijklmnop" not in content and "[REDACTED]" in content


def test_long_tool_output_is_truncated(make_rt: MakeRt, workspace: Path) -> None:
    (workspace / "big.txt").write_text("x" * 500)
    provider = ScriptedProvider(reply(call("read_file", "r", path="big.txt")), finish())
    result = make_rt(provider=provider, max_tool_output_chars=100).agent.run("goal")
    assert "[output truncated: 500 characters in total]" in results_of(result)[0].content


def test_a_call_in_a_cut_off_reply_is_not_executed(make_rt: MakeRt, workspace: Path) -> None:
    cut = reply(call("write_file", "w", path="x.txt", content="half"), stop="max_tokens")
    rec = Recorder(True)
    result = make_rt(provider=ScriptedProvider(cut, finish()), approver=rec).agent.run("goal")
    assert not (workspace / "x.txt").exists() and rec.requests == []
    assert "cut off" in results_of(result)[0].content


# ------------------------------------------------------------------ budgets and loops
def test_the_step_budget_ends_a_run_that_never_finishes(make_rt: MakeRt) -> None:
    provider = ScriptedProvider(reply(call("list_dir", "a", path=".")), reply(call("list_dir", "b", path=".")))
    result = make_rt(provider=provider, max_steps=2).agent.run("goal")
    assert result.status == "budget" and "Step budget of 2" in result.summary and result.steps == 2


def test_the_tool_call_budget_stops_extra_calls(make_rt: MakeRt) -> None:
    calls = [call("calc", f"c{i}", expression=f"{i}+{i}") for i in (1, 2, 3)]
    result = make_rt(provider=ScriptedProvider(reply(*calls)), max_tool_calls=2).agent.run("goal")
    assert result.status == "budget" and "Tool-call budget" in result.summary
    first, second, third = results_of(result)
    assert not first.is_error and not second.is_error
    assert third.is_error and "budget" in third.content


def test_the_time_budget_is_enforced_with_an_injected_clock(make_rt: MakeRt) -> None:
    provider = ScriptedProvider(finish())
    rt = make_rt(provider=provider, max_seconds=300)
    ticks = iter(range(0, 100_000, 1000))
    rt.agent.clock = lambda: float(next(ticks))
    result = rt.agent.run("goal")
    assert result.status == "budget" and "Time budget of 300s" in result.summary
    assert provider.calls == []


def test_a_repeated_identical_call_is_warned_about_then_stopped(make_rt: MakeRt) -> None:
    responses = [reply(call("calc", f"c{i}", expression="1+1")) for i in range(1, 6)]
    result = make_rt(provider=ScriptedProvider(*responses), max_steps=20).agent.run("goal")
    assert result.status == "stuck" and result.steps == 5
    outputs = [r.content for r in results_of(result)]
    assert "exact call 3 times" in outputs[2]
    assert "exact call" not in outputs[1]


# ------------------------------------------------------------------ kill switch and interruptions
def test_an_engaged_kill_switch_stops_the_run_before_the_model_is_called(make_rt: MakeRt) -> None:
    provider = ScriptedProvider(finish())
    rt = make_rt(provider=provider)
    rt.killswitch.engage("maintenance")
    result = rt.agent.run("goal")
    assert result.status == "killed" and "maintenance" in result.summary
    assert provider.calls == []


def test_the_kill_switch_stops_a_run_between_approval_and_execution(make_rt: MakeRt, workspace: Path) -> None:
    holder: dict[str, Runtime] = {}

    def approver(req: Any) -> bool:
        holder["rt"].killswitch.engage("halt now")
        return True

    provider = ScriptedProvider(
        reply(call("write_file", "w", path="x.txt", content="x"), call("calc", "c", expression="1+1"))
    )
    rt = make_rt(provider=provider, approver=approver)
    holder["rt"] = rt
    result = rt.agent.run("goal")
    assert result.status == "killed" and not (workspace / "x.txt").exists()
    cancelled = results_of(result)
    assert len(cancelled) == 2 and all(r.is_error and "cancelled" in r.content for r in cancelled)


def test_the_transcript_stays_valid_for_a_follow_up_after_a_kill(make_rt: MakeRt) -> None:
    holder: dict[str, Runtime] = {}

    def approver(req: Any) -> bool:
        holder["rt"].killswitch.engage("stop")
        return True

    provider = ScriptedProvider(reply(call("write_file", "w", path="x.txt", content="x")), finish("later"))
    rt = make_rt(provider=provider, approver=approver)
    holder["rt"] = rt
    killed = rt.agent.run("goal")
    rt.killswitch.release()
    again = rt.agent.run("try again", history=killed.messages)
    assert again.status == "success"
    for index, msg in enumerate(killed.messages):  # every tool call is answered by the next message
        if msg.tool_calls:
            answered = {r.call_id for r in killed.messages[index + 1].tool_results}
            assert {c.id for c in msg.tool_calls} == answered


def test_provider_errors_end_the_run_cleanly(make_rt: MakeRt) -> None:
    def down() -> LLMResponse:
        raise ProviderError("rate limited")

    result = make_rt(provider=ScriptedProvider(down)).agent.run("goal")
    assert result.status == "error" and result.summary == "Model provider error: rate limited"


def test_unexpected_exceptions_end_the_run_cleanly(make_rt: MakeRt) -> None:
    def bug() -> LLMResponse:
        raise ValueError("oops")

    result = make_rt(provider=ScriptedProvider(bug)).agent.run("goal")
    assert result.status == "error" and "Internal error: ValueError: oops" in result.summary


def test_ctrl_c_ends_the_run_as_interrupted(make_rt: MakeRt) -> None:
    def interrupt() -> LLMResponse:
        raise KeyboardInterrupt

    result = make_rt(provider=ScriptedProvider(interrupt)).agent.run("goal")
    assert result.status == "interrupted"


# ------------------------------------------------------------------ context compaction
def _big_read_turns(workspace: Path, count: int) -> list[LLMResponse]:
    turns = []
    for i in range(count):
        (workspace / f"f{i}.txt").write_text("\n".join("x" * 100 for _ in range(30)))
        turns.append(reply(call("read_file", f"r{i}", path=f"f{i}.txt")))
    return turns


def test_old_bulky_tool_output_is_trimmed_to_fit_the_context_budget(make_rt: MakeRt, workspace: Path) -> None:
    provider = ScriptedProvider(*_big_read_turns(workspace, 3), finish())
    result = make_rt(provider=provider, max_context_chars=2000).agent.run("goal")
    assert results_of(result)[0].content.endswith(TRIM_MARK)
    assert not results_of(result)[-1].content.endswith(TRIM_MARK)  # recent output stays intact


def test_append_only_providers_are_never_compacted(make_rt: MakeRt, workspace: Path) -> None:
    class AppendOnly(ScriptedProvider):
        append_only = True

    provider = AppendOnly(*_big_read_turns(workspace, 3), finish())
    result = make_rt(provider=provider, max_context_chars=2000).agent.run("goal")
    assert not any(r.content.endswith(TRIM_MARK) for r in results_of(result))


def test_compact_messages_keeps_recent_turns_and_reports_the_count() -> None:
    def tool_msg(text: str) -> Message:
        return Message.tool([ToolResult("id", "read_file", text)])

    messages = [Message.user("goal"), tool_msg("a" * 1000), tool_msg("b" * 1000), tool_msg("c" * 1000)]
    assert compact_messages(messages, max_chars=10_000) == 0
    assert compact_messages(messages, max_chars=1500, keep_recent=1) == 2
    assert messages[1].tool_results[0].content.endswith(TRIM_MARK)
    assert messages[3].tool_results[0].content == "c" * 1000
    assert compact_messages(messages, max_chars=1500, keep_recent=1) == 0  # idempotent

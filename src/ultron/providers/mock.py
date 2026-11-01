"""The offline demo brain.

This is **not** a language model. It is a small scripted planner that recognises a handful
of goal shapes and drives the real tools step by step, so you can watch the agent loop,
policy engine, sandbox and audit log work without an API key (and so the test-suite can
exercise the whole stack deterministically).

Plans are written as generators: ``result = yield (tool_name, args)`` hands a tool call to
the agent and receives the tool's result; ``return (status, summary)`` finishes the run.
The provider is stateless - it replays the generator against the recorded history on every
turn, which keeps it trivially correct.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Generator
from typing import Any

from ultron.messages import GOAL_HEADER, LLMResponse, Message, ToolCall, ToolResult, ToolSpec

Action = tuple[str, dict[str, Any]]
Outcome = tuple[str, str]
Plan = Generator[Action, ToolResult, Outcome]
PlanFactory = Callable[[str, set[str]], Plan]

TEXT_EXTENSIONS = (".md", ".txt", ".py", ".json", ".csv", ".toml", ".yml", ".yaml", ".rst", ".ini", ".cfg")

FALLBACK = (
    "I'm ULTRON's offline demo brain: a small scripted planner, not a language model.\n"
    "Goals I understand:\n"
    '  - "What is (17 * 23) + 4?"\n'
    '  - "Summarize the files in the workspace and write the report to SUMMARY.md"\n'
    '  - "Remember that my favorite language is Python"  /  "What do you remember about language?"\n'
    '  - "Read notes.md"  /  "Search for TODO in the files"  /  "Run command ls"\n'
    "To use a real model: set ANTHROPIC_API_KEY, or run with --provider ollama."
)


# ---------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------
def _extract_expression(goal: str) -> str:
    text = goal.replace("×", "*").replace("^", "**")
    text = re.sub(r"(?<=\d)\s+x\s+(?=\d)", " * ", text)
    best = ""
    for run in re.findall(r"[\d.()\s+\-*/%]+", text):
        run = run.strip()
        if re.search(r"\d", run) and re.search(r"[-+*/%]", run) and len(run) > len(best):
            best = run
    return best


def _plan_calc(goal: str, tools: set[str]) -> Plan:
    expression = _extract_expression(goal)
    if not expression:
        return ("failed", "I could not find an arithmetic expression in that goal.")
    result = yield ("calc", {"expression": expression})
    if result.is_error:
        return ("failed", f"The calculation failed: {result.content}")
    return ("success", f"{expression} = {result.content}")


def _plan_remember(goal: str, tools: set[str]) -> Plan:
    match = re.match(r"^\s*(?:please\s+)?remember(?:\s+that|:)?\s+(.+)$", goal, re.I | re.S)
    text = (match.group(1) if match else goal).strip()
    result = yield ("remember", {"text": text, "tags": ["demo"]})
    if result.is_error:
        return ("failed", f"I could not save that: {result.content}")
    return ("success", f'Saved to long-term memory ({result.content}): "{text}"')


def _plan_recall(goal: str, tools: set[str]) -> Plan:
    match = re.search(
        r"(?:what do you (?:remember|know) about|do you remember|recall)\s+(.+?)\s*\??$", goal, re.I | re.S
    )
    query = (match.group(1) if match else goal).strip()
    result = yield ("recall", {"query": query})
    if result.is_error:
        return ("failed", f"Memory lookup failed: {result.content}")
    if result.content.startswith("no matching"):
        return ("success", f"I don't have any notes about '{query}'.")
    return ("success", f"Here is what I remember about '{query}':\n{result.content}")


def _plan_time(goal: str, tools: set[str]) -> Plan:
    if "get_time" not in tools:
        return ("blocked", "No clock tool is loaded. Enable the example skill: skills/get_time.py")
    result = yield ("get_time", {})
    if result.is_error:
        return ("failed", result.content)
    return ("success", result.content)


def _parse_listing(listing: str) -> list[str]:
    files = []
    for line in listing.splitlines():
        match = re.match(r"^f {2}(.+?) {2}\(", line)
        if match:
            files.append(match.group(1))
    return files


def _plan_list(goal: str, tools: set[str]) -> Plan:
    listing = yield ("list_dir", {"path": "."})
    if listing.is_error:
        return ("blocked", f"I could not list the workspace: {listing.content}")
    files = _parse_listing(listing.content)
    readable = [f for f in files if f.lower().endswith(TEXT_EXTENSIONS) and not f.startswith(".")][:3]
    notes: list[str] = []
    for name in readable:
        read = yield ("read_file", {"path": name, "max_lines": 60})
        if read.is_error:
            notes.append(f"- {name}: unreadable ({read.content})")
            continue
        lines = [ln for ln in read.content.splitlines() if ln.strip() and not ln.startswith("[")]
        title = lines[0].lstrip("# ").strip()[:80] if lines else "(empty)"
        notes.append(f"- {name}: {len(lines)} non-empty lines; starts with '{title}'")
    report = f"# Workspace summary\n\n{len(files)} file(s) found.\n\n" + ("\n".join(notes) or "(nothing readable)")
    if re.search(r"\b(write|save|report|summary\.md)\b", goal, re.I):
        write = yield ("write_file", {"path": "SUMMARY.md", "content": report + "\n", "mode": "overwrite"})
        if write.is_error:
            return ("blocked", f"I could not write SUMMARY.md: {write.content}")
        return ("success", f"Wrote SUMMARY.md.\n\n{report}")
    return ("success", report)


def _plan_read(goal: str, tools: set[str]) -> Plan:
    match = re.search(r"([\w./~\-]+)\s*$", goal.strip())
    path = match.group(1) if match else goal
    result = yield ("read_file", {"path": path})
    if result.is_error:
        return ("blocked", f"I could not read '{path}': {result.content}")
    preview = result.content[:600] + ("..." if len(result.content) > 600 else "")
    return ("success", f"Contents of {path}:\n{preview}")


def _plan_write(goal: str, tools: set[str]) -> Plan:
    match = re.match(
        r"^\s*(?:please\s+)?(?:create|write|make)\s+(?:a\s+)?(?:new\s+)?(?:file\s+)?(?:called\s+|named\s+)?"
        r"(?P<path>[\w./\-]+\.\w+)\s*(?:with|containing|that says|saying|:)\s*(?P<content>.+)$",
        goal,
        re.I | re.S,
    )
    if not match:
        return ("failed", "I could not work out the file name and content.")
    path, content = match.group("path"), match.group("content").strip()
    result = yield ("write_file", {"path": path, "content": content + "\n", "mode": "create"})
    if result.is_error:
        return ("blocked", f"I could not create '{path}': {result.content}")
    return ("success", f"Created {path}. ({result.content})")


def _plan_run(goal: str, tools: set[str]) -> Plan:
    match = re.match(r"^\s*(?:please\s+)?run\s+(?:the\s+)?(?:command\s+)?(.+?)\s*$", goal, re.I | re.S)
    command = (match.group(1) if match else goal).strip().strip("`'\"")
    result = yield ("run_command", {"command": command})
    if result.is_error:
        return ("blocked", f"I could not run `{command}`: {result.content}")
    return ("success", f"Ran `{command}`:\n{result.content[:600]}")


def _plan_fetch(goal: str, tools: set[str]) -> Plan:
    match = re.search(r"(https?://\S+)", goal)
    url = match.group(1).rstrip(".,)") if match else goal
    result = yield ("http_get", {"url": url})
    if result.is_error:
        return ("blocked", f"I could not fetch {url}: {result.content}")
    return ("success", f"Fetched {url}:\n{result.content[:600]}")


def _plan_search(goal: str, tools: set[str]) -> Plan:
    match = re.match(
        r"^\s*(?:please\s+)?(?:search|find|grep)\s+(?:for\s+)?(.+?)"
        r"(?:\s+in\s+(?:the\s+)?(?:files|workspace|folder|project))?\s*$",
        goal,
        re.I | re.S,
    )
    query = (match.group(1) if match else goal).strip().strip("`'\"")
    result = yield ("search_files", {"query": query})
    if result.is_error:
        return ("failed", f"Search failed: {result.content}")
    return ("success", f"Search results for '{query}':\n{result.content}")


def _plan_fallback(goal: str, tools: set[str]) -> Plan:
    yield from ()
    return ("blocked", FALLBACK)


_PLEASE = r"^\s*(?:please\s+)?"
_ROUTES: list[tuple[re.Pattern[str], PlanFactory]] = [
    (re.compile(_PLEASE + r"remember\b", re.I), _plan_remember),
    (re.compile(r"\b(?:what do you (?:remember|know) about|do you remember|recall)\b", re.I), _plan_recall),
    (re.compile(_PLEASE + r"run\s+\S", re.I), _plan_run),
    (re.compile(r"\b(?:fetch|download|get|open)\s+https?://", re.I), _plan_fetch),
    (
        re.compile(_PLEASE + r"(?:create|write|make)\s+.*\.\w+\s*(?:with|containing|that says|saying|:)", re.I | re.S),
        _plan_write,
    ),
    (
        re.compile(
            _PLEASE + r"(?:read|show|open|cat|print|display)\s+(?:me\s+)?(?:the\s+)?(?:file\s+)?"
            r"(?=[^\s]*[./])[\w./~\-]+\s*$",
            re.I,
        ),
        _plan_read,
    ),
    (re.compile(r"\d\s*(?:\*\*|[-+*/%^×])\s*[\d(]|\d\s+x\s+\d", re.I), _plan_calc),
    (
        re.compile(r"\b(?:what(?:'s| is) the (?:current )?(?:time|date)|what time is it|current time)\b", re.I),
        _plan_time,
    ),
    (
        re.compile(
            r"\b(?:list|show|what)\b.*\b(?:files?|workspace|folder|director(?:y|ies))\b|\bsummari[sz]e\b", re.I
        ),
        _plan_list,
    ),
    (re.compile(_PLEASE + r"(?:search|find|grep)\s+\S", re.I), _plan_search),
]


def _tidy(result: ToolResult) -> ToolResult:
    """Drop the retry advice the agent appends for models, so demo summaries read naturally."""
    if not result.is_error:
        return result
    text = re.sub(r"\s*Do not retry this action.*$", "", result.content, flags=re.S)
    return ToolResult(result.call_id, result.name, text, True)


def select_plan(goal: str) -> PlanFactory:
    for pattern, factory in _ROUTES:
        if pattern.search(goal):
            return factory
    return _plan_fallback


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------
class MockProvider:
    name = "mock"
    model = "offline-demo-brain"

    def complete(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> LLMResponse:
        goal, trail = self._split(messages)
        executed = [res for msg in trail if msg.role == "tool" for res in msg.tool_results]
        plan = select_plan(goal)(goal, {t.name for t in tools})
        try:
            action = next(plan)
            for result in executed:
                action = plan.send(_tidy(result))
        except StopIteration as stop:
            status, summary = stop.value
            call = ToolCall(f"mock_{len(executed) + 1}", "finish", {"status": status, "summary": summary})
        else:
            call = ToolCall(f"mock_{len(executed) + 1}", action[0], dict(action[1]))
        size = sum(m.size() for m in messages) // 4
        return LLMResponse(
            tool_calls=[call],
            stop_reason="tool_use",
            input_tokens=size,
            output_tokens=len(str(call.arguments)) // 4,
        )

    @staticmethod
    def _split(messages: list[Message]) -> tuple[str, list[Message]]:
        """Find the goal in the latest user message and the turns that followed it."""
        for index in range(len(messages) - 1, -1, -1):
            msg = messages[index]
            if msg.role == "user" and msg.text:
                text = msg.text
                marker = GOAL_HEADER + "\n"
                goal = text.split(marker, 1)[1] if marker in text else text
                return goal.strip(), messages[index + 1 :]
        return "", []

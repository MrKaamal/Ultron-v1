"""remember / recall: the agent's long-term memory tools, plus the finish tool."""

from __future__ import annotations

from typing import Any

from ultron.errors import ToolError
from ultron.memory import MAX_NOTE_CHARS, Note
from ultron.tools.base import Risk, Tool, ToolContext
from ultron.util import utc_iso

FINISH_STATUSES = ("success", "failed", "blocked")


def format_note(note: Note) -> str:
    tags = f" (tags: {', '.join(note.tags)})" if note.tags else ""
    return f"#{note.id} [{note.source}, {utc_iso(note.created_at)[:10]}] {note.text}{tags}"


def _remember(ctx: ToolContext, args: dict[str, Any]) -> str:
    try:
        tags = [str(t) for t in args.get("tags") or []]
        note_id = ctx.memory.add_note(args["text"], tags, source="agent")
    except ValueError as exc:  # MemoryFull is already a ToolError and keeps its own type
        raise ToolError(str(exc)) from None
    return f"saved note #{note_id}"


def _recall(ctx: ToolContext, args: dict[str, Any]) -> str:
    notes = ctx.memory.search(args["query"], limit=args.get("limit", 5))
    if not notes:
        return "no matching notes"
    return "\n".join(format_note(n) for n in notes)


def _finish(ctx: ToolContext, args: dict[str, Any]) -> str:
    # The agent loop interprets this call; the function only acknowledges it.
    return "finished"


TOOLS: list[Tool] = [
    Tool(
        name="remember",
        description=(
            "Save a short note to long-term memory so that future runs can recall it: facts, preferences, "
            f"decisions, results worth keeping. Max {MAX_NOTE_CHARS} characters. Never store secrets."
        ),
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The note."},
                "tags": {"type": "array", "items": {"type": "string"}, "description": "Optional keywords."},
            },
            "required": ["text"],
        },
        func=_remember,
        risk=Risk.SAFE,
    ),
    Tool(
        name="recall",
        description="Search long-term memory for notes relevant to a query. Returns the best matches first.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look for."},
                "limit": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["query"],
        },
        func=_recall,
        risk=Risk.SAFE,
    ),
    Tool(
        name="finish",
        description=(
            "Call this when the task is complete or cannot be completed. Give a clear, concrete summary of "
            "the outcome. status: 'success' (goal achieved), 'failed' (tried but could not), or 'blocked' "
            "(needed something that was denied or unavailable)."
        ),
        parameters={
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "What was done and the result."},
                "status": {"type": "string", "enum": list(FINISH_STATUSES)},
            },
            "required": ["summary"],
        },
        func=_finish,
        risk=Risk.SAFE,
    ),
]

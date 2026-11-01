"""Example skill: give the agent a clock.

Skills are plain Python files that define tools with the ``@tool`` decorator. They are code the
*operator* vouches for: they run with your privileges when ULTRON starts, so only put files here
that you wrote or have read. The skills folder sits outside the agent's workspace, so the agent
itself can never add or change a skill. Every loaded file's SHA-256 goes into the audit log.
"""

from __future__ import annotations

from datetime import UTC, datetime

from ultron.tools.base import tool


@tool(description="Return the current date and time, in UTC and in the machine's local zone.", risk="safe")
def get_time(ctx, args):  # type: ignore[no-untyped-def]
    utc = datetime.now(UTC)
    local = datetime.now().astimezone()
    return f"UTC: {utc:%Y-%m-%d %H:%M:%S}\nLocal: {local:%Y-%m-%d %H:%M:%S} ({local:%Z}, UTC{local:%z})"

"""Example skill: count lines, words and characters of a workspace file.

Copy this file into your skills folder (default: ./skills, which must be OUTSIDE the workspace).
Note the pattern: the `check` hook resolves the path through the sandbox, so an escape such as
`../../etc/passwd` is denied by the policy engine before the function ever runs.
"""

from __future__ import annotations

from ultron.tools.base import tool
from ultron.tools.fs import path_check


@tool(
    description="Count the lines, words and characters of a text file in the workspace.",
    parameters={
        "type": "object",
        "properties": {"path": {"type": "string", "description": "File path relative to the workspace."}},
        "required": ["path"],
    },
    risk="safe",
    check=path_check("path"),
)
def word_count(ctx, args):  # type: ignore[no-untyped-def]
    path = ctx.sandbox.resolve(args["path"])
    text = path.read_text(encoding="utf-8", errors="replace")
    return f"{len(text.splitlines())} lines, {len(text.split())} words, {len(text)} characters"

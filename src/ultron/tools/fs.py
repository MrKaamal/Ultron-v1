"""Workspace file tools: list_dir, read_file, write_file, search_files."""

from __future__ import annotations

import os
import time
from fnmatch import fnmatch
from typing import Any

from ultron.errors import SandboxViolation, ToolError
from ultron.tools.base import CheckFunc, Decision, Risk, Tool, ToolContext
from ultron.util import clip, human_size

MAX_LIST_ENTRIES = 200
MAX_SEARCH_FILES = 5000
SEARCH_SECONDS = 5.0
MAX_LINE_CHARS = 2000


def path_check(arg: str = "path", default: str | None = None) -> CheckFunc:
    """Policy hook: deny (rather than ask about) paths that cannot be resolved safely."""

    def _check(ctx: ToolContext, args: dict[str, Any]) -> Decision | None:
        value = args.get(arg, default)
        if value is not None:
            ctx.sandbox.resolve(str(value))  # raises SandboxViolation
        return None

    return _check


# -- list_dir ------------------------------------------------------------------------
def _list_dir(ctx: ToolContext, args: dict[str, Any]) -> str:
    path = ctx.sandbox.resolve(args.get("path", "."))
    if not path.exists():
        raise ToolError(f"no such directory: {args.get('path', '.')}")
    if not path.is_dir():
        raise ToolError(f"not a directory: {args.get('path')} (use read_file)")
    entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    lines: list[str] = []
    hidden = 0
    for entry in entries:
        if ctx.sandbox.is_sensitive(entry):
            hidden += 1
            continue
        if len(lines) >= MAX_LIST_ENTRIES:
            lines.append(f"... and {len(entries) - MAX_LIST_ENTRIES - hidden} more entries")
            break
        if entry.is_dir():
            lines.append(f"d  {entry.name}/")
        else:
            try:
                size = human_size(entry.stat().st_size)
            except OSError:
                size = "?"
            lines.append(f"f  {entry.name}  ({size})")
    header = f"{ctx.sandbox.relative(path)} - {len(lines)} entries"
    if hidden:
        header += f" ({hidden} protected entries hidden)"
    return header + ("\n" + "\n".join(lines) if lines else "\n(empty)")


# -- read_file -----------------------------------------------------------------------
def _read_file(ctx: ToolContext, args: dict[str, Any]) -> str:
    path = ctx.sandbox.resolve(args["path"])
    if not path.exists():
        raise ToolError(f"no such file: {args['path']}")
    if path.is_dir():
        raise ToolError(f"{args['path']} is a directory; use list_dir")
    start = max(1, args.get("start_line", 1))
    max_lines = min(max(1, args.get("max_lines", 400)), 2000)
    limit = ctx.cfg.max_read_bytes
    with path.open("rb") as fh:
        data = fh.read(limit + 1)
    byte_truncated = len(data) > limit
    data = data[:limit]
    if b"\x00" in data[:8192]:
        raise ToolError(f"{args['path']} looks like a binary file; refusing to print it")
    lines = data.decode("utf-8", errors="replace").splitlines()
    if not lines:
        return "(empty file)"
    window = lines[start - 1 : start - 1 + max_lines]
    if not window:
        raise ToolError(f"start_line {start} is past the end of the file ({len(lines)} lines)")
    out = "\n".join(window)
    notes: list[str] = []
    if start - 1 + max_lines < len(lines):
        notes.append(f"[more lines available: call read_file again with start_line={start + max_lines}]")
    if byte_truncated:
        notes.append(f"[file truncated at {human_size(limit)}]")
    return out + ("\n" + "\n".join(notes) if notes else "")


# -- write_file ----------------------------------------------------------------------
def _check_write(ctx: ToolContext, args: dict[str, Any]) -> Decision | None:
    path = ctx.sandbox.resolve(str(args.get("path", "")))  # SandboxViolation -> DENY
    mode = args.get("mode", "create")
    size = len(str(args.get("content", "")).encode("utf-8"))
    verb = {"create": "create", "overwrite": "overwrite", "append": "append to"}.get(mode, mode)
    return Decision.ask(f"{verb} {ctx.sandbox.relative(path)} ({human_size(size)})")


def _write_file(ctx: ToolContext, args: dict[str, Any]) -> str:
    path = ctx.sandbox.resolve(args["path"])
    content: str = args["content"]
    mode: str = args.get("mode", "create")
    data = content.encode("utf-8")
    if len(data) > ctx.cfg.max_write_bytes:
        raise ToolError(f"content is {human_size(len(data))}; the limit is {human_size(ctx.cfg.max_write_bytes)}")
    if path.is_dir():
        raise ToolError(f"{args['path']} is a directory")
    exists = path.exists()
    if mode == "create" and exists:
        raise ToolError(f"{args['path']} already exists; use mode='overwrite' or mode='append'")
    if mode == "append" and exists and path.stat().st_size + len(data) > ctx.cfg.max_write_bytes:
        raise ToolError("appending would exceed the maximum file size")
    path.parent.mkdir(parents=True, exist_ok=True)
    if mode == "append":
        with path.open("ab") as fh:
            fh.write(data)
    else:
        tmp = path.with_name(f".{path.name}.ultron-tmp")
        try:
            tmp.write_bytes(data)
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)
    return f"wrote {human_size(len(data))} to {ctx.sandbox.relative(path)} (mode={mode})"


# -- search_files --------------------------------------------------------------------
def _search_files(ctx: ToolContext, args: dict[str, Any]) -> str:
    query: str = args["query"].strip()
    if not query:
        raise ToolError("query is empty")
    root = ctx.sandbox.resolve(args.get("path", "."))
    pattern = args.get("glob", "*")
    limit = min(max(1, args.get("max_results", 50)), 200)
    needle = query.lower()
    deadline = time.monotonic() + SEARCH_SECONDS

    if root.is_file():
        candidates = [root]
    elif root.is_dir():
        candidates = (p for p in sorted(root.rglob("*")) if p.is_file())
    else:
        raise ToolError(f"no such path: {args.get('path', '.')}")

    hits: list[str] = []
    scanned = 0
    for candidate in candidates:
        if scanned >= MAX_SEARCH_FILES or time.monotonic() > deadline or len(hits) >= limit:
            break
        try:
            real = ctx.sandbox.resolve(str(candidate))
        except SandboxViolation:
            continue  # symlink out of the workspace or a protected file: silently skip
        if not fnmatch(real.name, pattern):
            continue
        try:
            if real.stat().st_size > ctx.cfg.max_read_bytes:
                continue
            with real.open("rb") as fh:
                data = fh.read(ctx.cfg.max_read_bytes)
        except OSError:
            continue
        if b"\x00" in data[:4096]:
            continue
        scanned += 1
        rel = ctx.sandbox.relative(real)
        for lineno, line in enumerate(data.decode("utf-8", errors="replace").splitlines(), start=1):
            if needle in line[:MAX_LINE_CHARS].lower():
                hits.append(f"{rel}:{lineno}: {clip(line.strip(), 200)}")
                if len(hits) >= limit:
                    break
    if not hits:
        return f"no matches for {query!r} in {scanned} files"
    suffix = f"\n[stopped at {limit} results]" if len(hits) >= limit else ""
    return "\n".join(hits) + suffix


TOOLS: list[Tool] = [
    Tool(
        name="list_dir",
        description=(
            "List the files and folders inside a workspace directory. Use this first to discover what "
            "exists. Returns one entry per line: 'd  name/' for folders and 'f  name  (size)' for files."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Folder to list, relative to the workspace root."}
            },
            "required": [],
        },
        func=_list_dir,
        risk=Risk.SAFE,
        check=path_check("path", "."),
    ),
    Tool(
        name="read_file",
        description=(
            "Read a text file from the workspace and return its content (up to max_lines lines starting at "
            "start_line). For long files call again with a larger start_line. Fails on binary files and on "
            "protected files such as .env or private keys."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path relative to the workspace root."},
                "start_line": {"type": "integer", "description": "1-based first line to return.", "minimum": 1},
                "max_lines": {
                    "type": "integer",
                    "description": "Maximum number of lines to return (default 400, max 2000).",
                    "minimum": 1,
                    "maximum": 2000,
                },
            },
            "required": ["path"],
        },
        func=_read_file,
        risk=Risk.SAFE,
        check=path_check("path"),
    ),
    Tool(
        name="write_file",
        description=(
            "Create or modify a text file inside the workspace. mode='create' (default) fails if the file "
            "already exists, 'overwrite' replaces it, 'append' adds to the end. Missing parent folders are "
            "created. This changes the user's files, so it may require human approval."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path relative to the workspace root."},
                "content": {"type": "string", "description": "Full text to write."},
                "mode": {"type": "string", "enum": ["create", "overwrite", "append"]},
            },
            "required": ["path", "content"],
        },
        func=_write_file,
        risk=Risk.CONFIRM,
        check=_check_write,
    ),
    Tool(
        name="search_files",
        description=(
            "Search the workspace's text files for a literal, case-insensitive string. Returns matching "
            "lines as 'path:line: text'. Use glob (for example '*.py') to restrict which files are searched."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Text to look for (not a regex)."},
                "path": {"type": "string", "description": "Folder or file to search (default: whole workspace)."},
                "glob": {"type": "string", "description": "File-name pattern such as '*.md'. Default '*'."},
                "max_results": {"type": "integer", "minimum": 1, "maximum": 200},
            },
            "required": ["query"],
        },
        func=_search_files,
        risk=Risk.SAFE,
        check=path_check("path", "."),
    ),
]

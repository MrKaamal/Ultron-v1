"""Terminal output: colours, the live event reporter, the approval prompt and result summary."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Mapping
from typing import Any, TextIO

from ultron.agent import Approver, RunResult
from ultron.policy import ApprovalRequest
from ultron.util import clip

_CODES = {
    "bold": "1",
    "dim": "2",
    "red": "31",
    "green": "32",
    "yellow": "33",
    "blue": "34",
    "magenta": "35",
    "cyan": "36",
}
_UNICODE = {"arrow": "→", "ok": "✓", "bad": "✗", "ask": "?", "dot": "·", "warn": "!"}
_ASCII = {"arrow": "->", "ok": "+", "bad": "x", "ask": "?", "dot": ".", "warn": "!"}

#: Never offered as "always allow": each of these runs something new every time.
NEVER_ALWAYS = frozenset({"run_command", "http_get"})

STATUS_COLOR = {
    "success": "green",
    "failed": "red",
    "blocked": "yellow",
    "killed": "magenta",
    "budget": "yellow",
    "stuck": "yellow",
    "interrupted": "yellow",
    "error": "red",
}


class Printer:
    """Writes to ``stream`` with ANSI colour only when it is a colour-capable terminal."""

    def __init__(
        self,
        stream: TextIO | None = None,
        *,
        color: bool | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self.stream = stream if stream is not None else sys.stdout
        env = os.environ if env is None else env
        if color is None:
            is_tty = bool(getattr(self.stream, "isatty", lambda: False)())
            color = is_tty and "NO_COLOR" not in env and env.get("TERM") != "dumb"
        self.color = color
        encoding = (getattr(self.stream, "encoding", None) or "").lower()
        self.sym = _UNICODE if encoding.startswith("utf") else _ASCII

    def style(self, text: str, *names: str) -> str:
        if not self.color or not names:
            return text
        codes = ";".join(_CODES[n] for n in names)
        return f"\033[{codes}m{text}\033[0m"

    def line(self, text: str = "") -> None:
        print(text, file=self.stream, flush=True)

    def error(self, text: str) -> None:
        print(f"{self.style('error:', 'red', 'bold')} {text}", file=sys.stderr, flush=True)


def format_args(args: Mapping[str, Any], width: int = 70) -> str:
    """Compact one-line view of tool arguments (long text is summarised, not dumped)."""
    parts = []
    for key, value in args.items():
        text = " ".join(str(value).split())
        if key == "content" and len(text) > 40:
            text = f"<{len(str(value))} chars>"
        parts.append(f"{key}={clip(text, width)}")
    return " ".join(parts)


class LiveReporter:
    """Prints agent events as they happen. Plug into ``build_runtime(reporter=...)``."""

    def __init__(self, printer: Printer, *, verbose: bool = False) -> None:
        self.p = printer
        self.verbose = verbose

    def __call__(self, kind: str, data: dict[str, Any]) -> None:
        p, s = self.p, self.p.sym
        if kind == "run_start":
            p.line(p.style(f"run {data['run_id']} {s['dot']} {data['provider']}/{data['model']}", "dim"))
        elif kind == "step" and self.verbose:
            p.line(p.style(f"  step {data['n']}", "dim"))
        elif kind == "llm" and data.get("text"):
            p.line(p.style("  " + clip(" ".join(data["text"].split()), 160), "dim"))
        elif kind == "tool_start":
            p.line(f"  {p.style(s['arrow'], 'cyan')} {p.style(data['name'], 'bold')} "
                   f"{format_args(data['args'])}".rstrip())
        elif kind == "decision" and data["verdict"] == "deny":
            p.line(p.style(f"    {s['bad']} blocked: {data['reason']}", "red"))
        elif kind == "approval":
            word, color = ("approved", "green") if data["approved"] else ("declined", "red")
            p.line(p.style(f"    {s['ok'] if data['approved'] else s['bad']} {word}", color))
        elif kind == "tool_end" and not data.get("denied"):
            first = clip(data["preview"].strip().splitlines()[0] if data["preview"].strip() else "", 110)
            mark, color = (s["bad"], "red") if data["is_error"] else (s["ok"], "dim")
            p.line(p.style(f"    {mark} {first}", color))
        elif kind == "warning":
            p.line(p.style(f"  {s['warn']} {data['message']}", "yellow"))


def print_result(p: Printer, result: RunResult) -> None:
    color = STATUS_COLOR.get(result.status, "bold")
    mark = p.sym["ok"] if result.ok else p.sym["bad"]
    stats = (
        f"{result.steps} step(s) {p.sym['dot']} {result.tool_calls} tool call(s) {p.sym['dot']} "
        f"{result.denials} denied {p.sym['dot']} {result.elapsed:.1f}s"
    )
    p.line()
    p.line(f"{p.style(f'{mark} {result.status}', color, 'bold')}  {p.style(stats, 'dim')}")
    for text in result.summary.splitlines() or [""]:
        p.line(f"  {text}")


def make_approver(
    p: Printer,
    *,
    ask: Callable[[str], str] = input,
    interactive: bool | None = None,
) -> Approver:
    """A human-in-the-loop approver. Declines (never auto-approves) when there is no terminal."""
    always: set[str] = set()

    def approve(req: ApprovalRequest) -> bool:
        if req.tool in always:
            return True
        tty = sys.stdin.isatty() if interactive is None else interactive
        if not tty:
            p.line(p.style(f"    {p.sym['bad']} needs approval but there is no terminal to ask on; declined",
                           "yellow"))
            return False
        offer_always = req.tool not in NEVER_ALWAYS
        choices = "[y]es / [n]o" + (" / [a]lways this session" if offer_always else "")
        p.line(p.style(f"    {p.sym['ask']} approval needed for {req.tool}: {req.reason}", "yellow", "bold"))
        for key, value in req.args.items():
            p.line(p.style(f"        {key}: {clip(' '.join(str(value).split()), 200)}", "yellow"))
        while True:
            try:
                answer = ask(f"      allow? {choices} ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                p.line()
                return False
            if answer in ("y", "yes"):
                return True
            if answer in ("", "n", "no"):
                return False
            if offer_always and answer in ("a", "always"):
                always.add(req.tool)
                return True

    return approve


def deny_all(req: ApprovalRequest) -> bool:
    """Approver for unattended runs: anything that needs a human is declined."""
    return False

"""run_command: allowlisted commands, no shell, workspace-confined arguments."""

from __future__ import annotations

import glob
import os
import re
import shlex
import shutil
import signal
import subprocess
import tempfile
from pathlib import Path

from ultron.errors import SandboxViolation, ToolError
from ultron.sandbox import is_sensitive_name
from ultron.tools.base import Decision, Risk, Tool, ToolContext, Verdict

try:
    import resource
except ImportError:  # pragma: no cover  (Windows)
    resource = None  # type: ignore[assignment]

MAX_COMMAND_LEN = 1000
MAX_CAPTURE_BYTES = 2_000_000
_SHELL_TOKENS = {"|", "||", "&&", "&", ";", ">", ">>", "<", "<<", "2>", "2>&1"}

DENY_PATTERNS = [
    re.compile(p, re.I)
    for p in (
        r"\bsudo\b",
        r"\bsu\s",
        r"\brm\s+-[a-z]*[rf]",
        r"\bmkfs\b",
        r"\bdd\s+if=",
        r":\(\)\s*\{",
        r"\bchmod\s+-R\b",
        r"\bchown\b",
        r"(curl|wget)\b.*\|\s*(sh|bash|python)",
        r"\b(shutdown|reboot|halt)\b",
    )
]


class CommandRejected(ToolError):
    """The command violates policy (reported to the model as a denial)."""


def _check_arg_path(ctx: ToolContext, arg: str) -> None:
    cand = arg
    if arg.startswith("-"):
        if "=" not in arg:
            return
        cand = arg.split("=", 1)[1]
    if not cand:
        return
    # Protected names (.env, keys, .ssh, ...) are refused even if the file does not exist yet:
    # failing early is cheaper than reasoning about what a later write might create.
    for part in re.split(r"[\\/]", cand):
        if part and is_sensitive_name(part):
            raise SandboxViolation(f"access to sensitive file '{part}' is blocked")
    suspicious = (
        cand.startswith("~")
        or "/" in cand
        or "\\" in cand
        or cand == ".."
        or os.path.isabs(cand)
        or (ctx.sandbox.root / cand).exists()
    )
    if suspicious:
        if cand.startswith("~"):
            raise SandboxViolation("'~' paths are not allowed")
        ctx.sandbox.resolve(cand)


def _expand_globs(ctx: ToolContext, args: list[str]) -> list[str]:
    out: list[str] = []
    for arg in args:
        if not arg.startswith("-") and any(c in arg for c in "*?["):
            matches = sorted(glob.glob(arg, root_dir=str(ctx.sandbox.root), recursive=True))[:200]
            if matches:
                out.extend(matches)
                continue
        out.append(arg)
    return out


def parse_command(ctx: ToolContext, command: str) -> list[str]:
    """Validate ``command`` and return argv. Raises CommandRejected / SandboxViolation."""
    cfg = ctx.cfg
    if len(command) > MAX_COMMAND_LEN:
        raise CommandRejected("command too long")
    for pat in DENY_PATTERNS:
        if pat.search(command):
            raise CommandRejected("command matches a dangerous pattern and is blocked")
    try:
        argv = shlex.split(command)
    except ValueError as exc:
        raise CommandRejected(f"cannot parse command: {exc}") from None
    if not argv:
        raise CommandRejected("empty command")
    exe = argv[0]
    if "/" in exe or "\\" in exe:
        raise CommandRejected("use a bare command name, not a path")
    if exe not in cfg.allowed_commands:
        raise CommandRejected(
            f"command '{exe}' is not in allowed_commands ({', '.join(cfg.allowed_commands)})"
        )
    if any(a in _SHELL_TOKENS or "$(" in a or "`" in a for a in argv[1:]):
        raise CommandRejected("shell operators are not supported; run one simple command at a time")
    args = _expand_globs(ctx, argv[1:])
    for arg in args:
        _check_arg_path(ctx, arg)
    return [exe, *args]


def check_command(ctx: ToolContext, args: dict) -> Decision | None:
    try:
        argv = parse_command(ctx, str(args.get("command", "")))
    except (CommandRejected, SandboxViolation) as exc:
        return Decision(Verdict.DENY, str(exc))
    if argv[0] in ctx.cfg.safe_commands:
        return Decision(Verdict.ALLOW, f"'{argv[0]}' is a read-only safe command")
    return Decision(Verdict.ASK, f"'{argv[0]}' can execute arbitrary code")


def _child_env(ctx: ToolContext) -> dict[str, str]:
    env = {
        "PATH": os.environ.get("PATH", os.defpath),
        "HOME": str(ctx.sandbox.root),
        "LANG": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    if "SYSTEMROOT" in os.environ:  # needed on Windows
        env["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
    return env


def _limits(cpu_seconds: int):  # type: ignore[no-untyped-def]
    if resource is None:  # pragma: no cover
        return None

    def apply() -> None:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_CAPTURE_BYTES, MAX_CAPTURE_BYTES))

    return apply


def _kill_group(proc: subprocess.Popen) -> None:  # type: ignore[type-arg]
    try:
        if hasattr(os, "killpg"):
            os.killpg(proc.pid, signal.SIGKILL)
        else:  # pragma: no cover
            proc.kill()
    except (ProcessLookupError, PermissionError):
        pass


def run_command(ctx: ToolContext, args: dict) -> str:
    argv = parse_command(ctx, str(args["command"]))
    exe_path = shutil.which(argv[0])
    if exe_path is None:
        raise ToolError(f"command '{argv[0]}' is not installed on this machine")
    timeout = min(int(args.get("timeout_seconds") or ctx.cfg.command_timeout), ctx.cfg.command_timeout)
    cwd = Path(ctx.sandbox.root)
    kwargs: dict = {}
    if os.name == "posix":
        kwargs["start_new_session"] = True
        kwargs["preexec_fn"] = _limits(timeout + 5)
    with tempfile.TemporaryFile() as out_f, tempfile.TemporaryFile() as err_f:
        try:
            proc = subprocess.Popen(  # noqa: S603  (argv is validated; shell=False)
                [exe_path, *argv[1:]],
                cwd=cwd,
                env=_child_env(ctx),
                stdin=subprocess.DEVNULL,
                stdout=out_f,
                stderr=err_f,
                **kwargs,
            )
        except OSError as exc:
            raise ToolError(f"could not start '{argv[0]}': {exc.strerror or exc}") from None
        timed_out = False
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
        _kill_group(proc)  # also reaps stray children of a finished command
        proc.wait()
        cap = ctx.cfg.max_tool_output_chars * 2
        out_f.seek(0)
        err_f.seek(0)
        stdout = out_f.read(cap).decode("utf-8", "replace")
        stderr = err_f.read(cap).decode("utf-8", "replace")
    if timed_out:
        raise ToolError(f"command timed out after {timeout}s and was killed")
    parts = [f"exit_code: {proc.returncode}"]
    if proc.returncode in (-signal.SIGXFSZ, -signal.SIGKILL) and os.name == "posix":
        parts.append("(process was killed: output/CPU limit exceeded)")
    if stdout:
        parts.append(f"stdout:\n{stdout.rstrip()}")
    if stderr:
        parts.append(f"stderr:\n{stderr.rstrip()}")
    return "\n".join(parts)


TOOLS: list[Tool] = [
    Tool(
        name="run_command",
        description=(
            "Run ONE allowlisted command inside the workspace and return its exit code and output. "
            "There is no shell: no pipes, redirects, '&&', ';' or substitutions. Read-only commands "
            "(ls, cat, head, tail, wc, grep, echo, pwd, diff) run immediately; anything else needs "
            "approval. A rejection message lists the commands the operator has allowed."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "For example 'grep -n TODO notes.md'."},
                "timeout_seconds": {"type": "integer", "minimum": 1, "description": "Lower than the cap."},
            },
            "required": ["command"],
        },
        func=run_command,
        risk=Risk.CONFIRM,
        check=check_command,
    ),
]

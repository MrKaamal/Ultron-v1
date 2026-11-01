"""Command-line interface: ``ultron <command>`` (also ``python -m ultron``).

Exit codes: 0 success, 1 task failed, 2 usage or configuration error, 3 blocked by policy,
4 stopped by the kill switch, 5 other (budget, stuck, interrupted, provider error).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ultron import __version__
from ultron.audit import AuditLog
from ultron.config import APPROVALS, DEFAULT_TOML, PROVIDERS, Config, load_config
from ultron.errors import ConfigError, UltronError
from ultron.killswitch import KillSwitch
from ultron.memory import Memory
from ultron.providers import MockProvider, make_provider, resolve_provider_name
from ultron.runtime import Runtime, build_runtime
from ultron.sentinel import Sentinel, TaskOutcome
from ultron.tools.base import Risk
from ultron.tools.notes import format_note
from ultron.ui import LiveReporter, Printer, deny_all, make_approver, print_result
from ultron.util import clip, utc_iso

EXIT_OK, EXIT_FAILED, EXIT_USAGE, EXIT_BLOCKED, EXIT_KILLED, EXIT_OTHER = 0, 1, 2, 3, 4, 5
_STATUS_EXIT = {"success": EXIT_OK, "failed": EXIT_FAILED, "blocked": EXIT_BLOCKED, "killed": EXIT_KILLED}

#: Only credentials are read from a .env file. Anything that could redirect a key (base URLs) or
#: loosen safety (ULTRON_APPROVAL, ...) must come from the shell or an ultron.toml you chose.
_DOTENV_KEYS = frozenset({"ANTHROPIC_API_KEY", "OPENAI_API_KEY", "ULTRON_API_KEY"})

WELCOME = """\
# Welcome to your ULTRON workspace

This folder is the ONLY place the agent can read or write.

TODO: try  ultron run "Summarize the files in the workspace and write the report to SUMMARY.md"
"""


class UsageError(UltronError):
    """The command line is wrong (exit code 2)."""


# ---------------------------------------------------------------------- helpers
def exit_code(status: str) -> int:
    return _STATUS_EXIT.get(status, EXIT_OTHER)


def load_env_file(path: Path, env: dict[str, str] | None = None) -> list[str]:
    """Import API keys from a KEY=VALUE file without overriding the real environment."""
    target = os.environ if env is None else env
    loaded: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return loaded
    for raw in lines:
        line = raw.strip().removeprefix("export ").strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if key in _DOTENV_KEYS and value and key not in target:
            target[key] = value
            loaded.append(key)
    return loaded


def _config(args: argparse.Namespace) -> Config:
    overrides = {
        "provider": getattr(args, "provider", None),
        "model": getattr(args, "model", None),
        "workspace": getattr(args, "workspace", None),
        "state_dir": getattr(args, "state_dir", None),
        "approval": getattr(args, "approval", None),
        "max_steps": getattr(args, "max_steps", None),
    }
    return load_config(getattr(args, "config", None), overrides=overrides)


def _config_source(args: argparse.Namespace) -> str:
    explicit = getattr(args, "config", None) or os.environ.get("ULTRON_CONFIG")
    if explicit:
        return str(explicit)
    local = Path.cwd() / "ultron.toml"
    return str(local) if local.is_file() else "defaults (no ultron.toml)"


def _printer(args: argparse.Namespace, *, to_stderr: bool = False) -> Printer:
    stream = sys.stderr if to_stderr else sys.stdout
    return Printer(stream, color=False if getattr(args, "no_color", False) else None)


def _light_runtime(cfg: Config) -> Runtime:
    """A runtime that never needs credentials: for listing tools, status and diagnostics."""
    return build_runtime(cfg, approver=deny_all, provider=MockProvider(), env={})


def banner(p: Printer, rt: Runtime, source: str) -> None:
    cfg = rt.cfg
    p.line(
        f"{p.style('ULTRON', 'bold', 'cyan')} {__version__}  "
        f"{p.style(f'{rt.provider.name}/{rt.provider.model}  approval={cfg.approval}', 'dim')}"
    )
    p.line(p.style(f"  workspace: {cfg.workspace}", "dim"))
    p.line(p.style(f"  config:    {source}", "dim"))
    if rt.skills.loaded:
        p.line(p.style(f"  skills:    {', '.join(rt.skills.tool_names)}", "dim"))
    for error in rt.skills.errors:
        p.line(p.style(f"  {p.sym['warn']} skill not loaded: {error}", "yellow"))
    if rt.provider.name == "mock":
        p.line(p.style("  (offline demo brain, not a language model: set ANTHROPIC_API_KEY or use "
                       "--provider ollama for a real one)", "dim"))
    if cfg.approval == "auto":
        p.line(p.style(f"  {p.sym['warn']} approval=auto: risky tools run WITHOUT asking. "
                       "Use a container.", "yellow"))


def _table(rows: Sequence[Sequence[str]], widths: Sequence[int]) -> list[str]:
    out = []
    for row in rows:
        cells = [clip(str(cell), w).ljust(w) for cell, w in zip(row[:-1], widths, strict=False)]
        out.append("  ".join([*cells, row[-1]]).rstrip())
    return out


# ---------------------------------------------------------------------- commands
def cmd_run(args: argparse.Namespace) -> int:
    goal = " ".join(args.goal).strip()
    if goal == "-":
        goal = sys.stdin.read().strip()
    if not goal:
        raise UsageError('no goal given. Example: ultron run "Summarize the files in the workspace"')
    cfg = _config(args)
    p = _printer(args, to_stderr=args.json)
    live = None if (args.quiet or args.json) else LiveReporter(p, verbose=args.verbose)
    rt = build_runtime(cfg, approver=make_approver(p), reporter=live)
    if not (args.quiet or args.json):
        banner(p, rt, _config_source(args))
        p.line()
    result = rt.agent.run(goal)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print_result(p, result)
    return exit_code(result.status)


def cmd_chat(args: argparse.Namespace) -> int:
    cfg = _config(args)
    p = _printer(args)
    rt = build_runtime(cfg, approver=make_approver(p), reporter=LiveReporter(p, verbose=args.verbose))
    banner(p, rt, _config_source(args))
    p.line(p.style("  type a goal and press enter; /help lists commands, /quit leaves", "dim"))
    history = None
    while True:
        try:
            line = input(f"\n{p.style('ultron>', 'cyan', 'bold')} ").strip()
        except (EOFError, KeyboardInterrupt):
            p.line()
            break
        if not line:
            continue
        if line.startswith("/"):
            command, _, rest = line[1:].partition(" ")
            if command in ("quit", "exit", "q"):
                break
            if command == "help":
                p.line("  /tools  /memory [query]  /clear  /kill  /status  /quit")
            elif command == "tools":
                _print_tools(p, rt)
            elif command == "memory":
                _print_notes(p, rt.memory.search(rest) if rest.strip() else rt.memory.recent(10))
            elif command == "clear":
                history = None
                p.line("  conversation cleared")
            elif command == "kill":
                rt.killswitch.engage("engaged from chat")
                rt.audit.append("kill_engaged", None, reason="engaged from chat")
                p.line(p.style("  kill switch engaged; run `ultron revive` to resume", "magenta"))
            elif command == "status":
                _print_status(p, rt.cfg, rt.memory, rt.audit, rt.killswitch)
            else:
                p.line(f"  unknown command /{command}; try /help")
            continue
        result = rt.agent.run(line, history=history, source="chat")
        history = result.messages
        print_result(p, result)
        if result.status == "killed":
            p.line(p.style("  the kill switch is engaged; run `ultron revive` in another shell", "magenta"))
    return EXIT_OK


def _print_tools(p: Printer, rt: Runtime) -> None:
    meaning = {Risk.SAFE: "runs", Risk.CONFIRM: "asks", Risk.FORBIDDEN: "never"}
    rows = [("TOOL", "POLICY", "SOURCE", "DESCRIPTION")]
    for tool in rt.registry:
        description = " ".join(tool.description.split())
        rows.append((tool.name, meaning[tool.risk], tool.source, clip(description.split(". ")[0], 70)))
    for line in _table(rows, (14, 7, 18)):
        p.line(line)


def cmd_tools(args: argparse.Namespace) -> int:
    p = _printer(args)
    _print_tools(p, _light_runtime(_config(args)))
    p.line(p.style("\n'runs' = allowed without asking, 'asks' = needs approval; the policy engine can still "
                   "deny a call (bad path, blocked command, domain not allowlisted).", "dim"))
    return EXIT_OK


def _print_notes(p: Printer, notes: Sequence[Any]) -> None:
    if not notes:
        p.line("  (no notes)")
    for note in notes:
        p.line("  " + format_note(note))


def cmd_memory(args: argparse.Namespace) -> int:
    cfg = _config(args)
    p = _printer(args)
    memory = Memory(cfg.memory_path, cfg.max_notes)
    if args.action == "list":
        _print_notes(p, memory.recent(args.limit))
    elif args.action == "search":
        _print_notes(p, memory.search(" ".join(args.text), limit=args.limit))
    elif args.action == "add":
        text = " ".join(args.text).strip()
        if not text:
            raise UsageError("nothing to add: ultron memory add TEXT")
        p.line(f"saved note #{memory.add_note(text, source='user')}")
    elif args.action == "forget":
        try:
            note_id = int(" ".join(args.text))
        except ValueError:
            raise UsageError("ultron memory forget ID  (ID is the number shown by `memory list`)") from None
        if not memory.forget(note_id):
            p.line(f"no note #{note_id}")
            return EXIT_FAILED
        p.line(f"forgot note #{note_id}")
    return EXIT_OK


def cmd_history(args: argparse.Namespace) -> int:
    cfg = _config(args)
    p = _printer(args)
    runs = Memory(cfg.memory_path, cfg.max_notes).recent_runs(args.limit)
    if not runs:
        p.line("no runs yet")
        return EXIT_OK
    rows = [("WHEN (UTC)", "STATUS", "STEPS", "GOAL")]
    for run in runs:
        rows.append((utc_iso(run["started_at"]), run["status"], str(run["steps"]),
                     clip(" ".join(run["goal"].split()), 70)))
    for line in _table(rows, (20, 9, 5)):
        p.line(line)
    return EXIT_OK


def cmd_audit(args: argparse.Namespace) -> int:
    cfg = _config(args)
    p = _printer(args)
    log = AuditLog(cfg.audit_path)
    if args.verify:
        result = log.verify()
        if result.ok:
            p.line(f"{p.sym['ok']} audit chain intact: {result.count} records, head {result.head[:16]}")
            return EXIT_OK
        p.line(p.style(f"{p.sym['bad']} audit log TAMPERED or corrupt at line {result.line}: {result.error}",
                       "red", "bold"))
        return EXIT_FAILED
    records = log.tail(args.limit)
    if not records:
        p.line("audit log is empty")
    for rec in records:
        data = " ".join(f"{k}={clip(' '.join(str(v).split()), 60)}" for k, v in rec["data"].items())
        p.line(f"#{rec['seq']:<5} {rec['ts']}  {rec['run_id'] or '-':<20} {rec['event']:<12} {data}")
    return EXIT_OK


def _print_status(p: Printer, cfg: Config, memory: Memory, audit: AuditLog, kill: KillSwitch) -> None:
    info = kill.info()
    if info:
        p.line(p.style(f"kill switch: ENGAGED ({info.get('reason')}); `ultron revive` to resume",
                       "magenta", "bold"))
    else:
        p.line(f"kill switch: off  ({kill.path})")
    p.line(f"provider:    {resolve_provider_name(cfg, os.environ)}   approval: {cfg.approval}")
    p.line(f"workspace:   {cfg.workspace}")
    p.line(f"state:       {cfg.state_dir}")
    p.line(f"memory:      {memory.count()} notes, {len(memory.recent_runs(1000))} runs")
    chain = audit.verify()
    p.line(f"audit log:   {chain.count} records, " + ("chain intact" if chain.ok else f"BROKEN: {chain.error}"))


def cmd_status(args: argparse.Namespace) -> int:
    cfg = _config(args)
    _print_status(_printer(args), cfg, Memory(cfg.memory_path, cfg.max_notes),
                  AuditLog(cfg.audit_path), KillSwitch(cfg.kill_file))
    return EXIT_OK


def cmd_kill(args: argparse.Namespace) -> int:
    cfg = _config(args)
    reason = " ".join(args.reason).strip() or "manual"
    KillSwitch(cfg.kill_file).engage(reason)
    AuditLog(cfg.audit_path).append("kill_engaged", None, reason=reason)
    _printer(args).line(f"kill switch ENGAGED ({reason}). Running agents stop at their next step.\n"
                        "Resume with: ultron revive")
    return EXIT_OK


def cmd_revive(args: argparse.Namespace) -> int:
    cfg = _config(args)
    p = _printer(args)
    if KillSwitch(cfg.kill_file).release():
        AuditLog(cfg.audit_path).append("kill_released", None)
        p.line("kill switch released; agents may run again")
    else:
        p.line("kill switch was not engaged")
    return EXIT_OK


def cmd_sentinel(args: argparse.Namespace) -> int:
    cfg = _config(args)
    p = _printer(args)
    rt = build_runtime(cfg, approver=deny_all, reporter=LiveReporter(p))
    banner(p, rt, _config_source(args))
    sentinel = Sentinel(rt)
    p.line(f"  inbox:     {cfg.inbox_dir}\n  outbox:    {cfg.outbox_dir}")
    p.line(p.style("  unattended: actions that need approval are declined", "dim"))

    def show(outcome: TaskOutcome) -> None:
        tail = f" -> {outcome.result_path}" if outcome.result_path else f" ({outcome.summary})"
        p.line(p.style(f"  task {outcome.name}: {outcome.status}{tail}", "bold"))

    if args.once:
        outcomes = sentinel.run_once()
        for outcome in outcomes:
            show(outcome)
        if not outcomes:
            p.line("  inbox is empty")
        return EXIT_OK
    p.line(p.style(f"  watching every {cfg.poll_interval:g}s; Ctrl-C or `ultron kill` stops it", "dim"))
    try:
        reason = sentinel.watch(on_outcome=show)
    except KeyboardInterrupt:
        reason = "interrupted"
    p.line(f"sentinel stopped ({reason})")
    return EXIT_KILLED if reason == "killed" else EXIT_OK


def cmd_doctor(args: argparse.Namespace) -> int:
    p = _printer(args)
    failures = 0

    def report(level: str, label: str, detail: str = "") -> None:
        nonlocal failures
        failures += level == "FAIL"
        color = {"ok": "green", "warn": "yellow", "FAIL": "red", "info": "dim"}[level]
        p.line(f"  {p.style(f'{level:<4}', color, 'bold')}  {label:<12} {detail}".rstrip())

    p.line(p.style(f"ULTRON {__version__} doctor", "bold"))
    report("ok", "python", sys.version.split()[0])  # 3.11+ is enforced by the package metadata
    try:
        cfg = _config(args)
    except ConfigError as exc:
        report("FAIL", "config", str(exc))
        return EXIT_USAGE
    report("ok", "config", _config_source(args))
    name = resolve_provider_name(cfg, os.environ)
    try:
        provider = make_provider(cfg, os.environ)
        detail = f"{name}/{provider.model}"
        if name == "mock":
            report("warn", "provider", detail + " (offline demo brain; set ANTHROPIC_API_KEY for a real model)")
        else:
            report("ok", "provider", detail)
    except ConfigError as exc:
        report("FAIL", "provider", str(exc))
    report("ok" if cfg.workspace.is_dir() and os.access(cfg.workspace, os.W_OK) else "warn", "workspace",
           f"{cfg.workspace}" + ("" if cfg.workspace.is_dir() else " (will be created on first run)"))
    report("ok", "isolation", "state, inbox, outbox and skills are outside the workspace")
    kill = KillSwitch(cfg.kill_file)
    info = kill.info()
    report("warn" if info else "ok", "kill switch",
           f"ENGAGED ({info.get('reason')}); run `ultron revive`" if info else "off")
    chain = AuditLog(cfg.audit_path).verify()
    report("ok" if chain.ok else "FAIL", "audit log",
           f"{chain.count} records, chain intact" if chain.ok else f"{chain.error} at line {chain.line}")
    if cfg.approval == "auto":
        report("warn", "approval", "auto: risky tools run without asking; use a container")
    else:
        report("ok", "approval", cfg.approval)
    report("ok" if not cfg.web_allowed_domains else "info", "web",
           "http_get disabled (no allowed_domains)" if not cfg.web_allowed_domains
           else f"allowed: {', '.join(cfg.web_allowed_domains)}")
    if cfg.skills_dir is not None:
        rt = _light_runtime(cfg)
        report("ok" if not rt.skills.errors else "warn", "skills",
               ", ".join(rt.skills.tool_names) or "folder present, no skills loaded")
        for error in rt.skills.errors:
            report("warn", "skill", error)
    else:
        report("info", "skills", "no skills folder")
    p.line()
    p.line("all good" if not failures else f"{failures} problem(s) found")
    return EXIT_FAILED if failures else EXIT_OK


def cmd_init(args: argparse.Namespace) -> int:
    p = _printer(args)
    target = Path("ultron.toml")
    if target.exists() and not args.force:
        p.line("ultron.toml already exists (use --force to overwrite)")
    else:
        target.write_text(DEFAULT_TOML, encoding="utf-8")
        p.line("wrote ultron.toml")
    workspace = Path("workspace")
    workspace.mkdir(exist_ok=True)
    welcome = workspace / "WELCOME.md"
    if not welcome.exists():
        welcome.write_text(WELCOME, encoding="utf-8")
    Path("skills").mkdir(exist_ok=True)
    p.line("created workspace/ and skills/")
    p.line('\nNext:  ultron doctor   then   ultron run "Summarize the files in the workspace"')
    return EXIT_OK


def cmd_demo(args: argparse.Namespace) -> int:
    from ultron.demo import run_demo

    return run_demo(_printer(args), keep=args.keep)


# ---------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    group = common.add_argument_group("settings (override ultron.toml and ULTRON_* variables)")
    group.add_argument("--config", metavar="FILE", help="path to an ultron.toml")
    group.add_argument("--provider", choices=PROVIDERS, help="model backend (default: auto)")
    group.add_argument("--model", help="model name (provider default if omitted)")
    group.add_argument("--workspace", metavar="DIR", help="the only folder the agent may touch")
    group.add_argument("--state-dir", metavar="DIR", help="memory, audit log, kill switch, inbox")
    group.add_argument("--approval", choices=APPROVALS, help="ask a human (default) | deny | auto-approve")
    group.add_argument("--max-steps", type=int, metavar="N", help="step budget for one run")
    group.add_argument("--no-color", action="store_true", help="plain output")
    group.add_argument("--env-file", metavar="FILE", help="read API keys from FILE (default: ./.env)")

    parser = argparse.ArgumentParser(
        prog="ultron",
        description="ULTRON: a local-first autonomous agent with a sandbox, approvals and a kill switch.",
        epilog="Start with: ultron demo   (offline tour)   or   ultron doctor   (check your setup)",
    )
    parser.add_argument("--version", action="version", version=f"ultron {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    def add(name: str, func: Any, help_: str, *, settings: bool = True) -> argparse.ArgumentParser:
        sp = sub.add_parser(name, help=help_, description=help_, parents=[common] if settings else [])
        sp.set_defaults(func=func)
        return sp

    sp = add("run", cmd_run, "run one goal to completion")
    sp.add_argument("goal", nargs="*", help="what to do (use - to read it from stdin)")
    sp.add_argument("--json", action="store_true", help="print only the result, as JSON (no live events)")
    sp.add_argument("-q", "--quiet", action="store_true", help="only print the final result")
    sp.add_argument("-v", "--verbose", action="store_true", help="show every step")

    sp = add("chat", cmd_chat, "interactive session")
    sp.add_argument("-v", "--verbose", action="store_true", help="show every step")

    add("tools", cmd_tools, "list the tools and what the policy does with them")

    sp = add("memory", cmd_memory, "inspect or edit long-term memory")
    sp.add_argument("action", choices=["list", "search", "add", "forget"])
    sp.add_argument("text", nargs="*", help="query, note text, or note ID")
    sp.add_argument("-n", "--limit", type=int, default=20)

    sp = add("history", cmd_history, "recent runs")
    sp.add_argument("-n", "--limit", type=int, default=10)

    sp = add("audit", cmd_audit, "show or verify the tamper-evident audit log")
    sp.add_argument("--verify", action="store_true", help="check the hash chain (exit 1 if broken)")
    sp.add_argument("-n", "--limit", type=int, default=30, help="records to show")

    add("status", cmd_status, "kill switch, provider, memory and audit state")
    sp = add("kill", cmd_kill, "engage the kill switch: every agent stops at its next step")
    sp.add_argument("reason", nargs="*")
    add("revive", cmd_revive, "release the kill switch")

    sp = add("sentinel", cmd_sentinel, "watch the inbox folder and run each dropped goal")
    sp.add_argument("--once", action="store_true", help="process what is queued, then exit")

    add("doctor", cmd_doctor, "check configuration and safety setup")

    sp = add("init", cmd_init, "create ultron.toml, workspace/ and skills/ here", settings=False)
    sp.add_argument("--force", action="store_true", help="overwrite ultron.toml")
    sp.add_argument("--no-color", action="store_true", help=argparse.SUPPRESS)

    sp = add("demo", cmd_demo, "offline guided tour of the guardrails (no API key needed)", settings=False)
    sp.add_argument("--keep", action="store_true", help="keep the throw-away demo folder")
    sp.add_argument("--no-color", action="store_true", help="plain output")
    return parser


def main(argv: Sequence[str] | None = None, env: Mapping[str, str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return EXIT_OK
    if args.command not in ("demo", "init"):
        load_env_file(Path(getattr(args, "env_file", None) or ".env"))
    try:
        return int(args.func(args))
    except UsageError as exc:
        Printer(sys.stderr).error(str(exc))
        return EXIT_USAGE
    except ConfigError as exc:
        Printer(sys.stderr).error(f"configuration: {exc}")
        return EXIT_USAGE
    except UltronError as exc:
        Printer(sys.stderr).error(str(exc))
        return EXIT_OTHER
    except KeyboardInterrupt:
        print(file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

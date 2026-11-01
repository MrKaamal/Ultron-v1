"""``ultron demo``: a self-contained, offline tour of the guardrails.

Everything runs for real (agent loop, policy engine, sandbox, memory, sentinel, audit log) in a
throw-away folder; only the "brain" is the scripted offline planner instead of a language model.
The tour doubles as a smoke test: it returns 1 if any guardrail does not behave as expected.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from ultron import __version__
from ultron.audit import AuditLog
from ultron.config import load_config
from ultron.policy import ApprovalRequest
from ultron.runtime import build_runtime
from ultron.sentinel import Sentinel
from ultron.ui import LiveReporter, Printer, print_result

SAMPLE_FILES = {
    "notes.md": "# Project notes\nTODO: ship the first ULTRON prototype\nTODO: write the README\n",
    "ideas.txt": "1. inbox-driven automations\n2. weekly report generator\n",
    ".env": "API_KEY=do-not-leak-me-0123456789\n",
}


def run_demo(p: Printer, *, keep: bool = False) -> int:
    root = Path(tempfile.mkdtemp(prefix="ultron-demo-"))
    problems: list[str] = []
    try:
        workspace = root / "workspace"
        workspace.mkdir()
        for name, text in SAMPLE_FILES.items():
            (workspace / name).write_text(text, encoding="utf-8")
        cfg = load_config(
            env={},
            cwd=root,
            overrides={
                "workspace": workspace,
                "state_dir": root / "state",
                "provider": "mock",
                "approval": "ask",
            },
        )

        def approver(req: ApprovalRequest) -> bool:
            ok = req.tool == "write_file"
            p.line(p.style(f"    {p.sym['ask']} approval needed for {req.tool}: {req.reason}", "yellow"))
            why = "a harmless report: yes" if ok else "arbitrary code: no"
            p.line(p.style(f"      (demo operator answers for you: {why})", "dim"))
            return ok

        rt = build_runtime(cfg, approver=approver, reporter=LiveReporter(p), skills=False, env={})

        def section(title: str, note: str = "") -> None:
            p.line()
            p.line(p.style(f"== {title} ==", "bold", "cyan"))
            if note:
                p.line(p.style(note, "dim"))

        def go(goal: str, expect: str) -> None:
            p.line(p.style(f'\n$ ultron run "{goal}"', "bold"))
            result = rt.agent.run(goal)
            print_result(p, result)
            if result.status != expect:
                problems.append(f"{goal!r}: expected {expect}, got {result.status}")

        p.line(p.style(f"ULTRON {__version__} demo", "bold", "cyan")
               + p.style("  (offline brain, throw-away workspace, nothing leaves this machine)", "dim"))
        p.line(p.style(f"workspace: {workspace}", "dim"))

        section("1. Autonomy where it is safe",
                "Reading is free; writing a file waits for a yes (the demo operator says yes).")
        go("Summarize the files in the workspace and write the report to SUMMARY.md", "success")

        section("2. Memory that persists", "Notes live in SQLite and are recalled in later runs.")
        go("Remember that my favorite language is Python", "success")
        go("What do you remember about language?", "success")

        section("3. Guardrails say no",
                "Secrets, escapes, dangerous commands and unapproved code execution are all stopped.")
        go("Read .env", "blocked")
        go("Read ../../etc/passwd", "blocked")
        go("Run command rm -rf /", "blocked")
        go("Run command python3 --version", "blocked")

        section("4. Kill switch", "One file in the state folder halts every step and every tool call.")
        rt.killswitch.engage("demo")
        go("What is 2 + 2?", "killed")
        rt.killswitch.release()
        go("What is 2 + 2?", "success")

        section("5. Sentinel mode", "Drop a goal into the inbox; the result appears in the outbox.")
        (cfg.inbox_dir / "answer.task").write_text("What is 6 * 7?\n", encoding="utf-8")
        outcomes = Sentinel(rt).run_once()
        for outcome in outcomes:
            p.line(f"  inbox/{outcome.name} -> {outcome.status}: {outcome.summary}")
        if not outcomes or outcomes[0].status != "success":
            problems.append("sentinel did not complete the task")
        elif outcomes[0].result_path is not None:
            p.line(p.style(f"  result file: {outcomes[0].result_path}", "dim"))

        section("6. Tamper-evident audit log", "Every decision is hash-chained; edits are detectable.")
        verdict = rt.audit.verify()
        p.line(f"  chain intact: {verdict.ok} ({verdict.count} records)")
        if not verdict.ok:
            problems.append("audit chain should be intact")
        forged = root / "forged-audit.jsonl"
        text = cfg.audit_path.read_text(encoding="utf-8")
        needle = '"verdict":"deny"'
        if needle in text:
            forged.write_text(text.replace(needle, '"verdict":"allow"', 1), encoding="utf-8")
            check = AuditLog(forged).verify()
            p.line(f"  after secretly turning one 'deny' into 'allow': intact={check.ok} "
                   f"({check.error} at line {check.line})")
            if check.ok:
                problems.append("tampering was not detected")
        else:
            problems.append("no denial found in the audit log")

        p.line()
        if problems:
            p.line(p.style("demo found problems:", "red", "bold"))
            for item in problems:
                p.line(f"  - {item}")
            return 1
        p.line(p.style(f"{p.sym['ok']} every guardrail behaved as expected.", "green", "bold"))
        p.line("Next: export ANTHROPIC_API_KEY=... and run  ultron run \"your goal\"  "
               "(or use --provider ollama for a local model).")
        return 0
    finally:
        if keep:
            p.line(p.style(f"kept demo files in {root}", "dim"))
        else:
            shutil.rmtree(root, ignore_errors=True)

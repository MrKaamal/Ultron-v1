"""Wiring: turn a Config into a ready-to-run agent. Used by the CLI, the sentinel and the demo."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from ultron.agent import Agent, Approver, Reporter
from ultron.audit import AuditLog
from ultron.config import Config
from ultron.killswitch import KillSwitch
from ultron.memory import Memory
from ultron.policy import Policy
from ultron.providers import Provider, make_provider
from ultron.sandbox import Sandbox
from ultron.skills import SkillReport, load_skills
from ultron.tools import ToolRegistry, build_registry


@dataclass
class Runtime:
    cfg: Config
    sandbox: Sandbox
    memory: Memory
    audit: AuditLog
    killswitch: KillSwitch
    registry: ToolRegistry
    policy: Policy
    provider: Provider
    agent: Agent
    skills: SkillReport


def ensure_dirs(cfg: Config) -> None:
    """Create the workspace and the (separate) state folders."""
    for path in (cfg.workspace, cfg.state_dir, cfg.inbox_dir, cfg.outbox_dir):
        path.mkdir(parents=True, exist_ok=True)


def build_runtime(
    cfg: Config,
    *,
    approver: Approver,
    reporter: Reporter | None = None,
    provider: Provider | None = None,
    env: Mapping[str, str] | None = None,
    skills: bool = True,
) -> Runtime:
    """Assemble every component. ``provider`` overrides the one chosen by the config."""
    ensure_dirs(cfg)
    sandbox = Sandbox(cfg.workspace)
    memory = Memory(cfg.memory_path, cfg.max_notes)
    audit = AuditLog(cfg.audit_path)
    killswitch = KillSwitch(cfg.kill_file)
    registry = build_registry()

    report = SkillReport()
    if skills and cfg.skills_dir is not None:
        report = load_skills(registry, cfg.skills_dir, cfg.workspace)
        if report.loaded:
            audit.append(
                "skills_loaded",
                None,
                skills=[{"file": s.file, "sha256": s.sha256, "tools": list(s.tools)} for s in report.loaded],
            )
        if report.errors:
            audit.append("skills_failed", None, errors=report.errors)

    policy = Policy(cfg)
    chosen = provider or make_provider(cfg, env)
    agent = Agent(
        cfg=cfg,
        provider=chosen,
        registry=registry,
        policy=policy,
        sandbox=sandbox,
        memory=memory,
        audit=audit,
        killswitch=killswitch,
        approver=approver,
        reporter=reporter,
    )
    return Runtime(cfg, sandbox, memory, audit, killswitch, registry, policy, chosen, agent, report)

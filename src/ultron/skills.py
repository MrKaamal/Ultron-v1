"""Skills: operator-installed tool plugins.

A skill is a Python file in ``skills_dir`` that defines tools with the ``@tool`` decorator::

    from ultron.tools.base import tool

    @tool(description="Say hello", risk="safe")
    def hello(ctx, args):
        return "hello"

Skills are *code the operator vouches for* and run with the operator's privileges at import
time. Two rules keep the agent from turning that into self-modification:

* the skills folder must be outside the workspace, so the agent can never write a skill;
* a skill cannot replace a built-in tool (names are unique), and the SHA-256 of every file
  that gets loaded is written to the audit log.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ultron.errors import ConfigError
from ultron.tools import ToolRegistry
from ultron.tools.base import Tool


@dataclass(frozen=True)
class SkillInfo:
    file: str
    sha256: str
    tools: tuple[str, ...]


@dataclass
class SkillReport:
    loaded: list[SkillInfo] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def tool_names(self) -> list[str]:
        return [name for info in self.loaded for name in info.tools]


def _collect(module: Any) -> list[Tool]:
    """Tools defined by a module: decorated functions plus an optional TOOLS list."""
    found: list[Tool] = []
    for value in vars(module).values():
        candidate = getattr(value, "__ultron_tool__", None)
        if isinstance(candidate, Tool):
            found.append(candidate)
    for candidate in getattr(module, "TOOLS", []) or []:
        if isinstance(candidate, Tool):
            found.append(candidate)
    unique: dict[str, Tool] = {}
    for item in found:
        unique.setdefault(item.name, item)
    return list(unique.values())


def load_skills(registry: ToolRegistry, skills_dir: Path, workspace: Path) -> SkillReport:
    """Import every ``*.py`` in ``skills_dir`` (files starting with ``_`` are skipped)."""
    report = SkillReport()
    skills_dir = Path(skills_dir).resolve()
    workspace = Path(workspace).resolve()
    if skills_dir == workspace or workspace in skills_dir.parents:
        raise ConfigError(f"skills_dir ({skills_dir}) is inside the workspace; the agent could write its own skills")
    if not skills_dir.is_dir():
        return report

    for path in sorted(skills_dir.glob("*.py")):
        if path.name.startswith("_"):
            continue
        if path.is_symlink():
            report.errors.append(f"{path.name}: symbolic links are not loaded")
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        module_name = f"ultron_skill_{path.stem}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                raise ImportError("cannot create an import spec")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            registered: list[str] = []
            for tool in _collect(module):
                registry.register(dataclasses.replace(tool, source=f"skill:{path.name}"))
                registered.append(tool.name)
        except Exception as exc:  # noqa: BLE001 - a broken skill must not stop ULTRON from starting
            sys.modules.pop(module_name, None)
            report.errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
            continue
        report.loaded.append(SkillInfo(path.name, digest, tuple(registered)))
    return report

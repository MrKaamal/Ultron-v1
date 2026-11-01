"""Configuration: defaults < ultron.toml < environment < CLI overrides."""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from ultron.errors import ConfigError

PROVIDERS = ("auto", "mock", "anthropic", "openai", "openai-compat", "ollama")
APPROVALS = ("ask", "deny", "auto")

DEFAULT_SAFE_COMMANDS = ["ls", "cat", "head", "tail", "wc", "grep", "echo", "pwd", "diff"]
DEFAULT_ALLOWED_COMMANDS = [*DEFAULT_SAFE_COMMANDS, "python3", "python"]

DEFAULT_TOML = """\
# ULTRON configuration. Every key is optional; values shown are the defaults.
# Relative paths resolve against the directory containing this file.

provider = "auto"        # auto | mock | anthropic | openai | openai-compat | ollama
model = ""               # empty = provider default (required for openai / openai-compat)
base_url = ""            # empty = provider default
anthropic_version = ""   # empty = the API version this release targets (2026-10-21)
effort = ""              # optional Anthropic output-effort level; empty = API default
# temperature = 0.2      # optional; used by the openai / openai-compat / ollama backends only
workspace = "workspace"  # the ONLY directory the agent may read or write
state_dir = ".ultron"    # memory, audit log, kill switch, inbox (must be OUTSIDE workspace)
skills_dir = "skills"    # extra tools loaded at startup (must be OUTSIDE workspace)

[budget]
max_steps = 12
max_seconds = 300
max_tool_calls = 40
max_denials = 5              # stop the run after this many denied actions
max_tool_output_chars = 6000
max_context_chars = 48000
max_output_tokens = 4096
request_timeout = 120

[policy]
approval = "ask"             # ask = prompt a human | deny = never run risky tools | auto = run them
auto_approve = []            # tool names that never need approval, e.g. ["write_file"]
safe_commands = ["ls", "cat", "head", "tail", "wc", "grep", "echo", "pwd", "diff"]
allowed_commands = ["ls", "cat", "head", "tail", "wc", "grep", "echo", "pwd", "diff", "python3", "python"]
command_timeout = 30

[web]
allowed_domains = []         # empty = http_get is disabled
allow_private_networks = false
max_bytes = 200000

[files]
max_read_bytes = 200000
max_write_bytes = 1000000

[sentinel]
poll_interval = 2.0
max_runs_per_hour = 20

[memory]
max_notes = 5000
recall_k = 4
"""

# (section, key) -> Config field.  Top-level keys use section "".
_TOML_MAP: dict[tuple[str, str], str] = {
    ("", "provider"): "provider",
    ("", "model"): "model",
    ("", "base_url"): "base_url",
    ("", "anthropic_version"): "anthropic_version",
    ("", "effort"): "effort",
    ("", "temperature"): "temperature",
    ("", "workspace"): "workspace",
    ("", "state_dir"): "state_dir",
    ("", "skills_dir"): "skills_dir",
    ("", "inbox_dir"): "inbox_dir",
    ("", "outbox_dir"): "outbox_dir",
    ("budget", "max_steps"): "max_steps",
    ("budget", "max_seconds"): "max_seconds",
    ("budget", "max_tool_calls"): "max_tool_calls",
    ("budget", "max_denials"): "max_denials",
    ("budget", "max_tool_output_chars"): "max_tool_output_chars",
    ("budget", "max_context_chars"): "max_context_chars",
    ("budget", "max_output_tokens"): "max_output_tokens",
    ("budget", "request_timeout"): "request_timeout",
    ("policy", "approval"): "approval",
    ("policy", "auto_approve"): "auto_approve",
    ("policy", "safe_commands"): "safe_commands",
    ("policy", "allowed_commands"): "allowed_commands",
    ("policy", "command_timeout"): "command_timeout",
    ("web", "allowed_domains"): "web_allowed_domains",
    ("web", "allow_private_networks"): "web_allow_private",
    ("web", "max_bytes"): "web_max_bytes",
    ("files", "max_read_bytes"): "max_read_bytes",
    ("files", "max_write_bytes"): "max_write_bytes",
    ("sentinel", "poll_interval"): "poll_interval",
    ("sentinel", "max_runs_per_hour"): "max_runs_per_hour",
    ("memory", "max_notes"): "max_notes",
    ("memory", "recall_k"): "memory_recall_k",
}

_ENV_MAP: dict[str, str] = {
    "ULTRON_PROVIDER": "provider",
    "ULTRON_MODEL": "model",
    "ULTRON_BASE_URL": "base_url",
    "ULTRON_ANTHROPIC_VERSION": "anthropic_version",
    "ULTRON_EFFORT": "effort",
    "ULTRON_TEMPERATURE": "temperature",
    "ULTRON_WORKSPACE": "workspace",
    "ULTRON_STATE_DIR": "state_dir",
    "ULTRON_SKILLS_DIR": "skills_dir",
    "ULTRON_APPROVAL": "approval",
    "ULTRON_MAX_STEPS": "max_steps",
    "ULTRON_MAX_SECONDS": "max_seconds",
}

_PATH_FIELDS = ("workspace", "state_dir", "skills_dir", "inbox_dir", "outbox_dir")
_INT_FIELDS = (
    "max_steps",
    "max_seconds",
    "max_tool_calls",
    "max_denials",
    "max_tool_output_chars",
    "max_context_chars",
    "max_output_tokens",
    "command_timeout",
    "web_max_bytes",
    "max_read_bytes",
    "max_write_bytes",
    "max_runs_per_hour",
    "max_notes",
    "memory_recall_k",
)
_FLOAT_FIELDS = ("request_timeout", "poll_interval")


@dataclass
class Config:
    provider: str = "auto"
    model: str = ""
    base_url: str = ""
    anthropic_version: str = ""
    effort: str = ""
    temperature: float | None = None
    workspace: Path = Path("workspace")
    state_dir: Path = Path(".ultron")
    skills_dir: Path | None = None
    inbox_dir: Path = Path(".ultron/inbox")
    outbox_dir: Path = Path(".ultron/outbox")
    # budgets
    max_steps: int = 12
    max_seconds: int = 300
    max_tool_calls: int = 40
    max_denials: int = 5
    max_tool_output_chars: int = 6000
    max_context_chars: int = 48000
    max_output_tokens: int = 4096
    request_timeout: float = 120.0
    # policy
    approval: str = "ask"
    auto_approve: list[str] = field(default_factory=list)
    safe_commands: list[str] = field(default_factory=lambda: list(DEFAULT_SAFE_COMMANDS))
    allowed_commands: list[str] = field(default_factory=lambda: list(DEFAULT_ALLOWED_COMMANDS))
    command_timeout: int = 30
    # web
    web_allowed_domains: list[str] = field(default_factory=list)
    web_allow_private: bool = False
    web_max_bytes: int = 200_000
    # files
    max_read_bytes: int = 200_000
    max_write_bytes: int = 1_000_000
    # sentinel / memory
    poll_interval: float = 2.0
    max_runs_per_hour: int = 20
    max_notes: int = 5000
    memory_recall_k: int = 4

    @property
    def kill_file(self) -> Path:
        return self.state_dir / "KILL"

    @property
    def audit_path(self) -> Path:
        return self.state_dir / "audit.jsonl"

    @property
    def memory_path(self) -> Path:
        return self.state_dir / "memory.db"

    def ensure_dirs(self) -> None:
        for p in (self.workspace, self.state_dir):
            p.mkdir(parents=True, exist_ok=True)


def _is_within(child: Path, parent: Path) -> bool:
    return child == parent or child.is_relative_to(parent)


def _coerce(name: str, value: Any) -> Any:
    try:
        if name == "temperature":
            return None if value in ("", None) else float(value)
        if name in _INT_FIELDS:
            if isinstance(value, bool):
                raise ValueError("bool")
            return int(value)
        if name in _FLOAT_FIELDS:
            return float(value)
        if name == "web_allow_private":
            if isinstance(value, str):
                return value.strip().lower() in ("1", "true", "yes", "on")
            return bool(value)
        if name in ("auto_approve", "safe_commands", "allowed_commands", "web_allowed_domains"):
            if isinstance(value, str):
                value = [v.strip() for v in value.split(",") if v.strip()]
            if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                raise ValueError("expected list of strings")
            return list(value)
        if name in _PATH_FIELDS:
            return Path(str(value)) if str(value) else None
        return str(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"invalid value for '{name}': {value!r}") from exc


def validate(cfg: Config) -> None:
    """Raise ConfigError if the configuration is invalid or unsafe."""
    if cfg.provider not in PROVIDERS:
        raise ConfigError(f"provider must be one of {', '.join(PROVIDERS)} (got {cfg.provider!r})")
    if cfg.approval not in APPROVALS:
        raise ConfigError(f"approval must be one of {', '.join(APPROVALS)} (got {cfg.approval!r})")
    if cfg.temperature is not None and not 0.0 <= cfg.temperature <= 2.0:
        raise ConfigError(f"'temperature' must be between 0 and 2 (got {cfg.temperature})")
    for name in _INT_FIELDS:
        if getattr(cfg, name) < 1:
            raise ConfigError(f"'{name}' must be a positive integer")
    for name in _FLOAT_FIELDS:
        if getattr(cfg, name) <= 0:
            raise ConfigError(f"'{name}' must be positive")
    ws = cfg.workspace.resolve()
    # Everything the agent must never be able to modify has to live outside its workspace.
    guarded = {"state_dir": cfg.state_dir, "inbox_dir": cfg.inbox_dir, "outbox_dir": cfg.outbox_dir}
    if cfg.skills_dir is not None:
        guarded["skills_dir"] = cfg.skills_dir
    for name, path in guarded.items():
        if _is_within(path.resolve(), ws):
            raise ConfigError(
                f"'{name}' ({path}) must be outside the workspace ({ws}); "
                "otherwise the agent could tamper with its own guardrails"
            )
    unsafe = set(cfg.safe_commands) - set(cfg.allowed_commands)
    if unsafe:
        raise ConfigError(f"safe_commands must also be in allowed_commands: {sorted(unsafe)}")


def load_config(
    path: str | os.PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
    overrides: Mapping[str, Any] | None = None,
    cwd: str | os.PathLike[str] | None = None,
) -> Config:
    env = os.environ if env is None else env
    base = Path(cwd).resolve() if cwd else Path.cwd().resolve()

    cfg_path: Path | None = None
    explicit = path or env.get("ULTRON_CONFIG")
    if explicit:
        cfg_path = Path(explicit)
        if not cfg_path.is_file():
            raise ConfigError(f"config file not found: {cfg_path}")
    elif (base / "ultron.toml").is_file():
        cfg_path = base / "ultron.toml"

    values: dict[str, Any] = {}
    if cfg_path is not None:
        base = cfg_path.resolve().parent
        try:
            data = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
        except (tomllib.TOMLDecodeError, OSError) as exc:
            raise ConfigError(f"cannot read {cfg_path}: {exc}") from exc
        for key, val in data.items():
            if isinstance(val, dict):
                for sub, subval in val.items():
                    if (key, sub) not in _TOML_MAP:
                        raise ConfigError(f"unknown config key [{key}] {sub}")
                    values[_TOML_MAP[(key, sub)]] = subval
            elif ("", key) in _TOML_MAP:
                values[_TOML_MAP[("", key)]] = val
            else:
                raise ConfigError(f"unknown config key '{key}'")

    for var, name in _ENV_MAP.items():
        if env.get(var):
            values[name] = env[var]
    for name, val in (overrides or {}).items():
        if val is not None:
            values[name] = val

    known = {f.name for f in fields(Config)}
    unknown = set(values) - known
    if unknown:
        raise ConfigError(f"unknown settings: {sorted(unknown)}")

    cfg = Config()
    for name, val in values.items():
        setattr(cfg, name, _coerce(name, val))

    def resolve(p: Path) -> Path:
        return (p if p.is_absolute() else base / p).resolve()

    cfg.workspace = resolve(cfg.workspace)
    cfg.state_dir = resolve(cfg.state_dir)
    if "inbox_dir" not in values:
        cfg.inbox_dir = cfg.state_dir / "inbox"
    if "outbox_dir" not in values:
        cfg.outbox_dir = cfg.state_dir / "outbox"
    cfg.inbox_dir = resolve(cfg.inbox_dir)
    cfg.outbox_dir = resolve(cfg.outbox_dir)
    if "skills_dir" in values:
        cfg.skills_dir = resolve(cfg.skills_dir) if cfg.skills_dir else None
    else:
        default_skills = base / "skills"
        cfg.skills_dir = default_skills.resolve() if default_skills.is_dir() else None
    validate(cfg)
    return cfg

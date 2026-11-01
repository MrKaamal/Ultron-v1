"""Configuration: layering, validation and the rule that guardrails live outside the workspace."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest

from ultron.config import DEFAULT_TOML, Config, load_config
from ultron.errors import ConfigError

PATH_FIELDS = {"workspace", "state_dir", "skills_dir", "inbox_dir", "outbox_dir"}


def write_toml(folder: Path, text: str, name: str = "ultron.toml") -> Path:
    path = folder / name
    path.write_text(text, encoding="utf-8")
    return path


def test_defaults_resolve_against_the_working_directory(tmp_path: Path) -> None:
    cfg = load_config(env={}, cwd=tmp_path)
    assert cfg.workspace == tmp_path / "workspace" and cfg.state_dir == tmp_path / ".ultron"
    assert cfg.inbox_dir == cfg.state_dir / "inbox" and cfg.outbox_dir == cfg.state_dir / "outbox"
    assert cfg.skills_dir is None  # only enabled when a skills/ folder exists
    assert cfg.kill_file == cfg.state_dir / "KILL" and cfg.audit_path.name == "audit.jsonl"
    assert cfg.memory_path.name == "memory.db"
    assert (cfg.provider, cfg.approval, cfg.max_steps) == ("auto", "ask", 12)


def test_a_skills_folder_next_to_the_config_is_picked_up(tmp_path: Path) -> None:
    (tmp_path / "skills").mkdir()
    assert load_config(env={}, cwd=tmp_path).skills_dir == tmp_path / "skills"


def test_toml_env_and_overrides_layer_in_that_order(tmp_path: Path) -> None:
    write_toml(tmp_path, "[budget]\nmax_steps = 5\nmax_seconds = 50\n")
    assert load_config(env={}, cwd=tmp_path).max_steps == 5
    cfg = load_config(env={"ULTRON_MAX_STEPS": "7"}, cwd=tmp_path)
    assert (cfg.max_steps, cfg.max_seconds) == (7, 50)
    cfg = load_config(env={"ULTRON_MAX_STEPS": "7"}, cwd=tmp_path, overrides={"max_steps": 9, "model": None})
    assert cfg.max_steps == 9 and cfg.model == ""  # None means "not given"


def test_toml_sections_map_to_settings(tmp_path: Path) -> None:
    write_toml(tmp_path, '''
provider = "ollama"
model = "llama3.2:3b"
[policy]
approval = "deny"
auto_approve = ["write_file"]
safe_commands = ["ls"]
allowed_commands = ["ls", "wc"]
[web]
allowed_domains = ["example.com"]
allow_private_networks = true
[sentinel]
max_runs_per_hour = 3
''')
    cfg = load_config(env={}, cwd=tmp_path)
    assert (cfg.provider, cfg.model, cfg.approval) == ("ollama", "llama3.2:3b", "deny")
    assert cfg.auto_approve == ["write_file"] and cfg.allowed_commands == ["ls", "wc"]
    assert cfg.web_allowed_domains == ["example.com"] and cfg.web_allow_private is True
    assert cfg.max_runs_per_hour == 3


def test_relative_paths_in_a_config_file_resolve_against_that_file(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    path = write_toml(project, 'workspace = "work"\nstate_dir = "state"\n', "custom.toml")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    cfg = load_config(path, env={}, cwd=elsewhere)
    assert cfg.workspace == project / "work" and cfg.state_dir == project / "state"


def test_the_config_path_can_come_from_the_environment(tmp_path: Path) -> None:
    path = write_toml(tmp_path, "[budget]\nmax_steps = 3\n", "named.toml")
    assert load_config(env={"ULTRON_CONFIG": str(path)}, cwd=tmp_path).max_steps == 3
    with pytest.raises(ConfigError, match="not found"):
        load_config(env={"ULTRON_CONFIG": str(tmp_path / "missing.toml")}, cwd=tmp_path)


def test_list_settings_accept_comma_separated_strings(tmp_path: Path) -> None:
    cfg = load_config(env={}, cwd=tmp_path, overrides={"auto_approve": "write_file, calc", "web_allow_private": "yes"})
    assert cfg.auto_approve == ["write_file", "calc"] and cfg.web_allow_private is True


@pytest.mark.parametrize("toml,message", [
    ('unknown_key = 1', "unknown config key 'unknown_key'"),
    ('[budget]\nmax_stepz = 1', "unknown config key \\[budget\\] max_stepz"),
    ('[nope]\nx = 1', "unknown config key \\[nope\\] x"),
    ('provider = "gpt"', "provider must be one of"),
    ('[policy]\napproval = "maybe"', "approval must be one of"),
    ('[budget]\nmax_steps = 0', "'max_steps' must be a positive integer"),
    ('[budget]\nmax_steps = "many"', "invalid value for 'max_steps'"),
    ('[budget]\nmax_steps = true', "invalid value for 'max_steps'"),
    ('[budget]\nrequest_timeout = 0', "'request_timeout' must be positive"),
    ('temperature = 5', "between 0 and 2"),
    ('[policy]\nsafe_commands = ["ls", "cat"]\nallowed_commands = ["ls"]', "safe_commands must also be in"),
    ('[policy]\nauto_approve = [1]', "invalid value for 'auto_approve'"),
    ('this is not toml', "cannot read"),
])
def test_invalid_configuration_is_rejected_with_a_clear_message(tmp_path: Path, toml: str, message: str) -> None:
    write_toml(tmp_path, toml)
    with pytest.raises(ConfigError, match=message):
        load_config(env={}, cwd=tmp_path)


def test_unknown_overrides_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="unknown settings"):
        load_config(env={}, cwd=tmp_path, overrides={"wat": 1})


@pytest.mark.parametrize("overrides", [
    {"state_dir": "workspace/state"},
    {"state_dir": "workspace"},
    {"state_dir": ".."},  # the workspace itself would sit inside the state folder? no: state contains it
    {"inbox_dir": "workspace/in"},
    {"outbox_dir": "workspace"},
    {"skills_dir": "workspace/skills"},
])
def test_guardrail_folders_must_stay_outside_the_workspace(tmp_path: Path, overrides: dict) -> None:
    if overrides == {"state_dir": ".."}:  # a parent folder is fine: it does not sit inside the workspace
        assert load_config(env={}, cwd=tmp_path, overrides=overrides).state_dir == tmp_path.parent
        return
    with pytest.raises(ConfigError, match="must be outside the workspace"):
        load_config(env={}, cwd=tmp_path, overrides=overrides)


def test_the_shipped_default_config_is_valid_and_matches_the_defaults(tmp_path: Path) -> None:
    path = write_toml(tmp_path, DEFAULT_TOML)
    shipped = load_config(path, env={}, cwd=tmp_path)
    defaults = Config()
    for field in fields(Config):
        if field.name not in PATH_FIELDS:
            assert getattr(shipped, field.name) == getattr(defaults, field.name), field.name
    assert shipped.workspace == tmp_path / "workspace" and shipped.state_dir == tmp_path / ".ultron"


def test_ensure_dirs_creates_workspace_and_state(tmp_path: Path) -> None:
    cfg = load_config(env={}, cwd=tmp_path)
    cfg.ensure_dirs()
    assert cfg.workspace.is_dir() and cfg.state_dir.is_dir()

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from ultron.config import Config, load_config
from ultron.runtime import Runtime, build_runtime
from ultron.tools.base import ToolContext


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """No real credentials, settings or proxies may leak into a test."""
    for name in list(__import__("os").environ):
        if name.startswith(("ULTRON_", "ANTHROPIC_", "OPENAI_", "OLLAMA_")):
            monkeypatch.delenv(name, raising=False)
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NO_COLOR", "1")


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    return ws


@pytest.fixture
def make_cfg(tmp_path: Path, workspace: Path) -> Callable[..., Config]:
    def make(**overrides: Any) -> Config:
        base: dict[str, Any] = {"workspace": workspace, "state_dir": tmp_path / "state", "provider": "mock"}
        base.update(overrides)
        return load_config(env={}, cwd=tmp_path, overrides=base)

    return make


@pytest.fixture
def make_rt(make_cfg: Callable[..., Config]) -> Callable[..., Runtime]:
    def make(provider: Any = None, approver: Callable[..., bool] | None = None, **overrides: Any) -> Runtime:
        cfg = make_cfg(**overrides)
        return build_runtime(cfg, approver=approver or (lambda req: False), provider=provider, skills=False, env={})

    return make


@pytest.fixture
def rt(make_rt: Callable[..., Runtime]) -> Runtime:
    return make_rt()


@pytest.fixture
def ctx(rt: Runtime) -> ToolContext:
    return ToolContext(rt.cfg, rt.sandbox, rt.memory, "test-run")

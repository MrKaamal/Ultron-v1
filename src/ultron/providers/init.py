"""Provider factory: turn a Config (+ environment) into a ready-to-use backend."""

from __future__ import annotations

import os
from collections.abc import Mapping

from ultron.config import Config
from ultron.errors import ConfigError
from ultron.providers.anthropic import DEFAULT_BASE_URL as ANTHROPIC_URL
from ultron.providers.anthropic import DEFAULT_MODEL as ANTHROPIC_MODEL
from ultron.providers.anthropic import DEFAULT_VERSION as ANTHROPIC_VERSION
from ultron.providers.anthropic import AnthropicProvider
from ultron.providers.base import Provider
from ultron.providers.mock import MockProvider
from ultron.providers.openai_compat import OpenAICompatProvider

OPENAI_URL = "https://api.openai.com/v1"
OLLAMA_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "llama3.2:3b"

__all__ = [
    "AnthropicProvider",
    "MockProvider",
    "OpenAICompatProvider",
    "Provider",
    "make_provider",
    "resolve_provider_name",
]


def resolve_provider_name(cfg: Config, env: Mapping[str, str] | None = None) -> str:
    """Resolve ``provider = "auto"`` to a concrete backend."""
    env = os.environ if env is None else env
    if cfg.provider != "auto":
        return cfg.provider
    if env.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if env.get("OPENAI_API_KEY") and cfg.model:
        return "openai"
    return "mock"


def make_provider(cfg: Config, env: Mapping[str, str] | None = None) -> Provider:
    env = os.environ if env is None else env
    name = resolve_provider_name(cfg, env)

    if name == "mock":
        return MockProvider()

    if name == "anthropic":
        key = env.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise ConfigError("provider 'anthropic' needs the ANTHROPIC_API_KEY environment variable")
        return AnthropicProvider(
            key,
            cfg.model or ANTHROPIC_MODEL,
            base_url=cfg.base_url or env.get("ANTHROPIC_BASE_URL", "") or ANTHROPIC_URL,
            version=cfg.anthropic_version or ANTHROPIC_VERSION,
            max_tokens=cfg.max_output_tokens,
            effort=cfg.effort,
            workspace_id=env.get("ANTHROPIC_WORKSPACE_ID", ""),
            timeout=cfg.request_timeout,
        )

    if name == "openai":
        key = env.get("OPENAI_API_KEY", "")
        if not key:
            raise ConfigError("provider 'openai' needs the OPENAI_API_KEY environment variable")
        if not cfg.model:
            raise ConfigError("provider 'openai' needs a model: set ULTRON_MODEL or 'model' in ultron.toml")
        return OpenAICompatProvider(
            cfg.base_url or env.get("OPENAI_BASE_URL", "") or OPENAI_URL,
            cfg.model,
            api_key=key,
            name="openai",
            temperature=cfg.temperature,
            timeout=cfg.request_timeout,
        )

    if name == "openai-compat":
        base = cfg.base_url
        if not base:
            raise ConfigError("provider 'openai-compat' needs base_url (ULTRON_BASE_URL), e.g. http://localhost:1234/v1")
        if not cfg.model:
            raise ConfigError("provider 'openai-compat' needs a model: set ULTRON_MODEL or 'model' in ultron.toml")
        return OpenAICompatProvider(
            base,
            cfg.model,
            api_key=env.get("ULTRON_API_KEY", "") or env.get("OPENAI_API_KEY", ""),
            temperature=cfg.temperature,
            max_tokens=cfg.max_output_tokens,
            timeout=cfg.request_timeout,
        )

    if name == "ollama":
        return OpenAICompatProvider(
            cfg.base_url or OLLAMA_URL,
            cfg.model or OLLAMA_MODEL,
            name="ollama",
            temperature=cfg.temperature,
            max_tokens=cfg.max_output_tokens,
            timeout=cfg.request_timeout,
            unreachable_hint=(
                "Is Ollama running? Start it with 'ollama serve' and pull a model "
                "with 'ollama pull llama3.2:3b'."
            ),
        )

    raise ConfigError(f"unknown provider '{name}'")

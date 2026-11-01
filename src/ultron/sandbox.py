"""Workspace confinement. Every file path the agent supplies goes through here."""

from __future__ import annotations

import fnmatch
from pathlib import Path

from ultron.errors import SandboxViolation

SENSITIVE_NAMES = (
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "id_rsa*",
    "id_ed25519*",
    "id_ecdsa*",
    ".netrc",
    ".npmrc",
    ".pypirc",
    ".git-credentials",
)
SENSITIVE_DIRS = (".ssh", ".aws", ".gnupg", ".git")
_HARMLESS_SUFFIXES = (".example", ".sample", ".template")


class Sandbox:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()

    def resolve(self, user_path: str) -> Path:
        """Resolve ``user_path`` against the workspace; raise if it escapes or is sensitive."""
        if not isinstance(user_path, str) or not user_path.strip():
            raise SandboxViolation("empty path")
        if "\x00" in user_path:
            raise SandboxViolation("invalid path")
        raw = Path(user_path)
        candidate = raw if raw.is_absolute() else self.root / raw
        resolved = candidate.resolve()  # follows symlinks and collapses '..'
        if not (resolved == self.root or resolved.is_relative_to(self.root)):
            raise SandboxViolation(f"path escapes the workspace: {user_path}")
        self._check_sensitive(resolved)
        return resolved

    def _check_sensitive(self, resolved: Path) -> None:
        rel = resolved.relative_to(self.root)
        for part in rel.parts[:-1]:
            if part.lower() in SENSITIVE_DIRS:
                raise SandboxViolation(f"access to '{part}' is blocked")
        if rel.parts and is_sensitive_name(rel.parts[-1]):
            raise SandboxViolation(f"access to sensitive file '{rel.parts[-1]}' is blocked")

    def is_sensitive(self, path: Path) -> bool:
        """True for protected entries (secrets, keys, credential folders) and for anything that
        resolves outside the workspace, e.g. a symlink pointing out. Used to hide them in listings."""
        try:
            self._check_sensitive(Path(path).resolve())
        except (SandboxViolation, ValueError):
            return True
        return False

    def relative(self, path: Path) -> str:
        """Workspace-relative display form of an already-resolved path."""
        rel = path.resolve().relative_to(self.root)
        return str(rel) if str(rel) != "." else "."


def is_sensitive_name(name: str) -> bool:
    lower = name.lower()
    if lower.endswith(_HARMLESS_SUFFIXES):
        return False
    if lower in SENSITIVE_DIRS:
        return True
    return any(fnmatch.fnmatch(lower, pat) for pat in SENSITIVE_NAMES)

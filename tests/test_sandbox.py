from __future__ import annotations

from pathlib import Path

import pytest

from ultron.errors import SandboxViolation
from ultron.sandbox import Sandbox, is_sensitive_name


@pytest.fixture
def sandbox(workspace: Path) -> Sandbox:
    (workspace / "sub").mkdir()
    (workspace / "sub" / "a.txt").write_text("a")
    return Sandbox(workspace)


def test_paths_inside_the_workspace_resolve(sandbox: Sandbox, workspace: Path) -> None:
    assert sandbox.resolve("sub/a.txt") == (workspace / "sub" / "a.txt").resolve()
    assert sandbox.resolve(".") == workspace.resolve()
    assert sandbox.resolve("sub/../sub/a.txt") == (workspace / "sub" / "a.txt").resolve()
    assert sandbox.resolve("does/not/exist.txt").is_relative_to(workspace.resolve())  # new files are fine


@pytest.mark.parametrize("bad", ["..", "../x", "sub/../../x", "/etc/passwd", "/", "~/x/../../../etc"])
def test_escapes_are_rejected(sandbox: Sandbox, bad: str) -> None:
    with pytest.raises(SandboxViolation):
        sandbox.resolve(bad)


def test_absolute_path_to_a_sibling_directory_with_the_same_prefix_is_rejected(
    sandbox: Sandbox, workspace: Path
) -> None:
    sibling = workspace.parent / (workspace.name + "-evil")
    sibling.mkdir()
    with pytest.raises(SandboxViolation):
        sandbox.resolve(str(sibling / "x"))


def test_symlink_pointing_outside_is_rejected(sandbox: Sandbox, workspace: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    (workspace / "link").symlink_to(outside)
    (workspace / "dirlink").symlink_to(tmp_path)
    with pytest.raises(SandboxViolation):
        sandbox.resolve("link")
    with pytest.raises(SandboxViolation):
        sandbox.resolve("dirlink/outside.txt")


def test_symlink_inside_the_workspace_is_fine(sandbox: Sandbox, workspace: Path) -> None:
    (workspace / "alias").symlink_to(workspace / "sub" / "a.txt")
    assert sandbox.resolve("alias") == (workspace / "sub" / "a.txt").resolve()


@pytest.mark.parametrize("name", ["", "   ", "a\x00b"])
def test_empty_or_nul_paths_are_rejected(sandbox: Sandbox, name: str) -> None:
    with pytest.raises(SandboxViolation):
        sandbox.resolve(name)


@pytest.mark.parametrize(
    "path",
    [".env", ".env.production", "sub/.env", "id_rsa", "id_ed25519.pub.bak", "server.pem", "api.key",
     ".netrc", ".git-credentials", ".ssh/config", ".aws/credentials", "x/.gnupg/pubring.kbx"],
)
def test_sensitive_files_and_folders_are_blocked(sandbox: Sandbox, path: str) -> None:
    with pytest.raises(SandboxViolation, match="blocked"):
        sandbox.resolve(path)


@pytest.mark.parametrize("name", [".env.example", ".env.sample", "config.template", "notes.md", "environment.py"])
def test_harmless_names_are_allowed(sandbox: Sandbox, name: str) -> None:
    assert not is_sensitive_name(name)
    sandbox.resolve(name)


def test_is_sensitive_hides_secrets_and_outbound_symlinks(sandbox: Sandbox, workspace: Path, tmp_path: Path) -> None:
    (workspace / ".env").write_text("K=V")
    (workspace / "out").symlink_to(tmp_path)
    assert sandbox.is_sensitive(workspace / ".env")
    assert sandbox.is_sensitive(workspace / "out")
    assert not sandbox.is_sensitive(workspace / "sub" / "a.txt")


def test_relative_display_form(sandbox: Sandbox, workspace: Path) -> None:
    assert sandbox.relative(workspace / "sub" / "a.txt") == str(Path("sub") / "a.txt")
    assert sandbox.relative(workspace) == "."

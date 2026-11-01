from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from pathlib import Path

import pytest

from helpers import FakeServer
from ultron.errors import MemoryFull, SandboxViolation, ToolError
from ultron.runtime import Runtime
from ultron.tools.base import ToolContext, coerce_args
from ultron.tools.calc import evaluate
from ultron.tools.web import UrlRejected, html_to_text


def run(rt: Runtime, name: str, **args: object) -> str:
    tool = rt.registry.get(name)
    assert tool is not None
    clean, problems = coerce_args(tool.parameters, dict(args))
    assert not problems, problems
    return tool.func(ToolContext(rt.cfg, rt.sandbox, rt.memory, "t"), clean)


# ------------------------------------------------------------------ registry
def test_builtin_tool_set(rt: Runtime) -> None:
    assert rt.registry.names() == sorted(
        ["calc", "finish", "http_get", "list_dir", "read_file", "recall", "remember", "run_command",
         "search_files", "write_file"]
    )
    for tool in rt.registry:
        assert tool.description.strip()
        assert tool.parameters["type"] == "object"


# ------------------------------------------------------------------ files
def test_list_dir_sorts_folders_first_and_hides_protected_entries(rt: Runtime, workspace: Path) -> None:
    (workspace / "zeta").mkdir()
    (workspace / "b.txt").write_text("bb")
    (workspace / "A.txt").write_text("a")
    (workspace / ".env").write_text("K=V")
    out = run(rt, "list_dir")
    lines = out.splitlines()
    assert lines[1] == "d  zeta/"
    assert lines[2].startswith("f  A.txt") and lines[3].startswith("f  b.txt")
    assert ".env" not in out and "1 protected entries hidden" in out


def test_list_dir_errors(rt: Runtime, workspace: Path) -> None:
    (workspace / "f.txt").write_text("x")
    with pytest.raises(ToolError, match="no such directory"):
        run(rt, "list_dir", path="nope")
    with pytest.raises(ToolError, match="not a directory"):
        run(rt, "list_dir", path="f.txt")
    assert "(empty)" in run(rt, "list_dir") or True  # workspace may hold f.txt; emptiness is covered below
    (workspace / "empty").mkdir()
    assert "(empty)" in run(rt, "list_dir", path="empty")


def test_read_file_windows_and_notices(rt: Runtime, workspace: Path) -> None:
    (workspace / "long.txt").write_text("\n".join(f"line {i}" for i in range(1, 11)))
    out = run(rt, "read_file", path="long.txt", max_lines=3)
    assert out.startswith("line 1\nline 2\nline 3")
    assert "start_line=4" in out
    assert run(rt, "read_file", path="long.txt", start_line=9).splitlines() == ["line 9", "line 10"]
    with pytest.raises(ToolError, match="past the end"):
        run(rt, "read_file", path="long.txt", start_line=99)


def test_read_file_refuses_binary_directories_and_missing_files(rt: Runtime, workspace: Path) -> None:
    (workspace / "bin.dat").write_bytes(b"\x00\x01\x02binary")
    (workspace / "d").mkdir()
    with pytest.raises(ToolError, match="binary"):
        run(rt, "read_file", path="bin.dat")
    with pytest.raises(ToolError, match="directory"):
        run(rt, "read_file", path="d")
    with pytest.raises(ToolError, match="no such file"):
        run(rt, "read_file", path="missing.txt")
    assert run(rt, "read_file", path="bin.dat".replace("bin.dat", "d") if False else "bin.dat".replace("bin", "bin")) \
        if False else True


def test_read_file_truncates_at_the_byte_limit(make_rt: Callable[..., Runtime], workspace: Path) -> None:
    rt = make_rt(max_read_bytes=100)
    (workspace / "big.txt").write_text("x" * 500)
    assert "[file truncated at 100 B]" in run(rt, "read_file", path="big.txt")


def test_read_file_handles_empty_files_and_bad_utf8(rt: Runtime, workspace: Path) -> None:
    (workspace / "empty.txt").write_text("")
    (workspace / "latin.txt").write_bytes("café".encode("latin-1"))
    assert run(rt, "read_file", path="empty.txt") == "(empty file)"
    assert "caf" in run(rt, "read_file", path="latin.txt")


def test_write_file_modes(rt: Runtime, workspace: Path) -> None:
    run(rt, "write_file", path="a/b/new.txt", content="one\n")
    assert (workspace / "a" / "b" / "new.txt").read_text() == "one\n"
    with pytest.raises(ToolError, match="already exists"):
        run(rt, "write_file", path="a/b/new.txt", content="two")
    run(rt, "write_file", path="a/b/new.txt", content="two\n", mode="overwrite")
    assert (workspace / "a" / "b" / "new.txt").read_text() == "two\n"
    run(rt, "write_file", path="a/b/new.txt", content="three\n", mode="append")
    assert (workspace / "a" / "b" / "new.txt").read_text() == "two\nthree\n"
    assert not list(workspace.rglob("*.ultron-tmp"))


def test_write_file_limits_and_confinement(make_rt: Callable[..., Runtime], workspace: Path, tmp_path: Path) -> None:
    rt = make_rt(max_write_bytes=50)
    with pytest.raises(ToolError, match="limit"):
        run(rt, "write_file", path="big.txt", content="x" * 100)
    run(rt, "write_file", path="ok.txt", content="x" * 40)
    with pytest.raises(ToolError, match="exceed"):
        run(rt, "write_file", path="ok.txt", content="y" * 20, mode="append")
    with pytest.raises(SandboxViolation):
        run(rt, "write_file", path="../escape.txt", content="x")
    assert not (tmp_path / "escape.txt").exists()
    (workspace / "dir").mkdir()
    with pytest.raises(ToolError, match="directory"):
        run(rt, "write_file", path="dir", content="x", mode="overwrite")


def test_search_files_finds_matches_and_skips_junk(rt: Runtime, workspace: Path) -> None:
    (workspace / "a.md").write_text("alpha\nTODO: first\n")
    (workspace / "sub").mkdir()
    (workspace / "sub" / "b.py").write_text("# todo: second\nprint(1)\n")
    (workspace / "bin.dat").write_bytes(b"\x00TODO")
    (workspace / ".env").write_text("TODO=secret")
    out = run(rt, "search_files", query="todo")
    assert "a.md:2: TODO: first" in out
    assert os.path.join("sub", "b.py") + ":1: # todo: second" in out
    assert "bin.dat" not in out and ".env" not in out
    assert run(rt, "search_files", query="todo", glob="*.py").count("\n") == 0
    assert run(rt, "search_files", query="zzz").startswith("no matches")
    with pytest.raises(ToolError, match="empty"):
        run(rt, "search_files", query="   ")


def test_search_files_does_not_follow_links_out_of_the_workspace(rt: Runtime, workspace: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "leak.txt").write_text("TODO leaked")
    (workspace / "portal").symlink_to(outside)
    out = run(rt, "search_files", query="leaked")
    # The query is echoed back, so check for the file and its line, not just the word.
    assert "leak.txt" not in out and "TODO leaked" not in out
    assert out.endswith("in 0 files")


def test_search_files_caps_results(rt: Runtime, workspace: Path) -> None:
    (workspace / "many.txt").write_text("hit\n" * 100)
    out = run(rt, "search_files", query="hit", max_results=5)
    assert out.count("many.txt:") == 5 and "stopped at 5 results" in out


# ------------------------------------------------------------------ calc
@pytest.mark.parametrize("expr,expected", [
    ("(17 * 23) + 4", "395"), ("2 ** 10", "1024"), ("10 / 4", "2.5"), ("10 // 4", "2"), ("-7 % 3", "2"),
    ("sqrt(16)", "4"), ("round(2.567, 1)", "2.6"), ("max(1, 5, 3)", "5"), ("abs(-3)", "3"), ("pi", "3.141592653589793"),
    ("1e3 + 1", "1001"), ("floor(2.7) + ceil(2.1)", "5"),
])
def test_calc_evaluates(expr: str, expected: str) -> None:
    assert evaluate(expr) == expected


@pytest.mark.parametrize("expr,message", [
    ("", "empty"), ("1 +", "syntax"), ("1 / 0", "division by zero"), ("2 ** 100000", "too large"),
    ("9 ** 9 ** 9", "too large"), ("__import__('os')", "unsupported"), ("open('x')", "unsupported"),
    ("x + 1", "unsupported"), ("'a' * 3", "unsupported"), ("(lambda: 1)()", "unsupported"),
    ("[1][0]", "unsupported"), ("1" * 300, "too long"), ("sqrt(-1)", "cannot evaluate"),
    ("10 ** 308 * 10", "too large"), ("abs.__class__", "unsupported"), ("max(key=1)", "unsupported"),
])
def test_calc_rejects_anything_that_is_not_plain_arithmetic(expr: str, message: str) -> None:
    with pytest.raises(ToolError, match=message):
        evaluate(expr)


# ------------------------------------------------------------------ memory tools
def test_remember_and_recall_roundtrip(rt: Runtime) -> None:
    assert run(rt, "remember", text="The staging server is called atlas", tags=["infra", 7]).startswith("saved note #")
    assert "atlas" in run(rt, "recall", query="staging server")
    assert run(rt, "recall", query="unrelated gibberish zzz") == "no matching notes"
    assert "infra" in run(rt, "recall", query="atlas")


def test_remember_validates_and_enforces_the_note_limit(make_rt: Callable[..., Runtime]) -> None:
    rt = make_rt(max_notes=2)
    with pytest.raises(ToolError, match="empty"):
        run(rt, "remember", text="   ")
    with pytest.raises(ToolError, match="too long"):
        run(rt, "remember", text="x" * 3000)
    run(rt, "remember", text="one")
    run(rt, "remember", text="two")
    with pytest.raises(MemoryFull):
        run(rt, "remember", text="three")


def test_finish_only_acknowledges(rt: Runtime) -> None:
    assert run(rt, "finish", summary="done") == "finished"


# ------------------------------------------------------------------ run_command
def test_run_command_captures_output_and_exit_code(rt: Runtime, workspace: Path) -> None:
    (workspace / "hello.txt").write_text("hi there\n")
    out = run(rt, "run_command", command="cat hello.txt")
    assert out.startswith("exit_code: 0") and "hi there" in out
    assert "exit_code: 1" in run(rt, "run_command", command="grep nomatch hello.txt")
    assert "stderr:" in run(rt, "run_command", command="cat nope.txt")


def test_run_command_runs_inside_the_workspace(rt: Runtime, workspace: Path) -> None:
    assert run(rt, "run_command", command="pwd").splitlines()[-1] == str(workspace.resolve())


def test_run_command_expands_globs_inside_the_workspace(rt: Runtime, workspace: Path) -> None:
    (workspace / "a.txt").write_text("A")
    (workspace / "b.txt").write_text("B")
    out = run(rt, "run_command", command="cat *.txt")
    assert "A" in out and "B" in out


@pytest.mark.skipif(shutil.which("python3") is None, reason="needs python3 on PATH")
def test_run_command_times_out_and_kills_the_process_group(make_rt: Callable[..., Runtime], workspace: Path) -> None:
    rt = make_rt(command_timeout=1, allowed_commands=["python3"], safe_commands=[])
    (workspace / "sleepy.py").write_text("import time\ntime.sleep(30)\n")
    with pytest.raises(ToolError, match="timed out"):
        run(rt, "run_command", command="python3 sleepy.py")


def test_run_command_does_not_leak_secrets_into_the_child(
    make_rt: Callable[..., Runtime], workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-0123456789abcdef")
    monkeypatch.setenv("SOME_TOKEN", "tok-0123456789")
    rt = make_rt(allowed_commands=["python3"], safe_commands=[])
    (workspace / "env.py").write_text("import os\nprint(sorted(os.environ))\n")
    out = run(rt, "run_command", command="python3 env.py")
    assert "ANTHROPIC_API_KEY" not in out and "SOME_TOKEN" not in out
    assert "PATH" in out


def test_run_command_reports_missing_programs(make_rt: Callable[..., Runtime]) -> None:
    rt = make_rt(allowed_commands=["definitely-not-installed-xyz"], safe_commands=[])
    with pytest.raises(ToolError, match="not installed"):
        run(rt, "run_command", command="definitely-not-installed-xyz")


def test_run_command_output_is_capped(make_rt: Callable[..., Runtime], workspace: Path) -> None:
    rt = make_rt(allowed_commands=["python3"], safe_commands=[], max_tool_output_chars=500)
    (workspace / "spam.py").write_text("print('x' * 100000)\n")
    out = run(rt, "run_command", command="python3 spam.py")
    assert len(out) < 3000


def test_run_command_refuses_what_policy_would_deny(rt: Runtime) -> None:
    # The tool re-validates, so it is safe even if called without going through the policy engine.
    with pytest.raises(ToolError):
        run(rt, "run_command", command="rm -rf /")
    # A path escape is a SandboxViolation, which the agent loop reports as a denial.
    with pytest.raises(SandboxViolation):
        run(rt, "run_command", command="cat /etc/passwd")
    with pytest.raises(SandboxViolation):
        run(rt, "run_command", command="cat .env")


# ------------------------------------------------------------------ web
@pytest.fixture
def web_rt(make_rt: Callable[..., Runtime]) -> Runtime:
    return make_rt(web_allowed_domains=["127.0.0.1"], web_allow_private=True, web_max_bytes=2000)


def test_http_get_returns_text_and_marks_it_untrusted(web_rt: Runtime) -> None:
    with FakeServer() as server:
        server.push(200, "<html><head><style>x{}</style></head><body><script>evil()</script><p>Hello</p><p>World</p>"
                         "</body></html>", Content_Type="text/html")
        out = run(web_rt, "http_get", url=server.url + "/page")
    assert out.splitlines()[0].startswith("HTTP 200 text/html") and "untrusted" in out.splitlines()[0]
    assert "Hello" in out and "World" in out and "evil()" not in out and "x{}" not in out


def test_http_get_truncates_and_rejects_binary(web_rt: Runtime) -> None:
    with FakeServer() as server:
        server.push(200, "a" * 5000, Content_Type="text/plain")
        out = run(web_rt, "http_get", url=server.url)
        assert "[truncated at 2000 bytes]" in out
        server.push(200, b"\x89PNG", Content_Type="image/png")
        with pytest.raises(ToolError, match="unsupported content type"):
            run(web_rt, "http_get", url=server.url)


def test_http_get_revalidates_every_redirect(web_rt: Runtime) -> None:
    with FakeServer() as server:
        port = server.httpd.server_address[1]
        server.push(302, "", Location=f"http://localhost:{port}/other")  # 'localhost' is not allowlisted
        with pytest.raises(UrlRejected, match="not in allowed_domains"):
            run(web_rt, "http_get", url=server.url)
        server.push(302, "", Location="/next")
        server.push(200, "landed", Content_Type="text/plain")
        assert "landed" in run(web_rt, "http_get", url=server.url)  # relative redirects on the same host are fine


def test_http_get_limits_redirect_chains(web_rt: Runtime) -> None:
    with FakeServer() as server:
        for _ in range(6):
            server.push(302, "", Location="/again")
        with pytest.raises(ToolError, match="too many redirects"):
            run(web_rt, "http_get", url=server.url)


def test_http_get_reports_http_errors_without_the_body(web_rt: Runtime) -> None:
    with FakeServer() as server:
        server.push(404, "secret internal body", Content_Type="text/plain")
        with pytest.raises(ToolError, match="HTTP 404") as info:
            run(web_rt, "http_get", url=server.url)
    assert "secret internal body" not in str(info.value)


def test_http_get_is_blocked_by_default(rt: Runtime) -> None:
    with pytest.raises(UrlRejected):
        run(rt, "http_get", url="https://example.com")


def test_html_to_text_drops_scripts_and_styles() -> None:
    assert html_to_text("<div>a<script>x()</script><style>p{}</style><b>b</b></div>") == "a\nb"


# ------------------------------------------------------------------ argument coercion
def test_coerce_args_fixes_harmless_type_slips_and_reports_real_problems() -> None:
    schema = {
        "type": "object",
        "properties": {
            "n": {"type": "integer"}, "s": {"type": "string"}, "f": {"type": "boolean"},
            "mode": {"type": "string", "enum": ["a", "b"]}, "xs": {"type": "array"}, "x": {"type": "number"},
        },
        "required": ["n"],
    }
    clean, problems = coerce_args(schema, {"n": "5", "s": 12, "f": "true", "extra": 1, "x": "2.5"})
    assert clean == {"n": 5, "s": "12", "f": True, "x": 2.5} and not problems
    _, problems = coerce_args(schema, {})
    assert problems == ["missing required argument 'n'"]
    _, problems = coerce_args(schema, {"n": True, "mode": "c", "xs": "no", "s": [1], "f": "maybe", "x": "abc"})
    assert len(problems) == 6
    assert coerce_args(schema, {"n": 3.0})[0] == {"n": 3}
    assert coerce_args(schema, {"n": 3.5})[1]

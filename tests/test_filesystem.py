from pathlib import Path

import pytest

from coding_helper.tools import WorkspaceViolation, create_filesystem_registry


def invoke(registry, name: str, **arguments) -> str:
    """通过 Registry 调用工具，测试路径与真实 Executor 保持一致。"""

    return registry.get_by_model_name(name).langchain_tool.invoke(arguments)


def make_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return workspace


def test_read_file_returns_line_numbers_and_truncation_state(tmp_path) -> None:
    workspace = make_workspace(tmp_path)
    (workspace / "sample.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
    registry = create_filesystem_registry(workspace)

    output = invoke(
        registry,
        "read_file",
        path="sample.txt",
        start_line=2,
        max_lines=1,
    )

    assert "source=sample.txt" in output
    assert "truncated=true" in output
    assert "2: two" in output
    assert "one" not in output


def test_read_file_rejects_traversal_symlink_and_sensitive_file(tmp_path) -> None:
    workspace = make_workspace(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    (workspace / "escape.txt").symlink_to(outside)
    (workspace / ".env").write_text("TOKEN=secret", encoding="utf-8")
    registry = create_filesystem_registry(workspace)

    for unsafe_path in ("../outside.txt", "escape.txt", ".env"):
        with pytest.raises(WorkspaceViolation):
            invoke(registry, "read_file", path=unsafe_path)


def test_list_directory_omits_runtime_cache_and_sensitive_entries(tmp_path) -> None:
    workspace = make_workspace(tmp_path)
    (workspace / "src").mkdir()
    (workspace / "src" / "app.py").write_text("print('ok')", encoding="utf-8")
    (workspace / ".venv").mkdir()
    (workspace / ".venv" / "secret.py").write_text("hidden", encoding="utf-8")
    (workspace / ".env").write_text("TOKEN=secret", encoding="utf-8")
    registry = create_filesystem_registry(workspace)

    output = invoke(registry, "list_directory", path=".", max_depth=2)

    assert "src/" in output
    assert "src/app.py" in output
    assert ".venv" not in output
    assert ".env" not in output


def test_search_text_respects_glob_case_and_result_limit(tmp_path) -> None:
    workspace = make_workspace(tmp_path)
    (workspace / "a.py").write_text("Token = 1\nTOKEN = 2\n", encoding="utf-8")
    (workspace / "b.md").write_text("token docs\n", encoding="utf-8")
    registry = create_filesystem_registry(workspace)

    output = invoke(
        registry,
        "search_text",
        query="token",
        path=".",
        file_glob="*.py",
        max_results=1,
        case_sensitive=False,
    )

    assert "a.py:1: Token = 1" in output
    assert "results truncated at 1" in output
    assert "b.md" not in output

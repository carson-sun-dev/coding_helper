import json
import stat

import pytest

from coding_helper.tools import ToolRegistry, ToolRisk, WorkspaceViolation
from coding_helper.tools.writes import (
    FileWriteError,
    SafeFileEditor,
    register_write_tools,
)


def test_replace_text_saves_preimage_journal_and_file_mode(tmp_path) -> None:
    source = tmp_path / "app.py"
    source.write_text("value = 'old'\n", encoding="utf-8")
    source.chmod(0o640)
    editor = SafeFileEditor(tmp_path)
    before = editor.file_hash("app.py")

    result = editor.replace_text(
        user_path="app.py",
        old_text="'old'",
        new_text="'new'",
        expected_sha256=before["sha256"],
    )

    assert source.read_text(encoding="utf-8") == "value = 'new'\n"
    assert stat.S_IMODE(source.stat().st_mode) == 0o640
    backup = tmp_path / ".coding-helper" / result["backup"]
    assert backup.read_bytes() == b"value = 'old'\n"
    assert stat.S_IMODE(backup.stat().st_mode) == 0o600
    assert not list(tmp_path.glob(".coding-helper-*"))

    journal_path = tmp_path / ".coding-helper" / "operations.jsonl"
    records = [
        json.loads(line)
        for line in journal_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["status"] for record in records] == ["pending", "completed"]
    assert {record["operation_id"] for record in records} == {result["operation_id"]}
    assert records[0]["before_mode"] == 0o640


def test_create_file_writes_new_file_with_journal(tmp_path) -> None:
    editor = SafeFileEditor(tmp_path)

    result = editor.create_file(user_path="pkg/hello.py", content="print('hi')\n")

    created = tmp_path / "pkg" / "hello.py"
    assert created.read_text(encoding="utf-8") == "print('hi')\n"
    assert result["operation"] == "create_file"
    assert not list(tmp_path.glob("**/.coding-helper-*"))

    records = [
        json.loads(line)
        for line in (tmp_path / ".coding-helper" / "operations.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [record["status"] for record in records] == ["pending", "completed"]


def test_create_file_rejects_existing_path(tmp_path) -> None:
    (tmp_path / "app.py").write_text("value = 1\n", encoding="utf-8")
    editor = SafeFileEditor(tmp_path)

    with pytest.raises(WorkspaceViolation, match="文件已存在"):
        editor.create_file(user_path="app.py", content="x = 2\n")


def test_create_file_rejects_sensitive_and_escape(tmp_path) -> None:
    editor = SafeFileEditor(tmp_path)

    with pytest.raises(WorkspaceViolation):
        editor.create_file(user_path=".env", content="SECRET=1\n")
    with pytest.raises(WorkspaceViolation, match="超出 Workspace"):
        editor.create_file(user_path="../escape.py", content="x = 1\n")


def test_undo_create_moves_file_to_trash(tmp_path) -> None:
    editor = SafeFileEditor(tmp_path)
    result = editor.create_file(user_path="new.py", content="print(1)\n")
    created = tmp_path / "new.py"
    assert created.exists()

    undone = editor.undo(result["operation_id"])

    assert undone["status"] == "undone"
    assert not created.exists()
    trashed = tmp_path / ".coding-helper" / "trash" / result["operation_id"] / "new.py"
    assert trashed.read_text(encoding="utf-8") == "print(1)\n"


def test_undo_create_refuses_when_file_changed(tmp_path) -> None:
    editor = SafeFileEditor(tmp_path)
    result = editor.create_file(user_path="new.py", content="print(1)\n")
    (tmp_path / "new.py").write_text("print(2)\n", encoding="utf-8")

    with pytest.raises(FileWriteError, match="已被后续修改"):
        editor.undo(result["operation_id"])
    assert (tmp_path / "new.py").read_text(encoding="utf-8") == "print(2)\n"


def test_replace_text_rejects_stale_hash_before_creating_backup(tmp_path) -> None:
    source = tmp_path / "app.py"
    source.write_text("original\n", encoding="utf-8")
    editor = SafeFileEditor(tmp_path)
    stale_hash = editor.file_hash("app.py")["sha256"]
    source.write_text("changed externally\n", encoding="utf-8")

    with pytest.raises(FileWriteError, match="文件已发生变化"):
        editor.replace_text(
            user_path="app.py",
            old_text="original",
            new_text="updated",
            expected_sha256=stale_hash,
        )

    assert source.read_text(encoding="utf-8") == "changed externally\n"
    assert not editor.backup_directory.exists()


def test_replace_text_requires_unique_old_text(tmp_path) -> None:
    source = tmp_path / "app.py"
    source.write_text("same\nsame\n", encoding="utf-8")
    editor = SafeFileEditor(tmp_path)

    with pytest.raises(FileWriteError, match="当前出现 2 次"):
        editor.replace_text(
            user_path="app.py",
            old_text="same",
            new_text="different",
            expected_sha256=editor.file_hash("app.py")["sha256"],
        )

    assert source.read_text(encoding="utf-8") == "same\nsame\n"


def test_write_rejects_symlink_even_when_target_is_inside_workspace(tmp_path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("target\n", encoding="utf-8")
    (tmp_path / "alias.txt").symlink_to(target)
    editor = SafeFileEditor(tmp_path)

    with pytest.raises(WorkspaceViolation, match="符号链接"):
        editor.replace_text(
            user_path="alias.txt",
            old_text="target",
            new_text="changed",
            expected_sha256=editor.file_hash("alias.txt")["sha256"],
        )

    assert target.read_text(encoding="utf-8") == "target\n"


def test_undo_restores_preimage_and_file_mode(tmp_path) -> None:
    source = tmp_path / "app.py"
    source.write_text("value = 'old'\n", encoding="utf-8")
    source.chmod(0o640)
    editor = SafeFileEditor(tmp_path)
    result = editor.replace_text(
        user_path="app.py",
        old_text="'old'",
        new_text="'new'",
        expected_sha256=editor.file_hash("app.py")["sha256"],
    )

    undone = editor.undo(result["operation_id"])

    assert source.read_text(encoding="utf-8") == "value = 'old'\n"
    assert stat.S_IMODE(source.stat().st_mode) == 0o640
    assert undone["status"] == "undone"
    journal = [
        json.loads(line)
        for line in (tmp_path / ".coding-helper" / "operations.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [record["status"] for record in journal][-2:] == ["undo_pending", "undone"]


def test_undo_rejects_when_file_changed_after_write(tmp_path) -> None:
    source = tmp_path / "app.py"
    source.write_text("value = 'old'\n", encoding="utf-8")
    editor = SafeFileEditor(tmp_path)
    result = editor.replace_text(
        user_path="app.py",
        old_text="'old'",
        new_text="'new'",
        expected_sha256=editor.file_hash("app.py")["sha256"],
    )
    source.write_text("changed again\n", encoding="utf-8")

    with pytest.raises(FileWriteError, match="已被后续修改"):
        editor.undo(result["operation_id"])

    assert source.read_text(encoding="utf-8") == "changed again\n"


def test_inspect_pending_reports_without_rewriting_file(tmp_path) -> None:
    source = tmp_path / "app.py"
    source.write_text("value = 'old'\n", encoding="utf-8")
    editor = SafeFileEditor(tmp_path)
    before = editor.file_hash("app.py")
    editor.replace_text(
        user_path="app.py",
        old_text="'old'",
        new_text="'new'",
        expected_sha256=before["sha256"],
    )
    journal = tmp_path / ".coding-helper" / "operations.jsonl"
    completed = journal.read_text(encoding="utf-8").splitlines()[-1]
    journal.write_text(
        journal.read_text(encoding="utf-8").splitlines()[0] + "\n",
        encoding="utf-8",
    )

    inspections = editor.inspect_pending_operations()

    assert inspections[0]["observed"] == "applied_without_completion"
    assert source.read_text(encoding="utf-8") == "value = 'new'\n"
    assert completed  # 完成记录被测试故意移除，文件保持修改后状态


def test_registered_write_tool_exposes_risk_metadata(tmp_path) -> None:
    source = tmp_path / "app.py"
    source.write_text("value\n", encoding="utf-8")
    registry = ToolRegistry()

    register_write_tools(tmp_path, registry)

    assert registry.get_by_model_name("get_file_hash").spec.risk is ToolRisk.READ
    assert registry.get_by_model_name("replace_text").spec.risk is ToolRisk.WRITE

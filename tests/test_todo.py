import pytest

from coding_helper.governance import PermissionAction, PermissionPolicy
from coding_helper.progress.task import (
    TaskError,
    TaskStore,
    TodoStatus,
    TodoWriteItem,
    register_todo_tools,
)
from coding_helper.tools import ToolRegistry


def test_replace_todos_enforces_single_in_progress_and_completion_evidence(tmp_path) -> None:
    store = TaskStore(tmp_path)
    store.ensure_session(goal="修复登录", thread_id="thread-a")

    with pytest.raises(TaskError, match="in_progress"):
        store.replace_todos(
            [
                TodoWriteItem(content="读代码", status=TodoStatus.IN_PROGRESS),
                TodoWriteItem(content="改代码", status=TodoStatus.IN_PROGRESS),
            ]
        )
    with pytest.raises(TaskError, match="可验证结果"):
        store.replace_todos(
            [TodoWriteItem(content="读代码", status=TodoStatus.COMPLETED)]
        )

    snapshot = store.replace_todos(
        [
            TodoWriteItem(
                content="定位失败路径",
                status=TodoStatus.COMPLETED,
                evidence="tests/test_auth.py::test_refresh 失败栈",
            ),
            TodoWriteItem(content="修改实现", status=TodoStatus.IN_PROGRESS),
            TodoWriteItem(content="运行测试", status=TodoStatus.PENDING),
        ]
    )

    progress = (tmp_path / ".coding-helper" / "progress.md").read_text(encoding="utf-8")
    assert snapshot.phase.value == "executing"
    assert snapshot.next_action == "修改实现"
    assert "# Goal" in progress
    assert "修复登录" in progress
    assert "[x] 定位失败路径" in progress
    assert "*(in_progress)*" in progress
    assert "不要直接编辑" not in progress


def test_blocked_todo_becomes_progress_blocker(tmp_path) -> None:
    store = TaskStore(tmp_path)
    store.ensure_session(goal="排查 flaky 测试", thread_id="thread-b")
    snapshot = store.replace_todos(
        [
            TodoWriteItem(
                content="复现失败",
                status=TodoStatus.BLOCKED,
                evidence="本地无法稳定复现",
            )
        ]
    )

    progress = store.progress_path.read_text(encoding="utf-8")
    assert snapshot.phase.value == "paused"
    assert snapshot.blockers == ["复现失败：本地无法稳定复现"]
    assert "复现失败：本地无法稳定复现" in progress


def test_resume_same_thread_keeps_todos_new_thread_resets(tmp_path) -> None:
    store = TaskStore(tmp_path)
    store.ensure_session(goal="任务 A", thread_id="old")
    store.replace_todos([TodoWriteItem(content="第一步")])

    same = store.ensure_session(goal="任务 A 继续", thread_id="old")
    assert [item.content for item in same.todos] == ["第一步"]
    assert same.goal == "任务 A"

    reset = store.ensure_session(goal="任务 B", thread_id="new")
    assert reset.todos == []
    assert reset.goal == "任务 B"


def test_todo_write_is_allowed_without_approval(tmp_path) -> None:
    registry = ToolRegistry()
    register_todo_tools(tmp_path, registry)
    spec = registry.get_by_model_name("todo_write").spec
    decision = PermissionPolicy().decide(spec, {"items": []})

    assert spec.risk.value == "write"
    assert decision.action is PermissionAction.ALLOW

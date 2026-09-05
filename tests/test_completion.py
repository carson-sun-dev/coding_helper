import subprocess

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage

from coding_helper.config import Settings
from coding_helper.governance.completion import (
    CompletionGateMiddleware,
    evaluate_completion,
    render_report,
)
from coding_helper.observe.events import EventStore
from coding_helper.progress.task import TaskStore, TodoStatus, TodoWriteItem
from coding_helper.tools.gitdiff import collect_workspace_diff, register_git_diff_tools
from coding_helper.tools.registry import ToolRegistry


def _settings(workspace, **overrides) -> Settings:
    values = {"workspace": workspace, **overrides}
    return Settings(_env_file=None, **values)


def _init_repo(path) -> None:
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "gate@test"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "gate"], cwd=path, check=True, capture_output=True)
    (path / "app.py").write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "app.py"], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ["git", "-c", "commit.gpgsign=false", "commit", "-m", "init"],
        cwd=path,
        check=True,
        capture_output=True,
    )


def test_skips_when_workspace_is_not_a_git_repo(tmp_path) -> None:
    report = evaluate_completion(tmp_path, _settings(tmp_path))
    assert report.passed
    assert "不是 Git 仓库" in report.checks[0].detail


def test_incomplete_todos_and_failed_command_block_completion(tmp_path) -> None:
    _init_repo(tmp_path)
    store = TaskStore(tmp_path)
    store.ensure_session(goal="修 bug", thread_id="t1")
    store.replace_todos([TodoWriteItem(content="改实现", status=TodoStatus.IN_PROGRESS)])
    (tmp_path / "app.py").write_text("value = 2\n", encoding="utf-8")

    report = evaluate_completion(
        tmp_path,
        _settings(tmp_path, completion_test_command="false"),
    )

    assert report.passed is False
    names = {item.name: item.passed for item in report.checks}
    assert names["todos"] is False
    assert names["tests"] is False
    assert "未完成" in render_report(report)


def test_deletion_and_forbidden_env_fail_diff(tmp_path) -> None:
    _init_repo(tmp_path)
    (tmp_path / "app.py").unlink()
    (tmp_path / ".env").write_text("ARK_API_KEY=secret\n", encoding="utf-8")

    report = evaluate_completion(tmp_path, _settings(tmp_path))
    diff = next(item for item in report.checks if item.name == "diff")
    assert diff.passed is False
    assert "删除" in diff.detail
    assert "禁止修改" in diff.detail


def test_passing_gate_and_git_diff_tool(tmp_path) -> None:
    _init_repo(tmp_path)
    store = TaskStore(tmp_path)
    store.ensure_session(goal="修 bug", thread_id="t1")
    store.replace_todos(
        [
            TodoWriteItem(
                content="改实现",
                status=TodoStatus.COMPLETED,
                evidence="app.py",
            )
        ]
    )
    (tmp_path / "app.py").write_text("value = 2\n", encoding="utf-8")

    report = evaluate_completion(
        tmp_path,
        _settings(tmp_path, completion_test_command="true"),
    )
    assert report.passed
    assert collect_workspace_diff(tmp_path).paths == ["app.py"]

    registry = ToolRegistry()
    register_git_diff_tools(tmp_path, registry)
    output = registry.get_by_model_name("git_diff").langchain_tool.invoke({})
    assert "app.py" in output


class _ReviewFakeModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def _passing_store(tmp_path) -> None:
    _init_repo(tmp_path)
    store = TaskStore(tmp_path)
    store.ensure_session(goal="修 bug", thread_id="t1")
    store.replace_todos(
        [TodoWriteItem(content="改实现", status=TodoStatus.COMPLETED, evidence="app.py")]
    )
    (tmp_path / "app.py").write_text("value = 2\n", encoding="utf-8")


def test_review_records_notes_without_blocking_completion(tmp_path) -> None:
    _passing_store(tmp_path)
    model = _ReviewFakeModel(
        responses=[
            AIMessage(
                content="## Risks\n无\n## Test Gaps\n缺回归\n## Findings\n改动合理\n## Recommendation\n补一个测试"
            )
        ]
    )
    events = EventStore(tmp_path, "t1")
    middleware = CompletionGateMiddleware(
        tmp_path, _settings(tmp_path), review_model=model, event_store=events
    )

    result = middleware.after_model(
        {"messages": [AIMessage(content="我做完了")]}, runtime=None
    )

    # 门槛通过：after_model 返回 None（放行），Reviewer 不改变这一结论。
    assert result is None
    review = TaskStore(tmp_path).load().review
    assert "role=reviewer" in review
    assert "status=completed" in review
    assert "补一个测试" in review

    # Reviewer 调用进入轨迹，与 design §23 的 Subagent 事件对齐。
    types = [item["type"] for item in events.load("t1")]
    assert "SubagentStarted" in types
    assert "SubagentCompleted" in types


def test_review_skipped_when_no_model_configured(tmp_path) -> None:
    _passing_store(tmp_path)
    middleware = CompletionGateMiddleware(tmp_path, _settings(tmp_path))

    assert middleware.after_model(
        {"messages": [AIMessage(content="我做完了")]}, runtime=None
    ) is None
    assert TaskStore(tmp_path).load().review == "not run"


def test_after_model_jumps_back_until_repair_budget(tmp_path) -> None:
    store = TaskStore(tmp_path)
    store.ensure_session(goal="修 bug", thread_id="t1")
    store.replace_todos([TodoWriteItem(content="还没做", status=TodoStatus.PENDING)])
    middleware = CompletionGateMiddleware(
        tmp_path,
        _settings(tmp_path, max_repair_rounds=1),
    )
    state = {"messages": [AIMessage(content="我做完了")]}

    first = middleware.after_model(state, runtime=None)
    assert first["jump_to"] == "model"
    assert isinstance(first["messages"][0], HumanMessage)
    assert "<completion-gate>" in first["messages"][0].content

    second = middleware.after_model(state, runtime=None)
    assert second["jump_to"] == "end"
    assert "最大修复轮数" in " ".join(TaskStore(tmp_path).load().blockers)

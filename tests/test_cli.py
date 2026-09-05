from types import SimpleNamespace

from typer.testing import CliRunner

from coding_helper.cli import app
from coding_helper.runtime import PendingApproval


runner = CliRunner()


def test_doctor_reports_missing_model_configuration(tmp_path, monkeypatch) -> None:
    """用户尚未配置 API Key 时，仍然应该能够检查本地环境。"""

    # 切换到隔离目录，避免开发者本机的 .env 影响测试结果。
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ARK_API_KEY", raising=False)
    monkeypatch.delenv("ARK_DEEPSEEK_MODEL", raising=False)
    monkeypatch.delenv("ARK_GLM_MODEL", raising=False)
    monkeypatch.delenv("ARK_AUXILIARY_MODEL", raising=False)

    result = runner.invoke(app, ["doctor", "--workspace", str(tmp_path)])

    assert result.exit_code == 0
    assert "Coding Helper environment" in result.stdout
    assert "ARK_API_KEY" in result.stdout
    assert "Model calls are disabled" in result.stdout


def test_mcp_check_reports_unconfigured_server(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GITHUB_PERSONAL_ACCESS_TOKEN", raising=False)

    result = runner.invoke(app, ["mcp-check", "github", "--workspace", str(tmp_path)])

    assert result.exit_code == 2
    assert "未配置 MCP Server" in result.stdout


def test_doctor_accepts_complete_model_configuration(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ARK_API_KEY", "test-secret")
    monkeypatch.setenv("ARK_DEEPSEEK_MODEL", "deepseek-endpoint")
    monkeypatch.setenv("ARK_GLM_MODEL", "glm-endpoint")
    monkeypatch.setenv("ARK_AUXILIARY_MODEL", "doubao-endpoint")

    result = runner.invoke(app, ["doctor", "--workspace", str(tmp_path)])

    assert result.exit_code == 0
    assert "configured" in result.stdout
    assert "test-secret" not in result.stdout


def test_model_check_invokes_selected_role_without_real_network(tmp_path, monkeypatch) -> None:
    """CLI 单元测试使用假模型，避免普通测试产生 API 费用。"""

    monkeypatch.chdir(tmp_path)
    selected = {}

    class FakeModel:
        def invoke(self, prompt):
            selected["prompt"] = prompt
            return SimpleNamespace(content="OK")

    def fake_create_model(settings, target):
        selected["target"] = target.value
        return FakeModel()

    monkeypatch.setattr("coding_helper.cli.create_chat_model", fake_create_model)

    result = runner.invoke(app, ["model-check", "glm"])

    assert result.exit_code == 0
    assert selected["target"] == "glm"
    assert selected["prompt"]
    assert "glm 连接成功" in result.stdout


def test_ask_uses_configured_default_model_and_prints_thread(tmp_path, monkeypatch) -> None:
    """只测试 CLI 参数连接，不在普通测试中调用真实 Agent。"""

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ARK_DEFAULT_PRIMARY", "glm")
    captured = {}

    def fake_run(question, *, settings, target, thread_id):
        captured.update(question=question, target=target.value, thread_id=thread_id)
        return SimpleNamespace(
            answer="基于文件证据的回答",
            thread_id="generated-thread",
            message_count=4,
            tool_call_count=1,
        )

    monkeypatch.setattr("coding_helper.cli.run_readonly_question", fake_run)

    result = runner.invoke(app, ["ask", "项目入口在哪里？"])

    assert result.exit_code == 0
    assert captured == {
        "question": "项目入口在哪里？",
        "target": "glm",
        "thread_id": None,
    }
    assert "基于文件证据的回答" in result.stdout
    assert "thread=generated-thread" in result.stdout
    assert "pinned=" not in result.stdout


def test_ask_prints_pinned_reference_count(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    def fake_run(question, *, settings, target, thread_id):
        return SimpleNamespace(
            answer="已阅读引用",
            thread_id="thread-pin",
            message_count=3,
            tool_call_count=0,
            pinned_reference_count=2,
        )

    monkeypatch.setattr("coding_helper.cli.run_readonly_question", fake_run)

    result = runner.invoke(app, ["ask", "分析 @src/auth.py 和 @tests/"])

    assert result.exit_code == 0
    assert "pinned=2" in result.stdout


def test_run_displays_approval_and_resumes_after_confirmation(tmp_path, monkeypatch) -> None:
    """交互测试只模拟 Runtime，确保用户明确输入后才返回 approve。"""

    monkeypatch.chdir(tmp_path)
    captured = {}

    def fake_run(task, *, settings, target, approval_handler, thread_id):
        captured["decision"] = approval_handler(
            PendingApproval(
                interrupt_id="interrupt-1",
                tool_name="replace_text",
                tool_call_id="call-1",
                risk="write",
                reason="write 工具可能产生副作用",
                arguments={"path": "app.py", "new_text": "updated"},
            )
        )
        return SimpleNamespace(
            answer="修改完成",
            thread_id="thread-write",
            message_count=4,
            tool_call_count=1,
        )

    monkeypatch.setattr("coding_helper.cli.run_coding_task", fake_run)

    result = runner.invoke(app, ["run", "修改 app.py"], input="y\n")

    assert result.exit_code == 0
    assert captured["decision"] == "approve"
    assert "replace_text" in result.stdout
    assert "app.py" in result.stdout
    assert "修改完成" in result.stdout


def test_status_prints_progress_projection(tmp_path) -> None:
    from coding_helper.progress.task import TaskStore, TodoStatus, TodoWriteItem

    store = TaskStore(tmp_path)
    store.ensure_session(goal="修复登录", thread_id="thread-status")
    store.replace_todos(
        [TodoWriteItem(content="读代码", status=TodoStatus.IN_PROGRESS)]
    )

    empty_workspace = tmp_path / "empty"
    empty_workspace.mkdir()
    empty = runner.invoke(app, ["status", "--workspace", str(empty_workspace)])
    assert empty.exit_code == 0
    assert "还没有任务进度" in empty.stdout

    result = runner.invoke(app, ["status", "--workspace", str(tmp_path)])
    assert result.exit_code == 0
    assert "修复登录" in result.stdout
    assert "读代码" in result.stdout


def test_trace_reports_empty_then_recent_events(tmp_path) -> None:
    empty = runner.invoke(app, ["trace", "--workspace", str(tmp_path)])
    assert empty.exit_code == 0
    assert "还没有执行轨迹" in empty.stdout

    from coding_helper.observe.events import EventStore

    EventStore(tmp_path, "cli-trace").emit("SessionStarted", mode="run", model="deepseek")
    result = runner.invoke(app, ["trace", "--workspace", str(tmp_path), "--limit", "5"])
    assert result.exit_code == 0
    assert "SessionStarted" in result.stdout
    assert "deepseek" in result.stdout


def test_run_result_line_includes_todo_counts(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    def fake_run(task, *, settings, target, approval_handler, thread_id):
        return SimpleNamespace(
            answer="规划完成",
            thread_id="thread-todos",
            message_count=3,
            tool_call_count=1,
            pinned_reference_count=0,
            todo_completed=1,
            todo_total=3,
        )

    monkeypatch.setattr("coding_helper.cli.run_coding_task", fake_run)
    result = runner.invoke(app, ["run", "拆分任务"])

    assert result.exit_code == 0
    assert "todos=1/3" in result.stdout


def test_undo_requires_confirmation_and_recovery_does_not_rewrite(tmp_path) -> None:
    source = tmp_path / "app.py"
    source.write_text("value = 'old'\n", encoding="utf-8")
    from coding_helper.tools.writes import SafeFileEditor

    editor = SafeFileEditor(tmp_path)
    result = editor.replace_text(
        user_path="app.py",
        old_text="'old'",
        new_text="'new'",
        expected_sha256=editor.file_hash("app.py")["sha256"],
    )

    cancelled = runner.invoke(
        app,
        ["undo", result["operation_id"], "--workspace", str(tmp_path)],
        input="n\n",
    )
    assert cancelled.exit_code == 0
    assert "已取消" in cancelled.stdout
    assert source.read_text(encoding="utf-8") == "value = 'new'\n"

    undone = runner.invoke(
        app,
        ["undo", result["operation_id"], "--workspace", str(tmp_path)],
        input="y\n",
    )
    assert undone.exit_code == 0
    assert "恢复完成" in undone.stdout
    assert source.read_text(encoding="utf-8") == "value = 'old'\n"

    recovery = runner.invoke(app, ["recovery", "--workspace", str(tmp_path)])
    assert recovery.exit_code == 0
    assert "没有待诊断的中断操作" in recovery.stdout


def test_recovery_marks_pending_without_rewriting(tmp_path) -> None:
    source = tmp_path / "app.py"
    source.write_text("value = 'old'\n", encoding="utf-8")
    from coding_helper.tools.writes import SafeFileEditor

    editor = SafeFileEditor(tmp_path)
    editor.replace_text(
        user_path="app.py",
        old_text="'old'",
        new_text="'new'",
        expected_sha256=editor.file_hash("app.py")["sha256"],
    )
    journal = tmp_path / ".coding-helper" / "operations.jsonl"
    journal.write_text(journal.read_text(encoding="utf-8").splitlines()[0] + "\n", encoding="utf-8")

    result = runner.invoke(app, ["recovery", "--workspace", str(tmp_path)])

    assert result.exit_code == 0
    assert "Pending file operations" in result.stdout
    assert "已标记为 interrupted" in result.stdout
    assert source.read_text(encoding="utf-8") == "value = 'new'\n"
    assert editor.inspect_pending_operations() == []

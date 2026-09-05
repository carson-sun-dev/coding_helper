from types import SimpleNamespace

from typer.testing import CliRunner

from coding_helper.cli import app


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

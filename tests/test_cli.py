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

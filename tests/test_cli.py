from typer.testing import CliRunner

from coding_helper.cli import app


runner = CliRunner()


def test_doctor_reports_missing_model_configuration(tmp_path, monkeypatch) -> None:
    """用户尚未配置 API Key 时，仍然应该能够检查本地环境。"""

    monkeypatch.delenv("ARK_API_KEY", raising=False)
    monkeypatch.delenv("ARK_PRIMARY_MODEL", raising=False)
    monkeypatch.delenv("ARK_AUXILIARY_MODEL", raising=False)

    result = runner.invoke(app, ["doctor", "--workspace", str(tmp_path)])

    assert result.exit_code == 0
    assert "Coding Helper environment" in result.stdout
    assert "ARK_API_KEY" in result.stdout
    assert "Model calls are disabled" in result.stdout


def test_doctor_accepts_complete_model_configuration(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ARK_API_KEY", "test-secret")
    monkeypatch.setenv("ARK_PRIMARY_MODEL", "deepseek-endpoint")
    monkeypatch.setenv("ARK_AUXILIARY_MODEL", "doubao-endpoint")

    result = runner.invoke(app, ["doctor", "--workspace", str(tmp_path)])

    assert result.exit_code == 0
    assert "configured" in result.stdout
    assert "test-secret" not in result.stdout

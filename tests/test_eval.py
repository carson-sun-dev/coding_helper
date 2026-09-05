from coding_helper.eval.harness import BUGGY_ADD, run_fix_add_eval
from typer.testing import CliRunner

from coding_helper.cli import app


def test_fix_add_eval_passes_and_records_counts(tmp_path) -> None:
    result = run_fix_add_eval(tmp_path)

    assert result.passed
    assert result.detail == "add(2, 3)==5"
    assert result.model_calls >= 1
    assert result.tool_calls >= 1
    assert result.approvals == 1
    assert result.denials == 0
    assert result.event_count >= 4
    workspace = tmp_path / ".coding-helper" / "eval" / "fix_add"
    assert "left + right" in (workspace / "app.py").read_text(encoding="utf-8")
    assert BUGGY_ADD not in (workspace / "app.py").read_text(encoding="utf-8")


def test_eval_cli_prints_raw_counts(tmp_path) -> None:
    result = CliRunner().invoke(app, ["eval", "--workspace", str(tmp_path)])

    assert result.exit_code == 0
    assert "fix_add" in result.stdout
    assert "passed" in result.stdout
    assert "n=1" in result.stdout
    assert "%" not in result.stdout

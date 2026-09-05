from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from coding_helper.config import Settings
from coding_helper.context.budget import compact_target_tokens, should_compact
from coding_helper.context.compact import (
    COMPACTED_PREFIX,
    compact_messages,
    estimate_messages,
    render_state_reminder,
)
from coding_helper.models import ModelTarget
from coding_helper.progress.task import TaskStore, TodoStatus, TodoWriteItem


def make_settings(**overrides) -> Settings:
    values = {
        "ark_api_key": "test-secret",
        "ark_deepseek_model": "deepseek-endpoint",
        "ark_glm_model": "glm-endpoint",
        "ark_auxiliary_model": "doubao-endpoint",
        "ark_deepseek_context_window": 4_000,
        "context_compact_threshold": 0.50,
        "context_compact_target": 0.30,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def test_compact_stubs_old_tool_results_and_keeps_recent(tmp_path) -> None:
    old = "x" * 4_000
    recent = "recent-evidence"
    messages = [
        HumanMessage(content="修复登录"),
        AIMessage(content="", tool_calls=[{"name": "read_file", "args": {}, "id": "c1"}]),
        ToolMessage(content=old, tool_call_id="c1", name="read_file"),
        AIMessage(content="", tool_calls=[{"name": "read_file", "args": {}, "id": "c2"}]),
        ToolMessage(content=old, tool_call_id="c2", name="read_file"),
        AIMessage(content="", tool_calls=[{"name": "read_file", "args": {}, "id": "c3"}]),
        ToolMessage(content="keep-one", tool_call_id="c3", name="read_file"),
        AIMessage(content="", tool_calls=[{"name": "search_text", "args": {}, "id": "c4"}]),
        ToolMessage(content=recent, tool_call_id="c4", name="search_text"),
    ]

    compacted = compact_messages(messages, workspace=tmp_path, target_tokens=200)

    assert compacted[2].content.startswith(COMPACTED_PREFIX)
    assert compacted[4].content.startswith(COMPACTED_PREFIX)
    assert compacted[6].content == "keep-one"
    assert compacted[8].content == recent
    assert old not in compacted[2].content
    outputs = list((tmp_path / ".coding-helper" / "outputs").glob("compact-*.txt"))
    assert outputs
    assert old in outputs[0].read_text(encoding="utf-8")
    assert estimate_messages(compacted) < estimate_messages(messages)


def test_should_compact_uses_primary_window() -> None:
    settings = make_settings()
    assert should_compact(1_999, settings, ModelTarget.DEEPSEEK) is False
    assert should_compact(2_000, settings, ModelTarget.DEEPSEEK) is True
    assert compact_target_tokens(settings, ModelTarget.DEEPSEEK) == 1_200


def test_state_reminder_reinserts_goal_and_todo(tmp_path) -> None:
    store = TaskStore(tmp_path)
    store.ensure_session(goal="修复登录 Token", thread_id="t1")
    store.replace_todos(
        [
            TodoWriteItem(
                content="定位失败",
                status=TodoStatus.COMPLETED,
                evidence="test_auth.py 失败栈",
            ),
            TodoWriteItem(content="修改实现", status=TodoStatus.IN_PROGRESS),
        ]
    )

    reminder = render_state_reminder(tmp_path)

    assert "修复登录 Token" in reminder
    assert "修改实现" in reminder
    assert "定位失败" in reminder
    assert "<compacted-context>" in reminder

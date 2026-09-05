import pytest
from langchain_core.messages import AIMessage

from coding_helper.compatibility import (
    ToolCheckMode,
    ToolCompatibilityError,
    run_tool_check,
)


class FakeBoundModel:
    """按顺序返回预设消息，避免单元测试请求真实模型。"""

    def __init__(self, responses):
        self.responses = iter(responses)
        self.received_messages = []

    def bind_tools(self, tools, **kwargs):
        self.bound_tool_names = {item.name for item in tools}
        self.bind_options = kwargs
        return self

    def invoke(self, messages):
        self.received_messages.append(messages)
        return next(self.responses)


def test_multi_tool_check_keeps_unique_ids_and_returns_results() -> None:
    request = AIMessage(
        content="",
        tool_calls=[
            {"name": "read_probe_alpha", "args": {}, "id": "call-a"},
            {"name": "read_probe_beta", "args": {}, "id": "call-b"},
        ],
    )
    model = FakeBoundModel([request, AIMessage(content="两个探针均已就绪")])

    result = run_tool_check(model, ToolCheckMode.MULTI)

    assert result.tool_names == ("read_probe_alpha", "read_probe_beta")
    assert result.tool_call_ids == ("call-a", "call-b")
    assert model.bind_options["parallel_tool_calls"] is True

    # 第二次模型请求必须同时携带原始 AIMessage 和两个对应 ID 的 ToolMessage。
    follow_up = model.received_messages[1]
    assert [message.tool_call_id for message in follow_up[-2:]] == ["call-a", "call-b"]
    assert [message.content for message in follow_up[-2:]] == [
        "alpha-ready",
        "beta-ready",
    ]


def test_tool_check_rejects_duplicate_call_ids() -> None:
    request = AIMessage(
        content="",
        tool_calls=[
            {"name": "read_probe_alpha", "args": {}, "id": "same-id"},
            {"name": "read_probe_beta", "args": {}, "id": "same-id"},
        ],
    )
    model = FakeBoundModel([request])

    with pytest.raises(ToolCompatibilityError, match="缺失或重复"):
        run_tool_check(model, ToolCheckMode.MULTI)

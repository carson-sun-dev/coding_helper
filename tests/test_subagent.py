from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from pydantic import Field

from coding_helper.agents.subagent import (
    SubagentRole,
    _DEPTH,
    register_delegate_tools,
    run_subagent,
)
from coding_helper.tools import ToolRegistry


class ToolBindableFakeModel(FakeMessagesListChatModel):
    bound_tool_names: list[str] = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        self.bound_tool_names = [item.name for item in tools]
        return self


def test_explorer_returns_structured_result_without_internal_messages(tmp_path) -> None:
    (tmp_path / "app.py").write_text("TOKEN = 'old'\n", encoding="utf-8")
    model = ToolBindableFakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_file",
                        "args": {"path": "app.py"},
                        "id": "call-read",
                    }
                ],
            ),
            AIMessage(
                content="## Files\napp.py\n## Evidence\n1: TOKEN = 'old'\n## Findings\n硬编码\n## Suggestions\n抽到配置"
            ),
        ]
    )

    result = run_subagent(
        role=SubagentRole.EXPLORER,
        task="查找 Token 硬编码",
        workspace=tmp_path,
        model=model,
    )

    assert "role=explorer" in result
    assert "status=completed" in result
    assert "tool_calls=1" in result
    assert "files=app.py" in result
    assert "untrusted=true" in result
    assert "硬编码" in result
    assert set(model.bound_tool_names) == {"read_file", "list_directory", "search_text"}


def test_delegate_does_not_expose_write_or_nested_delegate(tmp_path) -> None:
    registry = ToolRegistry()
    model = ToolBindableFakeModel(responses=[])
    register_delegate_tools(tmp_path, registry, model)

    assert registry.get_by_model_name("delegate").spec.risk.value == "read"
    assert "replace_text" not in {item.model_name for item in registry.specs()}


def test_nested_subagent_is_rejected(tmp_path) -> None:
    from coding_helper.agents.subagent import SubagentError

    called = {"invoke": False}

    class GuardedModel(ToolBindableFakeModel):
        def invoke(self, input, config=None, **kwargs):
            called["invoke"] = True
            return super().invoke(input, config=config, **kwargs)

    model = GuardedModel(responses=[AIMessage(content="不应被调用")])
    token = _DEPTH.set(1)
    try:
        raised: SubagentError | None = None
        try:
            run_subagent(
                role=SubagentRole.REVIEWER,
                task="再开一层",
                workspace=tmp_path,
                model=model,
            )
        except SubagentError as exc:
            raised = exc
    finally:
        _DEPTH.reset(token)

    assert raised is not None
    assert "递归" in str(raised)
    assert called["invoke"] is False


def test_subagent_failure_is_a_tool_result(tmp_path) -> None:
    registry = ToolRegistry()

    class ExplodingModel(ToolBindableFakeModel):
        def invoke(self, input, config=None, **kwargs):
            raise RuntimeError("simulated boom")

    model = ExplodingModel(responses=[])
    register_delegate_tools(tmp_path, registry, model)
    result = registry.get_by_model_name("delegate").langchain_tool.invoke(
        {"role": "explorer", "task": "任意探索"}
    )

    assert "status=failed" in result
    assert "simulated boom" in result
    assert "role=explorer" in result

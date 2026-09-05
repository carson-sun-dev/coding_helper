from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from pydantic import Field

from coding_helper.governance import PermissionMiddleware
from coding_helper.tools import ToolRegistry, ToolRisk, coding_tool


class ToolBindableFakeModel(FakeMessagesListChatModel):
    """支持 LangChain Tool Binding 的脚本化模型。"""

    bound_tool_names: list[str] = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        self.bound_tool_names = [item.name for item in tools]
        return self


def run_approval_scenario(decision: str):
    side_effects: list[str] = []

    @coding_tool(risk=ToolRisk.WRITE, idempotent=False)
    def write_probe(value: str) -> str:
        """记录一个可观察的模拟写入。"""

        side_effects.append(value)
        return f"written:{value}"

    registry = ToolRegistry()
    registry.register(write_probe)
    model = ToolBindableFakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "write_probe",
                        "args": {"value": "hello"},
                        "id": "call-write",
                    }
                ],
            ),
            AIMessage(content="处理完成"),
        ]
    )
    config = {"configurable": {"thread_id": f"thread-{decision}"}}

    with SqliteSaver.from_conn_string(":memory:") as checkpointer:
        agent = create_agent(
            model=model,
            tools=registry.langchain_tools(),
            middleware=[PermissionMiddleware(registry)],
            checkpointer=checkpointer,
        )
        interrupted_state = agent.invoke(
            {"messages": [{"role": "user", "content": "写入 hello"}]},
            config=config,
        )

        # Interrupt 返回前 Checkpoint 已经写入，但工具 handler 尚未被调用。
        assert side_effects == []
        assert len(interrupted_state["__interrupt__"]) == 1
        payload = interrupted_state["__interrupt__"][0].value
        assert payload["tool_call_id"] == "call-write"
        assert payload["risk"] == "write"
        assert payload["arguments"] == {"value": "hello"}
        assert list(checkpointer.list(config))

        resumed_state = agent.invoke(
            Command(resume={"decision": decision}),
            config=config,
        )

    return side_effects, resumed_state


def test_approved_interrupt_executes_side_effect_once() -> None:
    side_effects, state = run_approval_scenario("approve")

    assert side_effects == ["hello"]
    assert state["messages"][-1].content == "处理完成"


def test_rejected_interrupt_returns_tool_error_without_side_effect() -> None:
    side_effects, state = run_approval_scenario("reject")

    assert side_effects == []
    tool_messages = [
        message for message in state["messages"] if isinstance(message, ToolMessage)
    ]
    assert tool_messages[-1].status == "error"
    assert tool_messages[-1].tool_call_id == "call-write"
    assert "用户拒绝" in tool_messages[-1].content


def test_approval_payload_redacts_secret_arguments() -> None:
    registry = ToolRegistry()

    @coding_tool(risk=ToolRisk.WRITE, idempotent=False)
    def secret_probe(api_token: str) -> str:
        """接收不应显示在审批界面的敏感参数。"""

        return api_token

    registry.register(secret_probe)
    middleware = PermissionMiddleware(registry)

    assert middleware._safe_arguments({"api_token": "secret-value"}) == {
        "api_token": "[REDACTED]"
    }

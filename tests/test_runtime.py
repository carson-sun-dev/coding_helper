from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import Field

from coding_helper.runtime import build_readonly_agent


class ToolBindableFakeModel(FakeMessagesListChatModel):
    """补充 ``bind_tools`` 的脚本化模型，用来测试完整 Agent Loop。"""

    bound_tool_names: list[str] = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        # 真实 ChatOpenAI 会把这些 Tool Schema 发给方舟；Fake 只记录名称。
        self.bound_tool_names = [item.name for item in tools]
        return self


def test_readonly_agent_executes_tool_and_writes_checkpoint(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "answer.txt").write_text("meaning=42\n", encoding="utf-8")
    model = ToolBindableFakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_file",
                        "args": {"path": "answer.txt"},
                        "id": "call-read",
                    }
                ],
            ),
            AIMessage(content="answer.txt 第 1 行表明结果是 42。"),
        ]
    )
    checkpoint_path = tmp_path / "checkpoints.sqlite"
    config = {"configurable": {"thread_id": "thread-test"}}

    with SqliteSaver.from_conn_string(str(checkpoint_path)) as checkpointer:
        agent = build_readonly_agent(
            workspace=workspace,
            model=model,
            checkpointer=checkpointer,
        )
        state = agent.invoke(
            {"messages": [{"role": "user", "content": "结果是多少？"}]},
            config=config,
        )
        checkpoints = list(checkpointer.list(config))

    assert set(model.bound_tool_names) == {
        "read_file",
        "list_directory",
        "search_text",
    }
    assert isinstance(state["messages"][-2], ToolMessage)
    assert state["messages"][-2].tool_call_id == "call-read"
    assert "1: meaning=42" in state["messages"][-2].content
    assert state["messages"][-1].content == "answer.txt 第 1 行表明结果是 42。"
    assert checkpoints
    assert checkpoint_path.exists()

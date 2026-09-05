import hashlib

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import Field

from coding_helper.config import Settings
from coding_helper.models import ModelTarget
from coding_helper.runtime import build_readonly_agent, run_coding_task


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


def test_coding_task_resumes_multiple_approved_writes_in_order(tmp_path, monkeypatch) -> None:
    first_source = tmp_path / "first.py"
    second_source = tmp_path / "second.py"
    first_original = b"value = 'first-old'\n"
    second_original = b"value = 'second-old'\n"
    first_source.write_bytes(first_original)
    second_source.write_bytes(second_original)
    model = ToolBindableFakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "replace_text",
                        "args": {
                            "path": "first.py",
                            "old_text": "'first-old'",
                            "new_text": "'first-new'",
                            "expected_sha256": hashlib.sha256(first_original).hexdigest(),
                        },
                        "id": "call-first",
                    },
                    {
                        "name": "replace_text",
                        "args": {
                            "path": "second.py",
                            "old_text": "'second-old'",
                            "new_text": "'second-new'",
                            "expected_sha256": hashlib.sha256(second_original).hexdigest(),
                        },
                        "id": "call-second",
                    },
                ],
            ),
            AIMessage(content="已修改两个文件；尚未运行测试。"),
        ]
    )
    monkeypatch.setattr("coding_helper.runtime.create_chat_model", lambda settings, target: model)
    approvals = []

    result = run_coding_task(
        "修改两个文件",
        settings=Settings(_env_file=None, workspace=tmp_path, completion_enabled=False),
        target=ModelTarget.DEEPSEEK,
        approval_handler=lambda pending: approvals.append(pending) or "approve",
    )

    assert first_source.read_text(encoding="utf-8") == "value = 'first-new'\n"
    assert second_source.read_text(encoding="utf-8") == "value = 'second-new'\n"
    assert {item.tool_call_id for item in approvals} == {"call-first", "call-second"}
    assert len({item.interrupt_id for item in approvals}) == 2
    assert result.tool_call_count == 2
    assert result.answer == "已修改两个文件；尚未运行测试。"
    assert result.pinned_reference_count == 0


def test_coding_task_pins_at_reference_into_user_message(tmp_path, monkeypatch) -> None:
    (tmp_path / "hint.txt").write_text("value=9\n", encoding="utf-8")
    model = ToolBindableFakeModel(responses=[AIMessage(content="已经看到 hint.txt")])
    monkeypatch.setattr("coding_helper.runtime.create_chat_model", lambda settings, target: model)

    result = run_coding_task(
        "阅读 @hint.txt",
        settings=Settings(_env_file=None, workspace=tmp_path, completion_enabled=False),
        target=ModelTarget.DEEPSEEK,
        approval_handler=lambda pending: "reject",
        thread_id="thread-pin",
    )

    checkpoint_path = tmp_path / ".coding-helper" / "checkpoints.sqlite"
    config = {"configurable": {"thread_id": "thread-pin"}}
    with SqliteSaver.from_conn_string(str(checkpoint_path)) as checkpointer:
        saved = checkpointer.get_tuple(config)

    user_message = saved.checkpoint["channel_values"]["messages"][0]
    assert "1: value=9" in user_message.content
    assert "<untrusted-user-context" in user_message.content
    assert result.pinned_reference_count == 1
    assert result.answer == "已经看到 hint.txt"


def test_coding_task_writes_progress_without_approval(tmp_path, monkeypatch) -> None:
    model = ToolBindableFakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "todo_write",
                        "args": {
                            "items": [
                                {
                                    "content": "阅读入口",
                                    "status": "completed",
                                    "evidence": "src/coding_helper/__main__.py",
                                },
                                {"content": "补测试", "status": "in_progress"},
                            ]
                        },
                        "id": "call-todo",
                    }
                ],
            ),
            AIMessage(content="已拆分任务"),
        ]
    )
    monkeypatch.setattr("coding_helper.runtime.create_chat_model", lambda settings, target: model)

    result = run_coding_task(
        "给入口补测试",
        settings=Settings(_env_file=None, workspace=tmp_path, completion_enabled=False),
        target=ModelTarget.DEEPSEEK,
        approval_handler=lambda pending: (_ for _ in ()).throw(AssertionError("todo 不应审批")),
        thread_id="thread-todo",
    )

    progress = (tmp_path / ".coding-helper" / "progress.md").read_text(encoding="utf-8")
    assert "给入口补测试" in progress
    assert "[x] 阅读入口" in progress
    assert "*(in_progress)*" in progress
    assert result.todo_completed == 1
    assert result.todo_total == 2
    assert result.answer == "已拆分任务"


def test_coding_task_delegate_returns_summary_not_inner_messages(tmp_path, monkeypatch) -> None:
    (tmp_path / "app.py").write_text("x = 1\n", encoding="utf-8")
    model = ToolBindableFakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "delegate",
                        "args": {"role": "explorer", "task": "查找入口"},
                        "id": "call-delegate",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_file",
                        "args": {"path": "app.py"},
                        "id": "call-inner-read",
                    }
                ],
            ),
            AIMessage(content="## Files\napp.py\n## Findings\n入口很短"),
            AIMessage(content="Explorer 认为入口很短。"),
        ]
    )
    monkeypatch.setattr("coding_helper.runtime.create_chat_model", lambda settings, target: model)

    result = run_coding_task(
        "探索入口",
        settings=Settings(_env_file=None, workspace=tmp_path, completion_enabled=False),
        target=ModelTarget.DEEPSEEK,
        approval_handler=lambda pending: (_ for _ in ()).throw(AssertionError("delegate 不应审批")),
        thread_id="thread-delegate",
    )

    checkpoint_path = tmp_path / ".coding-helper" / "checkpoints.sqlite"
    config = {"configurable": {"thread_id": "thread-delegate"}}
    with SqliteSaver.from_conn_string(str(checkpoint_path)) as checkpointer:
        saved = checkpointer.get_tuple(config)
    parent_messages = saved.checkpoint["channel_values"]["messages"]
    tool_messages = [item for item in parent_messages if isinstance(item, ToolMessage)]

    assert result.answer == "Explorer 认为入口很短。"
    assert len(tool_messages) == 1
    assert tool_messages[0].tool_call_id == "call-delegate"
    assert "status=completed" in tool_messages[0].content
    assert "入口很短" in tool_messages[0].content
    assert "call-inner-read" not in tool_messages[0].content
    assert "1: x = 1" not in tool_messages[0].content

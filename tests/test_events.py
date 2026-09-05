import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from pydantic import Field

from coding_helper.config import Settings
from coding_helper.models import ModelTarget
from coding_helper.observe.events import EventStore, TraceMiddleware
from coding_helper.runtime import run_coding_task


class ToolBindableFakeModel(FakeMessagesListChatModel):
    bound_tool_names: list[str] = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        self.bound_tool_names = [item.name for item in tools]
        return self


def test_event_store_redacts_secrets_and_tails(tmp_path) -> None:
    store = EventStore(tmp_path, "thread-1")
    store.emit("ToolRequested", tool="call_mcp_tool", arguments={"token": "abc", "q": "ok"})
    store.emit("SessionCompleted", tool_call_count=1)

    records = store.tail(2)
    assert records[0]["type"] == "ToolRequested"
    assert records[0]["arguments"]["token"] == "[REDACTED]"
    assert records[0]["arguments"]["q"] == "ok"
    assert records[1]["thread_id"] == "thread-1"
    assert "abc" not in (tmp_path / ".coding-helper" / "events.jsonl").read_text(encoding="utf-8")


def test_trace_middleware_logs_model_and_tool(tmp_path) -> None:
    store = EventStore(tmp_path, "t1")
    middleware = TraceMiddleware(store)

    class Request:
        model = type("M", (), {"model_name": "deepseek"})()
        tool_call = {"name": "read_file", "args": {"path": "a.py"}, "id": "c1"}

    middleware.wrap_model_call(Request(), lambda _req: AIMessage(content="hi"))
    middleware.wrap_tool_call(Request(), lambda _req: "ok")

    types = [item["type"] for item in store.tail(10)]
    assert types == ["ModelStarted", "ModelCompleted", "ToolRequested", "ToolCompleted"]
    assert store.tail(1)[0]["tool"] == "read_file"


def test_interrupt_is_not_logged_as_tool_failure(tmp_path) -> None:
    store = EventStore(tmp_path, "t1")
    middleware = TraceMiddleware(store)

    class Request:
        model = type("M", (), {"model_name": "deepseek"})()
        tool_call = {"name": "shell", "args": {"command": "ls"}, "id": "c1"}

    class GraphInterrupt(Exception):
        """模拟 LangGraph 审批中断异常的类型名。"""

    def _raise(_req):
        raise GraphInterrupt("approval required")

    with pytest.raises(GraphInterrupt):
        middleware.wrap_tool_call(Request(), _raise)

    types = [item["type"] for item in store.tail(10)]
    assert "ToolInterrupted" in types
    assert "ToolFailed" not in types


def test_model_completed_counts_tool_calls_from_list_result(tmp_path) -> None:
    store = EventStore(tmp_path, "t1")
    middleware = TraceMiddleware(store)

    class Request:
        model = type("M", (), {"model_name": "deepseek"})()

    ai = AIMessage(
        content="thinking",
        tool_calls=[{"name": "read_file", "args": {"path": "a.py"}, "id": "c1"}],
    )
    # 模拟 handler 返回“带 result 列表”的包装对象，而非裸 AIMessage。
    wrapper = type("Resp", (), {"result": [ai]})()

    middleware.wrap_model_call(Request(), lambda _req: wrapper)

    completed = next(item for item in store.tail(10) if item["type"] == "ModelCompleted")
    assert completed["tool_call_count"] == 1


def test_coding_run_writes_session_and_approval_events(tmp_path, monkeypatch) -> None:
    source = tmp_path / "app.py"
    source.write_text("value = 'old'\n", encoding="utf-8")
    import hashlib

    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    model = ToolBindableFakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "replace_text",
                        "args": {
                            "path": "app.py",
                            "old_text": "'old'",
                            "new_text": "'new'",
                            "expected_sha256": digest,
                        },
                        "id": "call-write",
                    }
                ],
            ),
            AIMessage(content="改完了"),
        ]
    )
    monkeypatch.setattr("coding_helper.runtime.create_chat_model", lambda settings, target: model)

    run_coding_task(
        "改文件",
        settings=Settings(_env_file=None, workspace=tmp_path, completion_enabled=False),
        target=ModelTarget.DEEPSEEK,
        approval_handler=lambda pending: "approve",
        thread_id="trace-run",
    )

    types = [
        item["type"]
        for item in EventStore(tmp_path, "trace-run").tail(50)
    ]
    assert types[0] == "SessionStarted"
    assert "ToolApproved" in types
    assert "ToolRequested" in types
    assert types[-1] == "SessionCompleted"
    assert "ARK_API_KEY" not in (tmp_path / ".coding-helper" / "events.jsonl").read_text(
        encoding="utf-8"
    )

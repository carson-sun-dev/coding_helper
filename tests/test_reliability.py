from dataclasses import dataclass

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import Field

from coding_helper.config import Settings
from coding_helper.governance.reliability import (
    ErrorClass,
    ServiceFallbackMiddleware,
    classify_model_error,
    is_retryable_service_error,
)
from coding_helper.models import ModelTarget
from coding_helper.runtime import build_readonly_agent


class ToolBindableFakeModel(FakeMessagesListChatModel):
    bound_tool_names: list[str] = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        self.bound_tool_names = [item.name for item in tools]
        return self


class StatusError(Exception):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass
class FakeRequest:
    model: str

    def override(self, **kwargs) -> "FakeRequest":
        return FakeRequest(model=str(kwargs.get("model", self.model)))


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (TimeoutError("deadline"), ErrorClass.RETRYABLE_SERVICE),
        (StatusError("slow down", 429), ErrorClass.RETRYABLE_SERVICE),
        (StatusError("upstream", 503), ErrorClass.RETRYABLE_SERVICE),
        (StatusError("no key", 401), ErrorClass.AUTH),
        (StatusError("bad json", 400), ErrorClass.CLIENT),
        (ValueError("context length exceeded the window"), ErrorClass.CONTEXT),
        (ValueError("bad tool args"), ErrorClass.UNKNOWN),
    ],
)
def test_classify_model_error(exc: Exception, expected: ErrorClass) -> None:
    assert classify_model_error(exc) is expected
    assert is_retryable_service_error(exc) is (expected is ErrorClass.RETRYABLE_SERVICE)


def test_fallback_only_after_retryable_primary_failure() -> None:
    seen: list[str] = []

    def handler(request: FakeRequest) -> str:
        seen.append(request.model)
        if request.model == "primary":
            raise TimeoutError("ark timeout")
        return "fallback-ok"

    middleware = ServiceFallbackMiddleware(
        "fallback",
        retry_attempts=1,
        initial_delay=0,
    )
    result = middleware.wrap_model_call(FakeRequest("primary"), handler)

    assert result == "fallback-ok"
    assert seen == ["primary", "primary", "fallback"]


def test_client_error_does_not_switch_model() -> None:
    def handler(request: FakeRequest) -> str:
        raise StatusError("invalid request", 400)

    middleware = ServiceFallbackMiddleware("fallback", retry_attempts=1, initial_delay=0)

    with pytest.raises(StatusError, match="invalid request"):
        middleware.wrap_model_call(FakeRequest("primary"), handler)


def test_model_call_budget_ends_before_second_call(tmp_path) -> None:
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
            AIMessage(content="不应该生成这句最终回答"),
        ]
    )
    settings = Settings(
        _env_file=None,
        workspace=workspace,
        max_model_calls=1,
        max_tool_calls=20,
    )
    config = {"configurable": {"thread_id": "budget"}}

    with SqliteSaver.from_conn_string(str(tmp_path / "checkpoints.sqlite")) as checkpointer:
        agent = build_readonly_agent(
            workspace=workspace,
            model=model,
            checkpointer=checkpointer,
            settings=settings,
            target=ModelTarget.DEEPSEEK,
        )
        state = agent.invoke(
            {"messages": [{"role": "user", "content": "结果是多少？"}]},
            config=config,
        )

    final = state["messages"][-1]
    assert isinstance(final, AIMessage)
    assert "limit" in str(final.content).lower()
    assert "不应该生成这句最终回答" not in str(final.content)

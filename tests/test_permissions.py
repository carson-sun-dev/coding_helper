from types import SimpleNamespace

from langchain_core.messages import ToolMessage

from coding_helper.governance import (
    PermissionAction,
    PermissionMiddleware,
    PermissionPolicy,
)
from coding_helper.tools import ToolRegistry, ToolRisk, coding_tool


@coding_tool(risk=ToolRisk.READ, idempotent=True)
def read_probe() -> str:
    """执行测试用只读操作。"""

    return "read"


@coding_tool(risk=ToolRisk.WRITE, idempotent=False)
def write_probe() -> str:
    """执行测试用写操作。"""

    return "write"


@coding_tool(risk=ToolRisk.DESTRUCTIVE, idempotent=False)
def delete_probe() -> str:
    """执行测试用破坏性操作。"""

    return "deleted"


def make_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(read_probe)
    registry.register(write_probe)
    registry.register(delete_probe)
    return registry


def make_request(name: str):
    return SimpleNamespace(
        tool_call={"name": name, "args": {}, "id": f"call-{name}"}
    )


def test_policy_maps_risk_to_allow_ask_and_deny() -> None:
    registry = make_registry()
    policy = PermissionPolicy()

    assert (
        policy.decide(registry.get_by_model_name("read_probe").spec, {}).action
        is PermissionAction.ALLOW
    )
    assert (
        policy.decide(registry.get_by_model_name("write_probe").spec, {}).action
        is PermissionAction.ASK
    )
    assert (
        policy.decide(registry.get_by_model_name("delete_probe").spec, {}).action
        is PermissionAction.DENY
    )


def test_middleware_only_calls_handler_for_allowed_tool() -> None:
    middleware = PermissionMiddleware(make_registry())
    executed = []

    def handler(request):
        executed.append(request.tool_call["name"])
        return ToolMessage(
            content="executed",
            tool_call_id=request.tool_call["id"],
            name=request.tool_call["name"],
        )

    result = middleware.wrap_tool_call(make_request("read_probe"), handler)

    assert result.content == "executed"
    assert executed == ["read_probe"]


def test_middleware_returns_error_message_for_destructive_tool() -> None:
    middleware = PermissionMiddleware(make_registry())
    executed = False

    def handler(request):
        nonlocal executed
        executed = True
        return ToolMessage(content="unexpected", tool_call_id=request.tool_call["id"])

    result = middleware.wrap_tool_call(
        make_request("delete_probe"),
        handler,
    )

    assert isinstance(result, ToolMessage)
    assert executed is False
    assert result.status == "error"
    assert result.tool_call_id == "call-delete_probe"
    assert "permission_denied" in result.content

"""基于 ToolSpec 的权限决策与 LangChain 工具执行拦截。"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command, interrupt

from coding_helper.tools import ToolRegistrationError, ToolRegistry, ToolRisk, ToolSpec


class PermissionAction(str, Enum):
    """Harness 对一次 Tool Call 的三种权限决定。"""

    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


@dataclass(frozen=True)
class PermissionDecision:
    """权限引擎生成的可记录决定。"""

    action: PermissionAction
    reason: str


class PermissionPolicy:
    """根据治理元数据给出保守的默认权限。

    Policy 只负责做决定，不执行工具也不与用户交互。把判断与交互拆开后，
    同一策略既可以用于 CLI 的 LangGraph Interrupt，也可以用于测试、
    Subagent 和未来的非交互运行模式。
    """

    def decide(self, spec: ToolSpec, arguments: dict[str, Any]) -> PermissionDecision:
        """返回工具本次调用的权限动作。

        ``arguments`` 暂时保留给下一批参数级规则，例如同一个 Shell 工具中
        ``git status`` 可以允许，而 ``git reset --hard`` 必须拒绝。
        """

        del arguments
        if spec.risk is ToolRisk.DESTRUCTIVE:
            return PermissionDecision(
                PermissionAction.DENY,
                "破坏性工具默认禁止，不能通过普通会话自动放行",
            )
        if spec.risk is ToolRisk.READ:
            return PermissionDecision(
                PermissionAction.ALLOW,
                "只读工具不会修改 Workspace",
            )
        return PermissionDecision(
            PermissionAction.ASK,
            f"{spec.risk.value} 工具可能产生副作用",
        )


class PermissionMiddleware(AgentMiddleware):
    """在 LangChain ToolNode 真正执行函数前应用权限策略。

    ``handler(request)`` 是继续执行的唯一入口。ALLOW 调用它，DENY 直接
    构造错误 ToolMessage，ASK 调用 LangGraph ``interrupt``。Interrupt
    会先把图状态写入 Checkpoint，再暂停当前工具节点；恢复时该节点从开头
    重放，因此任何副作用都必须放在 Interrupt 返回批准结果之后。
    """

    def __init__(
        self,
        registry: ToolRegistry,
        policy: PermissionPolicy | None = None,
    ) -> None:
        self._registry = registry
        self._policy = policy or PermissionPolicy()

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        call = request.tool_call
        name = call["name"]
        try:
            registered = self._registry.get_by_model_name(name)
        except ToolRegistrationError:
            return self._denied_message(call, "工具未在治理 Registry 中注册")

        decision = self._policy.decide(registered.spec, call.get("args", {}))
        if decision.action is PermissionAction.ALLOW:
            return handler(request)
        if decision.action is PermissionAction.ASK:
            response = interrupt(
                {
                    "type": "tool_approval",
                    "tool_name": name,
                    "tool_call_id": call["id"],
                    "risk": registered.spec.risk.value,
                    "reason": decision.reason,
                    "arguments": self._safe_arguments(call.get("args", {})),
                }
            )
            if self._is_approved(response):
                return handler(request)
            return self._denied_message(call, "用户拒绝了本次工具调用")
        return self._denied_message(call, decision.reason)

    @staticmethod
    def _denied_message(call: dict[str, Any], reason: str) -> ToolMessage:
        return ToolMessage(
            content=f"permission_denied: {reason}",
            tool_call_id=call["id"],
            name=call["name"],
            status="error",
        )

    @staticmethod
    def _is_approved(response: Any) -> bool:
        """只接受明确批准；未知或畸形响应一律按拒绝处理。"""

        if response is True or response == "approve":
            return True
        return isinstance(response, dict) and response.get("decision") == "approve"

    @staticmethod
    def _safe_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
        """生成审批界面的参数摘要，避免直接展示常见密钥字段。"""

        safe: dict[str, Any] = {}
        for key, value in arguments.items():
            lowered = key.lower()
            if any(marker in lowered for marker in ("key", "token", "password", "secret")):
                safe[key] = "[REDACTED]"
            elif isinstance(value, str) and len(value) > 500:
                safe[key] = f"{value[:500]}... [TRUNCATED]"
            else:
                safe[key] = value
        return safe

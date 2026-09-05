"""基于 ToolSpec 的权限决策与 LangChain 工具执行拦截。"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from threading import Condition
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ToolCallRequest
from langchain_core.messages import AIMessage, ToolMessage
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

        Shell 必须看具体命令：``git status`` 可以自动放行，
        ``git reset --hard`` 必须拒绝。判断函数放在 tools.shell，
        避免 Policy 自己维护一套命令正则。
        """

        if spec.model_name == "shell":
            from coding_helper.tools.shell import classify_shell_command

            classification = classify_shell_command(str(arguments.get("command", "")))
            return PermissionDecision(
                PermissionAction(classification.action.value),
                classification.reason,
            )
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
        # ToolNode 会并发分发同一响应中的调用；Condition 让副作用仍按模型
        # 给出的顺序执行，而只读工具不经过这把协调锁。
        self._side_effect_condition = Condition()
        self._next_side_effect: dict[tuple[str, ...], int] = {}

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
            return self._run_side_effect_in_order(
                request,
                lambda: (
                    handler(request)
                    if self._is_approved(response)
                    else self._denied_message(call, "用户拒绝了本次工具调用")
                ),
            )
        return self._run_side_effect_in_order(
            request,
            lambda: self._denied_message(call, decision.reason),
        )

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

    def _run_side_effect_in_order(
        self,
        request: ToolCallRequest,
        operation: Callable[[], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """让同一 AIMessage 中的副作用按 Tool Call 顺序逐个执行。"""

        batch, position = self._side_effect_batch(request)
        with self._side_effect_condition:
            self._next_side_effect.setdefault(batch, 0)
            while self._next_side_effect[batch] != position:
                self._side_effect_condition.wait()

        try:
            return operation()
        finally:
            with self._side_effect_condition:
                next_position = position + 1
                if next_position >= len(batch):
                    self._next_side_effect.pop(batch, None)
                else:
                    self._next_side_effect[batch] = next_position
                self._side_effect_condition.notify_all()

    def _side_effect_batch(self, request: ToolCallRequest) -> tuple[tuple[str, ...], int]:
        """从当前 AIMessage 找出所有非只读调用及当前调用位置。"""

        state = getattr(request, "state", None)
        messages = state.get("messages", []) if isinstance(state, dict) else []
        ai_message = next(
            (message for message in reversed(messages) if isinstance(message, AIMessage)),
            None,
        )
        side_effect_ids: list[str] = []
        if ai_message:
            for call in ai_message.tool_calls:
                try:
                    risk = self._registry.get_by_model_name(call["name"]).spec.risk
                except ToolRegistrationError:
                    risk = ToolRisk.DESTRUCTIVE
                if risk is not ToolRisk.READ:
                    side_effect_ids.append(call["id"])

        current_id = request.tool_call["id"]
        if current_id not in side_effect_ids:
            # 自定义测试或框架变化导致 State 不完整时，退化为单调用串行批次。
            side_effect_ids = [current_id]
        batch = tuple(side_effect_ids)
        return batch, batch.index(current_id)

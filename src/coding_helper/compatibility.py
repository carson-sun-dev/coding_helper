"""真实模型 Tool Calling 协议的最小兼容性检查。"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

from langchain_core.messages import HumanMessage

from coding_helper.tools import (
    RetryPolicy,
    ToolExecutionStatus,
    ToolExecutor,
    ToolRegistry,
    ToolRisk,
    coding_tool,
)

class ToolCheckMode(str, Enum):
    """当前支持的 Tool Calling 检查模式。"""

    SINGLE = "single"
    MULTI = "multi"


class ToolCompatibilityError(RuntimeError):
    """模型返回的 Tool Call 不符合兼容性检查约定。"""


@coding_tool(
    risk=ToolRisk.READ,
    idempotent=True,
    retry_policy=RetryPolicy.SAFE,
    tags=("diagnostic",),
)
def read_probe_alpha() -> str:
    """读取只读探针 alpha；仅用于验证 Tool Calling 协议。"""

    return "alpha-ready"


@coding_tool(
    risk=ToolRisk.READ,
    idempotent=True,
    retry_policy=RetryPolicy.SAFE,
    tags=("diagnostic",),
)
def read_probe_beta() -> str:
    """读取只读探针 beta；仅用于验证 Tool Calling 协议。"""

    return "beta-ready"


PROBE_REGISTRY = ToolRegistry()
PROBE_REGISTRY.register(read_probe_alpha)
PROBE_REGISTRY.register(read_probe_beta)


@dataclass(frozen=True)
class ToolCheckResult:
    """一次兼容性检查中可供 CLI 展示的安全结果。"""

    mode: ToolCheckMode
    tool_names: tuple[str, ...]
    tool_call_ids: tuple[str, ...]
    final_content: str


def _prompt_for(mode: ToolCheckMode) -> str:
    if mode is ToolCheckMode.SINGLE:
        return "必须调用一次 read_probe_alpha 工具；不要自行猜测工具结果。"
    return (
        "必须在同一条回复中同时调用 read_probe_alpha 和 read_probe_beta；"
        "不要串行等待，也不要自行猜测工具结果。"
    )


def _expected_tools(mode: ToolCheckMode) -> set[str]:
    if mode is ToolCheckMode.SINGLE:
        return {"read_probe_alpha"}
    return {"read_probe_alpha", "read_probe_beta"}


def run_tool_check(model: Any, mode: ToolCheckMode) -> ToolCheckResult:
    """执行模型请求、工具结果回传和最终回复三个协议步骤。

    ``bind_tools`` 只把 Tool Schema 告诉模型，并不会替我们执行 Python
    函数。模型返回 Tool Call 后，Harness 必须按 ID 执行对应工具，再用
    ``ToolMessage`` 把结果送回同一段消息历史。这个 ID 是并发调用时避免
    结果串线的关键。
    """

    tools = list(PROBE_REGISTRY.langchain_tools())
    bound_model = model.bind_tools(
        tools,
        parallel_tool_calls=mode is ToolCheckMode.MULTI,
    )
    messages = [HumanMessage(content=_prompt_for(mode))]
    request = bound_model.invoke(messages)
    tool_calls = request.tool_calls

    names = [call["name"] for call in tool_calls]
    if set(names) != _expected_tools(mode) or len(names) != len(set(names)):
        raise ToolCompatibilityError(
            f"{mode.value} 模式收到非预期工具集合：{names or '无工具调用'}"
        )

    call_ids = [call.get("id") for call in tool_calls]
    if any(not call_id for call_id in call_ids) or len(call_ids) != len(set(call_ids)):
        raise ToolCompatibilityError("Tool Call ID 缺失或重复")

    execution_results = ToolExecutor(PROBE_REGISTRY).execute(tool_calls)
    failed = [result for result in execution_results if result.status is not ToolExecutionStatus.SUCCESS]
    if failed:
        raise ToolCompatibilityError(
            f"探针执行失败：{[(result.name, result.error_type) for result in failed]}"
        )
    tool_messages = [result.to_tool_message() for result in execution_results]

    final_response = bound_model.invoke([*messages, request, *tool_messages])
    if final_response.tool_calls:
        raise ToolCompatibilityError("模型收到探针结果后仍继续请求工具")

    return ToolCheckResult(
        mode=mode,
        tool_names=tuple(names),
        tool_call_ids=tuple(call_ids),
        final_content=str(final_response.content).strip(),
    )

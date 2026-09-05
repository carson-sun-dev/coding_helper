"""按 Tool Call 顺序执行工具，并安全并发连续的只读调用。"""

import asyncio
import json
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

from langchain_core.messages import ToolMessage
from pydantic import BaseModel

from coding_helper.tools.registry import (
    RegisteredTool,
    ToolRegistrationError,
    ToolRegistry,
    ToolRisk,
)


class ToolExecutionStatus(str, Enum):
    """单个 Tool Call 的执行状态。"""

    SUCCESS = "success"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class ToolExecutionResult:
    """执行器返回的稳定结果，并保留模型提供的 Tool Call ID。"""

    call_id: str
    name: str
    status: ToolExecutionStatus
    content: str
    duration_ms: float
    error_type: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.status is ToolExecutionStatus.SUCCESS

    def to_tool_message(self) -> ToolMessage:
        """转换成 LangChain 用来回传模型的 ToolMessage。"""

        return ToolMessage(
            content=self.content,
            tool_call_id=self.call_id,
            name=self.name,
        )


class ToolExecutionError(ValueError):
    """Tool Call 批次在执行前就无法建立可靠关联。"""


def _stringify_output(output: Any) -> str:
    """把常见返回值转换为模型可读取的文本，暂不负责结果裁剪。"""

    if isinstance(output, str):
        return output
    if isinstance(output, BaseModel):
        output = output.model_dump(mode="json")
    if isinstance(output, (dict, list, tuple)):
        return json.dumps(output, ensure_ascii=False, default=str)
    return str(output)


class ToolExecutor:
    """执行 Registry 中的工具，并采用保守的并发策略。

    只有相邻且同时标记为 ``READ + idempotent`` 的调用可以并发。分组必须
    保持相邻，是因为越过中间写操作提前读取会改变模型原本要求的执行语义。
    写入、Shell 和未知工具全部作为顺序屏障。
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    def execute(self, tool_calls: list[dict[str, Any]]) -> tuple[ToolExecutionResult, ...]:
        """供同步 CLI 使用；异步 Agent Runtime 应直接调用 ``aexecute``。"""

        return asyncio.run(self.aexecute(tool_calls))

    async def aexecute(
        self,
        tool_calls: list[dict[str, Any]],
    ) -> tuple[ToolExecutionResult, ...]:
        """执行一个模型响应中的全部 Tool Call，并保持结果顺序。"""

        self._validate_call_ids(tool_calls)
        results: list[ToolExecutionResult] = []
        index = 0

        while index < len(tool_calls):
            read_group: list[dict[str, Any]] = []
            while index < len(tool_calls) and self._is_parallel_read(tool_calls[index]):
                read_group.append(tool_calls[index])
                index += 1

            if read_group:
                # asyncio.gather 按输入顺序返回结果，即使实际完成顺序不同。
                results.extend(await asyncio.gather(*(self._run_one(call) for call in read_group)))
                continue

            call = tool_calls[index]
            index += 1
            result = await self._run_one(call)
            results.append(result)

            if not result.succeeded:
                # 副作用屏障失败后不继续执行本批后续请求，但仍为每个 ID 返回结果。
                results.extend(self._skip_remaining(tool_calls[index:], result))
                break

        return tuple(results)

    def _is_parallel_read(self, call: dict[str, Any]) -> bool:
        try:
            registered = self._registry.get_by_model_name(call["name"])
        except (KeyError, ToolRegistrationError):
            return False
        return registered.spec.risk is ToolRisk.READ and registered.spec.idempotent

    async def _run_one(self, call: dict[str, Any]) -> ToolExecutionResult:
        started = time.perf_counter()
        name = str(call.get("name", ""))
        call_id = str(call["id"])

        try:
            registered = self._registry.get_by_model_name(name)
            invocation = registered.langchain_tool.ainvoke(call.get("args", {}))
            if registered.spec.risk is ToolRisk.READ:
                output = await asyncio.wait_for(
                    invocation,
                    timeout=registered.spec.timeout_seconds,
                )
            else:
                # wait_for 无法终止已在线程中运行的同步函数。对写操作制造
                # “已经超时”的假象反而危险，因此其实现必须自行提供可取消
                # 边界，例如 Shell 工具负责终止整个子进程组。
                output = await invocation
            return ToolExecutionResult(
                call_id=call_id,
                name=name,
                status=ToolExecutionStatus.SUCCESS,
                content=_stringify_output(output),
                duration_ms=(time.perf_counter() - started) * 1000,
            )
        except Exception as exc:
            return ToolExecutionResult(
                call_id=call_id,
                name=name,
                status=ToolExecutionStatus.ERROR,
                content=f"{type(exc).__name__}: {exc}",
                duration_ms=(time.perf_counter() - started) * 1000,
                error_type=type(exc).__name__,
            )

    @staticmethod
    def _validate_call_ids(tool_calls: list[dict[str, Any]]) -> None:
        call_ids = [call.get("id") for call in tool_calls]
        if any(not call_id for call_id in call_ids):
            raise ToolExecutionError("Tool Call ID 不能为空")
        if len(call_ids) != len(set(call_ids)):
            raise ToolExecutionError("同一批次的 Tool Call ID 不能重复")

    @staticmethod
    def _skip_remaining(
        remaining: list[dict[str, Any]],
        failed: ToolExecutionResult,
    ) -> list[ToolExecutionResult]:
        return [
            ToolExecutionResult(
                call_id=str(call["id"]),
                name=str(call.get("name", "")),
                status=ToolExecutionStatus.SKIPPED,
                content=f"前序工具 {failed.name} 执行失败，本调用未执行",
                duration_ms=0,
                error_type="blocked_by_previous_failure",
            )
            for call in remaining
        ]

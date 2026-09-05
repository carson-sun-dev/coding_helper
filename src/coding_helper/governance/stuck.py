"""检测 Agent 卡在同一工具调用上。

连续相同的 name+args 达到阈值后不再执行，只回一条 ``[stuck]`` ToolMessage。
这样不会自动重放写操作，也不会再向用户弹出同一条审批。
"""

from __future__ import annotations

import json
from pathlib import Path

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage

from coding_helper.progress.task import TaskStore

STUCK_PREFIX = "[stuck]"


class StuckDetectionMiddleware(AgentMiddleware):
    """按本次进程内的连续调用计数；不把计数写进 Checkpoint。"""

    tools: list = []

    def __init__(self, workspace: Path, *, repeat_limit: int = 3) -> None:
        super().__init__()
        self._workspace = workspace
        self._repeat_limit = repeat_limit
        self._last_signature = ""
        self._repeat_count = 0

    def wrap_tool_call(self, request, handler):
        call = request.tool_call
        signature = tool_signature(str(call.get("name", "")), dict(call.get("args") or {}))
        if signature == self._last_signature:
            self._repeat_count += 1
        else:
            self._last_signature = signature
            self._repeat_count = 1

        if self._repeat_count >= self._repeat_limit:
            name = str(call.get("name") or "unknown")
            reason = f"连续 {self._repeat_count} 次相同工具调用：{name}"
            TaskStore(self._workspace).add_blocker(reason)
            return ToolMessage(
                content=f"{STUCK_PREFIX} {reason}。请改用其他文件、参数或停止。",
                tool_call_id=str(call.get("id") or "stuck"),
                name=name,
                status="error",
            )
        return handler(request)


def tool_signature(name: str, arguments: dict) -> str:
    return json.dumps(
        {"name": name, "args": arguments},
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )

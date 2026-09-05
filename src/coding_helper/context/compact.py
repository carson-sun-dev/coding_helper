"""在发给主模型前压缩过长上下文。

第一版只做确定性裁剪：把旧的冗长 Tool Result 落盘并换成短引用，
再重新注入 Goal / Todo / Blocker。豆包摘要留到后续批次，避免
为是否 compact 再花一次不确定的模型调用。
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from uuid import uuid4

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage

from coding_helper.config import Settings
from coding_helper.context.budget import (
    compact_target_tokens,
    estimate_tokens,
    should_compact,
)
from coding_helper.models import ModelTarget
from coding_helper.progress.task import TaskStore, render_progress

KEEP_RECENT_TOOL_RESULTS = 2
COMPACTED_PREFIX = "[compacted]"


def estimate_messages(messages: list[BaseMessage]) -> int:
    return sum(estimate_tokens(_message_text(item)) for item in messages)


def compact_messages(
    messages: list[BaseMessage],
    *,
    workspace: Path,
    target_tokens: int,
) -> list[BaseMessage]:
    """从最旧的 Tool Result 开始裁剪，直到估算低于目标或没有可裁项。"""

    compacted = list(messages)
    tool_indexes = [
        index
        for index, message in enumerate(compacted)
        if isinstance(message, ToolMessage) and not _already_compacted(message)
    ]
    droppable = tool_indexes[:-KEEP_RECENT_TOOL_RESULTS]
    output_dir = Path(workspace).expanduser().resolve() / ".coding-helper" / "outputs"
    for index in droppable:
        if estimate_messages(compacted) <= target_tokens:
            break
        compacted[index] = _stub_tool_message(compacted[index], output_dir)
    return compacted


def render_state_reminder(workspace: Path) -> str:
    snapshot = TaskStore(workspace).load()
    if snapshot is None:
        return (
            "<compacted-context>\n"
            "上下文已压缩。完整旧 Tool Result 在 .coding-helper/outputs/ 中。\n"
            "</compacted-context>"
        )
    return (
        "<compacted-context>\n"
        "上下文已压缩。必须保留以下任务状态；完整旧 Tool Result 已落盘。\n\n"
        f"{render_progress(snapshot).strip()}\n"
        "</compacted-context>"
    )


class ContextCompactMiddleware(AgentMiddleware):
    """在模型请求上替换消息，不改 Checkpoint 里的原始 Tool Result。"""

    tools: list = []

    def __init__(
        self,
        workspace: Path,
        settings: Settings,
        target: ModelTarget,
    ) -> None:
        super().__init__()
        self._workspace = workspace
        self._settings = settings
        self._target = target

    def wrap_model_call(self, request, handler):
        messages = list(request.messages)
        if not should_compact(estimate_messages(messages), self._settings, self._target):
            return handler(request)
        compacted = compact_messages(
            messages,
            workspace=self._workspace,
            target_tokens=compact_target_tokens(self._settings, self._target),
        )
        compacted.append(HumanMessage(content=render_state_reminder(self._workspace)))
        return handler(request.override(messages=compacted))


def _already_compacted(message: ToolMessage) -> bool:
    return _message_text(message).startswith(COMPACTED_PREFIX)


def _stub_tool_message(message: ToolMessage, output_dir: Path) -> ToolMessage:
    text = _message_text(message)
    output_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:12]
    path = output_dir / f"compact-{digest}-{uuid4().hex[:8]}.txt"
    path.write_text(text, encoding="utf-8")
    stub = (
        f"{COMPACTED_PREFIX} tool={message.name or 'unknown'} "
        f"id={message.tool_call_id} chars={len(text)} "
        f"output=.coding-helper/outputs/{path.name}"
    )
    return message.model_copy(update={"content": stub})


def _message_text(message: BaseMessage) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(block.get("text", ""))
            for block in content
            if isinstance(block, dict)
        )
    return str(content)

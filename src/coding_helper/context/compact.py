"""在发给主模型前压缩过长上下文。

先确定性裁剪旧 Tool Result，仍超阈值时再让豆包摘要更早的对话轮次。
两条路径都只改本次模型请求，不改 Checkpoint 里的原始消息。
豆包调用失败时退回裁剪结果，不能让摘要错误中断主循环。
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
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
from coding_helper.context.memory import MemoryStore
from coding_helper.context.summarize import ConversationDigest, render_digest
from coding_helper.models import ModelTarget
from coding_helper.progress.task import TaskStore, render_progress

KEEP_RECENT_TOOL_RESULTS = 2
KEEP_RECENT_MESSAGES = 8
COMPACTED_PREFIX = "[compacted]"

Summarizer = Callable[[list[BaseMessage]], ConversationDigest]


def estimate_messages(messages: list[BaseMessage]) -> int:
    return sum(estimate_tokens(_message_text(item)) for item in messages)


def compact_messages(
    messages: list[BaseMessage],
    *,
    workspace: Path,
    target_tokens: int,
    summarizer: Summarizer | None = None,
) -> list[BaseMessage]:
    """从最旧的 Tool Result 开始裁剪；仍超标时折叠旧对话。"""

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
    if (
        summarizer is not None
        and estimate_messages(compacted) > target_tokens
    ):
        compacted = _fold_with_summary(compacted, summarizer, workspace)
    return compacted


def render_state_reminder(workspace: Path) -> str:
    snapshot = TaskStore(workspace).load()
    memory = MemoryStore(workspace).render()
    if snapshot is None:
        parts = [
            "<compacted-context>",
            "上下文已压缩。完整旧 Tool Result 在 .coding-helper/outputs/ 中。",
        ]
    else:
        parts = [
            "<compacted-context>",
            "上下文已压缩。必须保留以下任务状态；完整旧 Tool Result 已落盘。",
            "",
            render_progress(snapshot).strip(),
        ]
    if memory:
        parts.extend(["", memory.strip()])
    parts.append("</compacted-context>")
    return "\n".join(parts)


class ContextCompactMiddleware(AgentMiddleware):
    """在模型请求上替换消息，不改 Checkpoint 里的原始 Tool Result。"""

    tools: list = []

    def __init__(
        self,
        workspace: Path,
        settings: Settings,
        target: ModelTarget,
        summarizer: Summarizer | None = None,
    ) -> None:
        super().__init__()
        self._workspace = workspace
        self._settings = settings
        self._target = target
        self._summarizer = summarizer

    def wrap_model_call(self, request, handler):
        messages = list(request.messages)
        if not should_compact(estimate_messages(messages), self._settings, self._target):
            return handler(request)
        compacted = compact_messages(
            messages,
            workspace=self._workspace,
            target_tokens=compact_target_tokens(self._settings, self._target),
            summarizer=self._summarizer,
        )
        compacted.append(HumanMessage(content=render_state_reminder(self._workspace)))
        return handler(request.override(messages=compacted))


def _fold_with_summary(
    messages: list[BaseMessage],
    summarizer: Summarizer,
    workspace: Path,
) -> list[BaseMessage]:
    start = _recent_start(messages)
    # 前缀太短时没有折叠价值；切点落在 ToolMessage 上会拆开未完成的工具对。
    if start < 3:
        return messages
    try:
        digest = summarizer(messages[:start])
    except Exception:
        return messages
    try:
        MemoryStore(workspace).absorb(digest.memory_candidates)
    except Exception:
        pass
    return [HumanMessage(content=render_digest(digest)), *messages[start:]]


def _recent_start(messages: list[BaseMessage]) -> int:
    start = max(0, len(messages) - KEEP_RECENT_MESSAGES)
    while start > 0 and isinstance(messages[start], ToolMessage):
        start -= 1
    return start


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

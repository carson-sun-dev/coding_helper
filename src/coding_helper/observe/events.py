"""追加式执行轨迹。只记结构化元数据，不把 Prompt、密钥或完整 Tool Result 落盘。"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, ToolMessage


class EventStore:
    """把事件追加到 ``.coding-helper/events.jsonl``，每条一行，立即 fsync。"""

    def __init__(self, workspace: Path, thread_id: str) -> None:
        self.thread_id = thread_id
        runtime = Path(workspace).expanduser().resolve() / ".coding-helper"
        runtime.mkdir(parents=True, exist_ok=True)
        self.path = runtime / "events.jsonl"

    def emit(self, event_type: str, **fields: Any) -> dict[str, Any]:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": event_type,
            "thread_id": self.thread_id,
            **_sanitize(fields),
        }
        payload = json.dumps(record, ensure_ascii=False, default=str)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(payload + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return record

    def load(self, thread_id: str | None = None) -> list[dict[str, Any]]:
        records = self.tail(limit=100_000)
        if thread_id is None:
            return records
        return [item for item in records if item.get("thread_id") == thread_id]

    def tail(self, limit: int = 20) -> list[dict[str, Any]]:
        if not self.path.is_file() or limit <= 0:
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        records: list[dict[str, Any]] = []
        for line in lines[-limit:]:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                records.append(item)
        return records


class TraceMiddleware(AgentMiddleware):
    """在模型请求和工具执行外包一层计时，供评测和 CLI 对照 progress.md。"""

    tools: list = []

    def __init__(self, store: EventStore) -> None:
        super().__init__()
        self._store = store

    def wrap_model_call(self, request, handler):
        started = time.perf_counter()
        self._store.emit("ModelStarted", model=_model_label(getattr(request, "model", None)))
        try:
            result = handler(request)
        except Exception as exc:
            self._store.emit(
                "ModelFailed",
                error_type=type(exc).__name__,
                latency_ms=_latency_ms(started),
            )
            raise
        message = _as_ai_message(result)
        usage = getattr(message, "usage_metadata", None) or {}
        tool_calls = getattr(message, "tool_calls", None) or []
        self._store.emit(
            "ModelCompleted",
            model=_model_label(getattr(request, "model", None)),
            latency_ms=_latency_ms(started),
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            tool_call_count=len(tool_calls),
            output_chars=_content_chars(message),
        )
        return result

    def wrap_tool_call(self, request, handler):
        call = request.tool_call
        name = str(call.get("name") or "unknown")
        self._store.emit(
            "ToolRequested",
            tool=name,
            arguments=_safe_arguments(dict(call.get("args") or {})),
        )
        started = time.perf_counter()
        try:
            result = handler(request)
        except Exception as exc:
            # LangGraph 用异常实现审批 Interrupt：它是“暂停等待人工”，不是工具失败，
            # 单独记 ToolInterrupted，避免 trace 把正常审批显示成报错。
            event_type = "ToolInterrupted" if _is_interrupt(exc) else "ToolFailed"
            fields: dict[str, Any] = {"tool": name}
            if event_type == "ToolFailed":
                fields["error_type"] = type(exc).__name__
                fields["latency_ms"] = _latency_ms(started)
            self._store.emit(event_type, **fields)
            raise
        text = _message_text(result)
        self._store.emit(
            "ToolCompleted",
            tool=name,
            latency_ms=_latency_ms(started),
            output_chars=len(text),
            status=_tool_status(result, text),
        )
        return result


def _sanitize(fields: dict[str, Any]) -> dict[str, Any]:
    return {key: _safe_value(key, value) for key, value in fields.items() if value is not None}


def _safe_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    return {key: _safe_value(key, value) for key, value in arguments.items()}


def _safe_value(key: str, value: Any) -> Any:
    lowered = key.lower()
    if any(marker in lowered for marker in ("key", "token", "password", "secret")):
        return "[REDACTED]"
    if isinstance(value, str) and len(value) > 200:
        return value[:200] + "... [TRUNCATED]"
    if isinstance(value, dict):
        return _safe_arguments(value)
    return value


def _model_label(model: Any) -> str:
    return str(
        getattr(model, "model_name", None)
        or getattr(model, "model", None)
        or type(model).__name__
    )


def _is_interrupt(exc: BaseException) -> bool:
    """按类型名识别 LangGraph 的中断/冒泡异常，避免绑定具体版本的类路径。"""

    names = {cls.__name__ for cls in type(exc).__mro__}
    return bool(names & {"GraphInterrupt", "GraphBubbleUp", "ParentCommand"})


def _as_ai_message(result: Any) -> Any:
    """从模型响应中取出 AIMessage。

    不同 LangChain 版本 handler 可能返回 AIMessage、带 ``result``/``message``
    的包装对象，或一列消息。逐种形态提取，否则会漏掉 tool_calls，导致
    ModelCompleted 的 tool_call_count 恒为 0。
    """

    if isinstance(result, AIMessage):
        return result
    for attr in ("result", "message", "messages"):
        item = getattr(result, attr, None)
        if isinstance(item, AIMessage):
            return item
        if isinstance(item, (list, tuple)):
            for sub in reversed(item):
                if isinstance(sub, AIMessage):
                    return sub
    return result


def _content_chars(message: Any) -> int:
    return len(_message_text(message))


def _message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    return str(content or "")


def _tool_status(result: Any, text: str) -> str:
    status = getattr(result, "status", None)
    if status:
        return str(status)
    if isinstance(result, ToolMessage) and text.startswith(("permission_denied", "[stuck]")):
        return "error"
    return "ok"


def _latency_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)

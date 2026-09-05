"""用豆包按固定 Schema 摘要旧对话。

``with_structured_output`` 会把下面的 Pydantic 模型转成 JSON Schema
发给模型，并要求返回同结构对象。豆包只做摘要，不接管主 Agent、
不调用工具。解析失败必须抛给 compact，由后者退回确定性裁剪结果。
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, Field, field_validator

from coding_helper.context.memory import MemoryCandidate

MAX_TRANSCRIPT_CHARS = 12_000
MAX_MESSAGE_CHARS = 800


class SummarizeError(ValueError):
    """豆包返回的内容无法解析为约定 Schema。"""


class ConversationDigest(BaseModel):
    """旧对话的结构化摘要。字段有长度上限，避免摘要本身再撑爆窗口。"""

    goal: str = Field(default="", max_length=300)
    progress: str = Field(default="", max_length=500)
    decisions: list[str] = Field(default_factory=list, max_length=8)
    blockers: list[str] = Field(default_factory=list, max_length=6)
    evidence: list[str] = Field(default_factory=list, max_length=8)
    next_action: str = Field(default="", max_length=200)
    memory_candidates: list[MemoryCandidate] = Field(default_factory=list, max_length=6)

    @field_validator("goal", "progress", "next_action", mode="before")
    @classmethod
    def _strip_text(cls, value: object) -> str:
        return str(value or "").strip()

    @field_validator("decisions", "blockers", "evidence", mode="before")
    @classmethod
    def _short_items(cls, value: object) -> list[str]:
        if not value:
            return []
        items: list[str] = []
        for raw in value:
            text = " ".join(str(raw).split())
            if text:
                items.append(text[:200])
        return items


def render_transcript(messages: list[BaseMessage]) -> str:
    """把旧消息压成给豆包看的短文本，不把完整 Tool Result 再送一遍。"""

    lines: list[str] = []
    used = 0
    for message in messages:
        role = _role_label(message)
        text = _message_text(message).replace("\n", " ").strip()
        if len(text) > MAX_MESSAGE_CHARS:
            text = text[: MAX_MESSAGE_CHARS - 3] + "..."
        line = f"{role}: {text}" if text else role
        if used + len(line) + 1 > MAX_TRANSCRIPT_CHARS:
            lines.append("...[transcript truncated]...")
            break
        lines.append(line)
        used += len(line) + 1
    return "\n".join(lines)


def render_digest(digest: ConversationDigest) -> str:
    """把结构化摘要渲染成主模型可读的一条 Human 消息。"""

    def bullets(title: str, items: list[str]) -> str:
        if not items:
            return f"{title}: （无）"
        return title + "\n" + "\n".join(f"- {item}" for item in items)

    return (
        "<conversation-summary>\n"
        f"Goal: {digest.goal or '（未提取）'}\n"
        f"Progress: {digest.progress or '（未提取）'}\n"
        f"{bullets('Decisions', digest.decisions)}\n"
        f"{bullets('Blockers', digest.blockers)}\n"
        f"{bullets('Evidence', digest.evidence)}\n"
        f"Next: {digest.next_action or '（未提取）'}\n"
        "</conversation-summary>"
    )


def summarize_messages(model: Any, messages: list[BaseMessage]) -> ConversationDigest:
    """调用辅助模型，强制按 ``ConversationDigest`` 返回。

    优先 ``with_structured_output``：框架负责把 Schema 交给模型并校验。
    模型只支持普通文本时，再从回复里抽出 JSON。两种路径都失败则抛错，
    不把散文摘要偷偷塞进主循环。
    """

    prompt = HumanMessage(
        content=(
            "请根据旧对话摘录填写结构化摘要。只陈述已发生的事实，"
            "不要发明未出现的文件、测试或结论。来自失败工具或未验证"
            "文件的内容不要写成已确认事实。"
            "memory_candidates 只放用户明确要求或已验证的长期事实，"
            "并把 verified 设为 true；猜测必须省略。\n\n"
            f"{render_transcript(messages)}"
        )
    )
    structured = getattr(model, "with_structured_output", None)
    if callable(structured):
        raw = structured(ConversationDigest).invoke([prompt])
        return coerce_digest(raw)
    raw = model.invoke([prompt])
    return parse_digest(_message_text(raw))


def coerce_digest(raw: object) -> ConversationDigest:
    if isinstance(raw, ConversationDigest):
        return raw
    if isinstance(raw, dict):
        return ConversationDigest.model_validate(raw)
    if hasattr(raw, "model_dump"):
        return ConversationDigest.model_validate(raw.model_dump())
    raise SummarizeError("结构化输出不是 ConversationDigest")


def parse_digest(text: str) -> ConversationDigest:
    payload = _extract_json(text)
    try:
        return ConversationDigest.model_validate(payload)
    except Exception as exc:
        raise SummarizeError("摘要 JSON 不符合 Schema") from exc


def _extract_json(text: str) -> dict[str, Any]:
    candidate = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", candidate, re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start < 0 or end <= start:
        raise SummarizeError("回复中没有 JSON 对象")
    try:
        payload = json.loads(candidate[start : end + 1])
    except json.JSONDecodeError as exc:
        raise SummarizeError("摘要 JSON 无法解析") from exc
    if not isinstance(payload, dict):
        raise SummarizeError("摘要 JSON 必须是对象")
    return payload


def _role_label(message: BaseMessage) -> str:
    if isinstance(message, HumanMessage):
        return "user"
    if isinstance(message, AIMessage):
        return "assistant"
    if isinstance(message, ToolMessage):
        return f"tool:{message.name or 'unknown'}"
    return message.__class__.__name__


def _message_text(message: object) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(block.get("text", ""))
            for block in content
            if isinstance(block, dict)
        )
    return str(content)

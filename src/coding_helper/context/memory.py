"""保守的项目记忆：豆包只提候选，规则决定能否写入。

``memory.md`` 和 ``progress.md`` 一样是投影，不给模型直接改。来自失败
工具、未验证文件或未确认猜测的内容不能变成长期事实。
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

MAX_MEMORIES = 20
MAX_FACT_CHARS = 200
_REJECT_HINTS = (
    "error",
    "failed",
    "failure",
    "traceback",
    "denied",
    "timeout",
    "untrusted",
    "失败",
    "报错",
    "拒绝",
)
_PREFERENCE_HINTS = ("用户", "请记住", "长期", "prefer", "always", "不要再")


class MemoryKind(str, Enum):
    CONVENTION = "convention"
    PREFERENCE = "preference"
    VERIFIED_COMMAND = "verified_command"
    ARCHITECTURE = "architecture"


class MemoryCandidate(BaseModel):
    """豆包提出的记忆候选。``verified=false`` 的条目会被规则丢弃。"""

    kind: MemoryKind
    fact: str = Field(default="", max_length=MAX_FACT_CHARS)
    evidence: str = Field(default="", max_length=200)
    verified: bool = False

    @field_validator("fact", "evidence", mode="before")
    @classmethod
    def _strip(cls, value: object) -> str:
        return " ".join(str(value or "").split())


class MemoryEntry(BaseModel):
    kind: MemoryKind
    fact: str
    evidence: str = ""


class MemorySnapshot(BaseModel):
    entries: list[MemoryEntry] = Field(default_factory=list)


class MemoryStore:
    """结构化记忆存在 ``memory.json``，再投影成 ``memory.md``。"""

    def __init__(self, workspace: Path) -> None:
        self.runtime_directory = Path(workspace).expanduser().resolve() / ".coding-helper"
        self.state_path = self.runtime_directory / "memory.json"
        self.markdown_path = self.runtime_directory / "memory.md"

    def load(self) -> MemorySnapshot:
        if not self.state_path.is_file():
            return MemorySnapshot()
        try:
            return MemorySnapshot.model_validate_json(
                self.state_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            return MemorySnapshot()

    def absorb(self, candidates: list[MemoryCandidate]) -> list[MemoryEntry]:
        """验收候选：去重、限量，并拒绝未验证或失败证据。"""

        snapshot = self.load()
        known = {_normalize(item.fact) for item in snapshot.entries}
        accepted: list[MemoryEntry] = []
        for candidate in candidates:
            if len(snapshot.entries) >= MAX_MEMORIES:
                break
            if not acceptable_candidate(candidate):
                continue
            key = _normalize(candidate.fact)
            if not key or key in known:
                continue
            entry = MemoryEntry(
                kind=candidate.kind,
                fact=candidate.fact[:MAX_FACT_CHARS],
                evidence=candidate.evidence,
            )
            snapshot.entries.append(entry)
            known.add(key)
            accepted.append(entry)
        if accepted:
            self.save(snapshot)
        return accepted

    def save(self, snapshot: MemorySnapshot) -> None:
        self.runtime_directory.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(snapshot.model_dump(mode="json"), ensure_ascii=False, indent=2)
        self.state_path.write_text(payload + "\n", encoding="utf-8")
        self.markdown_path.write_text(render_memory(snapshot), encoding="utf-8")

    def render(self) -> str:
        snapshot = self.load()
        if not snapshot.entries:
            return ""
        return render_memory(snapshot)


def acceptable_candidate(candidate: MemoryCandidate) -> bool:
    if not candidate.verified or not candidate.fact:
        return False
    haystack = f"{candidate.fact} {candidate.evidence}".casefold()
    if any(hint in haystack for hint in _REJECT_HINTS):
        return False
    if candidate.kind is MemoryKind.PREFERENCE:
        return any(hint.casefold() in haystack for hint in _PREFERENCE_HINTS)
    return True


def render_memory(snapshot: MemorySnapshot) -> str:
    sections = [
        ("Conventions", MemoryKind.CONVENTION),
        ("Preferences", MemoryKind.PREFERENCE),
        ("Verified Commands", MemoryKind.VERIFIED_COMMAND),
        ("Architecture", MemoryKind.ARCHITECTURE),
    ]
    lines = [
        "# Project Memory",
        "",
        "只包含规则验收后的长期事实，失败工具结果不会写入。",
        "",
    ]
    for title, kind in sections:
        items = [item.fact for item in snapshot.entries if item.kind is kind]
        lines.append(f"## {title}")
        if items:
            lines.extend(f"- {fact}" for fact in items)
        else:
            lines.append("- （无）")
        lines.append("")
    return "\n".join(lines)


def attach_project_memory(prompt: str, workspace: Path) -> str:
    """把已有记忆追加到用户消息。没有记忆时不改原文。"""

    rendered = MemoryStore(workspace).render()
    if not rendered:
        return prompt
    return f"{prompt}\n\n<project-memory>\n{rendered.strip()}\n</project-memory>"


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())

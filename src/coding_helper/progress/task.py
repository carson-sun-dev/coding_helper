"""结构化 Todo，以及由状态渲染出的 ``progress.md``。"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from coding_helper.tools.registry import RetryPolicy, ToolRegistry, ToolRisk, coding_tool

MAX_TODOS = 16
MAX_GOAL_CHARS = 300


class TaskError(ValueError):
    """Todo 约束被破坏，例如同时出现多个 in_progress。"""


class TodoStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class SessionPhase(str, Enum):
    PLANNING = "planning"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    PAUSED = "paused"


class TodoItem(BaseModel):
    id: str
    content: str
    status: TodoStatus
    evidence: str = ""


class TodoWriteItem(BaseModel):
    """模型写入时使用的条目。id 由 Store 重新编号，避免模型伪造稳定主键。"""

    content: str = Field(min_length=1, max_length=200)
    status: TodoStatus = TodoStatus.PENDING
    evidence: str = Field(default="", max_length=400)


class TaskSnapshot(BaseModel):
    """Workspace 内当前写入型 Session 的结构化进度。"""

    goal: str
    thread_id: str
    phase: SessionPhase = SessionPhase.PLANNING
    todos: list[TodoItem] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    modified_files: list[str] = Field(default_factory=list)
    latest_verification: str = "not run"
    blockers: list[str] = Field(default_factory=list)
    next_action: str = ""


class TaskStore:
    """把 Todo 存在 ``task.json``，再投影成人类可读的 ``progress.md``。

    ``progress.md`` 不提供给模型直接编辑。模型只能通过 ``todo_write`` 改
    结构化状态；Harness 负责重写 Markdown，这样压缩后仍能按同一来源
    重新注入 Goal、Todo 和 Blocker。
    """

    def __init__(self, workspace: Path) -> None:
        self.runtime_directory = Path(workspace).expanduser().resolve() / ".coding-helper"
        self.state_path = self.runtime_directory / "task.json"
        self.progress_path = self.runtime_directory / "progress.md"

    def load(self) -> TaskSnapshot | None:
        if not self.state_path.is_file():
            return None
        try:
            return TaskSnapshot.model_validate_json(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def ensure_session(self, *, goal: str, thread_id: str) -> TaskSnapshot:
        current = self.load()
        if current is not None and current.thread_id == thread_id:
            return current
        snapshot = TaskSnapshot(goal=_clip(goal, MAX_GOAL_CHARS), thread_id=thread_id)
        self.save(snapshot)
        return snapshot

    def replace_todos(self, items: list[TodoWriteItem]) -> TaskSnapshot:
        if not items:
            raise TaskError("至少需要一条 Todo")
        if len(items) > MAX_TODOS:
            raise TaskError(f"Todo 不能超过 {MAX_TODOS} 条")
        in_progress = [item for item in items if item.status is TodoStatus.IN_PROGRESS]
        if len(in_progress) > 1:
            raise TaskError("同时最多一个 in_progress")

        todos: list[TodoItem] = []
        for index, item in enumerate(items, start=1):
            if item.status is TodoStatus.COMPLETED and not item.evidence.strip():
                raise TaskError(f"完成 {item.content!r} 前必须提供可验证结果")
            todos.append(
                TodoItem(
                    id=f"t{index}",
                    content=item.content.strip(),
                    status=item.status,
                    evidence=item.evidence.strip(),
                )
            )

        snapshot = self.load() or TaskSnapshot(goal="", thread_id="")
        snapshot = snapshot.model_copy(
            update={
                "todos": todos,
                "phase": _phase_from(todos),
                "blockers": _blockers_from(todos),
                "next_action": _next_action(todos),
            }
        )
        self.save(snapshot)
        return snapshot

    def save(self, snapshot: TaskSnapshot) -> None:
        self.runtime_directory.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(snapshot.model_dump(mode="json"), ensure_ascii=False, indent=2)
        self.state_path.write_text(payload + "\n", encoding="utf-8")
        self.progress_path.write_text(render_progress(snapshot), encoding="utf-8")


def render_progress(snapshot: TaskSnapshot) -> str:
    """把结构化状态渲染成固定章节的 Markdown。"""

    completed = sum(item.status is TodoStatus.COMPLETED for item in snapshot.todos)
    total = len(snapshot.todos)
    status = (
        f"{snapshot.phase.value} — {completed}/{total} tasks completed"
        if total
        else snapshot.phase.value
    )
    tasks = [_format_task(item) for item in snapshot.todos] or ["- (尚未拆分)"]
    return "\n".join(
        [
            "# Goal",
            snapshot.goal or "(未设置)",
            "",
            "## Status",
            status,
            "",
            "## Tasks",
            *tasks,
            "",
            "## Decisions",
            *_bullets(snapshot.decisions),
            "",
            "## Modified Files",
            *_bullets(snapshot.modified_files),
            "",
            "## Latest Verification",
            snapshot.latest_verification or "not run",
            "",
            "## Blockers",
            *_bullets(snapshot.blockers),
            "",
            "## Next Action",
            snapshot.next_action or "拆分或更新 Todo",
            "",
        ]
    )


def register_todo_tools(workspace: Path, registry: ToolRegistry) -> None:
    store = TaskStore(workspace)

    @coding_tool(
        risk=ToolRisk.WRITE,
        idempotent=False,
        retry_policy=RetryPolicy.NEVER,
        tags=("todo", "progress"),
    )
    def todo_write(items: list[TodoWriteItem]) -> str:
        """用完整 Todo 列表替换当前进度，并刷新 progress.md。

        同时只能有一个 in_progress。completed 必须填写 evidence，例如测试
        输出、文件路径或检查结论。不要直接编辑 progress.md。
        """

        snapshot = store.replace_todos(items)
        return render_progress(snapshot)

    registry.register(todo_write)


def _clip(text: str, limit: int) -> str:
    compact = " ".join(text.strip().split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1] + "…"


def _phase_from(todos: list[TodoItem]) -> SessionPhase:
    if any(item.status is TodoStatus.BLOCKED for item in todos):
        return SessionPhase.PAUSED
    if any(item.status is TodoStatus.IN_PROGRESS for item in todos):
        return SessionPhase.EXECUTING
    if todos and all(
        item.status in {TodoStatus.COMPLETED, TodoStatus.CANCELLED} for item in todos
    ):
        return SessionPhase.VERIFYING
    if any(item.status is TodoStatus.PENDING for item in todos):
        return SessionPhase.EXECUTING
    return SessionPhase.PLANNING


def _blockers_from(todos: list[TodoItem]) -> list[str]:
    blockers = []
    for item in todos:
        if item.status is TodoStatus.BLOCKED:
            detail = f"{item.content}：{item.evidence}" if item.evidence else item.content
            blockers.append(detail)
    return blockers


def _next_action(todos: list[TodoItem]) -> str:
    for item in todos:
        if item.status is TodoStatus.IN_PROGRESS:
            return item.content
    for item in todos:
        if item.status is TodoStatus.PENDING:
            return item.content
    return ""


def _format_task(item: TodoItem) -> str:
    checked = item.status in {TodoStatus.COMPLETED, TodoStatus.CANCELLED}
    mark = "[x]" if checked else "[ ]"
    label = f"~~{item.content}~~" if item.status is TodoStatus.CANCELLED else item.content
    suffix = ""
    if item.status is TodoStatus.IN_PROGRESS:
        suffix = " *(in_progress)*"
    elif item.status is TodoStatus.BLOCKED:
        suffix = " *(blocked)*"
    elif item.status is TodoStatus.COMPLETED and item.evidence:
        suffix = f" — {item.evidence}"
    return f"- {mark} {label}{suffix}"


def _bullets(values: list[str]) -> list[str]:
    return [f"- {item}" for item in values] or ["- None"]

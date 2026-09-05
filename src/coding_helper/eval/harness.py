"""内置冒烟任务：用 Fake Model 验证写路径、轨迹和自动验收。

这不是模型能力评测。样本为 1 个任务时只报告原始次数，不写成功率百分比。
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from pydantic import Field

from coding_helper.config import Settings
from coding_helper.models import ModelTarget
from coding_helper.observe.events import EventStore
from coding_helper.runtime import run_coding_task

BUGGY_ADD = "def add(left, right):\n    return left - right\n"
FIXED_ADD = "def add(left, right):\n    return left + right\n"
FIX_ADD_PROMPT = "修复 app.py 里 add 把加法做成减法的错误，使 add(2, 3) 等于 5。"


class ScriptedCodingModel(FakeMessagesListChatModel):
    bound_tool_names: list[str] = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        self.bound_tool_names = [item.name for item in tools]
        return self


@dataclass(frozen=True)
class EvalResult:
    name: str
    passed: bool
    model_calls: int
    tool_calls: int
    approvals: int
    denials: int
    duration_ms: int
    event_count: int
    detail: str


def run_fix_add_eval(workspace: Path) -> EvalResult:
    """在隔离目录运行 add 修复任务，自动批准写操作。"""

    root = Path(workspace).expanduser().resolve() / ".coding-helper" / "eval" / "fix_add"
    if root.exists():
        for child in root.rglob("*"):
            if child.is_file():
                child.unlink()
    root.mkdir(parents=True, exist_ok=True)
    (root / "app.py").write_text(BUGGY_ADD, encoding="utf-8")
    digest = hashlib.sha256(BUGGY_ADD.encode("utf-8")).hexdigest()
    model = ScriptedCodingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "replace_text",
                        "args": {
                            "path": "app.py",
                            "old_text": "left - right",
                            "new_text": "left + right",
                            "expected_sha256": digest,
                        },
                        "id": "eval-fix",
                    }
                ],
            ),
            AIMessage(content="已把减法改回加法。"),
        ]
    )
    settings = Settings(
        _env_file=None,
        workspace=root,
        completion_enabled=False,
    )
    started = time.perf_counter()
    run_coding_task(
        FIX_ADD_PROMPT,
        settings=settings,
        target=ModelTarget.DEEPSEEK,
        approval_handler=lambda _pending: "approve",
        thread_id="eval-fix-add",
        chat_model=model,
    )
    duration_ms = int((time.perf_counter() - started) * 1000)
    passed, detail = _accept_add(root)
    events = EventStore(root, "eval-fix-add").load("eval-fix-add")
    return EvalResult(
        name="fix_add",
        passed=passed,
        model_calls=sum(item.get("type") == "ModelCompleted" for item in events),
        tool_calls=sum(item.get("type") == "ToolCompleted" for item in events),
        approvals=sum(item.get("type") == "ToolApproved" for item in events),
        denials=sum(item.get("type") == "ToolDenied" for item in events),
        duration_ms=duration_ms,
        event_count=len(events),
        detail=detail,
    )


def _accept_add(workspace: Path) -> tuple[bool, str]:
    source = workspace / "app.py"
    if not source.is_file():
        return False, "app.py 不存在"
    namespace: dict[str, object] = {}
    exec(source.read_text(encoding="utf-8"), namespace)
    add = namespace.get("add")
    if not callable(add):
        return False, "没有 add 函数"
    try:
        value = add(2, 3)
    except Exception as exc:
        return False, f"调用失败：{type(exc).__name__}"
    if value != 5:
        return False, f"add(2, 3)={value!r}"
    if (workspace / ".env").exists():
        return False, "越权写入 .env"
    return True, "add(2, 3)==5"

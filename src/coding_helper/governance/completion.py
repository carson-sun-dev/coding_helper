"""Coding 任务的确定性完成门槛。

模型提出结束后由代码判定：diff 范围、删除、密钥文件、Todo 和配置的
测试/lint；失败结果有界打回主循环，不无限修。确定性检查全部通过后，
可选地再让 Reviewer Subagent 读一遍 diff——它只产出补充意见写进
progress.md，绝不改变放行结论，失败也不阻断完成。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage

from coding_helper.config import Settings
from coding_helper.observe.events import EventStore
from coding_helper.progress.task import TaskStore, TodoStatus
from coding_helper.tools.gitdiff import collect_workspace_diff, is_forbidden_path
from coding_helper.tools.shell import run_workspace_command


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class CompletionReport:
    checks: tuple[CheckResult, ...]

    @property
    def passed(self) -> bool:
        return all(item.passed for item in self.checks)


def evaluate_completion(workspace: Path, settings: Settings) -> CompletionReport:
    diff = collect_workspace_diff(workspace)
    prefixes = [
        part.strip().rstrip("/")
        for part in settings.completion_allowed_prefixes.split(",")
        if part.strip()
    ]
    snapshot = TaskStore(workspace).load()
    checks = [
        _check_diff(diff, prefixes),
        _check_todos(snapshot.todos if snapshot else []),
    ]
    if settings.completion_test_command.strip():
        checks.append(_check_command("tests", settings.completion_test_command, workspace))
    if settings.completion_lint_command.strip():
        checks.append(_check_command("lint", settings.completion_lint_command, workspace))
    return CompletionReport(tuple(checks))


def render_report(report: CompletionReport) -> str:
    title = "确定性验收通过。" if report.passed else "确定性验收未通过，请根据失败项修复后再次结束。"
    lines = ["<completion-gate>", title]
    for item in report.checks:
        mark = "PASS" if item.passed else "FAIL"
        lines.append(f"- [{mark}] {item.name}: {item.detail}")
    lines.append("</completion-gate>")
    return "\n".join(lines)


class CompletionGateMiddleware(AgentMiddleware):
    """模型不再请求工具时运行门槛；失败则跳回模型，次数用尽则结束。"""

    tools: list = []

    def __init__(
        self,
        workspace: Path,
        settings: Settings,
        *,
        review_model: BaseChatModel | None = None,
        event_store: EventStore | None = None,
    ) -> None:
        super().__init__()
        self._workspace = workspace
        self._settings = settings
        # review_model 为 None 时完全跳过 Reviewer；由 Harness 决定是否传入，
        # 这样默认路径不会为每次结束多打一次主模型。
        self._review_model = review_model
        self._event_store = event_store
        self._repairs = 0

    @hook_config(can_jump_to=["model", "end"])
    def after_model(self, state, runtime):
        last = (state.get("messages") or [None])[-1]
        if not isinstance(last, AIMessage) or last.tool_calls:
            return None
        report = evaluate_completion(self._workspace, self._settings)
        store = TaskStore(self._workspace)
        store.record_verification(
            "passed" if report.passed else "failed",
            files=_changed_paths(self._workspace),
        )
        if report.passed:
            self._run_review()
            return None
        if self._repairs >= self._settings.max_repair_rounds:
            store.add_blocker("完成门槛超过最大修复轮数")
            return {
                "messages": [HumanMessage(content=render_report(report))],
                "jump_to": "end",
            }
        self._repairs += 1
        return {
            "messages": [HumanMessage(content=render_report(report))],
            "jump_to": "model",
        }

    def _run_review(self) -> None:
        """确定性检查通过后跑一层 Reviewer；只记录意见，绝不影响放行。"""

        if self._review_model is None:
            return
        diff = collect_workspace_diff(self._workspace)
        if not diff.available or not diff.files:
            return
        # 延迟导入：subagent 依赖 governance，模块顶层导入会形成环。
        from coding_helper.agents.subagent import SubagentRole, run_subagent

        self._emit("SubagentStarted", role="reviewer")
        try:
            notes = run_subagent(
                role=SubagentRole.REVIEWER,
                task=_review_task(diff),
                workspace=self._workspace,
                model=self._review_model,
            )
        except Exception as exc:  # Reviewer 只是补充意见，任何异常都不能阻断完成。
            notes = f"role=reviewer status=failed error={type(exc).__name__}"
        TaskStore(self._workspace).record_review(notes)
        self._emit("SubagentCompleted", role="reviewer", status=_review_status(notes))

    def _emit(self, event_type: str, **fields) -> None:
        if self._event_store is not None:
            self._event_store.emit(event_type, **fields)


def _review_status(notes: str) -> str:
    """从 Subagent 结果头解析 completed / failed，供轨迹统计，不含正文。"""

    return "failed" if "status=failed" in notes else "completed"


def _review_task(diff) -> str:
    files = ", ".join(diff.paths[:12]) or "(无)"
    return (
        "审查本次改动的风险与测试缺口。"
        f"改动文件：{files}。"
        "请读取这些文件核对实现与测试是否一致，只给结论，不要修改仓库。"
    )[:480]


def _check_diff(diff, prefixes: list[str]) -> CheckResult:
    if not diff.available:
        return CheckResult("diff", True, diff.summary)
    deletions = [item.path for item in diff.files if "D" in item.status]
    forbidden = [item.path for item in diff.files if is_forbidden_path(item.path)]
    binaries = [item.path for item in diff.files if item.binary]
    outside = [
        item.path
        for item in diff.files
        if prefixes and not any(item.path == prefix or item.path.startswith(prefix + "/") for prefix in prefixes)
    ]
    failures: list[str] = []
    if deletions:
        failures.append("删除: " + ", ".join(deletions))
    if forbidden:
        failures.append("禁止修改: " + ", ".join(forbidden))
    if binaries:
        failures.append("二进制: " + ", ".join(binaries))
    if outside:
        failures.append("超出范围: " + ", ".join(outside))
    if failures:
        return CheckResult("diff", False, "; ".join(failures))
    changed = ", ".join(item.path for item in diff.files) or "无变更"
    return CheckResult("diff", True, changed)


def _check_todos(todos) -> CheckResult:
    if not todos:
        return CheckResult("todos", True, "没有 Todo")
    open_items = [
        item.content
        for item in todos
        if item.status not in {TodoStatus.COMPLETED, TodoStatus.CANCELLED}
    ]
    if open_items:
        return CheckResult("todos", False, "未完成: " + ", ".join(open_items))
    return CheckResult("todos", True, f"{len(todos)} 项已完成或取消")


def _check_command(name: str, command: str, workspace: Path) -> CheckResult:
    output = run_workspace_command(command, workspace, timeout_seconds=60)
    first = output.splitlines()[0] if output else ""
    passed = "exit_code=0" in first and "timed_out=false" in first
    return CheckResult(name, passed, first or "无输出")


def _changed_paths(workspace: Path) -> list[str]:
    return collect_workspace_diff(workspace).paths

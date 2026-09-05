"""受 Workspace 约束的受治理 Shell 工具。"""

from __future__ import annotations

import os
import re
import signal
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from uuid import uuid4

from coding_helper.tools.registry import RetryPolicy, ToolRegistry, ToolRisk, coding_tool
from coding_helper.tools.workspace import WorkspaceBoundary

MAX_VISIBLE_OUTPUT_CHARS = 8_000
DEFAULT_TIMEOUT_SECONDS = 30
MAX_TIMEOUT_SECONDS = 120
ENVIRONMENT_ALLOWLIST = {
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "LOGNAME",
    "PATH",
    "TERM",
    "TMPDIR",
    "TZ",
    "USER",
}

_DENIED_PATTERNS = (
    re.compile(r"\brm\s+-([A-Za-z]*r[A-Za-z]*f|[A-Za-z]*f[A-Za-z]*r)\b"),
    re.compile(r"\bgit\s+reset\s+--hard\b"),
    re.compile(r"\bgit\s+clean\b"),
    re.compile(r"\bfind\b.*-delete\b"),
    re.compile(r"\b(curl|wget)\b.*\|\s*(sh|bash)\b"),
    re.compile(r"\bsudo\b"),
    re.compile(r"\bchmod\s+777\b"),
    re.compile(r"\bmkfs\b"),
    re.compile(r"\bdd\s+if="),
    re.compile(r":\(\)\s*\{"),
)
_READONLY_PREFIXES = (
    "git status",
    "git diff",
    "git log",
    "git show",
    "git rev-parse",
    "git branch",
    "pytest",
    "python -m pytest",
    "python3 -m pytest",
    "ruff check",
    "mypy",
    "ls",
    "pwd",
)
_INTERACTIVE_COMMANDS = {"bash", "python", "python3", "node", "sh", "zsh", "irb"}


class ShellAction(str, Enum):
    """与 PermissionAction 对齐的命令分类，避免循环导入。"""

    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


class ShellPolicyError(ValueError):
    """命令在执行前就被策略拒绝。"""


@dataclass(frozen=True)
class ShellClassification:
    """把命令字符串映射到权限动作，供 Policy 和工具双重检查。"""

    action: ShellAction
    reason: str


def classify_shell_command(command: str) -> ShellClassification:
    """对命令做静态分类。这不是完整沙箱，只拦截常见高风险写法。"""

    compact = " ".join(command.strip().split())
    if not compact:
        return ShellClassification(ShellAction.DENY, "命令不能为空")
    if compact in _INTERACTIVE_COMMANDS or compact.endswith(" -i"):
        return ShellClassification(ShellAction.DENY, "拒绝可能无限等待的交互式命令")
    for pattern in _DENIED_PATTERNS:
        if pattern.search(compact):
            return ShellClassification(
                ShellAction.DENY,
                f"命令匹配破坏性模式：{pattern.pattern}",
            )
    lowered = compact.lower()
    if any(lowered == prefix or lowered.startswith(prefix + " ") for prefix in _READONLY_PREFIXES):
        return ShellClassification(
            ShellAction.ALLOW,
            "只读或测试命令，允许自动执行",
        )
    return ShellClassification(ShellAction.ASK, "命令可能产生副作用或未知影响")


def _filtered_environment() -> dict[str, str]:
    return {
        key: value
        for key, value in os.environ.items()
        if key in ENVIRONMENT_ALLOWLIST
    }


def _terminate_group(process: subprocess.Popen[bytes]) -> None:
    """终止整个进程组，避免 timeout 后留下后台子进程。"""

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.communicate(timeout=2)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()


def run_workspace_command(
    command: str,
    workspace: Path,
    *,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> str:
    """在 Workspace 内执行命令，并返回带截断提示的文本结果。"""

    if not 0 < timeout_seconds <= MAX_TIMEOUT_SECONDS:
        raise ShellPolicyError(f"timeout_seconds 必须在 0 到 {MAX_TIMEOUT_SECONDS} 之间")
    classification = classify_shell_command(command)
    if classification.action is ShellAction.DENY:
        raise ShellPolicyError(classification.reason)

    boundary = WorkspaceBoundary(workspace)
    runtime_directory = boundary.root / ".coding-helper" / "outputs"
    runtime_directory.mkdir(parents=True, exist_ok=True)
    output_id = uuid4().hex
    output_path = runtime_directory / f"{output_id}.txt"
    process = subprocess.Popen(
        ["/bin/bash", "-lc", command],
        cwd=str(boundary.root),
        env=_filtered_environment(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    timed_out = False
    try:
        raw_output, _ = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        _terminate_group(process)
        raw_output = exc.output or b""
    decoded = raw_output.decode("utf-8", errors="replace")
    output_path.write_text(decoded, encoding="utf-8")
    visible = decoded[:MAX_VISIBLE_OUTPUT_CHARS]
    truncated = len(decoded) > MAX_VISIBLE_OUTPUT_CHARS
    header = (
        f"exit_code={process.returncode} timed_out={str(timed_out).lower()} "
        f"cwd={boundary.root.as_posix()} truncated={str(truncated).lower()} "
        f"output=.coding-helper/outputs/{output_id}.txt"
    )
    return f"{header}\n{visible}"


def register_shell_tools(workspace: Path, registry: ToolRegistry) -> None:
    """把受治理 Shell 工具加入已有 Session Registry。"""

    @coding_tool(
        risk=ToolRisk.EXECUTE,
        idempotent=False,
        timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
        retry_policy=RetryPolicy.NEVER,
        tags=("shell", "execute"),
    )
    def shell(command: str, timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS) -> str:
        """在 Workspace 中执行一条 Shell 命令，并返回退出码与截断后的输出。"""

        return run_workspace_command(
            command,
            workspace,
            timeout_seconds=timeout_seconds,
        )

    registry.register(shell)

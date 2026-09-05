"""只读 Git 工作区快照，供完成门槛和模型查看 diff。"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from coding_helper.tools.registry import RetryPolicy, ToolRegistry, ToolRisk, coding_tool
from coding_helper.tools.workspace import WorkspaceBoundary

_RUNTIME_PREFIX = ".coding-helper/"
_FORBIDDEN_NAMES = {".env"}
_FORBIDDEN_SUFFIXES = (".env", ".pem", "credentials.json")


@dataclass(frozen=True)
class ChangedFile:
    path: str
    status: str
    binary: bool = False


@dataclass(frozen=True)
class WorkspaceDiff:
    available: bool
    files: tuple[ChangedFile, ...]
    summary: str

    @property
    def paths(self) -> list[str]:
        return [item.path for item in self.files]


def collect_workspace_diff(workspace: Path) -> WorkspaceDiff:
    root = Path(workspace).expanduser().resolve()
    # 只认 Workspace 自己的 .git，避免爬到开发者外层仓库。
    if not (root / ".git").exists():
        return WorkspaceDiff(False, (), "不是 Git 仓库，跳过 diff 检查")

    code, porcelain = _git(root, "status", "--porcelain=v1")
    if code != 0:
        return WorkspaceDiff(False, (), porcelain.strip() or "git status 失败")

    binaries = _binary_paths(root)
    files: list[ChangedFile] = []
    for raw in porcelain.splitlines():
        if len(raw) < 4:
            continue
        path = raw[3:].split(" -> ")[-1].strip()
        if not path or path.startswith(_RUNTIME_PREFIX):
            continue
        status = raw[:2].replace(" ", "")
        files.append(ChangedFile(path=path, status=status or "M", binary=path in binaries))
    _, stat = _git(root, "diff", "--stat", "HEAD")
    summary = stat.strip() or porcelain.strip() or "working tree clean"
    return WorkspaceDiff(True, tuple(files), summary)


def is_forbidden_path(path: str) -> bool:
    name = Path(path).name
    lowered = path.replace("\\", "/")
    if name in _FORBIDDEN_NAMES:
        return True
    return any(lowered.endswith(suffix) for suffix in _FORBIDDEN_SUFFIXES)


def register_git_diff_tools(workspace: Path, registry: ToolRegistry) -> None:
    @coding_tool(
        risk=ToolRisk.READ,
        idempotent=True,
        retry_policy=RetryPolicy.SAFE,
        tags=("git", "diff"),
    )
    def git_diff() -> str:
        """返回当前工作区相对 HEAD 的变更清单，不修改仓库。"""

        snapshot = collect_workspace_diff(workspace)
        if not snapshot.available:
            return snapshot.summary
        if not snapshot.files:
            return "working tree clean"
        lines = [snapshot.summary, ""]
        for item in snapshot.files:
            mark = " binary" if item.binary else ""
            lines.append(f"{item.status} {item.path}{mark}")
        return "\n".join(lines)

    registry.register(git_diff)


def _binary_paths(root: Path) -> set[str]:
    found: set[str] = set()
    _, numstat = _git(root, "diff", "--numstat", "HEAD")
    for line in numstat.splitlines():
        parts = line.split("\t")
        if len(parts) >= 3 and parts[0] == "-" and parts[1] == "-":
            found.add(parts[2])
    return found


def _git(workspace: Path, *args: str) -> tuple[int, str]:
    try:
        process = subprocess.run(
            ["git", *args],
            cwd=str(WorkspaceBoundary(workspace).root),
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, str(exc)
    return process.returncode, (process.stdout or "") + (process.stderr or "")

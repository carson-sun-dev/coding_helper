"""受 Workspace 边界保护的内置只读文件工具。"""

import fnmatch
import os
from pathlib import Path
from typing import Iterator

from coding_helper.tools.registry import RetryPolicy, ToolRegistry, ToolRisk, coding_tool
from coding_helper.tools.workspace import WorkspaceBoundary, WorkspaceViolation

MAX_TEXT_FILE_BYTES = 1_000_000
IGNORED_DIRECTORIES = {
    ".coding-helper",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "node_modules",
}


def _read_text(path: Path) -> str:
    """读取受限大小的 UTF-8 文本，并明确拒绝二进制输入。"""

    if path.stat().st_size > MAX_TEXT_FILE_BYTES:
        raise ValueError(f"文件超过 {MAX_TEXT_FILE_BYTES} 字节限制")
    data = path.read_bytes()
    if b"\x00" in data:
        raise ValueError("检测到二进制文件")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("文件不是有效 UTF-8 文本") from exc


def _iter_files(boundary: WorkspaceBoundary, directory: Path) -> Iterator[Path]:
    """遍历普通文件，不跟随目录符号链接和常见依赖/缓存目录。"""

    for current_root, directory_names, file_names in os.walk(directory, followlinks=False):
        current = Path(current_root)
        directory_names[:] = [
            name
            for name in directory_names
            if name not in IGNORED_DIRECTORIES and not (current / name).is_symlink()
        ]
        for name in sorted(file_names):
            candidate = current / name
            if candidate.is_symlink():
                continue
            try:
                yield boundary.resolve(str(candidate), expected="file")
            except WorkspaceViolation:
                # 搜索目录时跳过敏感文件；用户直接读取时仍返回明确拒绝原因。
                continue


def create_filesystem_registry(workspace: Path) -> ToolRegistry:
    """为指定 Workspace 创建三个只读工具及其独立 Registry。

    工具通过闭包共享同一个不可变根目录，模型参数中不会出现修改 Workspace
    的入口。不同 Session 可以各自创建 Registry，避免路径状态互相污染。
    """

    boundary = WorkspaceBoundary(workspace)
    registry = ToolRegistry()

    @coding_tool(
        risk=ToolRisk.READ,
        idempotent=True,
        retry_policy=RetryPolicy.SAFE,
        tags=("filesystem", "read"),
    )
    def read_file(path: str, start_line: int = 1, max_lines: int = 200) -> str:
        """读取 Workspace 内 UTF-8 文本文件的指定行，并返回稳定行号。"""

        if start_line < 1 or not 1 <= max_lines <= 1000:
            raise ValueError("start_line 必须大于 0，max_lines 必须在 1–1000 之间")
        resolved = boundary.resolve(path, expected="file")
        lines = _read_text(resolved).splitlines()
        selected = lines[start_line - 1 : start_line - 1 + max_lines]
        body = "\n".join(
            f"{line_number}: {line}"
            for line_number, line in enumerate(selected, start=start_line)
        )
        end_line = start_line + len(selected) - 1
        truncated = end_line < len(lines)
        header = (
            f"source={boundary.relative(resolved)} "
            f"lines={start_line}-{max(end_line, start_line - 1)} "
            f"total={len(lines)} truncated={str(truncated).lower()}"
        )
        return f"{header}\n{body}" if body else header

    @coding_tool(
        risk=ToolRisk.READ,
        idempotent=True,
        retry_policy=RetryPolicy.SAFE,
        tags=("filesystem", "directory"),
    )
    def list_directory(path: str = ".", max_depth: int = 2, max_entries: int = 200) -> str:
        """列出目录结构，不读取文件正文，也不递归符号链接。"""

        if not 0 <= max_depth <= 5 or not 1 <= max_entries <= 1000:
            raise ValueError("max_depth 必须在 0–5，max_entries 必须在 1–1000")
        root = boundary.resolve(path, expected="directory")
        entries: list[str] = []

        for current_root, directory_names, file_names in os.walk(root, followlinks=False):
            current = Path(current_root)
            depth = len(current.relative_to(root).parts)
            directory_names[:] = sorted(
                name
                for name in directory_names
                if name not in IGNORED_DIRECTORIES and not (current / name).is_symlink()
            )
            if depth >= max_depth:
                directory_names[:] = []

            for name in [*(f"{item}/" for item in directory_names), *sorted(file_names)]:
                candidate = current / name.rstrip("/")
                try:
                    boundary.resolve(str(candidate))
                except WorkspaceViolation:
                    continue
                relative = candidate.relative_to(root).as_posix()
                entries.append(f"{relative}{'/' if name.endswith('/') else ''}")
                if len(entries) >= max_entries:
                    return "\n".join([*entries, f"... entries truncated at {max_entries}"])

        return "\n".join(entries) if entries else "(empty directory)"

    @coding_tool(
        risk=ToolRisk.READ,
        idempotent=True,
        retry_policy=RetryPolicy.SAFE,
        tags=("filesystem", "search"),
    )
    def search_text(
        query: str,
        path: str = ".",
        file_glob: str = "*",
        max_results: int = 50,
        case_sensitive: bool = False,
    ) -> str:
        """在 Workspace 文本文件中执行字面量搜索并返回文件、行号和片段。"""

        if not query:
            raise ValueError("query 不能为空")
        if not 1 <= max_results <= 500:
            raise ValueError("max_results 必须在 1–500 之间")
        root = boundary.resolve(path)
        files = [root] if root.is_file() else _iter_files(boundary, root)
        needle = query if case_sensitive else query.casefold()
        matches: list[str] = []

        for file_path in files:
            relative = boundary.relative(file_path)
            if not fnmatch.fnmatch(relative, file_glob):
                continue
            try:
                lines = _read_text(file_path).splitlines()
            except (OSError, ValueError):
                continue
            for line_number, line in enumerate(lines, start=1):
                haystack = line if case_sensitive else line.casefold()
                if needle in haystack:
                    matches.append(f"{relative}:{line_number}: {line.strip()[:500]}")
                    if len(matches) >= max_results:
                        return "\n".join([*matches, f"... results truncated at {max_results}"])

        return "\n".join(matches) if matches else "No matches found"

    registry.register(read_file)
    registry.register(list_directory)
    registry.register(search_text)
    return registry

"""带冲突检测、Preimage 和原子替换的安全文本修改工具。"""

import hashlib
import json
import os
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from coding_helper.tools.registry import RetryPolicy, ToolRegistry, ToolRisk, coding_tool
from coding_helper.tools.workspace import WorkspaceBoundary

MAX_WRITE_BYTES = 1_000_000


class FileWriteError(ValueError):
    """文件无法安全修改或前置条件已经失效。"""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class SafeFileEditor:
    """管理现有文本文件的一次可恢复替换。

    Preimage 是修改前的原始字节副本。它既支持后续 Undo，也让程序崩溃后
    能判断文件处于修改前还是修改后。Journal 先写 pending，再写 completed；
    如果只留下 pending，恢复流程就知道本次操作可能在中途被打断。
    """

    def __init__(self, workspace: Path) -> None:
        self.boundary = WorkspaceBoundary(workspace)
        self.runtime_directory = self.boundary.root / ".coding-helper"
        self.backup_directory = self.runtime_directory / "backups"
        self.journal_path = self.runtime_directory / "operations.jsonl"

    def file_hash(self, user_path: str) -> dict[str, Any]:
        """返回现有文件 Hash，供模型建立乐观并发前置条件。"""

        path = self.boundary.resolve(user_path, expected="file")
        data = self._read_source(path)
        return {
            "path": self.boundary.relative(path),
            "sha256": _sha256(data),
            "bytes": len(data),
        }

    def replace_text(
        self,
        *,
        user_path: str,
        old_text: str,
        new_text: str,
        expected_sha256: str,
    ) -> dict[str, Any]:
        """唯一替换一段文本，并在 Hash 不一致时拒绝覆盖。"""

        if not old_text:
            raise FileWriteError("old_text 不能为空")
        path = self.boundary.resolve_existing_file_for_write(user_path)
        original = self._read_source(path)
        original_hash = _sha256(original)
        if original_hash != expected_sha256:
            raise FileWriteError(
                f"文件已发生变化：expected={expected_sha256}, actual={original_hash}"
            )

        text = self._decode(original)
        occurrences = text.count(old_text)
        if occurrences != 1:
            raise FileWriteError(f"old_text 必须唯一出现，当前出现 {occurrences} 次")
        updated = text.replace(old_text, new_text, 1).encode("utf-8")
        if len(updated) > MAX_WRITE_BYTES:
            raise FileWriteError(f"修改后文件超过 {MAX_WRITE_BYTES} 字节限制")

        operation_id = uuid4().hex
        relative_path = self.boundary.relative(path)
        self.backup_directory.mkdir(parents=True, exist_ok=True)
        backup_path = self.backup_directory / f"{operation_id}.bin"
        self._write_backup(backup_path, original)
        common_record = {
            "operation_id": operation_id,
            "operation": "replace_text",
            "path": relative_path,
            "before_sha256": original_hash,
            "after_sha256": _sha256(updated),
            "before_mode": stat.S_IMODE(path.stat().st_mode),
            "backup": backup_path.relative_to(self.runtime_directory).as_posix(),
        }
        self._append_journal({**common_record, "status": "pending"})

        try:
            self._atomic_replace(path, updated)
        except Exception:
            self._append_journal({**common_record, "status": "failed"})
            raise

        self._append_journal({**common_record, "status": "completed"})
        return {
            **common_record,
            "status": "completed",
            "replacements": 1,
        }

    @staticmethod
    def _read_source(path: Path) -> bytes:
        data = path.read_bytes()
        if len(data) > MAX_WRITE_BYTES:
            raise FileWriteError(f"文件超过 {MAX_WRITE_BYTES} 字节限制")
        if b"\x00" in data:
            raise FileWriteError("不能修改二进制文件")
        SafeFileEditor._decode(data)
        return data

    @staticmethod
    def _decode(data: bytes) -> str:
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise FileWriteError("文件不是有效 UTF-8 文本") from exc

    @staticmethod
    def _write_backup(path: Path, data: bytes) -> None:
        # Preimage 可能包含源码中的敏感信息，创建瞬间就限制为仅当前用户可读写。
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as backup:
            backup.write(data)
            backup.flush()
            os.fsync(backup.fileno())

    @staticmethod
    def _atomic_replace(path: Path, data: bytes) -> None:
        """在目标目录写临时文件，再原子替换并保留 Unix 权限位。"""

        mode = stat.S_IMODE(path.stat().st_mode)
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=".coding-helper-",
                dir=path.parent,
                delete=False,
            ) as temporary:
                temporary_name = temporary.name
                temporary.write(data)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.chmod(temporary_name, mode)
            os.replace(temporary_name, path)
        finally:
            if temporary_name and os.path.exists(temporary_name):
                os.unlink(temporary_name)

    def _append_journal(self, record: dict[str, Any]) -> None:
        self.runtime_directory.mkdir(parents=True, exist_ok=True)
        event = {
            **record,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with self.journal_path.open("a", encoding="utf-8") as journal:
            journal.write(json.dumps(event, ensure_ascii=False) + "\n")
            journal.flush()
            os.fsync(journal.fileno())


def register_write_tools(workspace: Path, registry: ToolRegistry) -> SafeFileEditor:
    """把 Hash 查询和安全替换工具加入已有 Session Registry。"""

    editor = SafeFileEditor(workspace)

    @coding_tool(
        risk=ToolRisk.READ,
        idempotent=True,
        retry_policy=RetryPolicy.SAFE,
        tags=("filesystem", "hash"),
    )
    def get_file_hash(path: str) -> dict[str, Any]:
        """返回 Workspace 文件的 SHA-256 和字节数，修改前必须先调用。"""

        return editor.file_hash(path)

    @coding_tool(
        risk=ToolRisk.WRITE,
        idempotent=False,
        retry_policy=RetryPolicy.NEVER,
        tags=("filesystem", "edit"),
    )
    def replace_text(
        path: str,
        old_text: str,
        new_text: str,
        expected_sha256: str,
    ) -> dict[str, Any]:
        """在 Hash 匹配时唯一替换 Workspace 文本，并保存可恢复 Preimage。"""

        return editor.replace_text(
            user_path=path,
            old_text=old_text,
            new_text=new_text,
            expected_sha256=expected_sha256,
        )

    registry.register(get_file_hash)
    registry.register(replace_text)
    return editor

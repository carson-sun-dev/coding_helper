"""带冲突检测、Preimage 和原子替换的安全文本修改工具。"""

import hashlib
import json
import os
import stat
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from coding_helper.tools.registry import RetryPolicy, ToolRegistry, ToolRisk, coding_tool
from coding_helper.tools.workspace import WorkspaceBoundary

MAX_WRITE_BYTES = 1_000_000


class FileWriteError(ValueError):
    """文件无法安全修改或前置条件已经失效。"""


@dataclass(frozen=True)
class UndoCandidate:
    """CLI 在真正恢复前展示的操作摘要。"""

    operation_id: str
    path: str
    before_sha256: str
    after_sha256: str


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

    def create_file(self, *, user_path: str, content: str) -> dict[str, Any]:
        """创建一个新文件并登记可恢复操作；目标已存在时拒绝。

        新建文件没有 Preimage（原本不存在），因此 Undo 通过把文件移入
        trash 实现，而不是恢复旧字节。这样模型创建文件也走同一条可回滚、
        可审计的副作用通道，不必退回 shell 重定向。
        """

        data = content.encode("utf-8")
        if len(data) > MAX_WRITE_BYTES:
            raise FileWriteError(f"内容超过 {MAX_WRITE_BYTES} 字节限制")
        if b"\x00" in data:
            raise FileWriteError("不能写入二进制内容")

        path = self.boundary.resolve_new_file_for_write(user_path)
        relative_path = self.boundary.relative(path)
        operation_id = uuid4().hex
        common_record = {
            "operation_id": operation_id,
            "operation": "create_file",
            "path": relative_path,
            "before_sha256": "",  # 创建前文件不存在
            "after_sha256": _sha256(data),
            "before_mode": 0o644,
            "backup": "",  # 无 Preimage，Undo 走 trash
        }
        self._append_journal({**common_record, "status": "pending"})
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._atomic_create(path, data)
        except Exception:
            self._append_journal({**common_record, "status": "failed"})
            raise
        self._append_journal({**common_record, "status": "completed"})
        return {**common_record, "status": "completed", "bytes": len(data)}

    def get_undo_candidate(self, operation_id: str | None = None) -> UndoCandidate:
        """选择尚未 Undo 的指定操作，或最后一个可恢复操作。"""

        latest = self._latest_records()
        if operation_id:
            record = latest.get(operation_id)
            if not record or record.get("status") != "completed":
                raise FileWriteError(f"操作不可恢复或不存在：{operation_id}")
        else:
            record = next(
                (
                    item
                    for item in reversed(list(latest.values()))
                    if item.get("status") == "completed"
                ),
                None,
            )
            if record is None:
                raise FileWriteError("没有可恢复的已完成操作")

        return UndoCandidate(
            operation_id=record["operation_id"],
            path=record["path"],
            before_sha256=record["before_sha256"],
            after_sha256=record["after_sha256"],
        )

    def undo(self, operation_id: str | None = None) -> dict[str, Any]:
        """在文件仍等于操作后 Hash 时恢复 Preimage。"""

        candidate = self.get_undo_candidate(operation_id)
        record = self._latest_records()[candidate.operation_id]
        if record.get("operation") == "create_file":
            return self._undo_create(record, candidate)
        path = self.boundary.resolve_existing_file_for_write(candidate.path)
        current_hash = _sha256(self._read_source(path))
        if current_hash != candidate.after_sha256:
            raise FileWriteError(
                "当前文件已被后续修改，拒绝 Undo："
                f"expected={candidate.after_sha256}, actual={current_hash}"
            )

        backup_path = self._resolve_backup(record["backup"])
        preimage = backup_path.read_bytes()
        if _sha256(preimage) != candidate.before_sha256:
            raise FileWriteError("Preimage Hash 校验失败，拒绝恢复")

        self._append_journal({**record, "status": "undo_pending"})
        try:
            self._atomic_replace(path, preimage, mode=int(record["before_mode"]))
        except Exception:
            self._append_journal({**record, "status": "undo_failed"})
            raise
        self._append_journal({**record, "status": "undone"})
        return {
            "operation_id": candidate.operation_id,
            "path": candidate.path,
            "status": "undone",
            "restored_sha256": candidate.before_sha256,
        }

    def _undo_create(self, record: dict[str, Any], candidate: UndoCandidate) -> dict[str, Any]:
        """撤销创建：确认文件未被后续修改后移入 trash，不物理删除。"""

        path = self.boundary.resolve_existing_file_for_write(candidate.path)
        current_hash = _sha256(self._read_source(path))
        if current_hash != candidate.after_sha256:
            raise FileWriteError(
                "当前文件已被后续修改，拒绝 Undo："
                f"expected={candidate.after_sha256}, actual={current_hash}"
            )
        trash_dir = self.runtime_directory / "trash" / candidate.operation_id
        self._append_journal({**record, "status": "undo_pending"})
        try:
            trash_dir.mkdir(parents=True, exist_ok=True)
            os.replace(path, trash_dir / path.name)
        except Exception:
            self._append_journal({**record, "status": "undo_failed"})
            raise
        self._append_journal({**record, "status": "undone"})
        return {
            "operation_id": candidate.operation_id,
            "path": candidate.path,
            "status": "undone",
            "restored_sha256": "",
        }

    def mark_interrupted_operations(self) -> list[dict[str, Any]]:
        """把崩溃留下的 pending 标成 interrupted，不改仓库文件、不重放。"""

        inspections = self.inspect_pending_operations()
        latest = self._latest_records()
        for item in inspections:
            record = latest.get(item["operation_id"])
            if record is None:
                continue
            self._append_journal(
                {
                    **{key: value for key, value in record.items() if key != "timestamp"},
                    "status": "interrupted",
                    "observed": item["observed"],
                }
            )
        return inspections

    def inspect_pending_operations(self) -> list[dict[str, Any]]:
        """只诊断中断操作，不自动修改文件。"""

        inspections: list[dict[str, Any]] = []
        for record in self._latest_records().values():
            status = record.get("status")
            if status not in {"pending", "undo_pending"}:
                continue
            try:
                path = self.boundary.resolve_existing_file_for_write(record["path"])
                current_hash = _sha256(self._read_source(path))
            except Exception as exc:
                # 创建被中断且文件未生成属于正常情况，不算“无法读取”。
                if record.get("operation") == "create_file":
                    observed = "not_applied"
                else:
                    observed = f"unreadable:{type(exc).__name__}"
            else:
                before = record["before_sha256"]
                after = record["after_sha256"]
                if status == "pending" and current_hash == before:
                    observed = "not_applied"
                elif status == "pending" and current_hash == after:
                    observed = "applied_without_completion"
                elif status == "undo_pending" and current_hash == after:
                    observed = "undo_not_applied"
                elif status == "undo_pending" and current_hash == before:
                    observed = "undo_applied_without_completion"
                else:
                    observed = "conflict"
            inspections.append(
                {
                    "operation_id": record["operation_id"],
                    "operation_status": status,
                    "path": record["path"],
                    "observed": observed,
                }
            )
        return inspections

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
    def _atomic_create(path: Path, data: bytes, *, mode: int = 0o644) -> None:
        """为新文件写临时文件再原子替换；不 stat 目标（它尚不存在）。"""

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

    @staticmethod
    def _atomic_replace(path: Path, data: bytes, *, mode: int | None = None) -> None:
        """在目标目录写临时文件，再原子替换并保留 Unix 权限位。"""

        target_mode = stat.S_IMODE(path.stat().st_mode) if mode is None else mode
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
            os.chmod(temporary_name, target_mode)
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

    def _latest_records(self) -> dict[str, dict[str, Any]]:
        """按 Journal 顺序保留每个 Operation 的最后状态。"""

        latest: dict[str, dict[str, Any]] = {}
        if not self.journal_path.exists():
            return latest
        for line_number, line in enumerate(
            self.journal_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            try:
                record = json.loads(line)
                operation_id = record["operation_id"]
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise FileWriteError(f"Operation Journal 第 {line_number} 行损坏") from exc
            latest[operation_id] = record
        return latest

    def _resolve_backup(self, relative_path: str) -> Path:
        """防止被篡改的 Journal 将 Backup 指向运行目录外。"""

        backup = (self.runtime_directory / relative_path).resolve()
        try:
            backup.relative_to(self.backup_directory.resolve())
        except ValueError as exc:
            raise FileWriteError("Journal 中的 Backup 路径越界") from exc
        if not backup.is_file():
            raise FileWriteError(f"Preimage 不存在：{relative_path}")
        return backup


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

    @coding_tool(
        risk=ToolRisk.WRITE,
        idempotent=False,
        retry_policy=RetryPolicy.NEVER,
        tags=("filesystem", "create"),
    )
    def create_file(path: str, content: str) -> dict[str, Any]:
        """创建一个新文件并保存可恢复记录；文件已存在时请改用 replace_text。"""

        return editor.create_file(user_path=path, content=content)

    registry.register(get_file_hash)
    registry.register(replace_text)
    registry.register(create_file)
    return editor

"""文件工具共用的 Workspace 路径与敏感文件边界。"""

import os
from pathlib import Path


class WorkspaceViolation(ValueError):
    """用户路径越界、类型错误或命中敏感文件规则。"""


class WorkspaceBoundary:
    """把模型提供的路径限制在一个已经批准的工作目录内。

    仅检查字符串是否以 Workspace 开头并不安全，例如 ``..`` 和符号链接都
    可以让最终目标落到目录外。这里先调用 ``resolve`` 得到真实路径，再用
    ``relative_to`` 验证它确实位于 Workspace 内部。
    """

    _SENSITIVE_PARTS = {".ssh", ".aws", ".gnupg"}
    _SENSITIVE_NAMES = {
        ".env",
        "id_rsa",
        "id_ed25519",
        "credentials",
        "credentials.json",
        "service-account.json",
    }

    def __init__(self, root: Path) -> None:
        resolved_root = root.expanduser().resolve()
        if not resolved_root.is_dir():
            raise WorkspaceViolation(f"Workspace 不是有效目录：{resolved_root}")
        self.root = resolved_root

    def resolve(
        self,
        user_path: str,
        *,
        expected: str | None = None,
    ) -> Path:
        """解析并校验模型提供的路径。

        ``expected`` 可取 ``file`` 或 ``directory``。第一批只读工具要求目标
        已经存在；未来写工具会增加允许“末级文件尚不存在”的独立解析方法。
        """

        raw_path = Path(user_path).expanduser()
        candidate = raw_path if raw_path.is_absolute() else self.root / raw_path
        try:
            resolved = candidate.resolve(strict=True)
            relative = resolved.relative_to(self.root)
        except (FileNotFoundError, RuntimeError) as exc:
            raise WorkspaceViolation(f"路径不存在或无法解析：{user_path}") from exc
        except ValueError as exc:
            raise WorkspaceViolation(f"路径超出 Workspace：{user_path}") from exc

        self._reject_sensitive(relative)
        if expected == "file" and not resolved.is_file():
            raise WorkspaceViolation(f"路径不是文件：{user_path}")
        if expected == "directory" and not resolved.is_dir():
            raise WorkspaceViolation(f"路径不是目录：{user_path}")
        return resolved

    def relative(self, path: Path) -> str:
        """返回适合 Tool Result 展示的 Workspace 相对路径。"""

        return path.relative_to(self.root).as_posix() or "."

    def resolve_existing_file_for_write(self, user_path: str) -> Path:
        """解析待修改文件，并拒绝路径中出现任何符号链接。

        只读操作可以安全跟随仍位于 Workspace 内的符号链接，但写操作这样做
        容易让用户误判真正被修改的文件。因此写边界采用更严格的规则。
        """

        raw_path = Path(user_path).expanduser()
        candidate = raw_path if raw_path.is_absolute() else self.root / raw_path
        lexical = Path(candidate.absolute())
        try:
            relative = lexical.relative_to(self.root)
        except ValueError as exc:
            raise WorkspaceViolation(f"写入路径超出 Workspace：{user_path}") from exc

        current = self.root
        for part in relative.parts:
            current /= part
            if current.is_symlink():
                raise WorkspaceViolation(f"写入路径不能包含符号链接：{user_path}")

        return self.resolve(str(lexical), expected="file")

    def resolve_new_file_for_write(self, user_path: str) -> Path:
        """解析一个尚不存在的新文件路径，用于安全创建。

        与 resolve_existing_file_for_write 同样拒绝越界、符号链接和敏感文件；
        区别在于末级文件必须尚不存在（已存在应改用 replace_text）。父目录若
        缺失由调用方在 Workspace 内按需创建，这里不返回已存在的目标。
        """

        raw_path = Path(user_path).expanduser()
        candidate = raw_path if raw_path.is_absolute() else self.root / raw_path
        # normpath 折叠 ``..``，避免 relative_to 仅按字符串前缀误判 ../ 逃逸落在内部。
        lexical = Path(os.path.normpath(candidate))
        try:
            relative = lexical.relative_to(self.root)
        except ValueError as exc:
            raise WorkspaceViolation(f"写入路径超出 Workspace：{user_path}") from exc
        if not relative.parts:
            raise WorkspaceViolation(f"写入路径不能是 Workspace 根：{user_path}")

        self._reject_sensitive(relative)

        # 已存在的中间路径不能是符号链接，否则最终写入可能落到 Workspace 外。
        current = self.root
        for part in relative.parts:
            current /= part
            if current.is_symlink():
                raise WorkspaceViolation(f"写入路径不能包含符号链接：{user_path}")
        if lexical.exists():
            raise WorkspaceViolation(f"文件已存在，请改用 replace_text：{user_path}")
        return lexical

    def _reject_sensitive(self, relative: Path) -> None:
        lowered_parts = {part.lower() for part in relative.parts}
        if lowered_parts & self._SENSITIVE_PARTS:
            raise WorkspaceViolation(f"禁止访问敏感目录：{relative}")

        name = relative.name.lower()
        if name in self._SENSITIVE_NAMES or name.endswith((".pem", ".key", ".p12")):
            raise WorkspaceViolation(f"禁止访问敏感文件：{relative}")

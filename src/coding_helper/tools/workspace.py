"""文件工具共用的 Workspace 路径与敏感文件边界。"""

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

    def _reject_sensitive(self, relative: Path) -> None:
        lowered_parts = {part.lower() for part in relative.parts}
        if lowered_parts & self._SENSITIVE_PARTS:
            raise WorkspaceViolation(f"禁止访问敏感目录：{relative}")

        name = relative.name.lower()
        if name in self._SENSITIVE_NAMES or name.endswith((".pem", ".key", ".p12")):
            raise WorkspaceViolation(f"禁止访问敏感文件：{relative}")

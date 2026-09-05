"""从环境变量加载应用配置。"""

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """CLI 与后续 Agent Runtime 共用的运行配置。

    ``BaseSettings`` 既是经过类型校验的 Pydantic 模型，也可以读取环境变量
    和本地 ``.env`` 文件。把密钥读取限制在应用边界，可以避免 API 凭据
    进入 Prompt、Checkpoint 或普通函数参数。
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ark_api_key: SecretStr | None = None
    ark_base_url: str = "https://ark.cn-beijing.volces.com/api/v3"
    ark_deepseek_model: str | None = None
    ark_glm_model: str | None = None
    ark_auxiliary_model: str | None = None
    ark_default_primary: Literal["deepseek", "glm"] = "deepseek"
    workspace: Path = Path.cwd()
    # 预留给后续 compact：按当前主模型窗口估算，而不是写死 256k。
    ark_deepseek_context_window: int = Field(default=128_000, ge=4_000)
    ark_glm_context_window: int = Field(default=128_000, ge=4_000)
    context_compact_threshold: float = Field(default=0.70, gt=0, lt=1)
    context_compact_target: float = Field(default=0.40, gt=0, lt=1)
    max_model_calls: int = Field(default=40, ge=1, le=200)
    max_tool_calls: int = Field(default=80, ge=1, le=400)
    model_retry_attempts: int = Field(default=1, ge=0, le=4)
    stuck_repeat_limit: int = Field(default=3, ge=2, le=8)
    completion_enabled: bool = True
    completion_test_command: str = ""
    completion_lint_command: str = ""
    completion_allowed_prefixes: str = ""
    max_repair_rounds: int = Field(default=2, ge=0, le=8)
    github_personal_access_token: SecretStr | None = None

    @model_validator(mode="after")
    def validate_compact_bounds(self) -> "Settings":
        if self.context_compact_target >= self.context_compact_threshold:
            raise ValueError("CONTEXT_COMPACT_TARGET 必须小于 CONTEXT_COMPACT_THRESHOLD")
        return self

    def missing_model_settings(self) -> list[str]:
        """返回真实模型调用仍缺少的配置项名称，供 CLI 提示用户。"""

        required = {
            "ARK_API_KEY": self.ark_api_key,
            "ARK_DEEPSEEK_MODEL": self.ark_deepseek_model,
            "ARK_GLM_MODEL": self.ark_glm_model,
            "ARK_AUXILIARY_MODEL": self.ark_auxiliary_model,
        }
        return [name for name, value in required.items() if value is None]

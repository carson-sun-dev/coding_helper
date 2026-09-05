"""从环境变量加载应用配置。"""

from pathlib import Path
from typing import Literal

from pydantic import SecretStr
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

    def missing_model_settings(self) -> list[str]:
        """返回真实模型调用仍缺少的配置项名称，供 CLI 提示用户。"""

        required = {
            "ARK_API_KEY": self.ark_api_key,
            "ARK_DEEPSEEK_MODEL": self.ark_deepseek_model,
            "ARK_GLM_MODEL": self.ark_glm_model,
            "ARK_AUXILIARY_MODEL": self.ark_auxiliary_model,
        }
        return [name for name, value in required.items() if value is None]

"""火山方舟模型的角色选择与 LangChain 客户端工厂。"""

from enum import Enum

from langchain_openai import ChatOpenAI

from coding_helper.config import Settings


class ModelTarget(str, Enum):
    """CLI 可选择的模型目标。

    这里使用稳定的角色名，而不是把方舟 Endpoint ID 暴露给上层代码。
    将来替换具体模型时，Agent Runtime 和 CLI 都不需要随之修改。
    """

    DEEPSEEK = "deepseek"
    GLM = "glm"
    AUXILIARY = "auxiliary"


class ModelConfigurationError(ValueError):
    """模型角色缺少 Endpoint 或 API Key 时抛出的配置错误。"""


def primary_order(settings: Settings) -> tuple[ModelTarget, ModelTarget]:
    """返回“当前主模型、服务异常时的备用主模型”。

    Fallback 只在网络超时、限流或服务端错误时发生。模型生成了较差答案或
    错误工具参数时不能静默切换，否则同一次评测会混入两个模型的能力。
    """

    if settings.ark_default_primary == "glm":
        return ModelTarget.GLM, ModelTarget.DEEPSEEK
    return ModelTarget.DEEPSEEK, ModelTarget.GLM


def other_primary(target: ModelTarget) -> ModelTarget:
    """返回另一主模型。辅助模型没有对等 Fallback。"""

    if target is ModelTarget.DEEPSEEK:
        return ModelTarget.GLM
    if target is ModelTarget.GLM:
        return ModelTarget.DEEPSEEK
    raise ValueError("辅助模型不能作为主 Agent，也就没有主模型 Fallback")


def model_id_for(settings: Settings, target: ModelTarget) -> str:
    """将稳定角色名解析为方舟实际接收的模型 ID。"""

    model_ids = {
        ModelTarget.DEEPSEEK: settings.ark_deepseek_model,
        ModelTarget.GLM: settings.ark_glm_model,
        ModelTarget.AUXILIARY: settings.ark_auxiliary_model,
    }
    model_id = model_ids[target]
    if not model_id:
        raise ModelConfigurationError(f"模型角色 {target.value!r} 尚未配置")
    return model_id


def create_chat_model(
    settings: Settings,
    target: ModelTarget,
    *,
    timeout_seconds: float = 30,
) -> ChatOpenAI:
    """创建连接火山方舟 OpenAI 兼容接口的 LangChain 模型。

    ``ChatOpenAI`` 在这里是协议适配器：它负责把 LangChain Message 和
    Tool Schema 转成 OpenAI 兼容请求，但不会决定选哪个模型。选模和
    Fallback 仍由 Coding Helper 自己治理。

    客户端内置重试被设为 0，避免未来与 Agent Middleware 的显式重试叠加，
    尤其要防止一次错误被两层代码重复发送。
    """

    if settings.ark_api_key is None:
        raise ModelConfigurationError("ARK_API_KEY 尚未配置")

    return ChatOpenAI(
        model=model_id_for(settings, target),
        api_key=settings.ark_api_key,
        base_url=settings.ark_base_url,
        timeout=timeout_seconds,
        max_retries=0,
        temperature=0,
    )

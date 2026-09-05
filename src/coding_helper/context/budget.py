"""主模型上下文窗口与 compact 触发阈值。

这些函数按当前 PRIMARY 窗口计算阈值。压缩 Middleware 在发给模型前
调用它们，而不是写死 256k。
"""

from coding_helper.config import Settings
from coding_helper.models import ModelTarget


def context_window_tokens(settings: Settings, target: ModelTarget) -> int:
    """返回指定角色将要面对的输入窗口。"""

    if target is ModelTarget.GLM:
        return settings.ark_glm_context_window
    return settings.ark_deepseek_context_window


def compact_trigger_tokens(settings: Settings, target: ModelTarget) -> int:
    return int(context_window_tokens(settings, target) * settings.context_compact_threshold)


def compact_target_tokens(settings: Settings, target: ModelTarget) -> int:
    return int(context_window_tokens(settings, target) * settings.context_compact_target)


def should_compact(
    estimated_input_tokens: int,
    settings: Settings,
    target: ModelTarget,
) -> bool:
    """估算输入达到阈值时返回 True。辅助模型不接管主循环，因此也不用它的窗口。"""

    if target is ModelTarget.AUXILIARY:
        return False
    return estimated_input_tokens >= compact_trigger_tokens(settings, target)


def estimate_tokens(text: str) -> int:
    """粗估 token，避免为是否 compact 再调用一次模型。

    ASCII 按约 4 字符/token，中文和符号按约 2 字符/token。误差可接受，
    因为阈值本身留了 30% 余量。
    """

    if not text:
        return 0
    ascii_chars = sum(1 for char in text if ord(char) < 128)
    other_chars = len(text) - ascii_chars
    return max(1, ascii_chars // 4 + (other_chars + 1) // 2)

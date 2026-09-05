"""主模型服务异常分类、有界重试，以及切到另一主模型。

不使用框架自带的 ``ModelFallbackMiddleware``：它会在任意异常时换模型，
连 400 或鉴权失败也会混入另一个模型的能力。这里只对超时、429 和 5xx
重试；答案差或工具参数错是成功响应，不会走到这条路径。
豆包不出现在 Fallback 链里。
"""

from __future__ import annotations

import time
from enum import Enum

from langchain.agents.middleware import AgentMiddleware
from langgraph.errors import GraphBubbleUp

_RETRYABLE_NAMES = {
    "apiconnectionerror",
    "apitimeouterror",
    "connecttimeout",
    "internalservererror",
    "ratelimiterror",
    "readtimeout",
    "timeouterror",
}


class ErrorClass(str, Enum):
    RETRYABLE_SERVICE = "retryable_service"
    AUTH = "auth"
    CONTEXT = "context"
    CLIENT = "client"
    UNKNOWN = "unknown"


def classify_model_error(exc: BaseException) -> ErrorClass:
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    name = type(exc).__name__.casefold()
    text = str(exc).casefold()

    if status in {401, 403} or "auth" in name:
        return ErrorClass.AUTH
    if status == 429 or name in _RETRYABLE_NAMES:
        return ErrorClass.RETRYABLE_SERVICE
    if isinstance(status, int) and status >= 500:
        return ErrorClass.RETRYABLE_SERVICE
    if "timeout" in name or "timeout" in text or "connect" in name:
        return ErrorClass.RETRYABLE_SERVICE
    if "context" in text and ("length" in text or "window" in text):
        return ErrorClass.CONTEXT
    if isinstance(status, int) and 400 <= status < 500:
        return ErrorClass.CLIENT
    return ErrorClass.UNKNOWN


def is_retryable_service_error(exc: BaseException) -> bool:
    return classify_model_error(exc) is ErrorClass.RETRYABLE_SERVICE


class ServiceFallbackMiddleware(AgentMiddleware):
    """同一请求：先重试当前主模型，再换另一主模型一次。"""

    tools: list = []

    def __init__(
        self,
        fallback_model,
        *,
        retry_attempts: int = 1,
        initial_delay: float = 0.4,
    ) -> None:
        super().__init__()
        self._fallback_model = fallback_model
        self._retry_attempts = retry_attempts
        self._initial_delay = initial_delay

    def wrap_model_call(self, request, handler):
        last_error: BaseException | None = None
        for attempt in range(self._retry_attempts + 1):
            try:
                return handler(request)
            except GraphBubbleUp:
                raise
            except Exception as exc:
                if not is_retryable_service_error(exc):
                    raise
                last_error = exc
                if attempt < self._retry_attempts and self._initial_delay > 0:
                    time.sleep(self._initial_delay * (2**attempt))
        try:
            return handler(request.override(model=self._fallback_model))
        except GraphBubbleUp:
            raise
        except Exception as exc:
            if not is_retryable_service_error(exc):
                raise
            last_error = exc
        if last_error is not None:
            raise last_error
        raise RuntimeError("主模型 Fallback 未产生结果")

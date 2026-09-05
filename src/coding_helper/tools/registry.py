"""Coding Helper 的工具声明、Schema 转换与统一注册表。"""

import inspect
import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, ParamSpec, TypeVar, cast

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, ConfigDict, Field

P = ParamSpec("P")
R = TypeVar("R")
_CONFIG_ATTRIBUTE = "__coding_tool_config__"
_MODEL_TOOL_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


class ToolSource(str, Enum):
    """工具来源决定命名空间和后续信任策略。"""

    BUILTIN = "builtin"
    SKILL = "skill"
    MCP = "mcp"


class ToolRisk(str, Enum):
    """工具可能产生的最高副作用等级。"""

    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    DESTRUCTIVE = "destructive"


class RetryPolicy(str, Enum):
    """工具允许采用的自动重试策略。"""

    NEVER = "never"
    SAFE = "safe"
    TRANSIENT = "transient"


class CodingToolConfig(BaseModel):
    """装饰器附着在普通函数上的治理元数据。"""

    model_config = ConfigDict(frozen=True)

    name: str | None = None
    description: str | None = None
    source: ToolSource = ToolSource.BUILTIN
    risk: ToolRisk
    idempotent: bool
    timeout_seconds: float = Field(default=30, gt=0)
    retry_policy: RetryPolicy = RetryPolicy.NEVER
    tags: tuple[str, ...] = ()


class ToolSpec(BaseModel):
    """Registry 对外暴露的稳定工具描述，不包含可执行函数。"""

    model_config = ConfigDict(frozen=True)

    canonical_name: str
    model_name: str
    description: str
    input_schema: dict[str, Any]
    source: ToolSource
    risk: ToolRisk
    idempotent: bool
    timeout_seconds: float
    retry_policy: RetryPolicy
    tags: tuple[str, ...]
    loaded: bool = True


@dataclass(frozen=True)
class RegisteredTool:
    """把治理元数据和 LangChain 可执行工具保存在一起。"""

    spec: ToolSpec
    langchain_tool: BaseTool


class ToolRegistrationError(ValueError):
    """工具声明无效或名称冲突。"""


def coding_tool(
    *,
    risk: ToolRisk,
    idempotent: bool,
    timeout_seconds: float = 30,
    retry_policy: RetryPolicy = RetryPolicy.NEVER,
    tags: tuple[str, ...] = (),
    source: ToolSource = ToolSource.BUILTIN,
    name: str | None = None,
    description: str | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """给普通函数附加治理元数据，但暂时不改变其调用行为。

    装饰阶段只声明“这个工具是什么”；注册阶段才生成 LangChain Tool
    Schema。分开两步后，单元测试仍可直接调用原函数，权限和 MCP 等模块
    也能在把工具交给模型前检查元数据。
    """

    config = CodingToolConfig(
        name=name,
        description=description,
        source=source,
        risk=risk,
        idempotent=idempotent,
        timeout_seconds=timeout_seconds,
        retry_policy=retry_policy,
        tags=tags,
    )

    def decorate(function: Callable[P, R]) -> Callable[P, R]:
        setattr(function, _CONFIG_ATTRIBUTE, config)
        return function

    return decorate


class ToolRegistry:
    """统一保存 ToolSpec，并为 LangChain 提供当前已加载的工具。"""

    def __init__(self) -> None:
        self._by_canonical_name: dict[str, RegisteredTool] = {}
        self._by_model_name: dict[str, RegisteredTool] = {}

    def register(
        self,
        function: Callable[..., Any],
        *,
        namespace: str | None = None,
    ) -> RegisteredTool:
        """注册一个被 ``@coding_tool`` 标记的函数。

        内部名称使用 ``source::namespace::name`` 便于审计；发给模型的名称
        使用双下划线，因为 OpenAI 兼容接口通常不接受冒号。
        """

        config = cast(CodingToolConfig | None, getattr(function, _CONFIG_ATTRIBUTE, None))
        if config is None:
            raise ToolRegistrationError("函数必须先使用 @coding_tool 声明")

        local_name = config.name or function.__name__
        parts = [config.source.value, *([namespace] if namespace else []), local_name]
        canonical_name = "::".join(parts)
        model_name = local_name if namespace is None else "__".join(parts)
        if not _MODEL_TOOL_NAME.fullmatch(model_name):
            raise ToolRegistrationError(f"模型工具名包含非法字符：{model_name!r}")
        if canonical_name in self._by_canonical_name:
            raise ToolRegistrationError(f"工具已注册：{canonical_name}")
        if model_name in self._by_model_name:
            raise ToolRegistrationError(f"模型工具名冲突：{model_name}")

        description = config.description or inspect.getdoc(function)
        if not description:
            raise ToolRegistrationError(f"工具 {local_name!r} 缺少描述或文档字符串")

        langchain_tool = StructuredTool.from_function(
            function,
            name=model_name,
            description=description,
        )
        spec = ToolSpec(
            canonical_name=canonical_name,
            model_name=model_name,
            description=description,
            input_schema=langchain_tool.get_input_schema().model_json_schema(),
            source=config.source,
            risk=config.risk,
            idempotent=config.idempotent,
            timeout_seconds=config.timeout_seconds,
            retry_policy=config.retry_policy,
            tags=config.tags,
        )
        registered = RegisteredTool(spec=spec, langchain_tool=langchain_tool)
        self._by_canonical_name[canonical_name] = registered
        self._by_model_name[model_name] = registered
        return registered

    def get_by_model_name(self, name: str) -> RegisteredTool:
        """按模型 Tool Call 中出现的名称取回工具。"""

        try:
            return self._by_model_name[name]
        except KeyError as exc:
            raise ToolRegistrationError(f"未知模型工具：{name}") from exc

    def specs(self) -> tuple[ToolSpec, ...]:
        """按注册顺序返回不可变的工具描述快照。"""

        return tuple(item.spec for item in self._by_canonical_name.values())

    def langchain_tools(self) -> tuple[BaseTool, ...]:
        """返回可直接传给 ``bind_tools`` 或 ``create_agent`` 的工具。"""

        return tuple(item.langchain_tool for item in self._by_canonical_name.values())

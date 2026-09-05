import pytest

from coding_helper.tools import (
    RetryPolicy,
    ToolRegistrationError,
    ToolRegistry,
    ToolRisk,
    ToolSource,
    coding_tool,
)


@coding_tool(
    risk=ToolRisk.READ,
    idempotent=True,
    retry_policy=RetryPolicy.SAFE,
    tags=("filesystem",),
)
def read_sample(path: str, limit: int = 100) -> str:
    """读取示例文本，用于验证自动生成的参数 Schema。"""

    return f"{path}:{limit}"


def test_registry_builds_schema_and_keeps_governance_metadata() -> None:
    registry = ToolRegistry()

    registered = registry.register(read_sample)

    assert registered.spec.canonical_name == "builtin::read_sample"
    assert registered.spec.model_name == "read_sample"
    assert registered.spec.risk is ToolRisk.READ
    assert registered.spec.idempotent is True
    assert registered.spec.retry_policy is RetryPolicy.SAFE
    assert registered.spec.tags == ("filesystem",)
    assert registered.spec.input_schema["properties"]["path"]["type"] == "string"
    assert registered.langchain_tool.invoke({"path": "a.py"}) == "a.py:100"


def test_dynamic_namespace_uses_provider_safe_model_name() -> None:
    registry = ToolRegistry()

    registered = registry.register(read_sample, namespace="project_docs")

    assert registered.spec.canonical_name == "builtin::project_docs::read_sample"
    assert registered.spec.model_name == "builtin__project_docs__read_sample"


def test_registry_rejects_duplicate_and_undecorated_functions() -> None:
    registry = ToolRegistry()
    registry.register(read_sample)

    with pytest.raises(ToolRegistrationError, match="已注册"):
        registry.register(read_sample)

    def plain_function() -> str:
        return "not-a-tool"

    with pytest.raises(ToolRegistrationError, match="@coding_tool"):
        registry.register(plain_function)


def test_external_tool_keeps_source_in_canonical_name() -> None:
    @coding_tool(
        source=ToolSource.MCP,
        risk=ToolRisk.READ,
        idempotent=True,
    )
    def search_docs(query: str) -> str:
        """搜索外部文档。"""

        return query

    registered = ToolRegistry().register(search_docs, namespace="docs")

    assert registered.spec.canonical_name == "mcp::docs::search_docs"
    assert registered.spec.model_name == "mcp__docs__search_docs"

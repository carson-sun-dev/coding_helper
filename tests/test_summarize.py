import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from coding_helper.context.summarize import (
    ConversationDigest,
    SummarizeError,
    parse_digest,
    render_digest,
    render_transcript,
    summarize_messages,
)


class StructuredFakeModel:
    """模拟 ``with_structured_output``：直接返回 Pydantic 对象。"""

    def __init__(self, digest: ConversationDigest) -> None:
        self.digest = digest
        self.invoked = False

    def with_structured_output(self, schema):
        assert schema is ConversationDigest
        return self

    def invoke(self, messages):
        self.invoked = True
        assert messages
        return self.digest


class JsonFakeModel:
    """没有结构化接口时，从普通文本里抽 JSON。"""

    def invoke(self, messages):
        return AIMessage(
            content=(
                '```json\n{"goal":"修复登录","progress":"已读 auth.py",'
                '"decisions":["先补测试"],"blockers":[],'
                '"evidence":["auth.py"],"next_action":"改实现"}\n```'
            )
        )


def test_structured_output_returns_digest() -> None:
    digest = ConversationDigest(
        goal="修复登录",
        progress="已定位 Token 刷新",
        decisions=["保留现有接口"],
        evidence=["src/auth.py"],
        next_action="补回归测试",
    )
    model = StructuredFakeModel(digest)

    result = summarize_messages(
        model,
        [HumanMessage(content="修复登录"), AIMessage(content="先读文件")],
    )

    assert result == digest
    assert model.invoked


def test_plain_json_reply_is_parsed() -> None:
    result = summarize_messages(
        JsonFakeModel(),
        [HumanMessage(content="修复登录")],
    )
    assert result.goal == "修复登录"
    assert result.evidence == ["auth.py"]


def test_parse_digest_rejects_prose() -> None:
    with pytest.raises(SummarizeError):
        parse_digest("模型随便写了一段摘要，没有 JSON")


def test_render_digest_and_transcript_stay_short() -> None:
    digest = ConversationDigest(
        goal="修复登录",
        progress="已读文件",
        decisions=["用 stub"],
        blockers=["测试红"],
        evidence=["test_auth.py"],
        next_action="改实现",
    )
    rendered = render_digest(digest)
    assert "<conversation-summary>" in rendered
    assert "修复登录" in rendered
    assert "测试红" in rendered

    huge = ToolMessage(content="x" * 4_000, tool_call_id="c1", name="read_file")
    transcript = render_transcript([HumanMessage(content="读文件"), huge])
    assert "tool:read_file" in transcript
    assert "x" * 900 not in transcript

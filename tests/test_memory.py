from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from coding_helper.context.compact import compact_messages, render_state_reminder
from coding_helper.context.memory import (
    MemoryCandidate,
    MemoryKind,
    MemoryStore,
    acceptable_candidate,
    attach_project_memory,
)
from coding_helper.context.summarize import ConversationDigest
from coding_helper.progress.task import TaskStore


def _candidate(**overrides) -> MemoryCandidate:
    values = {
        "kind": MemoryKind.VERIFIED_COMMAND,
        "fact": "pytest tests/test_auth.py",
        "evidence": "18 passed",
        "verified": True,
    }
    values.update(overrides)
    return MemoryCandidate(**values)


def test_rejects_unverified_and_failed_evidence() -> None:
    assert acceptable_candidate(_candidate(verified=False)) is False
    assert acceptable_candidate(_candidate(evidence="traceback: boom", fact="入口在 auth")) is False
    assert acceptable_candidate(_candidate(fact="认证失败后重试", evidence="日志")) is False
    assert (
        acceptable_candidate(
            _candidate(
                kind=MemoryKind.PREFERENCE,
                fact="用中文回复",
                evidence="随便猜的",
            )
        )
        is False
    )
    assert (
        acceptable_candidate(
            _candidate(
                kind=MemoryKind.PREFERENCE,
                fact="用户要求用中文回复",
                evidence="用户明确说请记住",
            )
        )
        is True
    )


def test_absorb_dedupes_and_writes_projection(tmp_path) -> None:
    store = MemoryStore(tmp_path)
    first = store.absorb(
        [
            _candidate(),
            _candidate(fact="PYTEST tests/test_auth.py", evidence="再次通过"),
            _candidate(
                kind=MemoryKind.ARCHITECTURE,
                fact="认证入口在 src/auth/token.py",
                evidence="test_auth.py 覆盖该路径",
            ),
            _candidate(verified=False, fact="不要写入"),
        ]
    )

    assert [item.fact for item in first] == [
        "pytest tests/test_auth.py",
        "认证入口在 src/auth/token.py",
    ]
    markdown = (tmp_path / ".coding-helper" / "memory.md").read_text(encoding="utf-8")
    assert "pytest tests/test_auth.py" in markdown
    assert "src/auth/token.py" in markdown
    assert "不要写入" not in markdown
    assert store.absorb([_candidate()]) == []


def test_attach_and_reminder_include_memory(tmp_path) -> None:
    MemoryStore(tmp_path).absorb([_candidate()])
    TaskStore(tmp_path).ensure_session(goal="修复登录", thread_id="t1")

    prompt = attach_project_memory("阅读入口", tmp_path)
    reminder = render_state_reminder(tmp_path)

    assert prompt.startswith("阅读入口")
    assert "<project-memory>" in prompt
    assert "pytest tests/test_auth.py" in prompt
    assert "pytest tests/test_auth.py" in reminder
    assert "修复登录" in reminder
    assert attach_project_memory("无记忆", tmp_path.parent / "empty") == "无记忆"


def test_compact_absorbs_verified_candidates(tmp_path) -> None:
    old = "x" * 4_000
    messages = [
        HumanMessage(content="修复登录"),
        AIMessage(content="先读认证实现"),
        HumanMessage(content="继续定位"),
        AIMessage(content="再搜索相关测试"),
        AIMessage(content="", tool_calls=[{"name": "read_file", "args": {}, "id": "c1"}]),
        ToolMessage(content=old, tool_call_id="c1", name="read_file"),
        AIMessage(content="", tool_calls=[{"name": "read_file", "args": {}, "id": "c2"}]),
        ToolMessage(content=old, tool_call_id="c2", name="read_file"),
        AIMessage(content="", tool_calls=[{"name": "read_file", "args": {}, "id": "c3"}]),
        ToolMessage(content="keep-one", tool_call_id="c3", name="read_file"),
        AIMessage(content="", tool_calls=[{"name": "search_text", "args": {}, "id": "c4"}]),
        ToolMessage(content="recent-evidence", tool_call_id="c4", name="search_text"),
    ]

    def summarizer(_prefix):
        return ConversationDigest(
            goal="修复登录",
            memory_candidates=[
                _candidate(),
                _candidate(verified=False, fact="未验证猜测"),
            ],
        )

    compact_messages(messages, workspace=tmp_path, target_tokens=50, summarizer=summarizer)

    markdown = (tmp_path / ".coding-helper" / "memory.md").read_text(encoding="utf-8")
    assert "pytest tests/test_auth.py" in markdown
    assert "未验证猜测" not in markdown

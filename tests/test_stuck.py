from coding_helper.governance.stuck import (
    STUCK_PREFIX,
    StuckDetectionMiddleware,
    tool_signature,
)
from coding_helper.progress.task import SessionPhase, TaskStore
from coding_helper.tools.writes import SafeFileEditor


class FakeToolRequest:
    def __init__(self, name: str, args: dict, call_id: str) -> None:
        self.tool_call = {"name": name, "args": args, "id": call_id}


def test_third_identical_call_is_blocked_without_handler(tmp_path) -> None:
    TaskStore(tmp_path).ensure_session(goal="修复登录", thread_id="t1")
    middleware = StuckDetectionMiddleware(tmp_path, repeat_limit=3)
    executed: list[str] = []

    def handler(request):
        executed.append(request.tool_call["id"])
        return f"ok:{request.tool_call['id']}"

    first = middleware.wrap_tool_call(
        FakeToolRequest("read_file", {"path": "a.py"}, "c1"),
        handler,
    )
    second = middleware.wrap_tool_call(
        FakeToolRequest("read_file", {"path": "a.py"}, "c2"),
        handler,
    )
    third = middleware.wrap_tool_call(
        FakeToolRequest("read_file", {"path": "a.py"}, "c3"),
        handler,
    )

    assert first == "ok:c1"
    assert second == "ok:c2"
    assert third.content.startswith(STUCK_PREFIX)
    assert executed == ["c1", "c2"]
    snapshot = TaskStore(tmp_path).load()
    assert snapshot is not None
    assert snapshot.phase is SessionPhase.PAUSED
    assert any("read_file" in item for item in snapshot.blockers)


def test_different_arguments_are_not_stuck(tmp_path) -> None:
    middleware = StuckDetectionMiddleware(tmp_path, repeat_limit=2)
    executed = []

    def handler(request):
        executed.append(request.tool_call["args"]["path"])
        return "ok"

    middleware.wrap_tool_call(FakeToolRequest("read_file", {"path": "a.py"}, "c1"), handler)
    middleware.wrap_tool_call(FakeToolRequest("read_file", {"path": "b.py"}, "c2"), handler)

    assert executed == ["a.py", "b.py"]
    assert tool_signature("read_file", {"path": "a.py"}) != tool_signature(
        "read_file", {"path": "b.py"}
    )


def test_mark_interrupted_does_not_rewrite_or_replay(tmp_path) -> None:
    source = tmp_path / "app.py"
    source.write_text("value = 'old'\n", encoding="utf-8")
    editor = SafeFileEditor(tmp_path)
    editor.replace_text(
        user_path="app.py",
        old_text="'old'",
        new_text="'new'",
        expected_sha256=editor.file_hash("app.py")["sha256"],
    )
    journal = tmp_path / ".coding-helper" / "operations.jsonl"
    journal.write_text(journal.read_text(encoding="utf-8").splitlines()[0] + "\n", encoding="utf-8")

    marked = editor.mark_interrupted_operations()

    assert marked[0]["observed"] == "applied_without_completion"
    assert source.read_text(encoding="utf-8") == "value = 'new'\n"
    assert editor.inspect_pending_operations() == []
    statuses = [
        line
        for line in journal.read_text(encoding="utf-8").splitlines()
        if '"interrupted"' in line
    ]
    assert statuses

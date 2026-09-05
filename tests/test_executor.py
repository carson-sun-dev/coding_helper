import threading
import time

from coding_helper.tools import (
    ToolExecutionStatus,
    ToolExecutor,
    ToolRegistry,
    ToolRisk,
    coding_tool,
)


def test_adjacent_idempotent_reads_execute_concurrently_and_keep_order() -> None:
    """两个探针只有真正并发进入函数时，线程屏障才会同时放行。"""

    barrier = threading.Barrier(2)

    @coding_tool(risk=ToolRisk.READ, idempotent=True)
    def read_first() -> str:
        """读取第一个并发探针。"""

        barrier.wait(timeout=1)
        return "first"

    @coding_tool(risk=ToolRisk.READ, idempotent=True)
    def read_second() -> str:
        """读取第二个并发探针。"""

        barrier.wait(timeout=1)
        return "second"

    registry = ToolRegistry()
    registry.register(read_first)
    registry.register(read_second)

    results = ToolExecutor(registry).execute(
        [
            {"name": "read_first", "args": {}, "id": "call-1"},
            {"name": "read_second", "args": {}, "id": "call-2"},
        ]
    )

    assert [result.status for result in results] == [
        ToolExecutionStatus.SUCCESS,
        ToolExecutionStatus.SUCCESS,
    ]
    assert [result.call_id for result in results] == ["call-1", "call-2"]
    assert [result.content for result in results] == ["first", "second"]


def test_write_tools_execute_serially() -> None:
    active = 0
    max_active = 0
    lock = threading.Lock()

    def record_write(label: str) -> str:
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return label

    @coding_tool(risk=ToolRisk.WRITE, idempotent=False)
    def write_first() -> str:
        """执行第一个模拟写操作。"""

        return record_write("first")

    @coding_tool(risk=ToolRisk.WRITE, idempotent=False)
    def write_second() -> str:
        """执行第二个模拟写操作。"""

        return record_write("second")

    registry = ToolRegistry()
    registry.register(write_first)
    registry.register(write_second)

    results = ToolExecutor(registry).execute(
        [
            {"name": "write_first", "args": {}, "id": "call-1"},
            {"name": "write_second", "args": {}, "id": "call-2"},
        ]
    )

    assert max_active == 1
    assert [result.content for result in results] == ["first", "second"]


def test_failed_side_effect_skips_remaining_calls() -> None:
    later_call_executed = False

    @coding_tool(risk=ToolRisk.WRITE, idempotent=False)
    def failed_write() -> str:
        """模拟执行失败的写操作。"""

        raise RuntimeError("write failed")

    @coding_tool(risk=ToolRisk.WRITE, idempotent=False)
    def later_write() -> str:
        """模拟不应继续执行的后续写操作。"""

        nonlocal later_call_executed
        later_call_executed = True
        return "unexpected"

    registry = ToolRegistry()
    registry.register(failed_write)
    registry.register(later_write)

    results = ToolExecutor(registry).execute(
        [
            {"name": "failed_write", "args": {}, "id": "call-1"},
            {"name": "later_write", "args": {}, "id": "call-2"},
        ]
    )

    assert results[0].status is ToolExecutionStatus.ERROR
    assert results[0].error_type == "RuntimeError"
    assert results[1].status is ToolExecutionStatus.SKIPPED
    assert later_call_executed is False
    assert results[1].to_tool_message().tool_call_id == "call-2"


def test_read_tool_timeout_returns_structured_error() -> None:
    @coding_tool(
        risk=ToolRisk.READ,
        idempotent=True,
        timeout_seconds=0.01,
    )
    def slow_read() -> str:
        """模拟可安全超时的只读操作。"""

        time.sleep(0.05)
        return "too-late"

    registry = ToolRegistry()
    registry.register(slow_read)

    result = ToolExecutor(registry).execute(
        [{"name": "slow_read", "args": {}, "id": "call-1"}]
    )[0]

    assert result.status is ToolExecutionStatus.ERROR
    assert result.error_type == "TimeoutError"

"""Coding Helper 的命令行入口。"""

import platform
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from coding_helper import __version__
from coding_helper.compatibility import (
    ToolCheckMode,
    ToolCompatibilityError,
    run_tool_check,
)
from coding_helper.config import Settings
from coding_helper.models import (
    ModelConfigurationError,
    ModelTarget,
    create_chat_model,
    primary_order,
)
from coding_helper.runtime import (
    PendingApproval,
    run_coding_task,
    run_readonly_question,
)
from coding_helper.tools.writes import FileWriteError, SafeFileEditor

app = typer.Typer(
    name="coding-helper",
    help="A governed and observable coding-agent CLI.",
    no_args_is_help=True,
)
console = Console()


@app.callback()
def root() -> None:
    """Coding Helper 的根命令组。

    Typer 会把只有一个命令的应用自动简化成单命令界面。显式声明 Callback
    可以让 ``doctor`` 保持为真正的子命令，后续增加 ``run`` 和 ``resume``
    时就不必破坏当前 CLI 接口。
    """


@app.command()
def doctor(workspace: Path = typer.Option(Path.cwd(), exists=True, file_okay=False)) -> None:
    """在不调用模型 API 的情况下检查本地环境。

    Typer 会根据函数签名生成 CLI 参数及校验逻辑。第一批代码只做到配置诊断；
    LangChain/LangGraph Runtime 将在独立、可审查的后续批次中引入。
    """

    settings = Settings(workspace=workspace.resolve())
    missing = settings.missing_model_settings()

    table = Table(title="Coding Helper environment")
    table.add_column("Check")
    table.add_column("Value")
    table.add_column("Status")
    table.add_row("Version", __version__, "OK")
    table.add_row("Python", platform.python_version(), "OK")
    table.add_row("Workspace", str(settings.workspace), "OK")
    table.add_row(
        "Ark configuration",
        "configured" if not missing else ", ".join(missing),
        "OK" if not missing else "MISSING",
    )
    console.print(table)

    if missing:
        console.print(
            "\n[yellow]Model calls are disabled until the missing values are "
            "added to .env.[/yellow]"
        )


@app.command("model-check")
def model_check(target: ModelTarget = typer.Argument(...)) -> None:
    """向指定模型发送一次最小请求，用于验证真实 API 连通性。

    该命令会产生少量模型费用。它只验证 LangChain 到方舟的基础文本链路，
    不代表 Tool Calling、流式响应和 Fallback 已经通过测试。
    """

    settings = Settings()
    try:
        model = create_chat_model(settings, target)
        with console.status(f"正在连接 {target.value}..."):
            response = model.invoke("这是连通性测试。请只回复 OK。")
    except ModelConfigurationError as exc:
        console.print(f"[red]配置错误：{exc}[/red]")
        raise typer.Exit(code=2) from exc
    except Exception as exc:
        # CLI 是异常边界：把 SDK 异常转换成简短提示，但保留异常类型方便排查。
        console.print(f"[red]{target.value} 调用失败：{type(exc).__name__}: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    # 不输出 response_metadata，避免未来某个供应商把敏感请求头放进元数据。
    content = str(response.content).strip()
    console.print(f"[green]{target.value} 连接成功[/green]：{content[:200]}")


@app.command("tool-check")
def tool_check(
    target: ModelTarget = typer.Argument(...),
    mode: ToolCheckMode = typer.Option(ToolCheckMode.SINGLE),
) -> None:
    """验证主模型的单工具或同响应多工具调用协议。

    该命令会产生两次小型模型请求：第一次让模型选择工具，第二次把带有
    Tool Call ID 的执行结果送回模型。辅助模型不承担主 Agent Tool Use，
    因此这里只允许检查 DeepSeek 和 GLM。
    """

    if target is ModelTarget.AUXILIARY:
        console.print("[red]辅助模型不参与主 Agent Tool Calling 检查。[/red]")
        raise typer.Exit(code=2)

    try:
        model = create_chat_model(Settings(), target)
        with console.status(f"正在检查 {target.value} {mode.value} Tool Calling..."):
            result = run_tool_check(model, mode)
    except (ModelConfigurationError, ToolCompatibilityError) as exc:
        console.print(f"[red]兼容性检查失败：{exc}[/red]")
        raise typer.Exit(code=2) from exc
    except Exception as exc:
        console.print(f"[red]模型调用失败：{type(exc).__name__}: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    console.print(
        f"[green]{target.value} {mode.value} 检查通过[/green]："
        f"tools={', '.join(result.tool_names)}, "
        f"unique_ids={len(set(result.tool_call_ids))}"
    )


@app.command()
def ask(
    question: str = typer.Argument(...),
    model: ModelTarget | None = typer.Option(None, "--model"),
    workspace: Path = typer.Option(Path.cwd(), exists=True, file_okay=False),
    thread_id: str | None = typer.Option(None, "--thread-id"),
) -> None:
    """让只读 Agent 使用文件工具回答一个仓库问题。

    不提供 ``--thread-id`` 时会创建新线程；重复传入输出中的 Thread ID，
    LangGraph 会从 SQLite Checkpoint 加载此前消息，再追加本轮问题。
    """

    settings = Settings(workspace=workspace.resolve())
    target = model or primary_order(settings)[0]
    if target is ModelTarget.AUXILIARY:
        console.print("[red]辅助模型不能作为主 Agent。[/red]")
        raise typer.Exit(code=2)

    try:
        with console.status(f"正在使用 {target.value} 分析仓库..."):
            result = run_readonly_question(
                question,
                settings=settings,
                target=target,
                thread_id=thread_id,
            )
    except (ModelConfigurationError, ValueError) as exc:
        console.print(f"[red]无法启动 Agent：{exc}[/red]")
        raise typer.Exit(code=2) from exc
    except Exception as exc:
        console.print(f"[red]Agent 运行失败：{type(exc).__name__}: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    _print_run_result(result)


def _interactive_approval(pending: PendingApproval) -> str:
    """展示脱敏审批载荷，并把用户选择转换成 Runtime 决定。"""

    console.print("\n[bold yellow]工具执行需要批准[/bold yellow]")
    console.print(f"工具：{pending.tool_name}")
    console.print(f"风险：{pending.risk}")
    console.print(f"原因：{pending.reason}")
    console.print("参数：")
    console.print_json(data=pending.arguments)
    return "approve" if typer.confirm("允许本次调用？", default=False) else "reject"


@app.command()
def run(
    task: str = typer.Argument(...),
    model: ModelTarget | None = typer.Option(None, "--model"),
    workspace: Path = typer.Option(Path.cwd(), exists=True, file_okay=False),
    thread_id: str | None = typer.Option(None, "--thread-id"),
) -> None:
    """运行可读取并安全修改文本文件的 Coding Agent。

    每个写 Tool Call 都会先保存 LangGraph Checkpoint 并暂停。CLI 展示经过
    脱敏的参数，用户明确批准后才从同一 Interrupt 恢复并执行副作用。
    """

    settings = Settings(workspace=workspace.resolve())
    target = model or primary_order(settings)[0]
    if target is ModelTarget.AUXILIARY:
        console.print("[red]辅助模型不能作为主 Agent。[/red]")
        raise typer.Exit(code=2)

    try:
        result = run_coding_task(
            task,
            settings=settings,
            target=target,
            approval_handler=_interactive_approval,
            thread_id=thread_id,
        )
    except (ModelConfigurationError, ValueError) as exc:
        console.print(f"[red]无法启动 Agent：{exc}[/red]")
        raise typer.Exit(code=2) from exc
    except Exception as exc:
        console.print(f"[red]Agent 运行失败：{type(exc).__name__}: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    _print_run_result(result)


def _print_run_result(result) -> None:
    console.print(result.answer)
    extras = ""
    pinned = getattr(result, "pinned_reference_count", 0)
    if pinned:
        extras += f" pinned={pinned}"
    todo_total = getattr(result, "todo_total", 0)
    if todo_total:
        extras += f" todos={getattr(result, 'todo_completed', 0)}/{todo_total}"
    console.print(
        f"\n[dim]thread={result.thread_id} "
        f"messages={result.message_count} tools={result.tool_call_count}"
        f"{extras}[/dim]"
    )


@app.command()
def status(
    workspace: Path = typer.Option(Path.cwd(), exists=True, file_okay=False),
) -> None:
    """打印当前 Workspace 的 progress.md 投影。"""

    progress_path = workspace.resolve() / ".coding-helper" / "progress.md"
    if not progress_path.is_file():
        console.print("还没有任务进度。")
        return
    console.print(progress_path.read_text(encoding="utf-8"))


@app.command()
def undo(
    operation_id: str | None = typer.Argument(None),
    workspace: Path = typer.Option(Path.cwd(), exists=True, file_okay=False),
) -> None:
    """在 Hash 未发生后续变化时恢复一次安全写操作。"""

    editor = SafeFileEditor(workspace.resolve())
    try:
        candidate = editor.get_undo_candidate(operation_id)
        console.print("[bold yellow]准备恢复文件[/bold yellow]")
        console.print(f"操作：{candidate.operation_id}")
        console.print(f"文件：{candidate.path}")
        console.print(f"当前应为：{candidate.after_sha256}")
        console.print(f"恢复至：{candidate.before_sha256}")
        if not typer.confirm("确认执行 Undo？", default=False):
            console.print("已取消，文件未修改。")
            return
        result = editor.undo(candidate.operation_id)
    except FileWriteError as exc:
        console.print(f"[red]无法恢复：{exc}[/red]")
        raise typer.Exit(code=2) from exc

    console.print(
        f"[green]恢复完成[/green]：{result['path']} "
        f"operation={result['operation_id']}"
    )


@app.command()
def recovery(
    workspace: Path = typer.Option(Path.cwd(), exists=True, file_okay=False),
) -> None:
    """检查因进程退出而只留下 pending 状态的文件操作。"""

    editor = SafeFileEditor(workspace.resolve())
    try:
        inspections = editor.inspect_pending_operations()
    except FileWriteError as exc:
        console.print(f"[red]无法读取 Operation Journal：{exc}[/red]")
        raise typer.Exit(code=2) from exc

    if not inspections:
        console.print("[green]没有待诊断的中断操作。[/green]")
        return
    table = Table(title="Pending file operations")
    table.add_column("Operation")
    table.add_column("Journal status")
    table.add_column("Path")
    table.add_column("Observed state")
    for item in inspections:
        table.add_row(
            item["operation_id"],
            item["operation_status"],
            item["path"],
            item["observed"],
        )
    console.print(table)


def main() -> None:
    """启动 Typer 应用。"""

    app()


if __name__ == "__main__":
    main()

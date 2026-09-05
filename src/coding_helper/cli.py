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
)

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


def main() -> None:
    """启动 Typer 应用。"""

    app()


if __name__ == "__main__":
    main()

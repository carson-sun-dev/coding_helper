"""Coding Helper 的命令行入口。"""

import platform
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from coding_helper import __version__
from coding_helper.config import Settings

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


def main() -> None:
    """启动 Typer 应用。"""

    app()


if __name__ == "__main__":
    main()

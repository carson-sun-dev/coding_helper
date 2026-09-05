# Coding Helper

一个面向本地代码仓库的命令行 Coding Agent。模型自己决定读什么、改什么、何时收工；外层负责它管得住的部分——权限、预算、副作用的可恢复、上下文压缩，以及由代码判定的完成条件。决策留给模型，边界交给工程。

它不是 Claude Code 的复刻，也不是流程写死的 Workflow。项目重点是把一个 Agent 真正投入本地改码时绕不开的工程问题走一遍：在模型保有决策权的前提下，让它的副作用可控、可观测、可恢复。

底层基于 LangChain 与 LangGraph。完整设计见 [design.md](design.md)，安装与用法见 [docs/USAGE.md](docs/USAGE.md)。

## 特点

- **副作用可恢复。** 改文件前自动备份、可回滚；进程中断后不会自动重放未完成的写。
- **先审批，后动手。** 写文件、执行命令、联网这类操作在真正生效之前暂停，等人工确认。
- **完成由代码判定。** 模型说"做完了"之后，由确定性检查（改动范围、测试等）决定是否放行，而不是听模型自述。
- **默认可观测。** 关键步骤都有结构化记录可回看，但不落密钥和完整提示词。
- **能力可扩展。** 内置工具、Skill、外部 MCP 都走同一套权限与治理。

## 局限

有意识的取舍，写在前面避免误解：

- 不是安全沙箱，不做容器隔离；Shell 命令的非文件副作用无法回滚。
- 对 Prompt Injection 只做来源标记与分层信任，不声称完全防御。

## 快速开始

需要 Python 3.11+ 和火山方舟 API。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env      # 填入 API Key 与模型 Endpoint
coding-helper doctor      # 本地配置体检
```

主要命令：`ask` 只读分析，`run` 执行可改文件的任务（写操作需确认），`status` / `trace` 查看进度与轨迹，`undo` 回滚，`eval` 跑内置冒烟任务。用法、工具清单和子 Agent 触发方式见 [docs/USAGE.md](docs/USAGE.md)。

## 测试

```bash
pytest
```

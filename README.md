# Coding Helper

面向本地代码仓库的 Coding Agent CLI。模型自己决定调用哪些工具、何时结束；Harness 负责权限、预算、副作用恢复、上下文压缩和可验证的完成条件。

这不是 Claude Code 复刻，也不是固定节点的 Workflow。详细设计见 [design.md](design.md)。

## 谁做什么

**LangChain / LangGraph 提供**

- `create_agent` 模型—工具循环
- `ChatOpenAI` 连接火山方舟 OpenAI 兼容接口
- SQLite Checkpoint、Interrupt / Resume
- 模型/工具调用次数上限

**本项目自研**

- `allow / ask / deny` 权限，写操作在副作用之前审批
- 文件 Hash 乐观锁、Preimage、原子写、Undo、崩溃后的 `interrupted` 标记（不自动重放）
- 确定性 Tool Result 裁剪 + 豆包结构化摘要；豆包不接管主 Agent
- 保守项目记忆（`memory.json` 为源，`memory.md` 是投影）
- Todo / `progress.md`、一层 Explorer/Reviewer Subagent
- Skill 按需加载、MCP 延迟连接与熔断、带 SSRF 防护的 `web_fetch`
- 完成门槛（Git diff、删除、禁止路径、Todo、配置的测试/lint）
- `events.jsonl` 轨迹与 `coding-helper eval` 冒烟评测

## 威胁边界（请勿过度宣传）

- Shell 不是容器沙箱，只做命令分类和超时杀进程组。
- 不能回滚任意 Shell 副作用。
- 仓库文件、`@` 引用、网页和 MCP 输出都是不可信数据，不能提升权限。
- Prompt Injection 没有被“完全防御”。
- Fallback 只在超时 / 429 / 5xx 时切到另一主模型，答案差不会换模型。

## 要求

- Python 3.11+
- 火山方舟 API：DeepSeek、GLM 作双主模型，Doubao 只做摘要/记忆

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

在 `.env` 填写 `ARK_API_KEY` 和三个模型 Endpoint，不要提交 `.env`。

```bash
coding-helper doctor
coding-helper model-check deepseek
coding-helper tool-check deepseek --mode multi
```

## 常用命令

| 命令 | 作用 |
|---|---|
| `ask "问题"` | 只读分析，可用 `@file` / `@dir/` |
| `run "任务"` | 可改文本文件；写操作需确认 |
| `status` | 打印 `progress.md` |
| `trace` | 最近 `events.jsonl` 事件 |
| `undo` | 在 Hash 未变时恢复一次安全写 |
| `recovery` | 诊断 pending 写操作并标为 interrupted |
| `eval` | 内置 Fake Model 冒烟任务 `fix_add`（n=1） |
| `mcp-check` | 检查已配置的 MCP Server |

示例：

```bash
coding-helper ask "入口在哪里？"
coding-helper run "参考 @src/auth.py 修复登录并补测试"
coding-helper eval
```

运行数据在工作区 `.coding-helper/`（已 gitignore）：Checkpoint、轨迹、备份、记忆和评测沙箱。

## 评测与指标

`coding-helper eval` 用脚本化 Fake Model 验证写路径、审批和自动验收，**不评价真实模型写代码的能力**。当前任务集规模为 **n=1**，报告原始次数，不写成功率百分比。

简历或对外数字必须来自后续真实任务测量，并注明任务集规模。本仓库尚未提供这类测量结果。

## 已知限制

- 完成门槛的 Reviewer Subagent 尚未接入。
- 消融评测（Todo / Subagent / Compact 分组）尚未做。
- Web Search 没有独立内置实现，可通过通用 MCP 挂搜索 Server。
- 真实 DeepSeek / GLM 带测试的端到端改码任务仍需人工跑，未写入本 README。
- 没有演示 GIF。

## 测试

```bash
pytest
```

核心路径由 Fake Model 覆盖。`model-check` / `tool-check` 会打真实 API，产生费用。

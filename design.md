# Coding Helper CLI 设计文档

## 1. 项目概述

Coding Helper 是一个面向本地代码仓库的可观测 Coding Agent CLI。项目使用 LangChain Agent 与 LangGraph Durable Runtime 承担通用的模型—工具循环、检查点、中断恢复和事件流，在其上实现 Coding Agent 特有的能力治理：

- 风险感知的工具权限与人工审批
- Built-in Tool、Skill、MCP 的统一注册和按需加载
- 隔离且受预算约束的 Subagent
- 长上下文压缩、会话记忆和任务恢复
- 可回滚的文件修改与副作用日志
- 长任务进度投影、执行轨迹和成本统计
- Git diff、测试与 lint 驱动的确定性完成检查
- 可复现任务集与 Harness 机制消融评测

本项目的目标不是复刻 Claude Code，也不是实现固定节点的 Workflow。决策权仍属于模型；Harness 负责向模型提供环境、边界、状态、恢复能力和可验证的完成条件。

## 2. 项目目标与简历定位

现有项目已经展示 FSM、LangGraph Workflow、RAG、模型路由、全栈交付和生产部署。本项目重点补充 Agent Harness 工程能力：

1. 模型自主决定工具调用与停止时机。
2. Harness 对模型行为实施权限、预算和副作用治理。
3. 长任务能够观察、暂停、恢复、回滚和验证。
4. 外部能力能够通过 Skill 和 MCP 动态扩展。
5. 通过轨迹和评测证明机制效果，而非仅罗列功能。

预期简历叙事：

- 基于 LangChain Agent 与 LangGraph Durable Runtime 构建 Coding Agent CLI，统一管理本地工具、按需 Skill、MCP 动态工具和隔离 Subagent。
- 设计风险感知的 `allow / ask / deny` 权限管线，并通过文件 Preimage、原子写入、Operation Journal 和 Undo 提供可恢复副作用执行。
- 实现分层 Tool Result 裁剪、豆包辅助上下文摘要、持久化任务和 Markdown 长任务进度投影。
- 建立代码任务评测及轨迹观测体系，对 Todo、Subagent、动态工具和上下文压缩进行消融比较。

所有最终简历指标必须来自实际测量，并注明任务集规模，不能预先虚构成功率或成本收益。

### 2.1 开发、审查与 GitHub 协作约束

项目采用小批次、人工审查后继续的开发方式。每个代码批次必须满足：

- 只解决一个清晰的子问题，不在同一批次横跨多个无关模块。
- 默认控制在 1–4 个核心文件和约 100–300 行可读净变更内；依赖锁文件、自动生成文件和测试夹具单独说明，不以机械行数作为唯一标准。
- 开始前说明本批次目标、预计修改文件和验收方法。
- 完成后运行与该批次匹配的最小测试，汇报变更、测试结果、已知限制和下一批建议。
- 完成一个批次后停止，等待人工审查和明确继续指令，不自动开始下一阶段。
- 用户提出修改意见时，先处理当前批次，不叠加后续功能。

项目同时承担框架学习用途。所有代码注释和文档字符串统一使用中文。首次出现 LangChain、LangGraph、Middleware、Checkpoint、Interrupt、Tool Schema 等概念时，应在相邻注释或文档字符串中用面向新手的语言说明它是什么、为什么在这里使用，以及它与相邻组件的边界。代码注释仍强调“为什么”和关键不变量，而不是逐行翻译代码。以下位置必须有简洁核心注释：

- LangGraph Resume/Replay 与副作用顺序。
- 多 Tool Call 并发、锁和冲突处理。
- 权限升级、路径边界和危险命令判断。
- 文件 Preimage、原子替换和 Undo。
- Tool Result 裁剪、上下文保留规则。
- Skill/MCP 不可信内容与指令优先级。

简单赋值、显然的控制流和由类型签名已经表达的内容不添加冗余注释。

GitHub 仓库创建、Remote 配置和 Push 暂由用户负责。开发过程不自行创建远程仓库或上传代码；提交前确保 `.env`、API Key、本地 Checkpoint、轨迹、备份和 Trash 已被 `.gitignore` 排除。

## 3. 范围

### 3.1 MVP 必做

- 火山方舟 DeepSeek-V4-Pro、GLM-5.2 双主模型
- Doubao-Seed-2.0-lite 负责摘要、压缩和结构化记忆提取
- LangChain Agent Loop
- LangGraph SQLite Checkpoint、Interrupt/Resume 和 Event Streaming
- 文件读取、搜索、补丁修改、Shell、Git 工具
- `@file`、`@directory` 和文本型 PDF 显式上下文引用
- 受治理的 Web Search/Web Fetch；搜索能力优先通过 MCP 接入
- 轻量 `@coding_tool` 装饰器和能力注册表
- Tool Middleware 治理链
- `allow / ask / deny` 权限策略
- Todo、持久化任务和 `progress.md`
- Explorer、Reviewer Subagent
- Skill 元数据扫描和按需加载
- MCP stdio Server 按需连接
- Tool Result 裁剪和历史摘要
- JSONL 执行轨迹、Token、时延及成本记录
- 文件备份、Trash 和 Undo
- 测试、lint、Git diff 完成门槛
- Fake Model 驱动的确定性运行时测试
- 小型代码任务集和消融评测

### 3.2 条件允许时添加

- 长时间 Shell 命令后台运行
- 流式模型输出和流式 Tool Call 参数
- 静态 HTML 轨迹报告
- LangChain Tool 兼容适配器
- Docker 隔离执行

### 3.3 非目标

- Cron 调度
- 自治 Agent Team
- 固定 Workflow Runtime
- LSP、Tree-sitter 或向量代码检索
- React 管理后台
- 完整 OS 级安全沙箱
- PDF 图片理解、OCR、音视频和复杂 Office 版式解析
- 完整浏览器自动化
- 训练、微调或强化学习

## 4. 技术选型

### 4.1 主要依赖

- Python 3.11+
- `langchain`
- `langchain-openai`
- `langgraph`
- `langgraph-checkpoint-sqlite`
- `langchain-mcp-adapters`
- `pydantic`
- `typer`
- `rich`
- `pytest`

### 4.2 使用框架的边界

框架负责通用运行时：

- Agent Loop 和消息协议
- 模型 Tool Call 解析
- Model Retry 和 Fallback
- 模型及工具调用次数限制
- 对话摘要基础设施
- Checkpoint、Interrupt、Resume
- Event Streaming
- MCP 基础协议适配

项目自行实现特有能力：

- Tool 风险、幂等、超时和来源元数据
- 动态能力目录和按需工具注入
- Shell 参数级风险识别
- Skill 指令边界
- MCP 命名空间、延迟连接和熔断
- 安全文件修改、Preimage、Trash 和 Undo
- Tool Result 分层裁剪与敏感信息脱敏
- 长任务 Markdown 进度投影
- Coding 完成检查
- 本地轨迹、评测和消融实验

不直接采用以下通用组件：

- 全局 Tool Retry：可能重复执行非幂等副作用。
- LLM Tool Selector：增加额外模型调用，优先使用确定性目录发现。
- 通用 Human-in-the-loop：无法充分识别 Shell 参数和目标路径风险。
- 通用 Todo：项目 Todo 需要驱动持久化状态和 `progress.md`。
- LangSmith：MVP 使用本地轨迹，不依赖外部 SaaS。

## 5. 总体架构

```text
User
  │
  ▼
Typer + Rich CLI
  │
  ├── progress/status/approval rendering
  └── session commands
  │
  ▼
LangChain create_agent
  │
  ▼
LangGraph Durable Runtime
  ├── SQLite Checkpointer
  ├── Interrupt / Resume
  └── Event Stream
  │
  ▼
Agent Middleware
  ├── capability discovery
  ├── dynamic tool injection
  ├── model routing
  ├── context compaction
  └── budget / stuck detection
  │
  ▼
Tool Governance Pipeline
  ├── validation
  ├── routing
  ├── risk classification
  ├── permission / approval
  ├── concurrency control
  ├── timeout / retry policy
  ├── execution
  ├── result clipping / redaction
  └── trajectory / progress update
  │
  ├── Built-in Tools
  ├── Skills
  ├── MCP Tools
  └── Subagents
```

## 6. 模型设计

### 6.1 双主模型与固定角色路由

用户可为每个 Session 选择 DeepSeek-V4-Pro 或 GLM-5.2 作为主模型；另一主模型只在超时、限流、5xx 等服务异常时作为 Fallback。Doubao-Seed-2.0-lite 固定承担低风险、结构明确的辅助任务。模型路由由 Harness 根据用户选择和任务类型确定，不让模型自行选择模型。

```python
class ModelRole(str, Enum):
    PRIMARY = "primary"
    PRIMARY_FALLBACK = "primary_fallback"
    COMPACTION = "compaction"
    MEMORY = "memory"
```

- DeepSeek-V4-Pro 或 GLM-5.2 `PRIMARY`
  - 主 Agent 决策
  - Todo 规划
  - 代码探索和修改
  - 测试失败分析
  - Explorer/Reviewer Subagent
  - 最终结果归纳
- 未被选择的另一主模型 `PRIMARY_FALLBACK`
  - 只在主模型发生可重试的服务异常时接管
  - 不因回答质量较差或工具参数错误自动切换
- Doubao-Seed-2.0-lite `COMPACTION`
  - 历史对话摘要
  - 超长结果语义压缩
- Doubao-Seed-2.0-lite `MEMORY`
  - 会话记忆候选提取

权限判断、测试是否通过、修改是否越界和任务是否达到确定性验收标准必须由代码判断。

### 6.2 配置

```yaml
models:
  deepseek: ${ARK_DEEPSEEK_MODEL}
  glm: ${ARK_GLM_MODEL}
  default_primary: ${ARK_DEFAULT_PRIMARY}
  compaction: ${ARK_DOUBAO_ENDPOINT}
  memory: ${ARK_DOUBAO_ENDPOINT}

ark:
  api_key: ${ARK_API_KEY}
  base_url: ${ARK_BASE_URL}
```

环境变量和 API Key 不写入仓库、轨迹或模型上下文。

### 6.3 方舟兼容性验证

正式实现前完成最小技术验证：

1. DeepSeek-V4-Pro 和 GLM-5.2 普通对话。
2. 两个主模型分别执行单 Tool Call。
3. 两个主模型分别执行同一响应中的多个 Tool Call。
4. Tool Call 参数流式增量。
5. Token Usage 获取。
6. DeepSeek 与 GLM 在模拟服务错误时互相 Fallback。
7. Doubao-Seed-2.0-lite 按指定 JSON Schema 生成摘要和记忆候选。
8. LangGraph Interrupt 后恢复。
9. SQLite Checkpoint 跨进程恢复。

如果 `ChatOpenAI + 火山方舟` 在 Tool Calling 或流式事件上不兼容，则实现薄层 `ArkChatModel`，上层架构不变。

## 7. CLI 与可解释进度

不展示或持久化完整思维链。CLI 展示结构化执行轨迹、简短行动理由和下一步：

```text
● Planning     任务已拆分为 4 步
● Exploring    正在定位认证相关代码
→ Search       "refresh_token" in src/
← Found        8 matches · 42ms
● Editing      已修改 2 个文件
● Verifying    正在运行 pytest
✓ Completed    18 tests passed
```

模型可以输出简短 Reason/Next，但高层阶段由 Harness 根据事件确定，避免依赖模型自报进度。

```text
Reason: 失败集中在 Token 过期分支，需要检查刷新逻辑和现有测试。
Next: 读取认证实现与相关测试。
```

CLI 模式：

```bash
coding-helper run "修复登录问题"
coding-helper run "修复登录问题" --verbose
coding-helper resume <session-id>
coding-helper status [session-id]
coding-helper history
coding-helper inspect <session-id>
coding-helper undo [operation-id]
```

默认模式展示阶段、Todo、工具名称和摘要。Verbose 模式额外展示安全参数、裁剪后的结果、权限决策和模型统计，但仍不输出密钥或完整思维链。

## 8. Tool Decorator 与能力注册表

### 8.1 装饰器

```python
@coding_tool(
    risk="write",
    idempotent=False,
    timeout=30,
    retry_policy="never",
    tags=["filesystem"],
)
def apply_patch(path: str, patch: str) -> str:
    ...
```

装饰器读取函数签名和 Pydantic 类型生成 JSON Schema，并转换为 LangChain `StructuredTool`。自定义元数据保存在独立注册表中，不能只依赖框架 Tool 对象。

### 8.2 ToolSpec

```python
class ToolSpec(BaseModel):
    name: str
    description: str
    input_schema: dict
    source: Literal["builtin", "skill", "mcp"]
    risk: Literal["read", "write", "execute", "destructive"]
    idempotent: bool
    timeout_seconds: float
    retry_policy: Literal["never", "safe", "transient"]
    tags: list[str]
    loaded: bool
    server_name: str | None = None
```

工具名称必须全局唯一。内部审计使用带冒号的 Canonical Name：

```text
builtin::read_file
skill::test-runner::detect_tests
mcp::github::get_issue
```

OpenAI 兼容接口通常只接受字母、数字、下划线和连字符，因此实际发给模型的动态工具名使用双下划线，例如 `mcp__github__get_issue`。Registry 负责维护两者映射，核心 Built-in Tool 可以继续使用简短名称 `read_file`。

## 9. 工具发现、路由与按需加载

### 9.1 常驻工具

模型始终可见：

- `read_file`
- `search`
- `apply_patch`
- `shell`
- `git_diff`
- `todo_write`
- `delegate`
- `discover_capabilities`

### 9.2 两阶段加载

```text
用户任务
  → 核心工具 + 精简能力目录
  → discover_capabilities(query)
  → 确定性筛选 Skill/MCP 候选
  → 加载完整指令或 Tool Schema
  → 下一轮注入选中工具
```

筛选依据：

- 查询词和能力标签
- 当前 Todo
- 用户权限
- Server 是否启用
- Subagent 角色
- Workspace 类型
- 工具风险

不额外调用 LLM 进行工具筛选。设置单轮和会话级动态工具上限，使用 LRU 淘汰长时间未使用的 Tool Schema。淘汰只移除模型可见 Schema，不销毁仍在执行的工具。

## 10. Skills

启动时只扫描 Skill 元数据，不加载完整正文：

```text
code-review：检查代码质量和潜在缺陷
test-runner：识别并运行项目测试
dependency-debug：排查依赖问题
```

Agent 选择 Skill 后才读取完整 `SKILL.md`，并在当前 Agent 或 Subagent 上下文中激活。

规则：

- 只从配置允许的项目目录和用户目录加载。
- Skill 内容是低于系统指令和权限策略的知识。
- 项目 Skill 不能覆盖用户 Skill，名称冲突时使用命名空间。
- 限制文件大小和一次加载数量。
- 缓存已加载内容，并记录来源、Hash、加载原因和 Token 增量。
- Skill 中的外部命令仍需经过工具权限管线。
- 仓库内 Skill 视为不可信内容，不能自行扩大权限。

## 11. MCP

### 11.1 延迟连接

启动时只加载 Server 名称、描述和启动配置，不连接全部 Server：

```yaml
mcp_servers:
  filesystem:
    transport: stdio
    command: ...
    description: Extended filesystem operations
```

首次匹配到某 Server 后：

1. 启动 Server。
2. 建立或复用 ClientSession。
3. 执行 `list_tools`。
4. 转换为统一 ToolSpec。
5. 为工具添加 `mcp::<server>::` 前缀。
6. 将选中工具注入下一轮。

### 11.2 生命周期和隔离

- 配置 Server Allowlist。
- 启动、发现和工具调用分别设置超时。
- 单 Server 故障不终止主 Agent。
- 断线后最多自动重连一次。
- 连续失败后打开 Circuit Breaker。
- MCP 工具仍经过统一权限、预算和轨迹管线。
- MCP 返回内容作为不可信外部数据，不能覆盖系统指令。
- 会话结束时关闭持久连接和 stdio 子进程。
- 区分无状态和有状态 MCP Session，不能错误地为有状态服务每次创建新会话。

## 12. Todo、任务和 Subagent

### 12.1 Todo

Todo 是长任务进度的结构化来源：

- `pending`
- `in_progress`
- `completed`
- `blocked`
- `cancelled`

约束：

- 同时最多一个 `in_progress`。
- 完成前必须存在可验证结果。
- 状态变化写入 Checkpoint、事件日志和 `progress.md`。
- 上下文压缩后重新注入当前 Goal、Todo 和 Blocker。

### 12.2 Subagent

MVP 提供：

- Explorer：只读探索代码，返回文件、符号、证据和建议。
- Reviewer：读取 diff 和测试结果，返回结构化审查意见。

Subagent 规则：

- 使用独立消息上下文和 Token 预算。
- 仅获得角色所需的最小工具集。
- 默认无写权限。
- 主 Agent 只接收结构化最终结果，不接收全部内部消息。
- 限制最大深度为一层，禁止递归创建。
- Subagent 的模型、Token、工具调用和耗时独立计量。
- Subagent 失败作为可恢复 Tool Result 返回，不直接终止主循环。

## 13. 多 Tool Call 并发

模型可能在同一响应中返回多个 Tool Call。调度器先分析读写集合和风险，再决定并发计划。

### 13.1 可并行

- 多个文件读取
- 多个搜索
- 无状态且只读的 MCP 调用
- 访问不同资源的只读 Git 查询

### 13.2 必须串行

- 文件写入和删除
- Shell 命令
- Git 修改操作
- 有状态 MCP 调用
- 访问相同路径或未知资源集合的调用
- 风险为 `destructive` 的任何调用

### 13.3 冲突规则

- 调度前规范化绝对路径并解析符号链接。
- 每个文件写操作携带读取时的内容 Hash。
- 两个调用写同一路径时按模型顺序串行执行。
- 一个调用读、另一个调用写同一路径时先读后写，或要求模型重新读取。
- 工作区级 Shell 命令与文件写入默认互斥。
- 并发结果按原始 Tool Call 顺序回填，保留 Tool Call ID 对应关系。
- 一个并行只读调用失败时，不取消其他安全只读调用；分别返回结果。
- 任一写调用失败后停止后续依赖写操作，并把已完成操作写入 Journal。
- 权限审批按调用分别执行，不允许一次批准隐式覆盖其他调用。

第一版使用保守调度：只并行相邻且明确标记为 `read + idempotent` 的工具，其余全部串行。只读调用不能跨过中间写操作提前执行，否则会破坏模型给出的顺序语义。通用 `asyncio` 超时无法终止已经在线程中运行的同步写函数，因此副作用工具必须在自身实现可取消边界，例如 Shell 工具负责终止完整子进程组；Executor 不能在后台写操作仍运行时伪装成已经安全超时。

## 14. Tool Governance Middleware

工具调用统一经过 Onion Middleware：

```text
Tool Request
  → Schema Validation
  → Capability/Namespace Resolution
  → Workspace and Path Validation
  → Risk Classification
  → Permission Decision
  → Interrupt Approval
  → Concurrency Scheduling
  → Idempotency/Conflict Check
  → Timeout/Retry Policy
  → Execution
  → Result Validation
  → Clipping/Redaction
  → Trajectory and Progress Update
```

中间件必须对成功、失败、拒绝、超时和取消都产生完整事件。不能因中途异常跳过清理、解锁或日志收尾。

## 15. 权限与信任边界

### 15.1 决策

- `allow`：无需用户确认。
- `ask`：通过 LangGraph Interrupt 暂停，展示影响后请求确认。
- `deny`：禁止执行并返回原因。

示例：

```text
git status                allow
pytest                    allow
apply_patch src/a.py      ask/allow by policy
rm file.txt               ask
git reset --hard          deny by default
curl ... | sh             deny
read ~/.ssh/id_rsa        deny
```

### 15.2 路径和敏感信息

- 所有路径相对于已批准 Workspace 解析。
- 检查 `..`、绝对路径和符号链接逃逸。
- 默认拒绝 `.env`、SSH Key、云凭据和系统密钥目录。
- Shell 只传递 Allowlist 环境变量。
- API Key 不进入模型消息和事件日志。
- 日志写入前执行敏感信息脱敏。

### 15.3 Prompt Injection

代码、README、Skill 和 MCP 输出都可能包含恶意指令：

- 明确标记内容来源。
- 外部内容不能提升权限。
- 文件中的命令建议不能跳过审批。
- Skill 不能覆盖系统和用户指令。
- MCP 返回值只能作为数据。
- 记录引发权限请求的来源。

项目不宣称彻底解决 Prompt Injection，只宣称建立来源标记、最小权限和分层信任边界。

## 16. 文件安全、删除和 Undo

### 16.1 写入事务

```text
create operation_id
  → record pending
  → validate path and current hash
  → save preimage + mode + hash
  → write temporary file
  → atomic replace
  → verify resulting hash
  → record completed
```

要求：

- 保留原始换行符和编码。
- 二进制文件默认拒绝编辑。
- 超大文件不整体加载。
- 文件在读取后被外部修改时拒绝覆盖。
- 每个 Operation 可定位到 Session、Tool Call 和 Todo。

### 16.2 删除

- 删除默认移动到 `.coding-helper/trash/<operation-id>/`。
- 不默认执行物理删除。
- 批量删除、工作区外删除和不可逆删除必须明确说明影响并获得批准。
- `rm`、`rmdir`、`git clean`、`git reset --hard`、`find -delete` 默认 ask 或 deny。

### 16.3 Undo

```bash
coding-helper undo
coding-helper undo <operation-id>
coding-helper recovery
```

恢复前检查当前文件 Hash。如果文件已经被后续操作修改，不直接覆盖，而是提示冲突并拒绝自动恢复。`recovery` 只诊断 Journal 中的 `pending`/`undo_pending` 记录，比较当前文件与 before/after Hash，不自动重放或回滚副作用。

Shell 字符串策略不是完整安全沙箱，脚本和间接命令可能绕过静态识别。MVP 只能称为策略防护和可恢复执行；强隔离需要 Docker 或 OS Sandbox。

## 17. Shell 执行

- 使用独立进程组。
- 设置执行超时。
- 超时或取消时终止整个子进程树。
- 异步读取 stdout/stderr，避免管道阻塞。
- 限制模型可见输出大小，完整输出落盘。
- 固定 Workspace CWD。
- 环境变量使用 Allowlist。
- 检测可能等待交互输入的命令。
- 默认串行执行。
- 非零退出码作为 Tool Result 返回，不视为 Harness 崩溃。
- 未完成的 Shell 操作在 Resume 时不自动重放。

后台任务若实现，必须有 Job ID、状态查询、日志读取、取消和完成通知，不能只启动无人管理的线程。

## 18. 重试、幂等和错误处理

### 18.1 Tool Error 类型

```text
validation_error
permission_denied
path_violation
conflict
timeout
process_error
mcp_unavailable
transient_error
internal_error
```

错误返回包含：

- 错误类型
- 安全的错误消息
- 是否可重试
- 建议修复动作
- Operation/Tool Call ID

### 18.2 重试边界

- Read/Search：瞬时失败可以安全重试。
- 模型调用：指数退避加随机抖动。
- Apply Patch：仅在确认未产生副作用且 Hash 未变化时重试。
- Shell：默认不自动重试。
- 删除和外部写入：绝不自动重试。
- MCP：只有声明为幂等且属于瞬时错误时重试。

相同工具、相同参数、相同错误连续出现两次后停止自动重试。

### 18.3 API 异常

- 429、超时、5xx：有界指数退避。
- 当前主模型出现可重试服务异常：切换到另一主模型并记录降级。
- Context 超限：执行压缩后重试一次。
- 非法 Tool Call：返回结构化纠正消息。
- 主模型 Fallback 不能绕过原模型的权限和预算限制，Doubao-Seed-2.0-lite 不接管主 Agent。

### 18.4 崩溃恢复

- Checkpointer 保存 Agent 状态。
- Tool Operation 独立保存 `pending/completed/failed/interrupted`。
- 重启时将未完成写操作标记为 `interrupted`。
- 不自动重复执行修改型工具。
- 比较当前文件和 Preimage Hash，再决定保留或恢复。
- LangGraph Interrupt 前不得执行非幂等副作用。
- 需要重放的操作使用 Idempotency Key 或 read-before-write。

## 19. 卡死检测与预算

检测条件：

- 连续调用相同工具和参数。
- Todo 长时间没有变化。
- 连续测试输出完全相同。
- 同一工具错误重复。
- 多轮只输出文本但不推进任务。
- 超过最大修复轮数。

Session 预算：

- 最大模型调用次数
- 最大工具调用次数
- 最大 Token
- 最大估算成本
- 最大运行时间
- 最大连续错误数
- 最大修复轮数
- 最大 Subagent 数量和深度

达到限制后：

1. 停止继续执行。
2. 保存 Checkpoint。
3. 更新 `progress.md` 的 Blocker。
4. 输出已完成内容、当前状态和推荐下一步。
5. 允许用户调整预算后 Resume。

## 20. 上下文管理与记忆

### 20.1 分层处理

达到预算阈值后依次：

1. Tool 自身限制原始输出。
2. 保留摘要，将完整输出落盘并提供引用。
3. 清除旧的冗长 Tool Result。
4. 使用豆包摘要旧对话。
5. 重新注入关键运行状态。

必须保留：

- 用户 Goal 和验收条件
- 当前 Todo 和 Blocker
- 权限决定
- 已修改文件及 Git diff 摘要
- 最近测试错误
- 未完成 Tool Call
- 已激活 Skill/MCP 名称
- 关键设计决策

### 20.2 记忆

MVP 只实现保守记忆：

- 项目约定
- 用户明确要求长期保留的偏好
- 已验证的构建和测试命令
- 经确认的重要架构事实

豆包只负责提取候选，写入前由规则去重并限制大小。来自未验证文件或失败工具结果的内容不能直接成为长期事实。

## 21. 长任务进度与持久化

```text
.coding-helper/
├── coding-helper.db
├── progress.md
├── events.jsonl
├── operations.jsonl
├── memory.md
├── outputs/
├── backups/
└── trash/
```

- `coding-helper.db`：LangGraph Checkpoint 和必要的 Session 元数据。
- `events.jsonl`：追加式 Agent、模型、工具、权限和上下文事件。
- `operations.jsonl`：文件和外部副作用操作。
- `progress.md`：结构化状态的人类可读投影。
- `outputs/`：被裁剪的完整工具输出。
- `backups/`：写入前 Preimage。
- `trash/`：软删除内容。

`progress.md` 不由模型直接自由编辑，而是由任务状态渲染：

```markdown
# Goal
修复登录 Token 刷新失败并添加测试

## Status
Executing — 2/4 tasks completed

## Tasks
- [x] 定位失败路径
- [x] 检查刷新逻辑
- [ ] 修改实现并添加测试
- [ ] 运行完整测试

## Decisions
- 问题由并发刷新覆盖新 Token 引起

## Modified Files
- src/auth/token.py
- tests/test_token.py

## Latest Verification
- Targeted tests: 6 passed
- Full suite: not run

## Blockers
None

## Next Action
运行认证模块完整测试
```

更新时机：

- Todo 变化
- 文件操作完成或失败
- 测试结束
- Subagent 返回
- 权限被拒绝
- 上下文压缩前
- Session 暂停、失败或完成

同一 Workspace 同时只允许一个写入型 Session，通过锁文件保护。

## 22. Coding 完成门槛

模型提出结束后，Harness 进入 Verification：

1. 获取 Git diff 和修改文件清单。
2. 检查是否修改允许范围外的文件。
3. 检查是否存在未解释的删除和二进制变化。
4. 运行目标测试。
5. 运行配置的 lint/type check。
6. 检查 Todo 是否全部完成或明确取消。
7. Reviewer Subagent 审查 diff。
8. 生成最终摘要。

失败结果重新送回主 Agent，允许有界修复。达到最大修复轮数后停止并报告，不无限循环。

LLM Reviewer 仅提供补充意见，测试结果、退出码、diff 范围和预算由代码判定。

## 23. 事件与可观测性

核心事件：

```text
SessionStarted
PhaseChanged
ModelStarted / ModelCompleted / ModelFailed
ToolRequested / ToolApproved / ToolDenied
ToolStarted / ToolCompleted / ToolFailed
TodoChanged
SkillLoaded
MCPConnected / MCPFailed
SubagentStarted / SubagentCompleted
ContextCompacted
OperationRecorded / OperationUndone
VerificationCompleted
SessionPaused / SessionCompleted / SessionFailed
```

每次模型调用记录：

- Model Role 和 Endpoint
- 输入/输出 Token
- 时延
- 重试次数
- Fallback 状态
- 估算成本

每次 Tool Call 记录：

- Tool、来源和命名空间
- 安全参数摘要
- 风险和权限决定
- 开始/结束时间
- 输出长度和裁剪比例
- 状态和错误类型
- 关联 Todo、Subagent 和 Operation

## 24. 测试策略

### 24.1 Fake Model

运行时逻辑使用脚本化 Fake Model 确定性测试：

```python
model = FakeChatModel([
    tool_call("read_file", {"path": "a.py"}),
    tool_call("apply_patch", {"path": "a.py", "patch": "..."}),
    final_answer("completed"),
])
```

覆盖：

- Agent Loop
- 单个和多个 Tool Call
- 多个只读工具并发
- 写工具串行和冲突
- 参数错误后修复
- 权限批准、拒绝和 Resume
- API Retry/Fallback
- Context Compact
- Subagent 隔离
- MCP 失败
- Tool Result 裁剪
- 重复调用检测
- 写操作中断和 Undo
- Checkpoint 跨进程恢复
- `@` 引用的路径越界、大文件截断、目录 Manifest 和无文本 PDF
- Web Fetch 的 SSRF、重定向、大小限制、超时和不可信内容标记

### 24.2 集成测试

使用真实方舟 Endpoint 验证：

- DeepSeek-V4-Pro 和 GLM-5.2 Tool Calling
- Doubao-Seed-2.0-lite 结构化摘要和记忆候选
- 两个主模型之间的 Fallback
- Token Usage
- 简单代码修改任务

集成测试默认不在普通单元测试中执行，避免产生不可控费用。

### 24.3 阶段四验收示例

在临时 Git 仓库中准备一个计算器 Bug，并让 Fake Model 依次读取文件、请求补丁、运行测试和结束。在补丁审批处主动终止进程，再通过 `resume` 恢复并批准。

必须验证：

- 补丁只执行一次，不因 LangGraph 重放重复写入。
- Operation Journal 只有一个完成的写操作。
- Preimage 能恢复修改前文件。
- Todo 和 `progress.md` 从 Paused 正确推进至 Verifying。
- 测试通过后才允许 Session 完成。
- 同一响应中的多个只读 Tool Call 并行执行，写操作仍然串行。

阶段四评价的是 Harness 机制正确性，不评价真实模型的代码能力。

## 25. 评测

准备 6–10 个可自动验证的小型代码任务：

- 修复单元测试失败
- 添加带测试的小功能
- 跨文件重命名
- 定位异常来源
- 修改配置并验证
- 权限限制下完成任务
- 长上下文代码库问答
- 需要 Explorer Subagent 的任务

比较：

1. 基础 Agent。
2. 增加 Todo。
3. 增加动态 Skill/Tool。
4. 增加 Subagent。
5. 增加上下文压缩。
6. 完整 Harness。

指标：

- 任务成功率
- 平均模型轮数
- 输入/输出 Token
- 时延和估算成本
- Tool Error Rate
- 重复 Tool Call 数
- 人工审批次数
- 越权操作拦截数
- Context Compact 次数
- 恢复后任务完成情况

为确保可复现，保存：

- Model Endpoint
- Prompt 版本
- Git Commit
- 配置 Hash
- 初始代码快照
- 任务验收命令
- 温度等推理参数
- 人工批准记录

### 25.1 阶段五评测示例

示例任务：

```text
修复并发刷新 Token 时旧 Token 覆盖新 Token 的问题，
添加回归测试，不允许修改数据库模块。
```

自动验收包括目标测试、修改文件范围、删除检查和预算检查。基础组只开放固定主模型与核心 Read/Search/Edit/Shell；完整组在相同模型、参数和初始 Git 快照上加入 Todo、Explorer Subagent、动态能力、Tool Result 裁剪、上下文压缩和完成门槛。

每组重复运行并记录成功、测试、越界修改、轮数、Token、耗时、成本、重复 Tool Call 和人工审批。样本较小时报告原始次数和任务集规模，不使用夸大的百分比结论。

专项机制评测：

- 多 Tool 并行：四个各延迟 500ms 的只读工具，对比串行和并发耗时，并核对 Tool Call ID 回填顺序。
- 动态加载：注册 30 个模拟工具，对比全部注入与按需注入的 Schema Token 和选取正确率。
- 权限：在仓库文档中放置恶意删除指令，验证其不能跳过审批。
- 上下文压缩：构造超长测试输出，验证压缩后仍保留失败用例、Todo 和修改文件。
- Undo：修改后恢复，验证内容 Hash、权限和 Git diff 回到操作前状态。

阶段五评价的是完整 Harness 对真实模型任务过程的影响，并负责形成可复现报告、README、演示和真实简历指标。

## 26. 实现顺序

### 阶段一：兼容性和垂直切片

1. 初始化 Python 项目和配置。
2. 验证方舟 DeepSeek-V4-Pro、GLM-5.2、Doubao-Seed-2.0-lite 与 LangChain。
3. 建立 `create_agent + SQLite Checkpointer + Event Stream`。
4. 实现 `@` 引用解析、一个只读 Tool 和一个写 Tool。
5. 验证文本文件、目录 Manifest 和文本型 PDF 输入。
6. 完成一次从任务到测试验证的端到端运行。

### 阶段二：工具治理

1. 实现 `@coding_tool` 和 ToolSpec Registry。
2. 实现 Middleware Chain。
3. 加入权限、Interrupt 和 Resume。
4. 加入文件 Hash、Preimage、原子写和 Undo。
5. 加入 Shell 进程治理。
6. 加入多 Tool Call 保守并发调度。

### 阶段三：复杂任务能力

1. Todo 和 `progress.md`。
2. Explorer/Reviewer Subagent。
3. Skill 目录和按需加载。
4. MCP 延迟连接和命名空间。
5. 通过 MCP 接入 Web Search，并实现受治理的 Web Fetch。
6. 上下文裁剪、摘要和项目记忆。

### 阶段四：可靠性和验收

1. 错误分类、Retry/Fallback 和预算。
2. 卡死检测和崩溃恢复。
3. Coding 完成门槛。
4. Fake Model 单元测试。
5. 真实模型集成测试。

### 阶段五：评测和项目包装

1. 构建代码任务集。
2. 执行消融评测。
3. 输出结果和已知限制。
4. 完成架构图、README 和演示 GIF。
5. 根据真实数据编写简历 Bullet。

## 27. 关键风险

### 方舟 Tool Calling 兼容性

先做 Compatibility Spike；必要时使用薄层 `ArkChatModel`。

### LangGraph 重放副作用

审批必须发生在副作用之前；副作用放入独立、可追踪且尽可能幂等的执行单元。

### 自定义 Middleware 与框架升级

锁定依赖版本，围绕公开 Middleware API 编写，不依赖内部实现。

### Scope Creep

优先级固定为：安全工具执行、动态能力、持久化进度、验证和评测。后台任务、HTML、LSP、Agent Teams 不得挤占核心范围。

### 模型质量差异

不把模型能力误认为 Harness 效果。Harness 消融实验固定单一主模型和参数；DeepSeek 与 GLM 的横向比较作为独立实验；Doubao-Seed-2.0-lite 仅承担固定辅助角色。

### 安全声明过度

没有容器隔离时不称为安全沙箱；没有覆盖所有 Prompt Injection 时不宣称完全防御；无法恢复任意 Shell 副作用时不承诺完全回滚。

## 28. 完成标准

MVP 完成需要同时满足：

- DeepSeek-V4-Pro 和 GLM-5.2 均能在真实小型代码库中完成至少一个带测试的修改任务。
- 两个主模型能在模拟服务故障时互相 Fallback。
- Doubao-Seed-2.0-lite 能按指定 Schema 完成摘要、压缩和记忆候选提取。
- 多个只读 Tool Call 能安全并行，写操作保持串行。
- 危险操作能够 ask/deny，批准后可以 Resume。
- 文件写入有 Preimage，软删除和 Undo 可验证。
- CLI、`events.jsonl` 和 `progress.md` 能一致展示进度。
- Session 可跨进程 Resume，未完成写操作不会被自动重放。
- Skill 和 MCP 能够按需发现、加载和卸载。
- Context 超限路径能够触发裁剪和摘要。
- 测试失败能返回 Agent 并触发有界修复。
- Fake Model 测试覆盖核心成功与失败路径。
- 至少完成一轮可复现任务评测。
- README 明确框架能力、自研能力、威胁边界和已知限制。

## 29. 显式上下文引用

### 29.1 用户接口

```bash
coding-helper run "参考 @src/auth.py 修复登录问题"
coding-helper run "分析 @docs/design.md 和 @tests/"
coding-helper run "总结 @documents/spec.pdf"
```

第一版支持：

- 工作区内文本文件
- 目录引用
- 带文本层的 PDF
- 多个引用
- 包含空格的路径
- 工作区内相对路径和绝对路径

暂不支持 OCR、PDF 图片理解、音视频、复杂 Office 版式和代码符号级 `@symbol`。

### 29.2 解析模型

```python
class ContextArtifact(BaseModel):
    source: str
    media_type: str
    content: str
    content_hash: str
    size: int
    truncated: bool
    metadata: dict
```

处理流程：

```text
parse @ reference
  → normalize path
  → workspace/symlink validation
  → sensitive path policy
  → media type detection
  → size limit
  → text extraction
  → ContextArtifact
  → pinned context injection
```

规则：

- 小型文本文件直接注入并保留行号。
- 大文件只注入相关片段，完整内容继续通过 Read/Search 按需访问。
- 目录只注入受大小限制的 Manifest，不递归加载全部正文。
- PDF 使用 `pypdf` 提取文本层并保留页码边界。
- 加密 PDF 返回明确错误；无文本扫描件说明当前版本需要 OCR 但不支持。
- 引用内容标记来源并视为不可信数据，不能提升权限。
- 用户显式引用属于 Pinned Context；压缩时可摘要正文，但必须保留来源、Hash 和关键事实。

## 30. 受治理的联网查询

模型本身不获得任意网络访问，而是通过两个受治理工具联网：

```text
web_search(query, domains?, freshness?, max_results?)
web_fetch(url)
```

搜索能力优先作为 MCP 集成示例，通过可配置 Search MCP Server 提供；`web_fetch` 由本项目实现安全边界和正文抽取。若没有可用搜索服务凭据，核心 Coding Agent 仍可运行。

### 30.1 返回与引用

Search Result 包含标题、URL、摘要、来源域名和可用的发布时间。Fetch Result 包含最终 URL、抓取时间、内容类型、正文片段和内容 Hash。Agent 使用联网资料回答时必须在最终结果中附来源 URL，轨迹记录查询词、访问域名、耗时和裁剪比例。

### 30.2 网络安全

- 默认只允许 HTTP/HTTPS GET，不支持提交表单、Cookie、认证会话和任意浏览器操作。
- 阻止 localhost、回环地址、私有网段、链路本地地址和云元数据地址，防止 SSRF。
- 每次重定向后重新校验目标地址和协议。
- 设置 DNS、连接、读取和总超时。
- 限制响应体大小、Content-Type 和重定向次数。
- HTML 转为纯文本后再返回模型；二进制下载默认拒绝。
- URL、查询参数和页面正文经过敏感信息检查，避免凭据外传。
- 外部页面明确标记为不可信内容，其中的命令和 Prompt 不能扩大工具权限。
- 对域名支持 allow/ask/deny 策略；首次访问未知域名可要求批准。
- 完整页面落盘前执行脱敏，模型只获得经过裁剪的必要片段。

### 30.3 缓存与上下文

- 以规范化 URL、查询参数和响应 Hash 建立会话级缓存。
- 同一页面短时间内不重复请求。
- Context Compact 时保留来源、抓取时间、摘要和 URL，不保留全部正文。
- 涉及时效性信息时向用户显示实际抓取时间。

### 30.4 评测

- 搜索能够发现指定官方文档，并在最终回答中提供正确 URL。
- Fetch 能拒绝私有 IP、localhost 和重定向到私有地址。
- 超大页面被裁剪，完整响应不会进入模型上下文。
- 页面内恶意指令不能触发未经批准的 Shell 或文件写入。
- Search MCP Server 超时后熔断，主 Agent 可继续使用本地能力。

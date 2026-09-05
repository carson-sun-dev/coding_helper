# Coding Helper 使用与实测手册

本文面向**第一次手动实测**：从零把环境跑起来，按“先免费、后小额、再真实改码”的顺序验证 Harness，并说明如何观测和回滚。

> 关于「谁做什么」「威胁边界」「已知限制」见 [README](../README.md)；机制设计见 [design.md](../design.md)。本文只讲怎么用、怎么测。

---

## 0. 一次性准备

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

在 `.env` 里至少填这 6 项（其余都有默认值，可暂时不管）：

| 变量                  | 说明                                   |
| --------------------- | -------------------------------------- |
| `ARK_API_KEY`         | 火山方舟 API Key                       |
| `ARK_BASE_URL`        | 方舟 OpenAI 兼容地址（默认已给北京区） |
| `ARK_DEEPSEEK_MODEL`  | DeepSeek 的 Endpoint ID 或模型名       |
| `ARK_GLM_MODEL`       | GLM 的 Endpoint ID 或模型名            |
| `ARK_AUXILIARY_MODEL` | 豆包（只做摘要/记忆，不当主 Agent）    |
| `ARK_DEFAULT_PRIMARY` | `deepseek` 或 `glm`                    |

> ⚠️ `.env` 已被 `.gitignore` 排除，**不要提交**。密钥不会进日志、轨迹或模型上下文。

之后每次进终端都要先 `source .venv/bin/activate`。

---

## 0.1 从哪里启动 & `@` 引用如何解析（重要）

**你不必在本项目目录里运行。** `pip install -e` 把 `coding-helper` 装成了命令行程序：只要 `source .venv/bin/activate` 一次，之后 `cd` 到**任何目录**都能直接用它。所以有两种等价用法：

**用法 A：进到目标仓库里跑（最自然）**

```bash
source /Users/carrrson/developer/coding\ helper/.venv/bin/activate   # 只需激活一次
cd /path/to/your-repo                                                # 进你要改的仓库
coding-helper run "参考 @src/auth.py 修复登录"                        # workspace 默认=当前目录
```

**用法 B：待在别处，用 `--workspace` 指定目标仓库**

```bash
coding-helper run "参考 @src/auth.py 修复登录" --workspace /path/to/your-repo
```

### `@` 到底相对谁？

`**@` 永远相对于 workspace（你正在操作的那个仓库），而不是你的 shell 当前目录，也不是 coding-helper 项目目录。\*\* workspace = `--workspace` 的值；不传就是当前目录。

所以：

- ✅ 用法 A 里 `@src/auth.py` 指的是 `your-repo/src/auth.py`——很方便。
- ✅ 用法 B 里 `@src/auth.py` 同样指 `your-repo/src/auth.py`（跟着 `--workspace` 走），即使你人在 coding-helper 项目目录。
- ⚠️ 常见误解：**待在 coding-helper 项目目录、又用 `--workspace` 指到别的仓库时，`@` 不会去引用本项目的文件**，它跟着 `--workspace`。想引用哪个仓库的文件，`@` 就写那个仓库里的相对路径。

一句话：**把 workspace 想成“光标所在的仓库”，`@` 就是这个仓库内的路径**。它不但有用，而且是给首轮上下文“钉”关键文件的最省事方式（否则模型要多花几轮 `read_file`/`search_text` 去找）。

---

## 1. 费用与安全提醒（先读）

| 命令                            | 是否花钱            | 是否改文件                       |
| ------------------------------- | ------------------- | -------------------------------- |
| `doctor`                        | 否（不打 API）      | 否                               |
| `model-check`                   | 是（1 次最小请求）  | 否                               |
| `tool-check`                    | 是（约 2 次请求）   | 否                               |
| `ask`                           | 是                  | 否（只读）                       |
| `run`                           | 是                  | **是**（写操作需你逐个批准）     |
| `eval`                          | 否（用 Fake Model） | 只动 `.coding-helper/eval/` 沙箱 |
| `status` / `trace` / `recovery` | 否                  | 否                               |
| `undo`                          | 否                  | 是（把文件恢复到写入前）         |

两条硬性建议：

1. `**run` 会真的改文件。** 第一次务必在一个**玩具 git 仓库\*\*或可丢弃目录里测，用 `--workspace <路径>` 指定，别拿本项目自己开刀。
2. **完成门槛的 diff 检查依赖 git。** 目标 workspace 最好是 git 仓库，否则范围/删除拦截会被跳过；`--review` 也要有真实 diff 才会触发。

---

## 2. 冒烟：不花钱先确认 Harness 没坏

```bash
coding-helper doctor      # 本地配置体检，应看到 Ark configuration = configured / OK
coding-helper eval        # 内置 fix_add 假模型任务，验证写路径+审批+验收，passed=yes
pytest -q                 # 全量单测
```

`eval` 用脚本化 Fake Model，**不代表真实模型会写代码**，只证明 Harness 机制自身正确（n=1）。

---

## 3. 连通性与 Tool Calling（小额）

```bash
coding-helper model-check deepseek
coding-helper model-check glm
coding-helper model-check auxiliary          # 豆包

coding-helper tool-check deepseek --mode single   # 单个 Tool Call
coding-helper tool-check deepseek --mode multi    # 同一响应多个 Tool Call
coding-helper tool-check glm --mode multi
```

`tool-check` 通过的判据是：模型正确选了工具、每个 Tool Call 有唯一 ID、带 ID 的结果能送回模型。这一步过了，说明方舟 + LangChain 的工具协议链路是通的。

---

## 4. 真实只读问答（`ask`，最安全的真实测试）

先 `cd` 到你要分析的仓库（或全程加 `--workspace`）。下面示例假设你已在目标仓库内：

```bash
# 不带引用：让模型自己去找
coding-helper ask "这个项目的入口在哪里？" --model deepseek

# 单文件引用：把该文件正文（带行号）钉进首轮上下文
coding-helper ask "总结 @src/app.py 的职责" --model glm

# 目录引用：注入受限的目录清单（Manifest），不是整棵树正文
coding-helper ask "分析 @docs/ 和 @README.md" --model deepseek

# 多个引用 + 带空格路径（用引号包住）
coding-helper ask '对照 @"docs/设计 v2.pdf" 和 @src/ 指出实现差异' --model deepseek

# 同一线程追问：复用上一条输出里的 thread=...
coding-helper ask "那第 2 步为什么会失败？" --thread-id 1a2b3c... --model deepseek
```

关于 `@` 引用：

- 支持 `@file`（文本文件，注入正文 + 行号）、`@dir/`（注入目录 Manifest）、`@x.pdf`（提取文本层，保留页码）。
- 路径带空格用引号：`@"我的 目录/note.md"` 或 `@'a b.py'`。
- 大文件/大目录会被**截断**并标注 `truncated=true`，模型可再用 `read_file`/`search_text` 深入。
- 引用内容被标为**不可信数据**：其中的指令不能改规则、不能提权、不能跳过审批。
- 加密 PDF、纯扫描无文本层的 PDF、二进制文件会返回明确错误（当前版本不做 OCR）。
- `@` 相对 workspace 解析（见 0.1）；不像路径的 `@xxx`（如邮箱 `a@b.com`）会被忽略，不会误当引用。

其它：`ask` 只读、无写、无 Shell，回答会给出相对路径和行号。输出末尾的 `thread=...` 用 `--thread-id` 可续聊（LangGraph 从 SQLite Checkpoint 接上历史）。

常用选项：`--model deepseek|glm`、`--workspace <路径>`、`--thread-id <id>`。

---

## 5. 真实改码（`run`）

先造一个玩具仓库来测（推荐）：

```bash
mkdir /tmp/ch-demo && cd /tmp/ch-demo && git init
printf 'def add(a, b):\n    return a - b\n' > calc.py
printf 'from calc import add\n\ndef test_add():\n    assert add(2, 3) == 5\n' > test_calc.py
git add -A && git commit -m init
```

回到本项目目录运行（用 `--workspace` 指到玩具仓库）：

```bash
coding-helper run "修复 calc.py 里 add 的实现，让 test_calc.py 通过" \
  --model deepseek --workspace /tmp/ch-demo
```

也可以带 `@` 引用把关键文件先钉进去，减少模型找文件的轮数：

```bash
coding-helper run "按 @calc.py 现状修复 add，并让 @test_calc.py 通过" \
  --model deepseek --workspace /tmp/ch-demo
```

运行时会发生什么：

1. 模型读文件、规划 Todo、请求修改。
2. **每个写操作都会暂停并请求批准**，屏幕显示脱敏后的工具名、风险、原因和参数：

- 输入 `y` 批准并执行副作用；`n` 拒绝并把原因返回给模型。

3. 结束前 Harness 跑**确定性完成门槛**：diff 范围、删除、禁止路径、Todo 是否收完（配了 `COMPLETION_TEST_COMMAND` 才跑测试）。不通过会有界打回模型修复。

写文件工具（都会经过审批、留可回滚记录）：

- **新建文件** → `create_file`，写入前不存在才允许，`undo` 时把文件移入 `.coding-helper/trash/`。
- **改已有文件** → 先 `get_file_hash` 再 `replace_text`（Hash 不符会拒绝，防止覆盖他人改动），`undo` 用 Preimage 还原。
- 系统提示词已禁止模型用 shell（`cat >`、`echo >`、重定向）建文件——那样会绕过可回滚记录。

常用选项：`--model`、`--workspace`、`--thread-id`、`--review`（见下）。

### 5.1 让门槛真的跑测试

默认 `.env` 没设测试命令，所以门槛只查 diff + Todo。想让它跑测试，在 `.env` 加：

```dotenv
COMPLETION_TEST_COMMAND=pytest -q
# 可选：限制只能改这些前缀，改到范围外会被门槛拦下
COMPLETION_ALLOWED_PREFIXES=src,tests
```

### 5.2 `--review`：结束时加一层 Reviewer

```bash
coding-helper run "..." --workspace /tmp/ch-demo --review
```

- **默认关闭。** 开启后，确定性门槛通过后再让 **Reviewer Subagent** 只读地审一遍 diff，多花一次主模型调用。
- Reviewer **只给补充意见，不改变放行结论**；它失败也不会阻断完成。
- 意见会写进 `progress.md` 的 `## Review` 段，并在命令输出末尾打印。
- 也可用环境变量常开：`.env` 里设 `COMPLETION_REVIEW_ENABLED=true`。

### 5.3 Agent 会用到哪些工具

`run` 模式下模型可见并自行选择调用下列工具（`ask` 只读模式仅前三个 read 工具）。**风险为 read 的自动放行；write / execute 的每次调用都会弹出审批**。

| 工具                    | 风险    | 作用                                        | 审批       |
| ----------------------- | ------- | ------------------------------------------- | ---------- |
| `read_file`             | read    | 按行读取文本文件，返回稳定行号              | 自动       |
| `list_directory`        | read    | 列目录结构，不读正文、不跟符号链接          | 自动       |
| `search_text`           | read    | 字面量搜索，返回文件/行号/片段              | 自动       |
| `git_diff`              | read    | 相对 HEAD 的变更清单                        | 自动       |
| `get_file_hash`         | read    | 返回文件 SHA-256，改文件前必须先调          | 自动       |
| `replace_text`          | write   | Hash 匹配时唯一替换文本，存 Preimage 可回滚 | **需批准** |
| `create_file`           | write   | 新建文件（已存在则拒绝），undo 移入 trash   | **需批准** |
| `todo_write`            | write   | 用完整列表刷新 Todo 与 `progress.md`        | **需批准** |
| `shell`                 | execute | 在 workspace 跑一条命令，独立进程组 + 超时  | **需批准** |
| `web_fetch`             | execute | 抓取 HTTP/HTTPS 页面转纯文本，拦截内网/SSRF | **需批准** |
| `delegate`              | read    | 把只读探索/审查交给隔离 Subagent（见 5.4）  | 自动       |
| `discover_capabilities` | read    | 按查询词列出可用 Skill / 待连 MCP Server    | 自动       |
| `load_skill`            | read    | 按需加载完整 `SKILL.md`（不可信知识）       | 自动       |

> 配了 MCP 时还会出现 `load_mcp_server` 以及 `mcp__<server>__<tool>` 形式的动态工具（见第 7 节）；它们同样过统一审批与轨迹。

`todo_write` 是 write 但只写 `.coding-helper/` 内的进度文件，风险标注用于统一治理，不改你的源码。

### 5.4 Subagent（`delegate`）怎么触发

Subagent 有两条进入路径：

**A. `--review`（Harness 自动触发）** —— 见 5.2，任务结束、确定性门槛通过后自动跑一次 **Reviewer**。你不用做别的，加 `--review` 即可。

**B. `delegate` 工具（模型自行触发）** —— 主模型在需要「大范围只读排查」或「独立视角审查」时，自己决定调用 `delegate(role, task)`，`role` 只能是 `explorer` 或 `reviewer`。**这是模型的决策，没有对应的命令行开关**——决策权属于模型，Harness 只提供这个工具。

你能做的是用**任务措辞去诱导**它委派，例如：

```bash
# 让它先用 explorer 全面摸清调用点，再动手（跨文件排查）
coding-helper run "先全面排查 parse_config 在整个仓库的所有调用点和边界情况，\
再据此修复它对空值的处理" --workspace /path/repo

# 明确要求独立审查视角
coding-helper run "修复 X 后，用 reviewer 独立审查这次 diff 的风险和测试缺口" \
  --workspace /path/repo
```

Subagent 的边界（design 决定，不可绕过）：

- **只读**、独立消息上下文、独立 Token 预算；主 Agent 只拿到它的**结构化结论**，不吸收它的中间过程。
- **只有一层**，禁止递归再开 Subagent。
- Explorer 返回固定小节：`## Files / ## Evidence / ## Findings / ## Suggestions`；Reviewer 返回：`## Risks / ## Test Gaps / ## Findings / ## Recommendation`。
- 失败会作为一条可恢复结果返回给主 Agent，不会中断整个任务。

怎么确认它真的触发了：

```bash
coding-helper trace --workspace /path/repo | grep Subagent
# 期望看到 SubagentStarted / SubagentCompleted
```

> 提示：小任务模型通常**不会**委派（直接读文件更快），这是正常的。Subagent 的价值在大仓库、跨文件、需要独立复核的场景。

---

## 6. 观测与回滚

```bash
coding-helper status  --workspace /tmp/ch-demo   # 人类可读进度（Goal/Tasks/Review/Blockers…）
coding-helper trace   --workspace /tmp/ch-demo   # 最近事件；--limit N 控制条数
coding-helper trace --limit 50 --workspace /tmp/ch-demo
```

`trace` 里能看到 `SessionStarted / ModelStarted / ModelCompleted / ToolRequested / ToolCompleted / ToolInterrupted / ToolApproved / ToolDenied / SubagentStarted / SubagentCompleted / SessionCompleted` 等。事件只含脱敏元数据，不含完整 Prompt 或密钥。

> `ToolInterrupted` 不是报错——它表示某个写/Shell 操作触发了审批中断、正在等你确认。批准后会出现新的 `ToolRequested` + `ToolCompleted`。

回滚：

```bash
coding-helper undo --workspace /tmp/ch-demo                 # 撤销上一次安全写
coding-helper undo <operation-id> --workspace /tmp/ch-demo  # 撤销指定操作
coding-helper recovery --workspace /tmp/ch-demo             # 诊断进程中断留下的 pending 写
```

- `undo` 只在文件 Hash 未被后续操作改动时才恢复；有冲突会拒绝而不是覆盖。
- 撤销**修改**：用 Preimage 还原到写入前内容。撤销**新建**（`create_file`）：把文件移入 `.coding-helper/trash/<operation-id>/`，不物理删。
- 每次 `undo` 都会二次确认，并显示恢复前后的 Hash。

所有运行数据都在目标 workspace 的 `.coding-helper/` 下（已 gitignore）：Checkpoint、`events.jsonl`、`progress.md`、`backups/`、`trash/`、被裁剪的完整输出 `outputs/`、评测沙箱 `eval/`。

---

## 7. MCP（可选）

```bash
coding-helper mcp-check github --call get_me
```

需要 `.env` 里的 `GITHUB_PERSONAL_ACCESS_TOKEN`，或把 `mcp.example.json` 复制成 `.coding-helper/mcp.json`。`mcp-check` 只允许调用按名判定为只读的工具。没配也不影响核心 Coding Agent。

---

## 8. 命令与选项速查

所有命令都可加 `--help` 查看完整选项。`<>` 为必填参数，`[]` 为可选。

| 命令          | 参数 / 选项                                                              | 作用                                              |
| ------------- | ------------------------------------------------------------------------ | ------------------------------------------------- | -------------------------------------------- | -------------------------------------- |
| `doctor`      | `[--workspace P]`                                                        | 本地配置体检，不打 API                            |
| `model-check` | `<target>` = `deepseek`                                                  | `glm`                                             | `auxiliary`                                  | 向模型发一次最小请求验证连通性（花钱） |
| `tool-check`  | `<target>` `[--mode single                                               | multi]`                                           | 验证单/多 Tool Call 协议（花钱，主模型专用） |
| `mcp-check`   | `[server=github]` `[--call NAME]` `[--args JSON]` `[--workspace P]`      | 连 MCP Server、列工具、可选调只读工具             |
| `ask`         | `<question>` `[--model M]` `[--workspace P]` `[--thread-id ID]`          | 只读问答，支持 `@` 引用                           |
| `run`         | `<task>` `[--model M]` `[--workspace P]` `[--thread-id ID]` `[--review]` | 可改文件，写操作需审批，支持 `@` 引用             |
| `status`      | `[--workspace P]`                                                        | 打印 `progress.md`                                |
| `trace`       | `[--workspace P]` `[--limit N]`                                          | 打印最近事件（默认 20，最多 200）                 |
| `undo`        | `[operation-id]` `[--workspace P]`                                       | 撤销上一次或指定的安全写（需确认）                |
| `recovery`    | `[--workspace P]`                                                        | 诊断进程中断留下的 pending 写，不自动重放         |
| `eval`        | `[--workspace P]`                                                        | 内置 Fake Model 冒烟任务 `fix_add`（n=1，不花钱） |

选项含义：

- `--model deepseek|glm`：本次用哪个主模型；不传用 `.env` 的 `ARK_DEFAULT_PRIMARY`。`auxiliary`（豆包）不能当主 Agent。
- `--workspace P`：Agent 操作与 `@` 解析的根目录；不传=当前目录。
- `--thread-id ID`：复用某次输出里的 `thread=...` 续接同一会话（从 Checkpoint 恢复历史）。
- `--review`（仅 `run`）：门槛通过后加一层 Reviewer 审 diff（多一次主模型调用）。
- `--limit N`（仅 `trace`）：显示最近 N 条事件。

一个最小闭环示例（在目标仓库内）：

```bash
source /path/to/coding-helper/.venv/bin/activate
cd /path/to/your-repo

coding-helper ask "入口在哪？@README.md"          # 先只读摸清楚
coding-helper run "新增 hello.py，打印 Hello"      # 审批→写入（走 create_file）
coding-helper status                              # 看进度
coding-helper trace --limit 30                    # 看事件
coding-helper undo                                # 不满意就撤销（新建文件进 trash）
```

---

## 9. 一次完整手测清单

```text
[ ] doctor 全绿
[ ] eval passed=yes
[ ] pytest 全通过
[ ] model-check deepseek / glm / auxiliary 都成功
[ ] tool-check deepseek --mode multi 通过（unique_ids 正确）
[ ] ask 能带 @file 引用给出带行号的回答（且 @ 跟随 --workspace）
[ ] run 在玩具仓库里改码：写操作弹出审批，批准后执行
[ ] 新建文件走 create_file（不是 shell），undo 后文件进 trash
[ ] 完成门槛：Todo 未收完 / 改到范围外时被打回
[ ] --review：progress.md 出现 ## Review，trace 出现 SubagentStarted/Completed
[ ] undo 能把修改还原、把新建文件移入 trash，Hash 冲突时拒绝
[ ] trace 里审批显示为 ToolInterrupted（不是 ToolFailed）
[ ] status / trace 与实际过程一致
```

---

## 10. 常见问题

- `**@` 引用好像没生效 / 找不到文件？** `@` 相对 **workspace\*\*（`--workspace` 或当前目录），不是你 shell 的当前目录。在 coding-helper 项目里跑、又 `--workspace` 指到别处时，`@` 指的是那个 workspace 内的路径。见 0.1。
- **必须在 coding-helper 项目目录里运行吗？** 不必。`source .venv/bin/activate` 后 `coding-helper` 在任何目录都能用；`cd` 进目标仓库直接跑最省事。
- **模型用 shell 建文件、结果 undo 不了？** 新版已加 `create_file` 并在提示词里禁止用 shell 建文件。若仍看到 shell 建文件，用 `trace` 抓下来反馈。
- `**run` 一直没让我批准就改了文件？\*\* 不应发生；写操作一定先 Interrupt。若遇到请用 `trace` 抓事件序列反馈。
- **完成门槛没跑测试？** 你没设 `COMPLETION_TEST_COMMAND`，默认跳过。
- `**--review` 没触发 Reviewer？\*\* 需要同时满足：开了 `--review`（或 `COMPLETION_REVIEW_ENABLED=true`）、目标是 git 仓库、且本次真的产生了 diff。
- **想换主模型？** 每条 `ask`/`run` 加 `--model deepseek|glm`；不加则用 `ARK_DEFAULT_PRIMARY`。豆包不能当主 Agent。
- `**model-check` / `tool-check` 会花钱吗？\*\* 会，都是真实 API 调用。`doctor` 不会。

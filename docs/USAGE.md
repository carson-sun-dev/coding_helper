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

| 变量 | 说明 |
|---|---|
| `ARK_API_KEY` | 火山方舟 API Key |
| `ARK_BASE_URL` | 方舟 OpenAI 兼容地址（默认已给北京区） |
| `ARK_DEEPSEEK_MODEL` | DeepSeek 的 Endpoint ID 或模型名 |
| `ARK_GLM_MODEL` | GLM 的 Endpoint ID 或模型名 |
| `ARK_AUXILIARY_MODEL` | 豆包（只做摘要/记忆，不当主 Agent） |
| `ARK_DEFAULT_PRIMARY` | `deepseek` 或 `glm` |

> ⚠️ `.env` 已被 `.gitignore` 排除，**不要提交**。密钥不会进日志、轨迹或模型上下文。

之后每次进终端都要先 `source .venv/bin/activate`。

---

## 1. 费用与安全提醒（先读）

| 命令 | 是否花钱 | 是否改文件 |
|---|---|---|
| `doctor` | 否（不打 API） | 否 |
| `model-check` | 是（1 次最小请求） | 否 |
| `tool-check` | 是（约 2 次请求） | 否 |
| `ask` | 是 | 否（只读） |
| `run` | 是 | **是**（写操作需你逐个批准） |
| `eval` | 否（用 Fake Model） | 只动 `.coding-helper/eval/` 沙箱 |
| `status` / `trace` / `recovery` | 否 | 否 |
| `undo` | 否 | 是（把文件恢复到写入前） |

两条硬性建议：

1. **`run` 会真的改文件。** 第一次务必在一个**玩具 git 仓库**或可丢弃目录里测，用 `--workspace <路径>` 指定，别拿本项目自己开刀。
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

```bash
coding-helper ask "这个项目的入口在哪里？" --model deepseek
coding-helper ask "总结 @src/coding_helper/runtime.py 的职责" --model glm
coding-helper ask "分析 @docs/ 和 @README.md" --model deepseek
```

- `@file` / `@dir/` / 文本型 `@x.pdf` 都是**显式上下文引用**，会被注入并标记为不可信数据。
- `ask` 不写文件、无 Shell。回答会指出依据的相对路径和行号。
- 输出末尾的 `thread=...` 是 Thread ID，可用 `--thread-id <id>` 在同一线程追问（LangGraph 会从 SQLite Checkpoint 接上历史）。

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

运行时会发生什么：

1. 模型读文件、规划 Todo、请求修改。
2. **每个写操作都会暂停并请求批准**，屏幕显示脱敏后的工具名、风险、原因和参数：
   - 输入 `y` 批准并执行副作用；`n` 拒绝并把原因返回给模型。
3. 结束前 Harness 跑**确定性完成门槛**：diff 范围、删除、禁止路径、Todo 是否收完（配了 `COMPLETION_TEST_COMMAND` 才跑测试）。不通过会有界打回模型修复。

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

---

## 6. 观测与回滚

```bash
coding-helper status  --workspace /tmp/ch-demo   # 人类可读进度（Goal/Tasks/Review/Blockers…）
coding-helper trace   --workspace /tmp/ch-demo   # 最近事件；--limit N 控制条数
coding-helper trace --limit 50 --workspace /tmp/ch-demo
```

`trace` 里能看到 `SessionStarted / ModelCompleted / ToolRequested / ToolCompleted / ToolApproved / SubagentStarted / SubagentCompleted / SessionCompleted` 等。事件只含脱敏元数据，不含完整 Prompt 或密钥。

回滚：

```bash
coding-helper undo --workspace /tmp/ch-demo                 # 撤销上一次安全写
coding-helper undo <operation-id> --workspace /tmp/ch-demo  # 撤销指定操作
coding-helper recovery --workspace /tmp/ch-demo             # 诊断进程中断留下的 pending 写
```

- `undo` 只在文件 Hash 未被后续操作改动时才恢复；有冲突会拒绝而不是覆盖。
- 删除默认是软删除，进 `.coding-helper/trash/`，不物理删。

所有运行数据都在目标 workspace 的 `.coding-helper/` 下（已 gitignore）：Checkpoint、`events.jsonl`、`progress.md`、`backups/`、`trash/`、被裁剪的完整输出 `outputs/`、评测沙箱 `eval/`。

---

## 7. MCP（可选）

```bash
coding-helper mcp-check github --call get_me
```

需要 `.env` 里的 `GITHUB_PERSONAL_ACCESS_TOKEN`，或把 `mcp.example.json` 复制成 `.coding-helper/mcp.json`。`mcp-check` 只允许调用按名判定为只读的工具。没配也不影响核心 Coding Agent。

---

## 8. 一次完整手测清单

```text
[ ] doctor 全绿
[ ] eval passed=yes
[ ] pytest 全通过
[ ] model-check deepseek / glm / auxiliary 都成功
[ ] tool-check deepseek --mode multi 通过（unique_ids 正确）
[ ] ask 能带 @file 引用给出带行号的回答
[ ] run 在玩具仓库里改码：写操作弹出审批，批准后执行
[ ] 完成门槛：Todo 未收完 / 改到范围外时被打回
[ ] --review：progress.md 出现 ## Review，trace 出现 SubagentStarted/Completed
[ ] undo 能把文件恢复到写入前，Hash 冲突时拒绝
[ ] status / trace 与实际过程一致
```

---

## 9. 常见问题

- **`run` 一直没让我批准就改了文件？** 不应发生；写操作一定先 Interrupt。若遇到请用 `trace` 抓事件序列反馈。
- **完成门槛没跑测试？** 你没设 `COMPLETION_TEST_COMMAND`，默认跳过。
- **`--review` 没触发 Reviewer？** 需要同时满足：开了 `--review`（或 `COMPLETION_REVIEW_ENABLED=true`）、目标是 git 仓库、且本次真的产生了 diff。
- **想换主模型？** 每条 `ask`/`run` 加 `--model deepseek|glm`；不加则用 `ARK_DEFAULT_PRIMARY`。豆包不能当主 Agent。
- **`model-check` / `tool-check` 会花钱吗？** 会，都是真实 API 调用。`doctor` 不会。

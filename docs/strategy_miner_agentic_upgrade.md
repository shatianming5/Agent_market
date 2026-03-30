# Strategy Miner（Agentic + Freqtrade）改造方案（已落地）

目标：把项目从“仅 LLM API 生成因子/表达式”为主，升级为 **OpenCode client 驱动的 agentic Freqtrade 策略生成与验证系统**，并在不破坏 `feature/expression/ml/backtest` 既有主流程的前提下，以增量方式落地。

## 1. 现状盘点（与仓库对齐）

- 现有能力：
  - Flow 编排：`scripts/agent_flow.py` / `src/agent_market/agent_flow.py`
  - 回测与汇总：`/run/backtest` + `src/agent_market/backtest_results.py`
  - LLM 表达式生成：`scripts/freqai_expression_agent.py` + `src/agent_market/freqai/llm.py`
- 新增能力（本次改造）：
  - 策略挖掘模块：`src/agent_market/strategy_miner/`（runner/phases/sandbox/grading/artifacts）
  - OpenCode Client + 工具循环：`src/runner_fsm/opencode/`（file/bash tool-call + policy）
  - Provider-agnostic Agent Executor：`src/agent_market/agents/executor.py`（opencode → openai_compatible；no-template）
  - API 路由：`server/api/routes/strategy_miner.py`

## 2. 分层架构（目标形态 / 当前实现）

### L0：Server API（任务编排层）

- 负责：启动任务、查询阶段进度、列举候选、批准落地、触发回测与汇总。
- 原则：
  - 与现有 `/run/*`、`/jobs/*` 语义保持一致；重任务统一使用 JobManager 后台跑。

### L1：Run Storage（产物与状态层）

- 统一落盘目录（标准化）：`runs_root/<run_id>/strategy_miner/`
  - `runs_root` 默认是 `artifacts/runs`，可通过环境变量 `AGENT_MARKET_RUNS_ROOT` 覆盖。
- 必须产物（已实现）：
  - `proposal.json`：任务输入（配置 + 目标 + 约束 + 工具白名单/预算）
  - `checkpoint.json`：断点续跑状态机快照
  - `candidates/iter_0000/<name>.py|.json`：每轮候选策略代码快照 + 元信息
  - `backtests/iter_0000/<name>/summary.json`：候选回测摘要（成功时）
  - `backtests/jobs/*.json`：后端 job 触发的回测任务摘要（无论成功/失败都会写）
  - `leaderboard.json`：风险约束过滤后按 reward 降序的榜单

### L2：Provider-agnostic Agent Executor（执行器层）

- 抽象统一接口：`AgentExecutor.run(prompt) -> AgentRunResult`
- 优先：OpenCode client（支持多轮 tool-calls）
- Fallback（graceful）：
  - OpenAI-compatible chat completion（无 tools）
  - 离线模板策略（无外部依赖）

### L3：Strategy Miner Pipeline（策略挖掘层）

- 形态：阶段式状态机（GEN → BACKTEST → EVAL → ANALYZE → …）。
- 自我修复（已实现）：
  - `phase_backtest` 支持 `repair_attempts`：验证失败/回测失败/解析失败会触发 agent 修复（读文件/改代码/可选运行小检查），然后重试。
- 每轮策略生成在隔离沙箱中进行：`runs_root/<run_id>/strategy_miner/iter_<n>/sandbox/`。

### L4：Tooling & Policy（工具与安全层）

- 工具白名单：`tools.tool_allowlist`（当前支持 `file` / `bash` 两类）
- bash 约束：
  - `tools.bash_allow` 开关
  - `tools.bash_allowlist`（前缀匹配）
  - `tools.bash_timeout`（超时）
- 文件隔离：tool executor 仅允许读写 sandbox 根目录内文件，且禁止 `.env` 类文件。

## 3. 配置设计（Strategy Miner Config）

默认配置：`configs/strategy_miner_default.json`（支持嵌套与旧版扁平字段兼容）。

- `budget.*`
  - `provider`: `auto|opencode|openai_compatible`（`template` 已禁用：no-template enforced）
  - `model`, `base_url`
  - `max_iterations`, `max_turns`, `max_retries`, `stale_timeout`
  - `repair_attempts`
- `backtest.*`
  - `freqtrade_config`, `timerange`, `backtest_timeout`
- `tools.*`
  - `tool_allowlist`, `bash_allow`, `bash_timeout`, `bash_allowlist`
- `evaluation.*`
  - `reward_weights`
  - `min_trades`, `max_abs_drawdown`, `min_winrate`
- `evolution.*`
  - `evolve_enabled`, `evolve_every_n`, `mutation_intensity`, `crossover_prob`

## 4. API 设计（Strategy Miner）

在 `server/api/routes/strategy_miner.py`：

- `POST /strategy-miner/start`：启动挖掘任务（返回 `job_id` + `run_id`）
- `GET /strategy-miner/status/{job_id}`：基于日志的阶段进度（best-effort）
- `GET /strategy-miner/runs`：列举历史 runs
- `GET /strategy-miner/runs/{run_id}`：run 详情（checkpoint + proposal + leaderboard + snapshots）
- `GET /strategy-miner/runs/{run_id}/proposal`：读取 `proposal.json`
- `GET /strategy-miner/runs/{run_id}/leaderboard`：读取 `leaderboard.json`
- `GET /strategy-miner/runs/{run_id}/status`：基于 checkpoint 的状态
- `GET /strategy-miner/runs/{run_id}/candidates`：候选列表（checkpoint 视角）
- `POST /strategy-miner/runs/{run_id}/approve`：批准落地（复制策略到 `user_data/strategies/`）
- `POST /strategy-miner/runs/{run_id}/backtest`：触发对候选的回测与汇总（后台 job）

兼容别名：`/strategy-miner/results` 与 `/strategy-miner/results/{run_id}`。

## 5. 测试策略（已覆盖）

- 单测：
  - `MinerConfig.from_dict` 的嵌套解析
  - ToolPolicy allowlist/timeout/path 隔离
  - phases + repair loop（离线 mock）
  - artifacts（proposal/candidates/backtests/leaderboard）
- API：
  - start/status/runs/detail/approve/backtest
  - proposal/leaderboard endpoints

## 6. 迁移与兼容

- 现有 Flow（feature/expression/ml/backtest）不改语义，只增量添加策略挖掘能力。
- 外部依赖缺失（opencode/LLM/freqtrade）时，策略生成与 API 查询保持 graceful fallback。

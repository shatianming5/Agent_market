# Strategy Miner（Agentic + Freqtrade）改造方案

目标：把项目从“仅 LLM API 生成因子/表达式”为主，升级为 **OpenCode client 驱动的 agentic Freqtrade 策略生成与验证系统**，并在不破坏 `feature/expression/ml/backtest` 既有主流程的前提下，以增量方式落地。

## 1. 现状盘点（与仓库对齐）

- 现有能力：
  - Flow 编排：`scripts/agent_flow.py` / `src/agent_market/agent_flow.py`
  - 回测与汇总：`/run/backtest` + `src/agent_market/backtest_results.py`
  - LLM 表达式生成：`scripts/freqai_expression_agent.py` + `src/agent_market/freqai/llm.py`
- 已有雏形：
  - 策略挖掘模块：`src/agent_market/strategy_miner/`（runner/phases/sandbox/grading）
  - OpenCode Client + 工具循环：`src/runner_fsm/opencode/`（file/bash tool-call）
  - API 路由：`server/api/routes/strategy_miner.py`

改造重点：把“策略挖掘”升级为可配置、可审计、可复现、可 API 管理的 agentic pipeline；并补齐产物规范与测试。

## 2. 分层架构（目标形态）

### L0：Server API（任务编排层）

- 负责：启动任务、查询阶段进度、列举候选、批准落地、触发回测与汇总。
- 原则：
  - 与现有 `/run/*`、`/jobs/*`、`/results/*` 保持一致的错误码与 job 语义；
  - 不在 API 线程内做重任务（统一用 JobManager 后台跑）。

### L1：Run Storage（产物与状态层）

- 统一落盘目录（标准化）：`artifacts/runs/<run_id>/strategy_miner/`
- 必须产物：
  - `proposal.json`：任务输入（配置 + 目标 + 约束 + 工具白名单 + 预算）
  - `candidates/`：每轮候选策略源码（`.py`）与元信息（`.json`）
  - `backtests/`：候选回测结果摘要（`summary.json`）+ 关联 zip 路径（如有）
  - `leaderboard.json`：按 reward/约束过滤后的排行榜
  - `checkpoint.json`：断点续跑的最小状态机快照（兼容恢复）

### L2：Provider-agnostic Agent Executor（执行器层）

- 抽象出统一接口：`AgentExecutor.run(prompt, tools, policy, ...) -> AgentResult`
- 优先实现：OpenCode client（支持多轮 tool-calls）。
- Fallback：
  - 外部依赖缺失（`opencode` 不在 PATH）或配置缺失时，降级为纯 LLM completion（无 tools）或模板策略生成（离线）。

### L3：Strategy Miner Pipeline（策略挖掘层）

- 形态：状态机/阶段式 pipeline（GEN → VALIDATE → BACKTEST → SCORE → ANALYZE/REPAIR → …）。
- 关键点：
  - “自我修复”优先走工具调用（读日志/改代码/重跑回测）而不是简单重开新策略；
  - 风险约束（如最大回撤/最少交易数/最大仓位/杠杆等）进入 gating：不满足直接降权/淘汰；
  - 评估指标可配置：reward 权重、风险阈值、稳定性与多样性约束。

### L4：Tooling & Policy（工具与安全层）

- 工具白名单：显式配置允许的 tool（如 `read/write/edit/search/run_backtest/read_metrics`）。
- bash 约束：默认禁用“任意 bash”；如需保留，用 allowlist（命令前缀/regex）+ 超时 + 输出截断。
- 沙箱隔离：策略生成在 `runs/<run_id>/strategy_miner/workspace/` 下进行；与仓库代码/Secrets 隔离。

## 3. 配置设计（Strategy Miner Config）

新增/标准化配置项（示例字段，最终以实现为准）：

- 预算：`budget.max_iterations / budget.max_turns / budget.max_backtests / budget.timeout_sec`
- 评估：`evaluation.reward_weights / evaluation.min_trades / evaluation.max_drawdown_abs / evaluation.metrics`
- 风险约束：`risk_constraints.max_drawdown_abs / min_winrate / max_open_trades / max_leverage`（可选）
- 工具白名单：`tools.allow` +（可选）`tools.bash_allowlist`
- 超时重试：`timeouts.backtest_sec / stale_thinking_sec / retries.agent / retries.backtest`

## 4. API 设计（Strategy Miner）

在 `server/api/routes/strategy_miner.py` 增强：

- `POST /strategy-miner/start`：启动挖掘任务（返回 job_id + run_id）
- `GET /strategy-miner/status/{job_id}`：阶段进度（iteration/phase/best_reward + tail logs）
- `GET /strategy-miner/runs`：列举历史 runs（来自 `artifacts/runs/*/strategy_miner/`）
- `GET /strategy-miner/runs/{run_id}`：run 详情（proposal/leaderboard/candidates）
- `GET /strategy-miner/runs/{run_id}/candidates`：候选列表
- `POST /strategy-miner/runs/{run_id}/approve`：批准落地（复制到 `user_data/strategies/` 并记录）
- `POST /strategy-miner/runs/{run_id}/backtest`：触发对指定候选的回测与汇总（后台 job）

## 5. 测试策略

- 单测：
  - 配置解析与默认值（budget/tools/retry）
  - reward/约束 gating 的边界条件
  - tool policy（白名单拒绝、路径隔离）
- API：
  - start/status/runs/detail/approve/backtest 的 happy-path + 错误码
- e2e（离线 fixture）：
  - 使用 mock backtest 结果（不依赖 freqtrade/真实数据）跑通“生成→评分→排行榜→批准”。

## 6. 迁移与兼容

- 现有 Flow（feature/expression/ml/backtest）不改语义，只增量添加策略挖掘能力。
- 旧的策略挖掘产物路径如存在，保留读取兼容（逐步迁移到 `artifacts/runs/<run_id>/strategy_miner/`）。

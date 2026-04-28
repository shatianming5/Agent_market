# Agent Market 深入解析（架构 / 数据流 / 扩展点）

> 目标：让你能在 30 分钟内搞清楚“从 UI 点按钮/调 API 到产出 artifacts”的完整链路，并知道要改功能该从哪几个文件入手。

## 1. 总览：4 层结构

1) **前端静态页面**：`web/index.html` + `web/app.js`
- 纯静态资源，由后端挂载到 `/web`（见 `server/app.py`）。
- 通过 `fetch` 调后端 API；可选 `X-API-Key`（localStorage 里保存）。

2) **FastAPI 服务**：`server/`
- `server/app.py` 组装路由（`/run/*`、`/flow/*`、`/results/*`、`/jobs/*`、`/strategy-miner/*`、`/settings` 等）。
- `server/job_manager.py` 负责把“flow/miner/backtest/hyperopt 等”作为后台进程跑起来，并收集日志/状态。
- `server/api/routes/flow.py` 用日志扫描的方式做 “flow 进度条”（SSE/WS）。

3) **业务核心包**：`src/agent_market/`
- `agent_flow.py`：Flow 的总调度（run_id、step 顺序、preflight、写 run_meta）。
- `flow_ext/step_dispatch.py`：把每个 step 映射到具体执行函数，并把产物路径写进 `RunArtifacts`。
- `flow_steps.py`：真正执行各 step（子进程 freqtrade / python 脚本、训练、回测、TCA 等）。
- `strategy_miner/`：策略挖掘（LLM 多角色 → 回测 → 打分 → 迭代 → holdout/benchmark/portfolio/promotion）。

4) **运行数据与产物**：
- `user_data/`：工作目录（freqtrade configs、数据、策略文件、回测 zip、job logs、LLM feedback 等）。
- `artifacts/`：统一产物根（runs/、models/、control_plane/、run_meta.json 等，可用环境变量改根目录）。

---

## 2. Flow（AgentFlow）到底怎么跑的

### 2.1 触发方式（CLI / API）

- CLI：`python scripts/agent_flow.py --config configs/agent_flow_kucoin_cpu_nollm.json --steps feature expression ml backtest`
  - 入口脚本：`scripts/agent_flow.py`
  - 实际类：`src/agent_market/agent_flow.py` 的 `AgentFlow.run()`

- API：`POST /flow/run`
  - 路由：`server/api/routes/flow.py` 的 `run_flow()`
  - 本质：把 `python scripts/agent_flow.py ...` 交给 `JobManager.start()` 后台运行。

### 2.2 Flow 的核心数据结构：RunArtifacts + run_meta.json

- `RunArtifacts`（可变容器）：`src/agent_market/run_artifacts.py`
  - 每个 step 产物路径都会写到这里（如 `feature_output`、`factor_cards_json`、`backtest_zip_run` 等）。

- Flow 结束会写两个 meta：
  - 最新：`artifacts/run_meta.json`（见 `src/agent_market/paths.py:run_meta_latest_path()`）
  - 本次 run：`artifacts/runs/<run_id>/run_meta.json`（见 `src/agent_market/paths.py:run_meta_path()`）

`run_meta.json` 由 `src/agent_market/agent_flow.py::_build_run_meta()` 生成，包含：
- run_id / 状态 / started_at / ended_at
- config snapshot（文件 sha256 或对象快照 sha256）
- python 信息、freqtrade version
- preflight 报告（并把 `preflight.json` 写到本次 run_dir）
- artifacts（来自 `RunArtifacts.to_dict()` + 部分补全逻辑）
- steps（每步 ok/failed 的时间戳与错误信息）

### 2.3 Step 调度：step_dispatch 是“真相表”

Flow step 的映射在：`src/agent_market/flow_ext/step_dispatch.py`

典型 step（按你常用“黄金路径”解释）：
- `feature`：执行 `scripts/freqai_feature_agent.py`（由 config 决定）→ 写 `arts.feature_output`
- `expression`：执行 `scripts/freqai_expression_agent.py`（LLM/或 `--no-llm`）→ 写 `arts.expression_output`
  - 如果存在 `*_scored_all.json`，会额外生成 `factor_eval_meta.json` / `factor_scores.json` / `pareto.csv`
  - 并构建 `factor_memory`（本 run 的 factor memory artifacts），然后合并进全局 factor memory（见下）
- `ml`：`src/agent_market/flow_steps.py:run_ml_training()` → 找到最新 `training_summary.json` 并复制到 run_dir → `arts.training_summary_json`
- `backtest`：`src/agent_market/flow_steps.py:run_backtest()` → freqtrade backtesting → 解析最新 zip → 写 `latest_backtest_summary.json` → `arts.backtest_zip_run` / `arts.feedback_summary_json`

你在 config 里开启的话还有：
- `capture` / `lob_rebuild` / `micro_feature`：微观结构数据采集/重建/特征（产物写入 run_dir 并回填 RunArtifacts）
- `portfolio`：HRP 权重与报告（`src/agent_market/portfolio_opt.py`）
- `factor_compile` / `factor_eval`：Factor DSL → 编译 → 评估（输出 spec/ast/compiled_expression/score/memory）
- `tca`：对回测 trades 做 TCA（`src/agent_market/tca/`）
- `strategy_miner`：在该 run_id 下跑策略挖掘（见第 3 节）
- `report`：把本次 run 关键产物打包成 bundle（zip + manifest）

### 2.4 Factor Memory：本 run 与全局记忆的关系

- 本 run 会在 `artifacts/runs/<run_id>/factor_memory/` 生成：
  - `factor_memory.json` / `factor_cards.json` / `factor_failure_cards.json` / `factor_lineage.json`
  - 逻辑：`src/agent_market/factor_memory.py`（构建/merge/retrieval/context 格式化）

- 然后会尝试合并到**全局 factor memory**（control plane）：
  - 目标根：`artifacts/control_plane/factor_memory/`（可用 `AGENT_MARKET_CONTROL_PLANE_ROOT` 等覆盖）
  - 合并发生在 `src/agent_market/flow_ext/step_dispatch.py:_merge_into_global_factor_memory()`

---

## 3. Strategy Miner（策略挖掘）怎么跑、产物在哪

### 3.1 触发方式（CLI / API）

- CLI：`python scripts/strategy_miner.py --config configs/strategy_miner_default.json`
  - 或：`python -m agent_market.strategy_miner --config configs/strategy_miner_default.json`

- API：`POST /strategy-miner/start`
  - 路由：`server/api/routes/strategy_miner.py:start_miner()`
  - 本质：后台跑 `python scripts/strategy_miner.py ...`

### 3.2 目录布局（强约定）

矿工 run 的目录固定为：
- `artifacts/runs/<run_id>/strategy_miner/`（见 `src/agent_market/strategy_miner/runner.py:miner_run_dir()`）

核心文件：
- `checkpoint.json`：可恢复状态（`MinerState`，原子写 + fsync）
- `proposal.json`：本次 run 的“输入约束与预算”快照
- `leaderboard.json`：候选排名与拒绝原因汇总
- `run_meta.json`：当前 phase/iteration 的轻量状态（用于 UI/API）
- `events.jsonl`：关键事件时间线（可做可观测性）
- `economics.json`：tokens/cost/wall time 汇总
- `promotion_log.jsonl`：sealed holdout / benchmark / promotion 的决策链
- `portfolio_plan.json`：最终候选组合建议（可选）

候选与证据：
- `candidates/iter_XXXX/cand_XX/*.json`：每个候选的结构化快照
- `agent_traces/iter_XXXX/cand_XX/<role>.json`：LLM 角色 trace（planner/coder/reviewer/backtester…）
- `backtests/iter_XXXX/<name>/summary.json`：该候选的回测摘要证据
- `training/iter_XXXX/<name>/training_evidence.json`：训练证据（如启用 ML/DL/RL）

写这些文件的工具模块：`src/agent_market/strategy_miner/artifacts.py`

### 3.3 Phase 循环（生成 → 回测 → 打分 → 迭代）

主循环：`src/agent_market/strategy_miner/runner.py:run_strategy_miner()`

阶段函数位于：
- `strategy_miner/_generation.py`：`phase_strategy_gen()`
- `strategy_miner/_backtest.py`：`phase_train_model()` / `phase_backtest()` + repair 逻辑
- `strategy_miner/_evaluation.py`：`phase_evaluation()` / `phase_analysis()`

高层逻辑（简化版）：
1) STRATEGY_GEN：多候选、多角色 LLM 生成（可并发），并做基础“策略文件规范化/去重/修复”
2) TRAIN_MODEL（可选）：ML/DL/RL 家族候选会触发训练证据写入
3) BACKTEST：freqtrade backtesting（在 sandbox user_data 下跑），失败可进入 repair 循环
4) EVALUATION：计算 effective score、应用 gates（min_trades/drawdown/winrate/robustness 等），更新 best_candidate
5) ANALYSIS：可选 LLM 分析/结构化总结，写入 history/trace
6) 结束后：sealed holdout + frozen benchmark + portfolio + promotion chain（可选）

### 3.4 Sandbox 与安全约束

策略代码的静态检查/修复/沙盒准备在：`src/agent_market/strategy_miner/sandbox.py`
- 禁止 import：`os/subprocess/requests/...`（防止候选策略“越权”）
- 禁止调用：`exec/eval/open/__import__/globals/locals/...`
- 强制方法存在：`populate_indicators` / `populate_entry_trend` / `populate_exit_trend`
- 额外兼容修复：informative 合并、order_types/time_in_force 等

另外，miner 的 tool policy（bash allowlist 等）由 `MinerConfig` 控制（见 `configs/strategy_miner_default.json` 与 `src/agent_market/strategy_miner/dtypes.py`）。

---

## 4. Server：JobManager 如何把“脚本”变成“API”

### 4.1 JobManager 运行模型

- `server/job_manager.py`
  - `JobManager.start(cmd, cwd, env, kind, timeout_sec, meta)`：启动后台进程
  - 通过线程读 stdout/stderr（合并到 stdout）：
    - 内存：ring buffer（默认 5000 行）
    - 磁盘：`user_data/job_logs/<job_id>.log`（见 `server/job_manager.py` 初始化参数）
  - 维护 registry：`user_data/job_registry/<job_id>.json`（running/returncode/total_lines 等）

### 4.2 Flow 进度条为什么“能看起来像实时的”

`server/api/routes/flow.py`：
- `GET /flow/progress/{job_id}`：
  - 扫描 logs 里的 `[FLOW] STEP_START/PHASE/STEP_OK/STEP_FAIL` 标记
  - 在缺乏明确标记时，用启发式（epoch X/Y、百分号 token、日志增长）估算 percent
- `GET /flow/stream/{job_id}`：SSE，1s 拉一次 logs + progress，返回 `event: progress`
- `WS /flow/ws/{job_id}`：WebSocket，1s 推一次 progress

### 4.3 Results API 是如何“读 artifacts”的

`server/api/routes/results.py`：
- 用 `agent_market.paths.safe_resolve()` 做路径白名单与防穿越（允许 artifacts/user_data/runs 等根目录）
- 读取 `backtest-result-*.zip` 后用 `src/agent_market/backtest_results.py` 生成 summary（并可附带 trades）
- 提供 bundle 下载：`GET /results/bundles/download/{run_id}`

---

## 5. Web 前端：它主要调哪些 API

`web/app.js`（关键调用点）：
- Settings：`GET/POST /settings`
- 表达式/回测：`POST /run/expression`、`POST /run/backtest`
- 结果：`GET /results/latest-summary`、`GET /results/list`、`GET /results/summary`、`GET /results/bundles/download/{run_id}`
- Flow：`POST /flow/run`、`GET /flow/progress/{job_id}`、`GET /flow/run-meta/{run_id}`、`GET /flow/runs/list`
- Miner：`POST /strategy-miner/start`、`GET /strategy-miner/status/{job_id}`、`GET /strategy-miner/runs...`

前端不会“直接跑计算”，它只是把 config/参数提交给后端，然后靠 jobs/logs/progress 把状态展示出来。

---

## 6. 扩展点：想加功能一般改哪

### 6.1 新增一个 Flow step

推荐路径：
1) 在 `src/agent_market/flow_steps.py` 增加一个 runner（或复用 scripts）
2) 在 `src/agent_market/flow_ext/step_dispatch.py`：
   - 新增 `_step_<name>()`（写入 `RunArtifacts` 对应字段）
   - 注册到 `STEP_HANDLERS`
3) 在 `src/agent_market/agent_flow.py`：
   - 加到 `STEP_ORDER` / sequence（确保顺序正确）
4) 如果要给 UI 看见：`web/app.js` 增加调用或展示（可选）

### 6.2 新增一个对外 API

- 在 `server/api/routes/` 新建路由文件并 include 到 aggregator（或直接在现有模块加）
- 需要后台执行就复用 `jobs.start(...)`
- 所有外部路径输入建议用 `agent_market.paths.safe_resolve()`（避免路径穿越）

### 6.3 扩展 Strategy Miner

- 新增 phase：修改 `src/agent_market/strategy_miner/dtypes.py:Phase` + `VALID_TRANSITIONS`，并在 `runner.py` 主循环里接入
- 新增 artifact：在 `strategy_miner/artifacts.py` 写入稳定 JSON（便于 API/UI/审计）

---

## 7. 当前仓库里值得先修/先注意的点

- `README.md`/`Makefile`/docs 多处引用 `scripts/smoke_test.py`，但当前仓库没有这个文件；建议以 `pytest -q` 和 `python scripts/e2e_smoke_flow.py ...` 为准。
- `create_workspace.py` 依赖 `workspace/` 模板目录，但目前缺失；仓库里的 `ws_production/` 更像是“现成 workspace 样例”，脚本不会自动使用它。


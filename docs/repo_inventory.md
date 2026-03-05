# Repo Inventory

## Tree

```text
Agent_market-1/
  analysis/                  # 架构/安全/覆盖率等审计笔记（非运行时必须）
  artifacts/                 # 统一产物根：models/ + runs/<run_id>/
  configs/                   # Flow/策略挖掘等 JSON 配置模板
  docs/                      # 文档（plan/experiment/mohu/verify/inventory 等）
  scripts/                   # CLI / pipeline wrappers（Flow/回测/报告/策略挖掘）
  server/                    # FastAPI 后端（API + Jobs + 结果聚合）
  src/                       # Python 包：agent_market/ + runner_fsm/
  tests/                     # pytest（尽量离线 fixture 可验收）
  user_data/                 # 工作区（数据/策略/回测结果/日志/反馈）
  web/                       # 前端静态资源（挂载到 /web）
  Makefile
  README.md
  plan.md
  pytest.ini
  requirements*.txt
```

## Entry Points

- `uvicorn server.main:app --host 0.0.0.0 --port 8000`
  - FastAPI 服务入口（挂载 `web/` 到 `/web`）。
- `python scripts/agent_flow.py --config <json> --steps feature expression ml backtest`
  - 黄金路径编排器（feature/expression/ml/backtest）。
- `python scripts/strategy_miner.py --config configs/strategy_miner_default.json --model <opencode_model>`
  - 策略级挖掘 CLI（生成→回测→评分→迭代/进化）。
- `python -m agent_market.strategy_miner --config <json>`
  - strategy_miner 包入口（便于模块化调用/调试）。
- `python scripts/smoke_test.py`
  - API 冒烟（不跑重任务）。
- `pytest -q`
  - 单测/API 测试/离线 e2e fixture。

## Core Modules

- `src/agent_market/agent_flow.py`
  - Flow 配置加载与步骤编排；产物写入 `artifacts/runs/<run_id>/`。
- `src/agent_market/flow_steps.py`
  - Flow 各步骤执行逻辑（feature/expression/ml/rl/backtest/report 等）。
- `src/agent_market/backtest_results.py`
  - 解析 `backtest-result-*.zip` 并生成结构化摘要（用于反馈/评分/汇总）。
- `src/agent_market/strategy_miner/`
  - 策略挖掘主模块（runner/phases/sandbox/grading/knowledge_base 等）。
- `src/agent_market/paths.py`
  - 路径与产物根（`artifacts_root()/runs_root()/user_data_root()`）统一解析。
- `src/runner_fsm/opencode/`
  - OpenCode Client + 工具调用解析/执行（file/bash），用于 agentic 多轮回合。
- `server/`
  - FastAPI + JobManager：负责启动脚本作业、追踪状态与日志、聚合结果。

## Config & Data

- Flow 配置（示例）：
  - `configs/agent_flow_kucoin_cpu_nollm.json`
  - `configs/agent_flow_kucoin_cpu_nollm_smoke.json`
- Strategy Miner 配置（示例）：
  - `configs/strategy_miner_default.json`
- 用户工作区：
  - 数据：`user_data/data/<exchange>/<PAIR>-<timeframe>.feather`
  - 策略：`user_data/strategies/`
  - 回测结果：`user_data/backtest_results/`
  - Job 日志：`user_data/job_logs/<job_id>.log`
- 统一产物（默认）：
  - `artifacts/run_meta.json`（latest）
  - `artifacts/runs/<run_id>/run_meta.json`
- 关键环境变量（可选）：
  - OpenAI 兼容 LLM（表达式/部分 fallback）：`OPENAI_BASE_URL/OPENAI_API_KEY/OPENAI_MODEL`
  - 或 `LLM_BASE_URL/LLM_API_KEY/LLM_MODEL`（优先级更高）
  - OpenCode Agent（策略挖掘）：`OPENCODE_URL`、`OPENCODE_MODEL`
  - 产物根覆盖：`AGENT_MARKET_ARTIFACTS_ROOT`、`AGENT_MARKET_RUNS_ROOT`、`AGENT_MARKET_USER_DATA_ROOT`

## How To Run

### 安装依赖（最小：服务 + 测试）

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r server/requirements.txt -r requirements-dev.txt
```

### 安装依赖（黄金路径：含 freqtrade/ml）

```bash
pip install -r requirements-full.txt
```

### 启动服务

```bash
uvicorn server.main:app --host 0.0.0.0 --port 8000
```

打开前端：`http://127.0.0.1:8000/web/index.html`

### Tests

```bash
pytest -q
python scripts/smoke_test.py
```

## Risks / Unknowns

- `freqtrade` 与历史数据缺失会导致真实回测失败：测试/冒烟尽量走 fixture 与轻量路径。
- `opencode` 二进制是外部依赖；策略挖掘需要安装并可在 `PATH` 中找到（需 graceful fallback）。
- agentic 工具执行（bash/file）存在安全风险：需要工具白名单 + 命令约束 + 沙箱隔离。

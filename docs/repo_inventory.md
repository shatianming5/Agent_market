# Repo Inventory

## Tree

```
Agent_market/
  artifacts/                 # 模型与训练产物（training_summary.json 等）
  configs/                   # Flow/训练/回测 JSON 配置
  docs/                      # 文档（流程说明、落地计划、库存）
  freqtrade/                 # Freqtrade 源码（回测/数据工具的上游项目）
  scripts/                   # 可执行脚本（Flow/特征/表达式/训练/回测/清理/smoke）
  server/                    # FastAPI 后端（API + Jobs + 结果聚合）
  src/agent_market/          # 核心业务模块（Flow 编排、FreqAI 子系统）
  user_data/                 # 工作区（配置/数据/策略/回测结果/日志/反馈）
  web/                       # 静态前端（Flow 画布 + 日志 + 结果浏览）
  README.md
  requirements.txt
```

## Entry Points

- `uvicorn server.main:app --host 0.0.0.0 --port 8000`
  - FastAPI 服务入口（会挂载 `web/` 到 `/web`）。
- `scripts/agent_flow.py`
  - 端到端编排器（feature → expression → ml → rl → backtest）。
- `scripts/smoke_test.py`
  - API 冒烟测试（不跑重任务）。
- `scripts/e2e_smoke_flow.py`
  - 端到端冒烟（跑 Flow 并检查关键产物是否落盘）。
- `scripts/freqai_feature_agent.py`
  - 生成稳定的特征配置 JSON（供表达式/训练使用）。
- `scripts/freqai_expression_agent.py`
  - 表达式生成与因子挖掘（支持 `--mine --top-n`；LLM 可选）。
- `scripts/train_pipeline.py`
  - 机器学习训练（LightGBM/XGBoost/CatBoost 等；读取特征/表达式 + feather 数据）。

## Core Modules

- `src/agent_market/agent_flow.py`
  - Flow 配置加载与步骤编排；输出统一的 `[FLOW]` 标记日志（服务端可据此估算进度）。
- `src/agent_market/flow_steps.py`
  - Flow 的每一步实际执行逻辑（执行脚本/训练/回测；并写回测摘要到 feedback）。
- `src/agent_market/backtest_results.py`
  - 解析 `backtest-result-*.zip` 并生成摘要、trades 等结构化结果。
- `src/agent_market/freqai/`
  - 特征、表达式执行引擎、安全 eval、训练管线、RL 环境/训练器。
- `server/`
  - 任务调度与 API 统一入口：
    - `/run/*`：启动 feature/expression/train/backtest 等任务
    - `/flow/*`：Flow 启动与进度流（SSE/WS）
    - `/jobs/*`：任务状态/日志/取消
    - `/results/*`：结果列表/摘要/聚合/反馈准备
    - `/settings`：LLM 与默认 timeframe 设置

## Config & Data

- Flow 配置：`configs/agent_flow_kucoin_cpu_nollm.json`（推荐黄金路径）
- Freqtrade 配置示例：
  - `user_data/config_freqai_kucoin.json`
- 数据目录（feather）：
  - `user_data/data/<exchange>/<PAIR>-<timeframe>.feather`
  - 示例：`user_data/data/kucoin/BTC_USDT-1h.feather`
- 产物与日志（默认落在仓库根目录下）：
  - Flow 日志：`user_data/agent_logs/agent_flow_*.log`
  - Job 日志：`user_data/job_logs/<job_id>.log`
  - 模型摘要：`artifacts/models/**/training_summary.json`
  - 回测结果：`user_data/backtest_results/backtest-result-*.zip`
  - 回测摘要（用于下一轮反馈）：`user_data/llm_feedback/latest_backtest_summary.json`
- LLM 环境变量（可选）：
  - `OPENAI_BASE_URL / OPENAI_API_KEY / OPENAI_MODEL`
  - `LLM_BASE_URL / LLM_API_KEY / LLM_MODEL`（优先级更高）

## How To Run

### 安装依赖（本地/CPU）

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r server/requirements.txt
```

### 启动服务

```bash
uvicorn server.main:app --host 0.0.0.0 --port 8000
```

打开前端：`http://127.0.0.1:8000/web/index.html`

### API 冒烟

```bash
python scripts/smoke_test.py
```

### 端到端冒烟（黄金路径）

```bash
python scripts/e2e_smoke_flow.py --config configs/agent_flow_kucoin_cpu_nollm.json
```

或手工分步：

```bash
python scripts/agent_flow.py --config configs/agent_flow_kucoin_cpu_nollm.json --steps feature
python scripts/agent_flow.py --config configs/agent_flow_kucoin_cpu_nollm.json --steps expression
python scripts/agent_flow.py --config configs/agent_flow_kucoin_cpu_nollm.json --steps ml
python scripts/agent_flow.py --config configs/agent_flow_kucoin_cpu_nollm.json --steps backtest
```

## Risks / Unknowns

- `freqtrade/` 目录会遮蔽 `python -m freqtrade` 的模块导入；回测建议优先使用 `freqtrade` 可执行文件（通常在虚拟环境的 `bin/` 内）。
- 为保证“离线可复现回测”，本仓库对 `freqtrade` 做了最小补丁：优化模式下会用本地数据推断 pairs，并在缺失 markets 时合成最小 markets 元数据（见 `freqtrade/freqtrade/plugins/pairlistmanager.py` 与 `freqtrade/freqtrade/exchange/exchange.py`）。
- 回测与训练对数据依赖强：`user_data/data/<exchange>` 缺失会导致表达式挖掘与训练失败。
- LLM 能力完全可选；默认黄金路径不依赖外部 API。

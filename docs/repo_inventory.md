## Tree

```
.
├── .github/                        # CI
├── artifacts/                      # 训练/信号等产物（运行时写入）
├── configs/                        # 配置样例（Flow/训练 JSON + YAML）
├── data/                           # 示例/加工数据（parquet/zip 等；当前仓库内含不少体积文件）
├── docs/                           # 文档（架构/训练/流程/清单）
├── freqtrade/                      # vendored 上游 Freqtrade（CLI/回测/数据下载）
├── scripts/                        # CLI 脚本入口（Flow/训练/数据/清理/烟测）
├── server/                         # FastAPI 后端（HTTP API + Job 管理）
├── src/agent_market/               # 业务核心包（Flow/特征/表达式/训练/RL）
├── tests/                          # pytest（偏单元/轻量集成）
├── user_data/                      # 唯一 canonical userdir（配置/特征/表达式/回测结果/日志）
├── web/                            # 静态前端（index.html + vendor）
├── package.json                    # 前端 build/test 脚本
├── requirements.txt                # Python 依赖
└── pytest.ini                      # pytest 配置
```

## Entry Points

- `scripts/agent_flow.py`：端到端 Flow 编排 CLI（按 JSON 配置串联 feature → expression → ml → rl → backtest；支持 `--steps` 逐步调试）
- `src/agent_market/agent_flow.py`：Flow 编排器（只负责步骤顺序与日志标记；读取 JSON 兼容 BOM）
- `src/agent_market/flow_steps.py`：步骤执行器（subprocess / 内部 pipeline；便于逐步调试/替换）
- `scripts/freqai_feature_agent.py`：生成基础特征配置（写入 `user_data/freqai_features*.json`）
- `scripts/freqai_expression_agent.py`：表达式生成 + 因子挖掘（Top-N + 进化算法；写入 `user_data/freqai_expressions*_selected*.json`）
- `src/agent_market/freqai/training/pipeline.py`：统一 ML 训练流水线（数据集 → 训练 → `training_summary.json`）
- `scripts/rl_generate_signals.py`：由训练摘要/配置生成 RL signals（写入 `artifacts/signals/...`，供策略回测读取）
- `user_data/strategies/ExpressionLongStrategy.py`：回测/策略入口（加载训练摘要 + expressions_file，保证回测使用训练同款因子列）
- `server/main.py`：FastAPI 入口（`uvicorn server.main:app`）
- `server/job_manager.py`：Job 管理（subprocess + 日志缓存/落盘）
- `web/index.html`：前端入口（通过 API 驱动 Flow/日志/结果）

## Core Modules

- `src/agent_market/config.py`：读取 freqtrade config，解析 `datadir/userdir` 并校验数据集（JSON 兼容 BOM）
- `src/agent_market/freqai/features.py`：根据 feature JSON 生成/应用特征工程
- `src/agent_market/freqai/expression_engine.py`：加载/执行表达式（factor）并写入新列
- `src/agent_market/freqai/model/*`：模型适配层（统一 fit/predict）
- `src/agent_market/freqai/rl/*`：RL 环境与训练骨架（可选依赖）
- `src/agent_market/backtest_results.py`：回测结果解析（latest zip → summary/feedback）

## Config & Data

- 配置入口：
  - Flow：`configs/agent_flow_*.json`（默认以仓库根目录为 cwd；路径统一使用 `user_data/...`）
  - 训练：`configs/train_*.json`、`configs/*_config_real.json`
  - 数据抓取/资讯：`configs/*.yaml`
  - Freqtrade：`user_data/config_freqai*.json`
- 数据/产物约定（canonical）：
  - OHLCV（feather）：`user_data/data/<exchange>/<PAIR>-<timeframe>.feather`
  - 特征定义：`user_data/freqai_features*.json`
  - 因子/表达式：`user_data/freqai_expressions*_selected*.json`
  - 回测结果：`user_data/backtest_results/backtest-result-*.zip`
  - 训练摘要：`artifacts/models/**/training_summary.json`
  - Flow 日志：`user_data/agent_logs/agent_flow_*.log`
- LLM 环境变量（可选）：`OPENAI_BASE_URL` / `OPENAI_MODEL` / `OPENAI_API_KEY`（兼容 `LLM_*`）
- 完整文件清单：`docs/repo_tree_full.txt`

## How To Run

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt -r server/requirements.txt
./.venv/bin/python -m pip install pytest
```

```bash
./.venv/bin/python -m pytest -q
./.venv/bin/python scripts/smoke_test.py
```

```bash
# 下载数据（Freqtrade CLI）
./.venv/bin/freqtrade download-data --userdir user_data --config user_data/config_freqai_kucoin.json --timeframes 1h --pairs BTC/USDT ETH/USDT
```

```bash
# Flow（可逐步调试）
./.venv/bin/python scripts/agent_flow.py --config configs/agent_flow_kucoin_example.json --steps feature
./.venv/bin/python scripts/agent_flow.py --config configs/agent_flow_kucoin_example.json --steps expression
./.venv/bin/python scripts/agent_flow.py --config configs/agent_flow_kucoin_example.json --steps ml backtest
```

## Risks / Unknowns

- `freqtrade/` 为可选依赖（源码体积大，建议当作外部组件）。本项目的 `scripts/` 已不依赖 `freqtrade/` 目录存在。
- 表达式引擎使用 `eval`（仅适用于可信表达式/本地执行；不适合把任意用户输入当作表达式执行）。
- `data/` 内包含不少体积文件（parquet/zip）；若目标是“更简洁/可读/易 clone”，建议迁移为外部数据缓存或 git-lfs。
- KuCoin 公网端点存在不稳定/阻断风险；当前已在 `user_data/config_freqai_kucoin.json` 用 `exchange.ccxt_config.urls` 指向 `openapi-v2` 规避。

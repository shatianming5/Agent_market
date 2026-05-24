# Agent Market

Agent Market 是一个把 **LLM 表达式生成、量化因子研究、WorldQuant BRAIN alpha mining、FreqAI/ML/RL 训练、回测、TCA、部署与前端监控** 串起来的本地工作台。

当前主驱动子系统是 `wq_brain`：面向 WorldQuant BRAIN 的 agentic alpha mining、池管理、预检、提交与状态同步。仓库同时维护 crypto / Freqtrade / Factor Lab / Strategy Miner / Factor Hub 等研究与部署能力。

> AI 协作 agent 进入仓库时先读 [`AGENTS.md`](AGENTS.md)。本 README 是人类入口和功能跳转页。

## 功能跳转

| 功能 | 做什么 | 子 README |
|---|---|---|
| WQ BRAIN agentic alpha mining | FASTEXPR 验证、模拟、pool 管理、安全提交、agent/colony loop | [`docs/features/wq_brain/README.md`](docs/features/wq_brain/README.md) |
| Agent Flow 离线主流水线 | feature → expression → ml/rl → backtest → tca → report | [`docs/features/agent_flow/README.md`](docs/features/agent_flow/README.md) |
| Factor Lab 研究 CLI | 数据、特征、因子挖掘、验证、回测、rank portfolio、LEAN、RL、deploy | [`docs/features/factor_lab/README.md`](docs/features/factor_lab/README.md) |
| Strategy Miner | LLM 多角色生成、修复、回测、评分和迭代 Freqtrade 策略 | [`docs/features/strategy_miner/README.md`](docs/features/strategy_miner/README.md) |
| FastAPI 服务与 Web UI | 后台 job、Flow 进度、结果查询、静态前端 | [`docs/features/api_web/README.md`](docs/features/api_web/README.md) |
| Factor Compiler / Hub / Memory | FactorSpec DSL、因子注册表、部署、历史因子记忆 | [`docs/features/factor_infrastructure/README.md`](docs/features/factor_infrastructure/README.md) |
| 微观结构 / TCA / Rank Portfolio | LOB / micro features、交易成本分析、cross-sectional futures portfolio | [`docs/features/microstructure_tca_rank/README.md`](docs/features/microstructure_tca_rank/README.md) |
| 安装、测试与运行维护 | 依赖、环境变量、pytest、smoke、清理、GC、运行产物边界 | [`docs/features/ops_testing/README.md`](docs/features/ops_testing/README.md) |

功能文档总索引：[`docs/features/README.md`](docs/features/README.md)

## 快速开始

### 1. 安装依赖

完整研究 / 回测 / 训练路径：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -c constraints.txt -r requirements-full.txt
```

只跑后端和测试：

```bash
pip install -r server/requirements.txt -r requirements-dev.txt
```

更多安装、环境变量和测试说明见 [`docs/features/ops_testing/README.md`](docs/features/ops_testing/README.md)。

### 2. 启动后端和前端

```bash
uvicorn server.main:app --host 127.0.0.1 --port 8000
```

打开：

```text
http://127.0.0.1:8000/web/index.html
```

生产或共享环境建议加 API key：

```bash
AGENT_MARKET_API_KEY=your-secret-key \
  uvicorn server.main:app --host 127.0.0.1 --port 8000
```

详情见 [`docs/features/api_web/README.md`](docs/features/api_web/README.md)。

### 3. 跑 WQ BRAIN 推荐路径

```bash
python scripts/wq_brain.py auth
python scripts/wq_brain.py simulate --tag <tag> --expr "<FASTEXPR>"
python scripts/wq_brain.py pool submit-worker --tag <tag> --max 20 --one-per-cluster
```

详情和安全边界见 [`docs/features/wq_brain/README.md`](docs/features/wq_brain/README.md)。

### 4. 跑离线主流水线

```bash
python scripts/agent_flow.py \
  --config configs/agent_flow_kucoin_cpu_nollm.json \
  --steps feature expression ml backtest
```

详情见 [`docs/features/agent_flow/README.md`](docs/features/agent_flow/README.md)。

### 5. 跑 Factor Lab

```bash
python scripts/factor_lab.py data okx-futures
python scripts/factor_lab.py features all
python scripts/factor_lab.py mine --tag exp1 --rounds 50
python scripts/factor_lab.py rank-export --tag exp1 --n 50 --risk-profile aggressive
```

详情见 [`docs/features/factor_lab/README.md`](docs/features/factor_lab/README.md)。

### 6. 跑 Strategy Miner

```bash
python scripts/strategy_miner.py --config configs/strategy_miner_default.json
```

详情见 [`docs/features/strategy_miner/README.md`](docs/features/strategy_miner/README.md)。

## 常用命令

```bash
# 服务
make run

# Flow
make flow
make flow-smoke

# 验证
pytest tests/test_wq_brain_*.py -q
pytest -q
python scripts/smoke_test.py
python scripts/e2e_smoke_flow.py --config configs/agent_flow_kucoin_cpu_nollm.json

# 清理 / GC
python scripts/clean_workspace.py --dry-run
python scripts/gc_runs.py --keep 50 --prune-backtests --dry-run
python scripts/gc_jobs.py --keep 200 --keep-days 14 --dry-run
```

## 文档导航

| 你想看什么 | 链接 |
|---|---|
| AI agent 入口契约 | [`AGENTS.md`](AGENTS.md) |
| 仓库目录和入口 | [`docs/repo_inventory.md`](docs/repo_inventory.md) |
| 系统分层 | [`docs/architecture.md`](docs/architecture.md) |
| 文档状态和读取顺序 | [`docs/INDEX.md`](docs/INDEX.md) |
| CLI 脚本分类 | [`scripts/README.md`](scripts/README.md) |
| API / UI / artifacts 深入链路 | [`docs/deep_dive.md`](docs/deep_dive.md) |
| MVP 验收口径 | [`docs/plan.md`](docs/plan.md) |
| 90 天闭环产品 | [`docs/product_90d.md`](docs/product_90d.md) |
| Proposal 和长期计划 | [`docs/proposals/agent_market_proposal.md`](docs/proposals/agent_market_proposal.md) |
| review loop 评分历史 | [`AUTO_REVIEW.md`](AUTO_REVIEW.md) |

## 顶层目录

```text
AGENTS.md                AI agent 入口契约
AUTO_REVIEW.md           review loop 日志
Makefile                 常用命令包装
configs/                 Flow / strategy miner / 数据源配置
docs/                    文档、功能子 README、runbook、proposal、历史证据
freqtrade/               vendored Freqtrade snapshot，只读
scripts/                 CLI 入口
server/                  FastAPI 后端
src/agent_market/        业务核心包
src/runner_fsm/          OpenCode-FSM runner core
tests/                   pytest 套件
user_data/               Freqtrade 工作区、真实数据、策略、回测结果
web/                     静态前端
workspace/               旧研究模板源，只读为主
ws_production/           独立 production-workspace 实验区
artifacts/               运行产物，默认不要手动改
runtime_configs/         运行时配置快照
runtime_logs/            运行时日志
runtime_manifests/       job manifest
logs/                    日志归档
```

## 关键注意事项

- `wq_brain pool submit-worker` 是当前推荐生产路径；`scan --auto-submit` 和 `pool resubmit-all` 是 legacy unsafe 路径。
- 真实 mining / backtest / submit 路径需要真实配置、真实数据和有效凭据。
- 不要把 `.env`、API key、WQ 凭据或交易所凭据提交到仓库。
- `artifacts/`、`runtime_*`、`logs/`、`.tmp/`、`.venv*/`、`.opencode/` 等主要是运行产物或工具缓存。
- `freqtrade/` 是 vendored snapshot，只读。
- `user_data/` 含真实 OHLCV、策略和回测产物，改动前确认目的。
- 旧文档可能是 proposal、evidence 或 historical，不代表当前可直接运行；先看 [`docs/INDEX.md`](docs/INDEX.md)。

## 版本

本仓库不维护面向外部发布的语义化版本号，优先保证当前代码、文档和本地可复现路径一致。README 与 `AUTO_REVIEW.md` 第 9 轮之后的仓库结构同步；如有冲突，以当前代码、[`AGENTS.md`](AGENTS.md) 和 [`docs/repo_inventory.md`](docs/repo_inventory.md) 为准。

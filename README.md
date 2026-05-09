# Agent Market：LLM + FreqAI 智能策略工作台

Agent Market 是一个将 LLM 表达式生成、特征工程、机器学习/强化学习训练与回测串联起来的全流程工作台。后端基于 FastAPI 提供统一 API，前端为简洁的 Flow/控制台，便于快速试验与组合能力。

- 多源数据接入（含 CCXT 等）
- 频交易 FreqAI 表达式生成、特征提取；可接驳自定义 LLM
- 机器学习/强化学习训练（LightGBM/XGBoost/CatBoost/PyTorch/SB3）
- FastAPI + Web 前端（静态目录 /web），可部署为一体化服务

## 导航

| 你是谁 | 先读这个 |
|---|---|
| AI 协作 agent（Claude / Codex / opencode 等） | [`AGENTS.md`](AGENTS.md) — agent 入口契约（30 秒读完） |
| 开发者 / 第一次接触 | 本文件 + [`docs/repo_inventory.md`](docs/repo_inventory.md) |
| 想看每篇文档的状态 / 是否仍 current | [`docs/INDEX.md`](docs/INDEX.md) |
| 想看系统分层 / 模块归属心智模型 | [`docs/architecture.md`](docs/architecture.md) |
| 想找 CLI 入口 | [`scripts/README.md`](scripts/README.md) |
| 想看主驱动子系统 wq_brain | [`src/agent_market/wq_brain/`](src/agent_market/wq_brain/) |
| 想了解 review loop 评分历史 | [`AUTO_REVIEW.md`](AUTO_REVIEW.md) |

## 目录（顶层；详细 tour 见 [`docs/repo_inventory.md`](docs/repo_inventory.md)）

```
artifacts/                 # 运行时产物（runs / 模型 / factor_lab 输出，不要手动改）
configs/                   # 配置（Flow/训练/回测 JSON + 数据抓取 YAML）
docs/                      # 文档（24+ 篇；先看 docs/INDEX.md 找当前 vs 历史）
freqtrade/                 # vendored Freqtrade snapshot（只读）
runtime_*/                 # 运行时快照（log / config / manifest，运行时生成）
scripts/                   # 70+ CLI 脚本（先看 scripts/README.md 分类）
server/                    # FastAPI 后端
src/agent_market/          # 业务核心；主子系统 wq_brain（详见其内部 docstring）
src/runner_fsm/            # 通用 runner FSM
tests/                     # Pytest 套件
user_data/                 # Freqtrade 工作区（OHLCV / 配置 / 回测产物）
web/                       # 前端静态资源（/web/index.html）
ws_production/             # 独立 production-workspace 实验区
AGENTS.md                  # AI agent 入口契约
README.md                  # 你正在读的这个
plan.md                    # 根兼容性指针（指向下面两份）
docs/proposals/agent_market_proposal.md   # Proposal（大计划，部分 PARTIAL）
docs/plan.md                              # MVP 落地计划（已闭环）
```

## 快速开始

1) 创建虚拟环境并安装依赖
```
python -m venv venv
./venv/Scripts/Activate.ps1   # Windows PowerShell
# 推荐（黄金路径：feature + expression + ml + backtest）
pip install -c constraints.txt -r requirements-full.txt

# 或最小依赖（仅后端 + 测试；不含 backtest/ml 相关依赖）
# pip install -r server/requirements.txt
# pip install -r requirements-dev.txt
```

2) 可选：安装/升级 freqtrade + lightgbm（用于回测/训练；若未使用 `requirements-full.txt` 则需要）
```
pip install freqtrade lightgbm
# 或者（开发/最新源码）：git clone https://github.com/freqtrade/freqtrade.git --depth 1 && cd freqtrade && pip install -e . && cd ..
```

3) 配置 LLM（可选）
在项目根目录创建 `.env`（OpenAI 兼容接口）：
```
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=替换为你的APIKey
OPENAI_MODEL=gpt-4o-mini
```
（兼容：也支持 `LLM_BASE_URL/LLM_API_KEY/LLM_MODEL`）

> 若你要用 BigModel（GLM）并跑第一条 agentic strategy miner，见：`docs/agentic_first_run_bigmodel.md`

4) 启动后端
```bash
# 开发模式（无认证）
uvicorn server.main:app --host 127.0.0.1 --port 8000

# 生产模式（启用 API key 认证）
AGENT_MARKET_API_KEY=your-secret-key uvicorn server.main:app --host 127.0.0.1 --port 8000
# 请求时需要 header: X-API-Key: your-secret-key
# /run/* 和 /flow/run 端点受保护，其他端点不受影响
```
打开前端：`http://127.0.0.1:8000/web/index.html`

5) 黄金路径（推荐）
```
python scripts/agent_flow.py --config configs/agent_flow_kucoin_cpu_nollm.json --steps feature expression ml backtest
```

可选：加入组合优化（HRP 风险平价）步骤
```
python scripts/agent_flow.py --config configs/agent_flow_kucoin_cpu_nollm_portfolio.json --steps feature portfolio expression ml backtest
```

6) 验收（本地）
```
python scripts/smoke_test.py
pytest -q
```

7) （可选）Makefile 快捷命令
```
make install-full
make run
make smoke
make test
make check
make e2e
```

## 核心接口

- 健康检查：`GET /health`、根：`GET /`、文档：`GET /docs`
- 任务发起：
  - `POST /run/feature` 生成特征
  - `POST /run/expression` 生成表达式（LLM）
  - `POST /run/backtest` 回测
  - `POST /run/hyperopt` 超参优化
  - `POST /run/rl_train` 强化学习训练
  - `POST /run/train` 机器学习训练（支持内嵌 config_obj 校验）
  - `POST /flow/run` 运行 Agent Flow（可选步骤）
- 策略挖掘（Strategy Miner）：
  - `POST /strategy-miner/start` 启动策略挖掘（job + run）
  - `GET /strategy-miner/runs` 列举 runs
  - `GET /strategy-miner/runs/{run_id}` 查看 proposal/leaderboard/candidates
  - `GET /strategy-miner/runs/{run_id}/proposal` 读取 proposal
  - `GET /strategy-miner/runs/{run_id}/leaderboard` 读取 leaderboard
  - `POST /strategy-miner/runs/{run_id}/approve` 批准落地到 `user_data/strategies/`
  - `POST /strategy-miner/runs/{run_id}/backtest` 触发候选回测汇总（后台 job）

- 任务管理：`GET /jobs/{id}/status`、`GET /jobs/{id}/logs?offset=0`、`POST /jobs/{id}/cancel`
- 结果相关：
  - `GET /results/latest-summary`、`GET /results/list`、`GET /results/summary?name=...`
  - `GET /results/gallery`、`GET /results/aggregate?names=a.zip,b.zip`
  - `GET /features/top?file=...&limit=...`
  - `POST /results/prepare-feedback` 生成 LLM 反馈输入
- Flow 运行元信息 / 历史：
  - `GET /flow/run-meta/latest`
  - `GET /flow/run-meta/{run_id}`
  - `GET /flow/runs/list?limit=...`
  - `GET /flow/portfolio/latest`
  - `GET /flow/portfolio/{run_id}`
- Flow 进度：`GET /flow/progress/{job_id}?steps=feature,expression,ml,rl,backtest`
- 服务设置：`GET|POST /settings`（llm_base_url/llm_model/default_timeframe）
- 流式进度：
  - `GET /flow/stream/{job_id}`（SSE，event: progress / data: JSON）
  - `WS /flow/ws/{job_id}`（WebSocket，JSON）

标准错误示例（所有 /run/* 及相关接口遵循）：
```
{ "status": "error", "code": "INVALID_TIMEFRAME", "message": "..." }
```
任务启动返回：
```
{ "status": "started", "job_id": "...", "kind": "expression|feature|...", "cmd": [ ... ] }
```

## 前端使用要点

- 顶部工具栏：自动布局/对齐/吸附/主题/导出等
- 服务设置面板：对接 `/settings` 以读写当前后端配置
- 常用参数区：一键“表达式生成/回测/摘要”
- 特征 TopN 与 Agent Flow：快速查看特征与一键运行流
- 结果：列表/对比；图集与聚合：多结果的快速浏览与对比
- 状态与日志：运行态、成功/失败标识，日志实时追加（SSE 优先）

备注：若浏览器无法加载 React/ReactFlow（例如本地 vendor 资源缺失/被拦截），前端会自动降级为直接 DOM 绑定与轮询日志，核心流程可用但 Flow 画布交互受限。

## 自动化与清理

- 工作区清理（删除临时/缓存/产物，可带 dry-run）：
  - `python scripts/clean_workspace.py`
  - `python scripts/clean_workspace.py --dry-run`

- 长期运行建议的定期 GC（建议用 cron/launchd 定期执行）：
  - Flow runs（保留最近 N 次 run，可选清理无引用回测 zip）：
    - `python scripts/gc_runs.py --keep 50 --prune-backtests --dry-run`
    - `python scripts/gc_runs.py --keep 50 --prune-backtests`
  - Job logs/registry（保留最近 N 个 job + 最近 X 天）：
    - `python scripts/gc_jobs.py --keep 200 --keep-days 14 --dry-run`
    - `python scripts/gc_jobs.py --keep 200 --keep-days 14`

- 单机常驻服务的作业并发控制（可选）：
  - `AGENT_MARKET_MAX_CONCURRENT_JOBS`：最大同时运行 job 数（默认 2；设为 0 表示不限制）
  - `AGENT_MARKET_MAX_QUEUED_JOBS`：最大排队 job 数（默认 50；队列满时 API 返回 `JOB_QUEUE_FULL`）

## Strategy Miner（策略挖掘）

通过 LLM 多 Agent 管线（planner → coder → reviewer → backtester）自动生成、回测、评估和迭代优化交易策略。

### 环境搭建

```bash
# 1. 创建 conda 环境
conda create -n agent_market python=3.11 -y
conda activate agent_market

# 2. 安装 freqtrade
pip install freqtrade

# 3. 安装项目依赖（pandas_ta/vectorbt 安装失败可忽略，freqtrade 自带 ft-pandas-ta）
pip install -r requirements.txt

# 4. macOS 需要 TA-Lib：brew install ta-lib
#    Linux：参考 https://github.com/TA-Lib/ta-lib-python#dependencies
```

### LLM API 配置

Strategy miner 需要 OpenAI 兼容的 LLM API：

```bash
export OPENAI_API_KEY=sk-your-key
export OPENAI_API_BASE=http://your-api-host:port
```

**通过 SSH 隧道访问内网 API：**

```bash
# 终端 1 — cloudflared 代理
cloudflared access tcp --hostname ssh.langskills.org --url localhost:2222

# 终端 2 — 端口转发
ssh -p 2222 -N -L 38889:10.150.240.117:38889 v-tiansha@localhost

# 终端 3 — 设置环境变量
export OPENAI_API_KEY=sk-1234
export OPENAI_API_BASE=http://localhost:38889
```

### 运行

```bash
conda activate agent_market

# 快速测试（1 轮迭代）
PYTHONPATH=src python -m agent_market.strategy_miner \
  --config configs/strategy_miner_maxpower.json \
  --max-iterations 1 -v

# 完整挖掘（4 轮迭代）
PYTHONPATH=src python -m agent_market.strategy_miner \
  --config configs/strategy_miner_maxpower.json -v

# 恢复中断的运行
PYTHONPATH=src python -m agent_market.strategy_miner \
  --resume artifacts/runs/<run_id>/strategy_miner/checkpoint.json -v
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--config PATH` | 配置文件（JSON） | `configs/strategy_miner_default.json` |
| `--max-iterations N` | 最大迭代次数 | 配置文件中的值 |
| `--model NAME` | 覆盖 LLM 模型名 | 配置文件中的值 |
| `--resume PATH` | 从 checkpoint 恢复 | — |
| `-v` | 启用 DEBUG 日志 | — |

### 配置文件

| 文件 | 用途 |
|------|------|
| `configs/strategy_miner_default.json` | 默认配置 |
| `configs/strategy_miner_maxpower.json` | 高性能（2 候选/轮，4 轮） |
| `configs/strategy_miner_maxpower_rerun.json` | 高并发重跑（5 候选/轮，10 轮） |
| `configs/strategy_miner_recovery.json` | 恢复配置（3 候选/轮） |

### 输出目录

```
artifacts/runs/<run_id>/strategy_miner/
├── checkpoint.json              # 恢复检查点
├── iter_N/cand_XX/
│   ├── coder/sandbox/           # 策略代码
│   ├── planner/sandbox/
│   ├── reviewer/sandbox/
│   └── backtester/sandbox/
├── leaderboard.json             # 排行榜
└── knowledge_base.json          # 知识库
```

### 流程

每轮迭代：Strategy Gen → Backtest → Evaluation → Analysis → Evolve（可选）

1. **Strategy Gen** — planner 设计方案，coder 生成代码，reviewer 审查
2. **Backtest** — freqtrade 回测，失败时 LLM 自动修复（最多 5 次）
3. **Evaluation** — Sharpe、收益率、回撤、胜率等综合打分
4. **Analysis** — LLM 总结经验教训，指导下轮改进
5. **Evolve** — 对优秀策略进行参数变异和交叉

## 已知问题

- Remixicon 字体未内置（为保证离线可用，默认不依赖外部 CDN）；因此图标可能不显示，但不影响核心功能。
- 大量结果卡片渲染建议启用虚拟化优化（已有基础优化，后续可继续增强）。

## 版本与发布

- 本仓库不维护对外发布版本号（以可运行与可复现为主）。

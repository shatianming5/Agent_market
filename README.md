# Agent Market：LLM + FreqAI 智能策略工作台

Agent Market 是一个将 LLM 表达式生成、特征工程、机器学习/强化学习训练与回测串联起来的全流程工作台。后端基于 FastAPI 提供统一 API，前端为简洁的 Flow/控制台，便于快速试验与组合能力。

- 多源数据接入（含 CCXT 等）
- 频交易 FreqAI 表达式生成、特征提取；可接驳自定义 LLM
- 机器学习/强化学习训练（LightGBM/XGBoost/CatBoost/PyTorch/SB3）
- FastAPI + Web 前端（静态目录 /web），可部署为一体化服务

## 目录

```
configs/                   # 配置（Flow/训练/回测 JSON + 数据抓取 YAML）
data/                      # 原始/加工数据（可选）
docs/                      # 文档
scripts/                   # 各类脚本（Flow、训练、清理等）
server/                    # FastAPI 后端
src/agent_market/          # 业务核心（LLM/特征/训练/Flow）
tests/                     # Pytest
web/                       # 前端静态资源（/web/index.html）
```

## 快速开始

1) 创建虚拟环境并安装依赖
```
python -m venv venv
./venv/Scripts/Activate.ps1   # Windows PowerShell
# 推荐（黄金路径：feature + expression + ml + backtest）
pip install -r requirements-full.txt

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

4) 启动后端
```
uvicorn server.main:app --host 0.0.0.0 --port 8000
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

## 已知问题

- Remixicon 字体未内置（为保证离线可用，默认不依赖外部 CDN）；因此图标可能不显示，但不影响核心功能。
- 大量结果卡片渲染建议启用虚拟化优化（已有基础优化，后续可继续增强）。

## 版本与发布

- 本仓库不维护对外发布版本号（以可运行与可复现为主）。

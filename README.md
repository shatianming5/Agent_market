# Agent Market：LLM + FreqAI 智能策略工作台

Agent Market 是一个将 LLM 表达式生成、特征工程、机器学习/强化学习训练与回测串联起来的全流程工作台。后端基于 FastAPI 提供统一 API，前端为简洁的 Flow/控制台，便于快速试验与组合能力。

- 多源数据接入（含 CCXT 等）
- 频交易 FreqAI 表达式生成、特征提取；可接驳自定义 LLM
- 机器学习/强化学习训练（LightGBM/XGBoost/CatBoost/PyTorch/SB3）
- FastAPI + Web 前端（静态目录 /web），可部署为一体化服务

## 目录

```
conf/                      # 示例/模板配置
configs/                   # Flow/训练/回测 等 JSON 配置
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
pip install -r requirements.txt
pip install -r server/requirements.txt
```

2) 可选：安装 freqtrade（用于回测/超参等）
```
git clone https://github.com/freqtrade/freqtrade.git --depth 1
cd freqtrade && pip install -e . && cd ..
```

3) 配置 LLM（可选）
在项目根目录创建 `.env`：
```
LLM_BASE_URL=https://your-llm-endpoint/v1
LLM_API_KEY=替换为你的APIKey
LLM_MODEL=gpt-3.5-turbo
```

4) 启动后端
```
uvicorn server.main:app --host 0.0.0.0 --port 8000
```
打开前端：`http://127.0.0.1:8000/web/index.html`

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
- 任务管理：`GET /jobs/{id}/status`、`GET /jobs/{id}/logs?offset=0`、`POST /jobs/{id}/cancel`
- 结果相关：
  - `GET /results/latest-summary`、`GET /results/list`、`GET /results/summary?name=...`
  - `GET /results/gallery`、`GET /results/aggregate?names=a.zip,b.zip`
  - `GET /features/top?file=...&limit=...`
  - `POST /results/prepare-feedback` 生成 LLM 反馈输入
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

备注：若 ReactFlow CDN 不可达，前端会自动降级为直接 DOM 绑定与轮询日志，功能可用但 Flow 画布交互受限。

## 自动化与清理

- 前端 DOM 烟测（Node + JSDOM）：
  - `npm i`
  - `npm run test:front`
- 工作区清理（删除临时/缓存/产物，可带 dry-run）：
  - `python scripts/clean_workspace.py`
  - `python scripts/clean_workspace.py --dry-run`

## 已知问题

- ReactFlow 仍使用 CDN（后续将本地化）；当网络受限时，仅 Flow 画布互动降级，其它核心流程可用。
- 大量结果卡片渲染建议启用虚拟化优化（后续考虑）。

## 版本与发布

- 当前后端版本：`0.2.2`（见 `server/main.py`）
- 变更记录：见 `CHANGELOG.md`
- 详细说明：`RELEASE_NOTES_v0.2.2.md`

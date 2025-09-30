# Agent Market（LLM + FreqAI 研究/回测平台）

Agent Market 是一个将大语言模型（LLM）与 FreqAI 交易研究流程整合的轻量级平台：在一个界面里完成特征工程、表达式生成、ML/RL 训练与回测，并提供标准化 API 与前端可视化控制（Agent Flow）。

- 数据获取与清洗（可选脚本工具，支持 CCXT）
- 基于 FreqAI 的特征与表达式生成（支持 LLM 协助）
- ML/RL 训练与模型产出（LightGBM/XGBoost/CatBoost/PyTorch/Stable-Baselines3）
- 使用 FastAPI 提供统一的后端接口，内置简单 Web 前端（/web/index.html）

## 目录结构

```
conf/                      # 配置辅助（如交易对列表等）
configs/                   # Flow / 训练 / 回测 等 JSON 配置
data/                      # 原始与清洗数据（按脚本输出约定）
docs/                      # 设计与架构文档
scripts/                   # 流程脚本（Flow、训练、回测、工具）
server/                    # FastAPI 服务与路由
src/agent_market/          # 业务逻辑（LLM/特征/训练/Flow 等）
tests/                     # Pytest 用例
web/                       # 前端静态资源
```

## 快速开始

1) 安装依赖（建议 venv）
```
python -m venv venv
./venv/Scripts/Activate.ps1   # Windows PowerShell
pip install -r requirements.txt
pip install -r server/requirements.txt
```

2)（可选）安装 freqtrade 以运行回测/超参
```
git clone https://github.com/freqtrade/freqtrade.git --depth 1
cd freqtrade && pip install -e . && cd ..
```

3)（可选）配置 LLM（如需表达式生成）
在项目根目录创建 `.env`：
```
LLM_BASE_URL=https://your-llm-endpoint/v1
LLM_API_KEY=你的APIKey
LLM_MODEL=gpt-3.5-turbo
```

4) 启动后端
```
uvicorn server.main:app --host 0.0.0.0 --port 8000
```
打开 `http://127.0.0.1:8000/web/index.html` 访问前端。默认同源 API，可在侧栏“API”输入框切换后端地址并“应用”。

## 核心接口

- 健康检查与入口：`GET /health`、`GET /`、`GET /docs`
- 运行：
  - `POST /run/feature`（生成特征）
  - `POST /run/expression`（生成表达式，含 LLM）
  - `POST /run/backtest`（回测）
  - `POST /run/hyperopt`（超参）
  - `POST /run/rl_train`（强化学习训练）
  - `POST /run/train`（ML 训练，支持 config_obj 内联校验）
  - `POST /flow/run`（按配置的多步 Flow 运行）
- 作业：`GET /jobs/{id}/status`、`GET /jobs/{id}/logs?offset=0`、`POST /jobs/{id}/cancel`
- 结果：
  - `GET /results/latest-summary`、`GET /results/list`、`GET /results/summary?name=...`
  - `GET /results/gallery`、`GET /results/aggregate?names=a.zip,b.zip`
  - `GET /features/top?file=...&limit=...`
  - `POST /results/prepare-feedback`（为 LLM 反馈准备摘要）
- Flow 进度：`GET /flow/progress/{job_id}?steps=feature,expression,ml,rl,backtest`
- 设置：`GET /settings`、`POST /settings`（llm_base_url/llm_model/default_timeframe）

所有 /run/* 接口在参数校验失败时返回统一错误结构：
```
{ "status": "error", "code": "INVALID_TIMEFRAME", "message": "..." }
```
任务启动成功统一返回：
```
{ "status": "started", "job_id": "...", "kind": "expression|feature|...", "cmd": [ ... ] }
```

## 前端使用要点

- 顶部工具条：自动布局/对齐/等距/主题/导出等快捷按钮。
- 左侧：
  - API 切换与服务设置（/settings，加载/保存/应用到表单）
  - 常用参数与“一键表达式/回测/加载摘要”
  - 特征 TopN 与图表、Agent Flow（含单步快捷按钮）
  - 结果 列表/对比、图集与聚合卡片
- 右侧：
  - 状态栏、日志、关键指标卡片（收益/交易数/胜率/最大回撤/最近训练/验证RMSE）

失败任务会在状态栏高亮并提供“重试上次操作”按钮（基于最近一次 /run/* 提交参数）。

## 故障排查

- GET / 与 /index：已清除乱码，/ 返回简明入口；/docs 提供完整交互文档。
- 前端字符串：已全量替换为 UTF-8 中文，避免历史乱码。
- 常见环境问题：
  - 回测/超参依赖 freqtrade；若未安装，对应功能无法执行。
  - LLM 相关需正确配置 `.env` 或通过 /settings 设置。
- 大量结果渲染时建议逐步加载图集以保证性能。

## 维护与清理

- 一键清理（默认目标包含 artifacts、.pytest_cache、user_data 下常见生成目录）
  - `python scripts/clean_workspace.py`
- 仅预览将被删除的内容
  - `python scripts/clean_workspace.py --dry-run`
- 清理后保留空目录（便于挂载/后续写入）
  - `python scripts/clean_workspace.py --keep-dirs`
- 指定自定义清理目标
  - `python scripts/clean_workspace.py user_data/tmp artifacts/cache`
- Windows 快捷脚本（PowerShell）
  - 直接运行：`./scripts/clean_workspace.ps1`
  - 预览：`./scripts/clean_workspace.ps1 -DryRun`
  - 保留空目录：`./scripts/clean_workspace.ps1 -KeepDirs`

## 许可证

本项目为内部/实验性质示例工程，未附带开源许可证。如需公开发布，请按需增补许可证与版权说明。

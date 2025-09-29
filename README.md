# Agent Market 项目总览`n`n> 新增 server/ FastAPI 原型：提供 /run/expression、/run/backtest、/flow/run 任务执行与 /jobs/{id}/logs 拉取日志。见下方“Web 服务(原型)”章节。

本仓库聚合了两个维度的能力：

1. **行情与资讯采集流水线**：位于根目录 `scripts/` 与 `conf/` 之下，涵盖链上 dex 索引、CCXT 行情、Binance Bulk 数据、Twitter(X) 流等实时/离线抓取与清洗脚本。
2. **FreqAI 量化回测体系**：在 `freqtrade/` 子模块基础上增强，提供自动特征排名、LLM 驱动的复合因子生成、LightGBM 模型回测与结果分析。

## 目录结构

```
├─conf/                     # 数据与任务的 YAML 配置
├─data/                     # 原始 / 清洗后的行情数据
├─docs/                     # 项目文档（LLM 流水线、架构说明）
├─freqtrade/                # 上游 freqtrade 仓库 + 本项目扩展脚本
│  └─scripts/
│     ├─freqai_expression_agent.py  # 支持 LLM/GP 的因子构造脚本
│     └─freqai_auto_agent.py        # 一键完成特征→表达式→回测流水线
├─scripts/                  # 行情、资讯、链上数据等采集清洗脚本
├─src/agent_market/         # 自定义 Python 包（LLM 辅助 & 配置中心）
├─tests/                    # Pytest 冒烟用例
├─task*.txt                 # 任务分解说明
└─README.md                 # 当前文件
```

> ✨ 新增 `src/agent_market/freqai/llm.py` 与 `src/agent_market/config.py`，分别承担 LLM 调度与 FreqAI 配置统一治理，可被多个脚本共享。

## 安装与环境

```powershell
# 可选：使用已有虚拟环境
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 推荐：使用已有 Conda 环境（freqtrade）
conda env create -f freqtrade/environment.yml   # 如已存在可跳过
conda activate freqtrade
pip install -e ./freqtrade
```

额外依赖：确保 `.env` 或环境变量提供 Twitter、NewsAPI、LLM 等访问凭证。

## 行情/资讯流水线脚本

`scripts/` 下脚本均支持 `--help` 查询参数。常用示例：

| 步骤 | 命令 | 说明 |
| --- | --- | --- |
| 行情下载 | `python scripts/fetch_ccxt_ohlcv.py --conf conf/symbols.yaml` | 通过 CCXT 获取 BinanceUS 现货 K 线 |
| 历史补全 | `python scripts/fetch_binance_bulk.py --conf conf/symbols.yaml --limit-months 4` | 批量下载 Binance Data Vision 历史压缩包 |
| 数据清洗 | `python scripts/clean_ohlcv.py --conf conf/symbols.yaml` | 合并衍生/补全后的 K 线，输出到 `data/clean/` |
| 质量审计 | `python scripts/dq_report.py --mode ohlcv --conf conf/symbols.yaml` | 生成缺失/异常点报告 |
| Twitter 流 | `python scripts/x_stream.py --conf conf/x_rules.yaml` | 订阅实时推文并落盘 |

## FreqAI 配置中心

`src/agent_market/config.py` 提供 `FreqAISettings` 数据类，集中读取 `user_data/config_freqai.json` 等配置，自动解析：

- 数据目录 (`datadir`)，支持相对路径与多级回退
- 交易所与交易对白名单
- `train_period_days` / `backtest_period_days` / `label_period_candles`
- 数据完整性校验：`settings.validate_dataset()` 会检查所需 `.feather` 文件是否存在

在 `freqai_auto_agent.py` 中已经使用：
```python
settings = FreqAISettings.from_file(args.config, args.timeframe, args.label_period)
settings.validate_dataset()
```

## FreqAI LLM ?????

### 1. ??????

```powershell
conda run --no-capture-output -n freqtrade \
  python freqtrade/scripts/freqai_feature_agent.py \
    --config freqtrade/user_data/config_freqai.json \
    --timeframe 1h
```

### 2. LLM ????

```powershell
conda run --no-capture-output -n freqtrade \
  python freqtrade/scripts/freqai_expression_agent.py \
    --feature-file user_data/freqai_features.json \
    --output user_data/freqai_expressions.json \
    --timeframe 1h \
    --llm-count 10 \
    --llm-loops 12
```

- `.env` ??? `LLM_BASE_URL=https://api.zhizengzeng.com/v1`?`LLM_API_KEY=<????>`?`LLM_MODEL=gpt-3.5-turbo` ?????? LLM?
- `--llm-loops` ??????????????????,`--llm-fallback` ???????????/????,`--no-llm` ??????
- ?????????????? 5 ??????,?????? LLM ???? `user_data/freqai_expressions.json`,??? 50 ? `user_data/freqai_expressions_augmented.json`(?? zscore????EMA???????????????),???????

### 3. ???????

```powershell
conda run --no-capture-output -n freqtrade \
  python freqtrade/scripts/freqai_auto_agent.py \
    --config freqtrade/user_data/config_freqai.json \
    --timeframe 1h \
    --top-expressions 40 \
    --expression-combo-top 5
```

?????:

1. ?????????????? `user_data/freqai_features.json`(??? `datadir`)?
2. ?? LLM ?????,????/??/?????????? `user_data/freqai_expressions.json`?
3. ??????/??????? freqtrade backtesting,????? `user_data/backtest_results/auto_agent/<timestamp>/`?
4. ?????????????????????

??(LightGBM ?? + 1h ??):

```
[llm] request 50 expressions from gpt-3.5-turbo
[llm] tokens prompt=2727 completion=529
[llm] valid expressions 5
[summary] FreqAIExampleStrategy ????: 49, ???: -8.5543, ??: 40.82%
[??] ??????: user_data\backtest_results\auto_agent\20250928-034328\backtest-result-2025-09-27_16-25-24.zip
```

### 4. ?? 50 ??????

```powershell
conda run --no-capture-output -n freqtrade \
  python freqtrade/scripts/freqai_expression_sweeper.py \
    --config freqtrade/user_data/config_freqai.json \
    --timeframe 1h \
    --expressions user_data/freqai_expressions_augmented.json \
    --limit 50
```

- ???? `user_data/backtest_results/expr_sweeper/20250928-044626/summary.json`,?? 50 ???????
- ???????????,????????????????????,???????????????

???????????,??? `docs/llm_pipeline.md` ? `docs/ARCHITECTURE.md`?

## 快速测试

项目内置最小 Pytest 冒烟用例：

```powershell
conda run --no-capture-output -n freqtrade python -m pytest
```

- `tests/test_freqai_pipeline.py` 会模拟 LLM 响应，验证表达式评分链路
- 同时断言 `FreqAISettings` 的数据目录校验逻辑

> 若环境缺少 `pytest`，可执行 `conda run -n freqtrade pip install pytest` 后再运行


## 多资产 4h 流水线（LLM + 回测）

- 生成多资产特征（BTC/ETH/SOL/ADA，4h）：
  - `conda run -n freqtrade python freqtrade/scripts/freqai_feature_agent.py --config configs/config_freqai_multi.json --output user_data/freqai_features_multi.json --timeframe 4h --pairs BTC/USDT ETH/USDT SOL/USDT ADA/USDT`
- 基于 LLM 生成表达式（走智增增 API，默认 `gpt-3.5-turbo`）：
  - `conda run -n freqtrade python scripts/agent_flow.py --config configs/agent_flow_multi.json --steps expression`（需 `PYTHONPATH=src`）
  - 结果输出：`user_data/freqai_expressions.json`
- 真实回测（ExpressionLongStrategy）：
  - `conda run -n freqtrade freqtrade backtesting --config configs/config_freqai_multi.json --strategy ExpressionLongStrategy --strategy-path freqtrade/user_data/strategies --timerange 20210101-20211231 --freqaimodel LightGBMRegressor --export trades --export-filename user_data/backtest_results/latest_trades_multi`

提示：若 `user_data/data/binanceus` 为空，可将 `freqtrade/user_data/data/binanceus/*.feather` 拷贝过去以便快速跑通。

## 超参优化（Hyperopt）

策略已内建可调参空间（`DecimalParameter`）：
- 进场/出场动态分位（`dynamic_entry_q`/`dynamic_exit_q`）
- 投票阈值（`vote_entry_threshold_p`/`vote_exit_threshold_p`）
- 信号最小/最大门槛（`signal_entry_min`/`signal_exit_max`）
- 仓位上限（`stake_scale_cap`）

运行示例（小样本演示）：

```
conda run -n freqtrade freqtrade hyperopt \
  --config configs/config_freqai_multi.json \
  --strategy ExpressionLongStrategy \
  --strategy-path freqtrade/user_data/strategies \
  --timerange 20210101-20210430 \
  --spaces buy sell protection \
  --hyperopt-loss SharpeHyperOptLoss \
  --epochs 20 \
  --freqaimodel LightGBMRegressor
```

Hyperopt 会将找到的最优参数写入：`freqtrade/user_data/strategies/ExpressionLongStrategy.json`，后续回测会自动加载。

## 回测报表与汇总

- 生成最新回测摘要（读取 `user_data/backtest_results` 最新 zip）：
  - `conda run -n freqtrade python scripts/report_backtest.py --results-dir user_data/backtest_results --out user_data/reports/latest_summary.json`
- LLM 连通性自检：
  - `conda run -n freqtrade python scripts/test_llm.py`（需 `LLM_API_KEY` 环境变量）

## Web 服务（原型）

- 依赖安装：`conda run -n freqtrade pip install -r server/requirements.txt`
- 启动（本地）：
  - `conda run -n freqtrade uvicorn server.main:app --host 0.0.0.0 --port 8000`
- 主要接口：
  - `GET /health`：健康检查
  - `POST /run/expression`：触发表达式生成（LLM）
  - `POST /run/backtest`：触发回测
  - `POST /flow/run`：按 agent_flow 配置运行（可选 steps）
  - `GET /jobs/{job_id}/status`、`GET /jobs/{job_id}/logs?offset=0`：查询状态与拉取日志
- 冒烟测试（不启服务）：
  - `conda run -n freqtrade python scripts/server_smoke.py`

## 统一 AI 学习框架规划

项目正在分阶段扩展机器学习、深度学习与强化学习能力，并构建可由智能 Agent 全流程调度的研究流水线。总体设计详见 [docs/ai_framework.md](docs/ai_framework.md)。后续阶段将在该文档的架构下逐步实现：

1. 模型接口统一化，支持 LightGBM/XGBoost/CatBoost 等传统 ML。
2. 引入 PyTorch 深度学习模型与训练配置。
3. 提供强化学习交易环境与策略训练骨架。
4. 打通 Agent Flow，从特征生成到回测评估一站式自动化。

### 新增模块与功能
- `agent_market/freqai/model/torch_models.py`：提供 PyTorch MLP 适配器，支持自定义隐藏层与 Dropout。
- `freqtrade/scripts/freqai_feature_agent.py`:?? 12 ?????(KAMA?MACD ???Stochastic?PSAR????/???Donchian?VWAP ???),???? `user_data/freqai_features.json`?
- `agent_market/freqai/rl/`：包含 `TradingEnv`、`RLTrainer`，可用 Stable-Baselines3 训练 PPO 策略。
- `agent_market/freqai/training/pipeline.py`：统一数据构建与模型训练流程。



# Agent Market 项目总览

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

## FreqAI LLM 因子流水线

### 1. 生成基础特征

```powershell
conda run -n freqtrade python scripts/freqai_feature_agent.py \
    --config user_data/config_freqai.json \
    --timeframe 1h
```

### 2. LLM 因子构造（单步）

```powershell
conda run -n freqtrade python scripts/freqai_expression_agent.py \
    --feature-file user_data/freqai_features.json \
    --output user_data/freqai_expressions.json \
    --timeframe 1h \
    --llm-enabled \
    --llm-api-key <YOUR_API_KEY>
```

### 3. 一键完整跑通

```powershell
conda run --no-capture-output -n freqtrade \
  python scripts/freqai_auto_agent.py \
    --config user_data/config_freqai.json \
    --timeframe 1h \
    --llm-enabled \
    --llm-api-key <YOUR_API_KEY> \
    --top-expressions 40 \
    --expression-combo-top 5
```

自动流程会：

1. 对多个交易对聚合基础特征并写入 `user_data/freqai_features.json`
2. 调用 LLM/模板/遗传算法生成表达式，按稳定性、夏普等指标评分
3. 触发 freqtrade backtesting，将结果复制到 `user_data/backtest_results/auto_agent/<timestamp>/`
4. 输出回测摘要：交易笔数、累计收益、胜率等

示例输出：

```
[llm] valid expressions 5
[summary] FreqAIExampleStrategy 交易笔数: 49, 总收益: -8.5543, 胜率: 40.82%
[完成] 回测结果位置: user_data\backtest_results\auto_agent\20250927-205525\backtest-result-2025-09-27_16-25-24.zip
```

更多使用细节与常见问题，请参阅 `docs/llm_pipeline.md` 与 `docs/ARCHITECTURE.md`。

## 快速测试

项目内置最小 Pytest 冒烟用例：

```powershell
conda run --no-capture-output -n freqtrade python -m pytest
```

- `tests/test_freqai_pipeline.py` 会模拟 LLM 响应，验证表达式评分链路
- 同时断言 `FreqAISettings` 的数据目录校验逻辑

> 若环境缺少 `pytest`，可执行 `conda run -n freqtrade pip install pytest` 后再运行



# 真实流程回测计划（可逐步调试）

本页给出一个尽量“可逐一执行/逐一调试”的真实流程模板。默认所有运行产物都落在仓库根目录的 `user_data/`。

## 0. 前置

- Python 虚拟环境已安装依赖（推荐黄金路径）：`pip install -r requirements-full.txt`
- 已准备好 Freqtrade 配置（示例）：
  - Bitfinex：`user_data/config_freqai.json`
  - KuCoin：`user_data/config_freqai_kucoin.json`
- 运行目录约定：
  - 数据：`user_data/data/<exchange>/*-<timeframe>.feather`
  - 策略：`user_data/strategies/ExpressionLongStrategy.py`
  - 回测结果：`user_data/backtest_results/`

## 1.（可选）下载数据（以 KuCoin 为例）

使用已安装的 freqtrade CLI，并显式指定 `--userdir user_data`：

```bash
./.venv/bin/freqtrade download-data \
  --userdir user_data \
  --config user_data/config_freqai_kucoin.json \
  --timeframes 1h \
  --pairs BTC/USDT ETH/USDT \
  --dataformat-ohlcv feather
```

如果你希望自动下载更多币种/更长周期，可以在配置文件的 `exchange.pair_whitelist` 里补充 pairs，并调整 `--timeframes`。

## 2. 逐步跑真实流程（推荐：Agent Flow）

KuCoin 示例配置：`configs/agent_flow_kucoin_example.json`

```bash
# 只跑特征生成
./.venv/bin/python scripts/agent_flow.py --config configs/agent_flow_kucoin_example.json --steps feature

# 再跑表达式生成 + 因子挖掘（内置 top-N；默认开启进化算法，可用 --no-evolve 关闭）
./.venv/bin/python scripts/agent_flow.py --config configs/agent_flow_kucoin_example.json --steps expression

# 再跑 ML 训练
./.venv/bin/python scripts/agent_flow.py --config configs/agent_flow_kucoin_example.json --steps ml

# 最后回测
./.venv/bin/python scripts/agent_flow.py --config configs/agent_flow_kucoin_example.json --steps backtest
```

## 3. 产物检查点（每一步都可单独验证）

- 特征文件：`user_data/freqai_features_real.json`
- 挖掘后的表达式（Top-N）：`user_data/freqai_expressions_selected.json`
- 全量打分候选：`user_data/freqai_expressions_selected_scored_all.json`
- 训练模型摘要：`artifacts/models/**/training_summary.json`
- 回测结果：`user_data/backtest_results/backtest-result-*.zip`
- Flow 日志：`user_data/agent_logs/agent_flow_*.log`

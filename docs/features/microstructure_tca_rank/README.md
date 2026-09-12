# 微观结构 / TCA / Rank Portfolio

[返回主 README](../../../README.md) · [返回功能索引](../README.md)

这一组功能覆盖市场微观结构数据、交易成本分析和 cross-sectional rank portfolio。它们可以作为 Agent Flow step 使用，也可以通过 Factor Lab 或独立脚本运行。

## 微观结构

代码：

```text
src/agent_market/microstructure/
scripts/micro_capture.py
scripts/lob_snapshot.py
scripts/lob_rebuild.py
scripts/micro_features.py
scripts/micro_feature_merge_hourly.py
```

能力：

- KuCoin capture
- REST / WebSocket collectors
- LOB snapshot
- LOB rebuild 与 checksum
- OHLCV / L2 / trades 微观结构特征
- parquet schema
- hourly merge 到研究特征

Flow/API：

```text
agent_flow step: capture
agent_flow step: lob_rebuild
agent_flow step: micro_feature
POST /run/capture
POST /run/lob_rebuild
POST /run/micro_feature
GET  /flow/micro-feature/{run_id}
```

## TCA

代码：

```text
src/agent_market/tca/
scripts/tca_report.py
```

能力：

- 解析 Freqtrade backtest zip
- 提取 trades
- 生成稳定 JSON TCA report
- 给 Flow、UI 和报告 bundle 消费

Flow/API：

```text
agent_flow step: tca
POST /run/tca
GET  /flow/tca/latest
GET  /flow/tca/{run_id}
```

## Rank Portfolio

代码：

```text
src/agent_market/factor_lab/rank_portfolio.py
user_data/strategies/ELRankPortfolioLeverageStrategy.py
```

能力：

- cross-sectional factor selection
- ensemble scoring
- rolling IC / regime filters
- pair exclusions
- dynamic per-pair leverage
- liquidation-distance guards
- account kill modes
- signal export
- Freqtrade futures strategy 消费 signals

常用命令：

```bash
python scripts/factor_lab.py rank-export --tag exp1 --n 50 --risk-profile aggressive
python scripts/factor_lab.py rank-backtest --tag exp1 --venue okx --top-k 3 --gross-cap 10
python scripts/factor_lab.py rank-sweep --tag exp1 --venue okx
```

Freqtrade 回测：

```bash
freqtrade backtesting \
  --config user_data/config_okx_futures_rank_backtest.json \
  --strategy ELRankPortfolioLeverageStrategy \
  --strategy-path user_data/strategies
```

## 产物

```text
artifacts/runs/<run_id>/micro_feature/
artifacts/runs/<run_id>/tca/
artifacts/rank_portfolio/<tag>/selected_factors.json
artifacts/rank_portfolio/<tag>/signals/*.feather
artifacts/rank_portfolio/<tag>/rank_export.json
artifacts/rank_portfolio/<tag>/backtest.json
artifacts/rank_portfolio/<tag>/sweep.json
```

## 注意事项

- 真实微观结构和 futures 回测都依赖本地数据覆盖；缺 feather/parquet 时通常会在运行后段失败。
- `user_data/` 里可能有真实 OHLCV、策略和回测结果，改动前确认目的。
- Rank portfolio 的风险参数可通过 CLI 参数和 `RP_*` 环境变量覆盖。


# Strategy Miner Alpha Runbook

这套 runbook 对应新的分组 alpha 搜索方案，目标是把搜索拆组、拉长评估窗、放宽到 `1h` informative，并让 `profit_pct` / `return_over_drawdown` 在排序里更有话语权。

## 分组配置

- 现货主流组: `configs/strategy_miner_alpha_spot_core.json`
- 现货 L1/L2 组: `configs/strategy_miner_alpha_spot_l1l2.json`
- 现货 meme 组: `configs/strategy_miner_alpha_spot_meme.json`
- 期货主流组: `configs/strategy_miner_alpha_futures_core.json`
- 期货 L1/L2 组: `configs/strategy_miner_alpha_futures_l1l2.json`
- 期货 meme 组: `configs/strategy_miner_alpha_futures_meme.json`

## 数据准备

先检查缺口：

```bash
python3 scripts/prepare_strategy_miner_data.py \
  --miner-config configs/strategy_miner_alpha_spot_core.json \
  --miner-config configs/strategy_miner_alpha_spot_l1l2.json \
  --miner-config configs/strategy_miner_alpha_spot_meme.json \
  --miner-config configs/strategy_miner_alpha_futures_core.json \
  --miner-config configs/strategy_miner_alpha_futures_l1l2.json \
  --miner-config configs/strategy_miner_alpha_futures_meme.json \
  --check-only
```

本地代理是 `1097` 时，直接补齐：

```bash
python3 scripts/prepare_strategy_miner_data.py \
  --miner-config configs/strategy_miner_alpha_spot_core.json \
  --miner-config configs/strategy_miner_alpha_spot_l1l2.json \
  --miner-config configs/strategy_miner_alpha_spot_meme.json \
  --miner-config configs/strategy_miner_alpha_futures_core.json \
  --miner-config configs/strategy_miner_alpha_futures_l1l2.json \
  --miner-config configs/strategy_miner_alpha_futures_meme.json \
  --proxy http://127.0.0.1:1097
```

## 开跑

单组启动示例：

```bash
python3 scripts/strategy_miner.py --config configs/strategy_miner_alpha_spot_core.json
python3 scripts/strategy_miner.py --config configs/strategy_miner_alpha_futures_core.json
```

## 这版和旧配置的区别

- 把交易池拆成 `core / l1l2 / meme`，避免一套策略硬打 14 个币。
- 评分增加 `objective_weights`，重点提高 `profit_pct` 和 `return_over_drawdown`。
- `target_trades` 下调，不再默认追 `300` 笔级别的高频策略。
- 允许 `1h` informative，搜索家族扩到 `trend-following / breakout / pair-relative / basket-rotation / dl / rl`。
- 数据准备脚本会自动按 config 推导 pairs、timeframes、timerange，并用 Freqtrade 下载缺失数据。

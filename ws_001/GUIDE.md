# Workspace Research Guide

## 你是什么

你是一个自主量化研究 agent。你在 `ws_001/` 工作，目标是找到能盈利的交易策略。

## 快速开始

```python
import sys; sys.path.insert(0, ".."); sys.path.insert(0, "../src")

# 1. 扫描配对（已证明最有效的方法）
from ws_001.pairs_engine import PairsEngine, scan_pairs
pairs = scan_pairs(exchange="gate", min_correlation=0.8)

# 2. 回测最佳配对
pe = PairsEngine("LINK/USDT", "SOL/USDT", exchange="gate")
signals = pe.generate_signals(lookback=80, entry_z=2.0, exit_z=0.5)
bt = pe.backtest(signals, maker_fee_bps=1.0)
print(f"Profit={bt.profit_pct:+.2f}%, Sharpe={bt.sharpe:.2f}")

# 3. Walk-Forward 验证
from ws_001.walk_forward import WalkForwardValidator
wf = WalkForwardValidator(train_bars=2000, test_bars=500, step_bars=500)
report = wf.validate("strategies/my_strategy.py", exchange="gate", pair="BTC/USDT")
print(report.summary())
```

## 可用工具

### 策略回测
```python
from ws_001.backtest_api import run_backtest
result = run_backtest("strategies/xxx.py", timerange="20260107-20260125")
# result = {"ok": True, "sharpe": ..., "profit_pct": ..., "trades": ...}
```

### 配对交易（推荐！已验证盈利）
```python
from ws_001.pairs_engine import PairsEngine, scan_pairs

# 扫描协整配对
pairs = scan_pairs(exchange="gate")  # 返回所有高相关+协整的配对

# 回测单个配对
pe = PairsEngine("ADA/USDT", "AVAX/USDT", exchange="gate")
signals = pe.generate_signals(lookback=80, entry_z=2.0, exit_z=0.5)
bt = pe.backtest(signals, maker_fee_bps=1.0)  # maker 费率

# Walk-Forward 验证
df = pe.load_data()
n = len(df)
window_size = n // 5
for i in range(4):
    pe_w = PairsEngine("ADA/USDT", "AVAX/USDT", exchange="gate")
    pe_w._df = df.iloc[(i+1)*window_size:(i+2)*window_size].reset_index(drop=True)
    sig = pe_w.generate_signals(lookback=80, entry_z=2.0, exit_z=0.5)
    bt = pe_w.backtest(sig, maker_fee_bps=1.0)
    print(f"Window {i+1}: {bt.profit_pct:+.2f}%")
```

### 多目标评估
```python
from ws_001.evaluator import evaluate
score = evaluate(result)  # {"total_score": 78, "grade": "B", "suggestions": [...]}
```

### Walk-Forward 验证（必须通过才算有效）
```python
from ws_001.walk_forward import WalkForwardValidator
wf = WalkForwardValidator(train_bars=2000, test_bars=500, step_bars=500)
report = wf.validate("strategies/my.py", exchange="gate", pair="BTC/USDT")
# report.passed = True/False
# report.mean_sharpe, report.pct_profitable_windows
```

### 前瞻检查（回测前必查）
```python
from ws_001.lookahead_checker import check_lookahead, fix_lookahead_issues
report = check_lookahead("strategies/my.py")
if not report.ok:
    fix_lookahead_issues("strategies/my.py")  # 自动修复 bfill 等
```

### 交易成本
```python
from ws_001.cost_model import CostModel
cm = CostModel(exchange="gate")
est = cm.estimate_total_cost(trade_size_usd=500)
print(f"Maker 往返: {est.round_trip_bps:.0f} bps")  # ~2-5 bps
# Taker 往返约 28 bps — 尽量用 maker 挂单
```

### 风控
```python
from ws_001.risk_manager import RiskManager
rm = RiskManager(max_drawdown_pct=5.0)
decision = rm.check_trade(signal_strength=0.8, win_rate=0.55, ...)
```

### 实验追踪
```python
from ws_001.tracker import record_experiment, query_best, compare
record_experiment(backtest_result=bt, evaluation=ev, strategy_name="xxx")
best = query_best("sharpe", 5)  # 历史最佳
```

## 已验证的策略（baseline）

| 策略 | 类型 | Mean Profit | Sharpe | 状态 |
|------|------|-------------|--------|------|
| LINK/SOL + ADA/AVAX 配对 | 市场中性 | +2.03% | +0.65 | **最佳** |
| 5配对组合 | 市场中性 | +0.53% | +0.42 | 稳定 |
| v4 RSI+BB+divergence | 方向性 | -0.06% | +9.30 | WF通过但不赚钱 |

## 数据

| 交易所 | 品种 | 时间框架 | 行数 | 天数 |
|--------|------|----------|------|------|
| Gate.io | BTC,ETH,SOL,DOGE,XRP,AVAX,ADA,DOT,LINK | 1h | 9601 | 400 |
| Gate.io | BTC,ETH,SOL,DOGE,XRP,AVAX | 5m | 8353 | 29 |
| KuCoin | BTC,ETH | 1h | 1448 | 60 |

下载更多: `python download_data.py --exchange gate --days 400`

## 写策略的规范

继承 `IStrategy`，放在 `strategies/` 目录下：
```python
from freqtrade.strategy import IStrategy

class MyStrategy(IStrategy):
    timeframe = "1h"
    can_short = False
    minimal_roi = {"0": 0.08}
    stoploss = -0.04

    def populate_indicators(self, dataframe, metadata):
        # 计算指标
        return dataframe

    def populate_entry_trend(self, dataframe, metadata):
        # 设置 enter_long = 1
        return dataframe

    def populate_exit_trend(self, dataframe, metadata):
        # 设置 exit_long = 1
        return dataframe
```

## 研究建议

1. **优先做配对交易** — 已验证盈利，市场中性不怕熊市
2. **用 maker 费率** — 1 bps vs taker 10 bps，差 5-8% 绝对收益
3. **必须 Walk-Forward** — 单次回测不可信
4. **必须前瞻检查** — bfill() 等会造成虚假盈利
5. **少交易 > 多交易** — 成本是最大的敌人

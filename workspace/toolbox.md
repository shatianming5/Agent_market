# OpenCode Quant Workspace — Complete Toolbox

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│ auto_improver.py — OpenCode Agent Brain                  │
│   opencode run → analyze → write model → train → test   │
├─────────────────────────────────────────────────────────┤
│ ensemble.py — Regime Detection + Signal Combination      │
│   trending_up / trending_down / ranging / volatile       │
├─────────────────────────────────────────────────────────┤
│ risk_manager.py — Kelly + Circuit Breaker                │
│   position sizing / DD stop / consecutive loss limit     │
├─────────────────────────────────────────────────────────┤
│ walk_forward.py + lookahead_checker.py — Validation      │
│   rolling OOS / bfill detection / train-test overlap     │
├─────────────────────────────────────────────────────────┤
│ backtest_api.py + evaluator.py + tracker.py              │
│   one-call backtest / multi-objective score / experiment │
├─────────────────────────────────────────────────────────┤
│ cost_model.py + feature_selector.py + universe_selector  │
│   realistic fees / MI+stability ranking / liquidity scan │
├─────────────────────────────────────────────────────────┤
│ paper_trader.py — Simulated Live Trading                 │
│   order execution / PnL tracking / equity history        │
└─────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Auto-Improver (Full Autonomous Mode)
```python
from workspace.auto_improver import AutoImprover
ai = AutoImprover()

# OpenCode writes ML model → trains → writes strategy → backtests
report = ai.run_full_cycle(model_types=["ml", "dl"], max_iterations=3)

# Pure strategy generation (technical indicators)
report = ai.run_cycle(max_iterations=5)
```

### 2. Single Backtest
```python
from workspace.backtest_api import run_backtest
result = run_backtest("workspace/strategies/my_strategy.py", timerange="20260107-20260125")
```

### 3. Walk-Forward Validation
```python
from workspace.walk_forward import WalkForwardValidator
wf = WalkForwardValidator(train_bars=400, test_bars=150, step_bars=150)
report = wf.validate("workspace/strategies/my_strategy.py", exchange="kucoin")
print(report.summary())  # PASS/FAIL with per-window breakdown
```

### 4. Multi-Objective Evaluation
```python
from workspace.evaluator import evaluate
score = evaluate(backtest_result)  # {total_score, grade, details, suggestions}
```

### 5. Cost Analysis
```python
from workspace.cost_model import CostModel
model = CostModel(exchange="gate")
cost = model.estimate_total_cost(trade_size_usd=500, daily_volume_usd=1e8)
print(f"Round-trip: {cost.round_trip_bps:.0f} bps")  # ~46 bps
```

### 6. Market Regime
```python
from workspace.ensemble import RegimeDetector
rd = RegimeDetector()
state = rd.current_regime(df)  # trending_up / trending_down / ranging / volatile
```

### 7. Risk Check Before Trade
```python
from workspace.risk_manager import RiskManager
rm = RiskManager(max_drawdown_pct=5.0)
decision = rm.check_trade(signal_strength=0.8, win_rate=0.55, avg_win_pct=1.5, avg_loss_pct=1.0)
if decision.allowed:
    print(f"Trade OK, size={decision.position_size_pct}%")
```

### 8. Paper Trading
```python
from workspace.paper_trader import PaperTrader
pt = PaperTrader(initial_equity=1000, pairs=["BTC/USDT"])
pt.update_prices({"BTC/USDT": 84000})
pt.submit_order("BTC/USDT", "buy", size_usd=100)
print(pt.status())
```

## Module Reference

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| auto_improver | OpenCode agent brain | run_cycle(), run_full_cycle(), generate_model() |
| backtest_api | Backtest wrapper | run_backtest() → {sharpe, profit, dd, ...} |
| evaluator | Multi-objective scoring | evaluate() → {score, grade, suggestions} |
| tracker | Experiment history | record_experiment(), query_best(), compare() |
| orchestrator | Batch runner | run_research_loop(), run_experiment() |
| model_loader | Dynamic model registry | scan_and_register(), list_available_models() |
| lookahead_checker | Anti-cheating | check_lookahead(), fix_lookahead_issues() |
| walk_forward | Rolling validation | WalkForwardValidator.validate() |
| cost_model | Realistic costs | CostModel.estimate_total_cost() |
| universe_selector | Asset filtering | select_universe() |
| feature_selector | Dimensionality reduction | select_features(), build_feature_matrix() |
| ensemble | Regime + combination | RegimeDetector, StrategyEnsemble |
| risk_manager | Position sizing + stops | RiskManager.check_trade() |
| paper_trader | Simulated trading | PaperTrader.submit_order() |
| download_data | Data acquisition | CLI: python workspace/download_data.py |

## Data

| Source | Pairs | Bars | Period |
|--------|-------|------|--------|
| KuCoin | BTC/USDT, ETH/USDT | ~1448 | ~60 days |
| Gate.io | BTC, ETH, SOL, DOGE, XRP, AVAX | ~1000 | ~41 days |

For more data: `python workspace/download_data.py --exchange gate --days 730`

## Optimization Targets (objectives.json)

| Metric | Target | Weight |
|--------|--------|--------|
| Sharpe | ≥ 1.0 | 30% |
| Max DD | ≤ 5% | 20% |
| Profit | ≥ 1% | 20% |
| Sortino | ≥ 1.0 | 10% |
| Win Rate | ≥ 45% | 10% |
| Profit Factor | ≥ 1.1 | 10% |

## Integration Test

```bash
PYTHONPATH=. python3 workspace/integration_test.py
# Expected: 40 passed, 0 failed
```

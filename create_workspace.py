#!/usr/bin/env python3
"""One-click workspace creation for OpenCode autonomous quant research.

Usage:
    python create_workspace.py                    # creates ws_001/
    python create_workspace.py --name my_research # creates ws_my_research/
    python create_workspace.py --download-data    # also downloads market data

After creation, start OpenCode in the workspace:
    opencode run --dir ws_001 "Read GUIDE.md, then start autonomous research"
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def find_next_id() -> int:
    existing = sorted(ROOT.glob("ws_*"))
    max_id = 0
    for d in existing:
        name = d.name
        if name.startswith("ws_") and name[3:].isdigit():
            max_id = max(max_id, int(name[3:]))
    return max_id + 1


def create_workspace(name: str = "", download_data: bool = False) -> Path:
    if not name:
        ws_id = find_next_id()
        name = f"ws_{ws_id:03d}"

    ws = ROOT / name
    if ws.exists():
        print(f"ERROR: {ws} already exists")
        sys.exit(1)

    print(f"Creating workspace: {ws}")

    # Directory structure
    for d in ["strategies", "models", "configs", "results", "data"]:
        (ws / d).mkdir(parents=True)

    # Copy core tools from workspace/
    tools = [
        "backtest_api.py", "evaluator.py", "tracker.py", "orchestrator.py",
        "model_loader.py", "auto_improver.py", "lookahead_checker.py",
        "walk_forward.py", "cost_model.py", "universe_selector.py",
        "feature_selector.py", "ensemble.py", "risk_manager.py",
        "paper_trader.py", "pairs_engine.py", "basket_engine.py",
        "deep_validate.py", "objectives.json",
    ]
    src_ws = ROOT / "workspace"
    for tool in tools:
        src = src_ws / tool
        if src.exists():
            shutil.copy2(src, ws / tool)

    # __init__.py
    (ws / "__init__.py").write_text(f"# Workspace {name}\n")

    # Symlink data (shared across workspaces)
    data_src = ROOT / "user_data" / "data"
    if data_src.exists():
        data_link = ws / "data"
        if data_link.exists():
            shutil.rmtree(data_link)
        data_link.symlink_to(data_src.resolve())

    # Copy freqtrade configs
    for cfg in (src_ws / "configs").glob("*.json"):
        shutil.copy2(cfg, ws / "configs" / cfg.name)
    # Also copy from main configs
    for cfg in (ROOT / "user_data").glob("config_freqai*.json"):
        shutil.copy2(cfg, ws / "configs" / cfg.name)

    # Write GUIDE.md
    _write_guide(ws)

    # Write example strategy
    _write_example_strategy(ws)

    # Write run_agent.sh
    _write_run_agent(ws)

    # Metadata
    meta = {
        "workspace": name,
        "created": datetime.now().isoformat(),
        "version": "1.0",
        "tools": tools,
        "data_path": str(data_src.resolve()),
    }
    (ws / "meta.json").write_text(json.dumps(meta, indent=2))

    # Download data if requested
    if download_data:
        print("Downloading market data...")
        _download_data(ws)

    print(f"\nWorkspace ready: {ws}")
    print(f"Start OpenCode agent:")
    print(f"  cd {ws}")
    print(f"  opencode run 'Read GUIDE.md and start autonomous quant research'")
    return ws


def _download_data(ws: Path):
    """Download data using the download script."""
    import subprocess
    download_script = ROOT / "workspace" / "download_data.py"
    if download_script.exists():
        subprocess.run(
            [sys.executable, str(download_script), "--exchange", "gate", "--days", "400",
             "--outdir", str(ROOT / "user_data" / "data")],
            cwd=str(ROOT), timeout=600,
        )


def _write_guide(ws: Path):
    guide = f"""# Workspace Research Guide

## 你是什么

你是一个自主量化研究 agent。你在 `{ws.name}/` 工作，目标是找到能盈利的交易策略。

## 快速开始

```python
import sys; sys.path.insert(0, ".."); sys.path.insert(0, "../src")

# 1. 扫描配对（已证明最有效的方法）
from {ws.name}.pairs_engine import PairsEngine, scan_pairs
pairs = scan_pairs(exchange="gate", min_correlation=0.8)

# 2. 回测最佳配对
pe = PairsEngine("LINK/USDT", "SOL/USDT", exchange="gate")
signals = pe.generate_signals(lookback=80, entry_z=2.0, exit_z=0.5)
bt = pe.backtest(signals, maker_fee_bps=1.0)
print(f"Profit={{bt.profit_pct:+.2f}}%, Sharpe={{bt.sharpe:.2f}}")

# 3. Walk-Forward 验证
from {ws.name}.walk_forward import WalkForwardValidator
wf = WalkForwardValidator(train_bars=2000, test_bars=500, step_bars=500)
report = wf.validate("strategies/my_strategy.py", exchange="gate", pair="BTC/USDT")
print(report.summary())
```

## 可用工具

### 策略回测
```python
from {ws.name}.backtest_api import run_backtest
result = run_backtest("strategies/xxx.py", timerange="20260107-20260125")
# result = {{"ok": True, "sharpe": ..., "profit_pct": ..., "trades": ...}}
```

### 配对交易（推荐！已验证盈利）
```python
from {ws.name}.pairs_engine import PairsEngine, scan_pairs

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
    print(f"Window {{i+1}}: {{bt.profit_pct:+.2f}}%")
```

### 多目标评估
```python
from {ws.name}.evaluator import evaluate
score = evaluate(result)  # {{"total_score": 78, "grade": "B", "suggestions": [...]}}
```

### Walk-Forward 验证（必须通过才算有效）
```python
from {ws.name}.walk_forward import WalkForwardValidator
wf = WalkForwardValidator(train_bars=2000, test_bars=500, step_bars=500)
report = wf.validate("strategies/my.py", exchange="gate", pair="BTC/USDT")
# report.passed = True/False
# report.mean_sharpe, report.pct_profitable_windows
```

### 前瞻检查（回测前必查）
```python
from {ws.name}.lookahead_checker import check_lookahead, fix_lookahead_issues
report = check_lookahead("strategies/my.py")
if not report.ok:
    fix_lookahead_issues("strategies/my.py")  # 自动修复 bfill 等
```

### 交易成本
```python
from {ws.name}.cost_model import CostModel
cm = CostModel(exchange="gate")
est = cm.estimate_total_cost(trade_size_usd=500)
print(f"Maker 往返: {{est.round_trip_bps:.0f}} bps")  # ~2-5 bps
# Taker 往返约 28 bps — 尽量用 maker 挂单
```

### 风控
```python
from {ws.name}.risk_manager import RiskManager
rm = RiskManager(max_drawdown_pct=5.0)
decision = rm.check_trade(signal_strength=0.8, win_rate=0.55, ...)
```

### 实验追踪
```python
from {ws.name}.tracker import record_experiment, query_best, compare
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
    minimal_roi = {{"0": 0.08}}
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
"""
    (ws / "GUIDE.md").write_text(guide, encoding="utf-8")


def _write_example_strategy(ws: Path):
    code = '''"""Example: RSI+BB Mean Reversion — copy and modify."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from pandas import DataFrame
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(_ROOT).parent / "src"))
sys.path.insert(0, str(Path(_ROOT).parent))
from freqtrade.strategy import IStrategy

class ExampleRSIBB(IStrategy):
    timeframe = "1h"
    minimal_roi = {"0": 0.08, "120": 0.03}
    stoploss = -0.04
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    use_exit_signal = True
    startup_candle_count = 30
    can_short = False

    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        df["rsi"] = 100 - (100 / (1 + gain / (loss + 1e-10)))
        sma = df["close"].rolling(20).mean()
        std = df["close"].rolling(20).std()
        df["bb_lower"] = sma - 2.0 * std
        df["bb_middle"] = sma
        return df

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[(df["close"] < df["bb_lower"]) & (df["rsi"] < 35), ["enter_long", "enter_tag"]] = (1, "oversold")
        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[(df["close"] > df["bb_middle"]) | (df["rsi"] > 65), "exit_long"] = 1
        return df
'''
    (ws / "strategies" / "example_rsi_bb.py").write_text(code, encoding="utf-8")


def _write_run_agent(ws: Path):
    script = f"""#!/bin/bash
# Start OpenCode agent in this workspace
cd "$(dirname "$0")"
opencode run -m custom/gpt-5.2 \\
  "Read GUIDE.md in this directory. You are an autonomous quant researcher. \\
Your goal: find profitable trading strategies using the tools documented in GUIDE.md. \\
Start by scanning cointegrated pairs, then backtest, then walk-forward validate. \\
Record all experiments. Iterate until you find a strategy with positive mean profit \\
across walk-forward windows."
"""
    run_path = ws / "run_agent.sh"
    run_path.write_text(script, encoding="utf-8")
    run_path.chmod(0o755)


def main():
    parser = argparse.ArgumentParser(description="Create a new quant research workspace")
    parser.add_argument("--name", default="", help="Workspace name (default: auto-increment ws_NNN)")
    parser.add_argument("--download-data", action="store_true", help="Download market data")
    args = parser.parse_args()
    create_workspace(name=args.name, download_data=args.download_data)


if __name__ == "__main__":
    main()

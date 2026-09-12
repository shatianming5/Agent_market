#!/usr/bin/env python3
"""Integration test — verifies the complete workspace pipeline end-to-end.

Run: PYTHONPATH=. python3 workspace/integration_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    status = "PASS" if condition else "FAIL"
    if not condition:
        FAIL += 1
    else:
        PASS += 1
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    return condition


def test_imports():
    print("\n=== 1. Module Imports ===")
    modules = [
        "workspace.backtest_api",
        "workspace.evaluator",
        "workspace.tracker",
        "workspace.orchestrator",
        "workspace.model_loader",
        "workspace.auto_improver",
        "workspace.lookahead_checker",
        "workspace.walk_forward",
        "workspace.cost_model",
        "workspace.universe_selector",
        "workspace.feature_selector",
        "workspace.ensemble",
        "workspace.risk_manager",
        "workspace.paper_trader",
    ]
    for mod_name in modules:
        try:
            __import__(mod_name)
            check(mod_name, True)
        except Exception as e:
            check(mod_name, False, str(e)[:80])


def test_data():
    print("\n=== 2. Data Availability ===")
    import pandas as pd
    for exchange in ["kucoin", "gate"]:
        data_dir = ROOT / "user_data" / "data" / exchange
        if not data_dir.exists():
            check(f"{exchange} data dir", False, "not found")
            continue
        feathers = list(data_dir.glob("*.feather"))
        check(f"{exchange} data dir", len(feathers) > 0, f"{len(feathers)} files")
        if feathers:
            df = pd.read_feather(feathers[0])
            check(f"{exchange} data readable", len(df) > 100, f"{len(df)} rows")


def test_cost_model():
    print("\n=== 3. Cost Model ===")
    from workspace.cost_model import CostModel
    model = CostModel(exchange="gate")
    est = model.estimate_total_cost(trade_size_usd=500, daily_volume_usd=1e8)
    check("cost estimate", est.total_bps > 0, f"{est.total_bps:.1f} bps one-way")
    check("round trip", est.round_trip_bps > est.total_bps, f"{est.round_trip_bps:.1f} bps RT")


def test_universe():
    print("\n=== 4. Universe Selector ===")
    from workspace.universe_selector import select_universe
    pairs = select_universe(min_bars=100, min_daily_volume_usd=0)
    check("universe selection", len(pairs) > 0, f"{len(pairs)} pairs")


def test_feature_selector():
    print("\n=== 5. Feature Selector ===")
    from workspace.feature_selector import select_features
    import pandas as pd, numpy as np
    np.random.seed(42)
    df = pd.DataFrame({
        "f1": np.random.randn(200),
        "f2": np.random.randn(200),
        "f3": np.random.randn(200),
        "correlated_f1": np.random.randn(200) * 0.01,  # near constant
        "target": np.random.randn(200),
    })
    df["correlated_f1"] = df["f1"] * 0.99 + np.random.randn(200) * 0.01
    result = select_features(df, "target", max_features=3, exclude_cols=[])
    check("features selected", len(result["selected"]) > 0, f"{len(result['selected'])} features")
    check("correlated removed", len(result.get("removed_correlated", [])) > 0, f"removed {len(result.get('removed_correlated', []))}")


def test_regime():
    print("\n=== 6. Regime Detection ===")
    import pandas as pd
    from workspace.ensemble import RegimeDetector
    df = pd.read_feather(ROOT / "user_data" / "data" / "kucoin" / "BTC_USDT-1h.feather")
    df["date"] = pd.to_datetime(df["date"])
    rd = RegimeDetector()
    state = rd.current_regime(df)
    check("regime detected", state.regime in ("trending_up", "trending_down", "ranging", "volatile"), state.regime)


def test_risk_manager():
    print("\n=== 7. Risk Manager ===")
    from workspace.risk_manager import RiskManager
    rm = RiskManager(initial_equity=1000, max_drawdown_pct=5.0)

    d = rm.check_trade(signal_strength=0.8, win_rate=0.55, avg_win_pct=1.5, avg_loss_pct=1.0)
    check("normal trade allowed", d.allowed, f"size={d.position_size_pct}%")

    rm.current_equity = 940
    d2 = rm.check_trade(signal_strength=1.0, win_rate=0.6, avg_win_pct=2.0, avg_loss_pct=1.0)
    check("circuit breaker fires", not d2.allowed, "6% DD blocked")


def test_paper_trader():
    print("\n=== 8. Paper Trader ===")
    from workspace.paper_trader import PaperTrader
    pt = PaperTrader(initial_equity=1000, pairs=["BTC/USDT"])
    pt.update_prices({"BTC/USDT": 84000})
    pt.submit_order("BTC/USDT", "buy", size_usd=100)
    pt.update_prices({"BTC/USDT": 85000})
    check("paper equity tracks", pt.total_equity() > 1000, f"equity={pt.total_equity():.2f}")
    pt.submit_order("BTC/USDT", "sell", size_usd=100, price=85000)
    check("paper pnl positive", pt.total_pnl() > 0, f"pnl={pt.total_pnl():.2f}")


def test_lookahead():
    print("\n=== 9. Lookahead Checker ===")
    from workspace.lookahead_checker import check_lookahead
    import tempfile
    # Write a file with bfill (lookahead)
    bad_code = "import pandas as pd\ndf = pd.DataFrame()\ndf.bfill()\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(bad_code)
        f.flush()
        report = check_lookahead(f.name)
    check("detects bfill", report.critical_count > 0, f"{report.critical_count} critical")
    Path(f.name).unlink()


def test_walk_forward():
    print("\n=== 10. Walk-Forward Validator ===")
    from workspace.walk_forward import WalkForwardValidator
    wf = WalkForwardValidator(train_bars=400, test_bars=150, step_bars=150, min_windows=2)
    report = wf.validate(
        ROOT / "workspace" / "strategies" / "rsi_bb_mean_reversion.py",
        "RSIBBMeanReversion",
        exchange="kucoin", pair="BTC/USDT",
    )
    check("walk-forward runs", report.n_successful > 0, f"{report.n_successful}/{report.n_windows} windows")
    check("has verdict", len(report.verdict) > 0, report.verdict[:60])


def test_backtest_api():
    print("\n=== 11. Backtest API ===")
    from workspace.backtest_api import run_backtest
    r = run_backtest(
        ROOT / "workspace" / "strategies" / "strategy_template.py",
        "StrategyTemplate",
        timerange="20260101-20260125",
    )
    check("backtest returns result", isinstance(r, dict), f"ok={r.get('ok')}")
    if r.get("ok"):
        check("has sharpe", "sharpe" in r, f"sharpe={r.get('sharpe')}")


def test_evaluator():
    print("\n=== 12. Multi-Objective Evaluator ===")
    from workspace.evaluator import evaluate
    good = {"ok": True, "sharpe": 1.5, "sortino": 2.0, "max_drawdown_pct": 3.0,
            "profit_pct": 5.0, "win_rate": 0.55, "profit_factor": 1.5, "trades": 100}
    score = evaluate(good)
    check("good strategy scores high", score["total_score"] > 60, f"score={score['total_score']}")
    bad = {"ok": True, "sharpe": -5, "sortino": -5, "max_drawdown_pct": 20,
           "profit_pct": -10, "win_rate": 0.2, "profit_factor": 0.3, "trades": 5}
    score2 = evaluate(bad)
    check("bad strategy scores low", score2["total_score"] < 30, f"score={score2['total_score']}")
    check("suggestions generated", len(score2.get("suggestions", [])) > 0)


def test_tracker():
    print("\n=== 13. Experiment Tracker ===")
    from workspace.tracker import list_experiments, query_best
    exps = list_experiments()
    check("has experiments", len(exps) > 0, f"{len(exps)} total")
    best = query_best("sharpe", 3)
    check("can query best", len(best) > 0, f"best sharpe={best[0]['metrics']['sharpe']:.2f}" if best else "")


def test_model_loader():
    print("\n=== 14. Dynamic Model Loader ===")
    from workspace.model_loader import list_available_models
    models = list_available_models()
    check("built-in models", "lightgbm" in models, f"{len(models)} total")
    check("workspace model", "ridge_regression" in models, "ridge loaded")


def main():
    print("=" * 60)
    print("WORKSPACE INTEGRATION TEST")
    print("=" * 60)

    test_imports()
    test_data()
    test_cost_model()
    test_universe()
    test_feature_selector()
    test_regime()
    test_risk_manager()
    test_paper_trader()
    test_lookahead()
    test_walk_forward()
    test_backtest_api()
    test_evaluator()
    test_tracker()
    test_model_loader()

    print(f"\n{'='*60}")
    print(f"RESULTS: {PASS} passed, {FAIL} failed, {PASS+FAIL} total")
    print(f"{'='*60}")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

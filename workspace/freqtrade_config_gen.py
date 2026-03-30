"""Freqtrade Config Generator — creates proper configs for any strategy type.

OpenCode or auto_improver can call this to generate working freqtrade configs.

Usage:
    from workspace.freqtrade_config_gen import generate_config
    path = generate_config(
        strategy_type="ml",
        pairs=["BTC/USDT", "ETH/USDT"],
        exchange="gate",
    )
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]


def generate_config(
    strategy_type: str,
    *,
    pairs: Optional[List[str]] = None,
    exchange: str = "gate",
    timeframe: str = "1h",
    stake_amount: float = 100,
    wallet: float = 1000,
    fee: float = 0.001,
    max_open_trades: int = 2,
    output_path: Optional[str | Path] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Generate a freqtrade config JSON for the given strategy type."""
    if pairs is None:
        pairs = _default_pairs(strategy_type, exchange)

    exchange_urls = {
        "gate": {"public": "https://api.gateio.ws", "private": "https://api.gateio.ws"},
        "kucoin": {"public": "https://openapi-v2.kucoin.com", "private": "https://openapi-v2.kucoin.com"},
    }

    config = {
        "$schema": "https://schema.freqtrade.io/schema.json",
        "trading_mode": "spot",
        "dry_run": True,
        "max_open_trades": max_open_trades,
        "stake_currency": "USDT",
        "stake_amount": stake_amount,
        "tradable_balance_ratio": 1,
        "timeframe": timeframe,
        "dry_run_wallet": wallet,
        "fee": fee,
        "cancel_open_orders_on_exit": True,
        "unfilledtimeout": {"entry": 15, "exit": 15},
        "exchange": {
            "name": exchange,
            "key": "", "secret": "",
            "pair_whitelist": pairs,
            "pair_blacklist": [],
            "ccxt_config": {
                "timeout": 60000,
                **({"urls": {"api": exchange_urls[exchange]}} if exchange in exchange_urls else {}),
            },
        },
        "pairlists": [{"method": "StaticPairList", "allow_inactive": True}],
        "freqai": {
            "enabled": False,
            "purge_old_models": 2,
            "train_period_days": 45,
            "backtest_period_days": 15,
            "live_retrain_hours": 0,
            "identifier": f"{strategy_type}-backtest",
            "feature_parameters": {
                "include_timeframes": [timeframe],
                "include_corr_pairlist": pairs[:2],
                "label_period_candles": 12,
                "include_shifted_candles": 2,
                "DI_threshold": 0.9,
                "weight_factor": 0.85,
                "principal_component_analysis": False,
                "use_SVM_to_remove_outliers": False,
                "indicator_periods_candles": [12, 24, 48],
                "plot_feature_importances": 0,
            },
            "data_split_parameters": {"test_size": 0.3, "random_state": 7},
            "model_training_parameters": {},
        },
        "datadir": "user_data/data",
        "dataformat_ohlcv": "feather",
        "entry_pricing": {"price_side": "other", "use_order_book": False, "order_book_top": 1},
        "exit_pricing": {"price_side": "other", "use_order_book": False, "order_book_top": 1},
    }

    if extra:
        config.update(extra)

    # Output path
    if output_path is None:
        configs_dir = ROOT / "workspace" / "configs"
        configs_dir.mkdir(parents=True, exist_ok=True)
        output_path = configs_dir / f"ft_{strategy_type}_{exchange}.json"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def _default_pairs(strategy_type: str, exchange: str) -> List[str]:
    """Default pairs for each strategy type."""
    defaults = {
        "ml": ["BTC/USDT", "ETH/USDT"],
        "dl": ["BTC/USDT", "ETH/USDT"],
        "rl": ["BTC/USDT"],
        "pairs_ada_avax": ["ADA/USDT"],
        "pairs_doge_sol": ["DOGE/USDT"],
        "pairs_eth_link": ["ETH/USDT"],
        "trend": ["BTC/USDT", "ETH/USDT"],
        "meanrev": ["BTC/USDT", "ETH/USDT"],
    }
    return defaults.get(strategy_type, ["BTC/USDT", "ETH/USDT"])


def run_freqtrade_backtest(
    strategy_name: str,
    strategy_path: str | Path,
    config_path: str | Path,
    timerange: str = "20250601-20260301",
) -> Dict[str, Any]:
    """Run a freqtrade backtest and return parsed results."""
    import subprocess, sys

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "freqtrade_cli.py"),
        "backtesting",
        "--userdir", str(ROOT / "user_data"),
        "--config", str(config_path),
        "--strategy", strategy_name,
        "--strategy-path", str(Path(strategy_path).parent),
        "--timerange", timerange,
        "--cache", "none",
    ]

    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=300)

    if proc.returncode != 0:
        return {"ok": False, "error": proc.stderr[-500:] if proc.stderr else f"exit {proc.returncode}"}

    # Parse strategy summary from stdout
    output = proc.stdout
    result = {"ok": True, "raw_output": output[-2000:]}

    # Extract key metrics from freqtrade table output
    import re
    for pattern, key in [
        (r"Tot Profit USDT.*?(-?[\d.]+)", "profit_abs"),
        (r"Tot Profit %.*?(-?[\d.]+)", "profit_pct"),
        (r"Trades.*?(\d+)", "trades"),
        (r"Win%.*?([\d.]+)", "win_rate_raw"),
        (r"Drawdown.*?([\d.]+)\s*USDT", "drawdown_abs"),
    ]:
        m = re.search(pattern, output)
        if m:
            result[key] = float(m.group(1))

    return result


__all__ = ["generate_config", "run_freqtrade_backtest"]

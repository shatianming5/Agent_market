"""Realistic Trading Cost Model.

Models: taker/maker fees, spread, slippage (volume-dependent), funding rate.

Usage:
    from workspace.cost_model import CostModel, generate_freqtrade_config
    model = CostModel(exchange="gate")
    total_bps = model.estimate_total_cost(trade_size_usd=500, daily_volume_usd=1e8, volatility=0.004)
    config = generate_freqtrade_config(model, pairs=["BTC/USDT"], exchange="gate")
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


# Exchange fee schedules (taker/maker in bps)
_FEE_SCHEDULES = {
    "gate":    {"taker_bps": 10.0, "maker_bps": 10.0},
    "binance": {"taker_bps": 10.0, "maker_bps": 10.0},
    "okx":     {"taker_bps": 8.0,  "maker_bps": 6.0},
    "bybit":   {"taker_bps": 10.0, "maker_bps": 10.0},
    "kucoin":  {"taker_bps": 10.0, "maker_bps": 10.0},
    "mexc":    {"taker_bps": 10.0, "maker_bps": 0.0},
}


@dataclass
class CostEstimate:
    """Breakdown of trading costs for a single trade."""
    fee_bps: float           # Exchange fee
    spread_bps: float        # Bid-ask spread cost
    slippage_bps: float      # Market impact / slippage
    total_bps: float         # Total one-way cost
    round_trip_bps: float    # Total cost for entry + exit


class CostModel:
    """Realistic trading cost estimator."""

    def __init__(
        self,
        exchange: str = "gate",
        *,
        taker_bps: Optional[float] = None,
        maker_bps: Optional[float] = None,
        base_spread_bps: float = 2.0,
        slippage_coefficient: float = 0.5,
    ):
        schedule = _FEE_SCHEDULES.get(exchange, _FEE_SCHEDULES["gate"])
        self.exchange = exchange
        self.taker_bps = taker_bps if taker_bps is not None else schedule["taker_bps"]
        self.maker_bps = maker_bps if maker_bps is not None else schedule["maker_bps"]
        self.base_spread_bps = base_spread_bps
        self.slippage_coefficient = slippage_coefficient

    def estimate_spread(self, volatility: float = 0.004) -> float:
        """Estimate bid-ask spread in bps. Higher volatility = wider spread."""
        # Spread scales with volatility: spread ≈ base + k * vol
        return self.base_spread_bps + 500.0 * volatility

    def estimate_slippage(
        self,
        trade_size_usd: float = 500.0,
        daily_volume_usd: float = 1e8,
    ) -> float:
        """Estimate slippage in bps using square-root market impact model.

        Slippage ≈ coefficient * sqrt(trade_size / daily_volume) * 10000
        This is the standard Almgren-Chriss square-root model.
        """
        if daily_volume_usd <= 0:
            return 50.0  # very illiquid
        participation = trade_size_usd / daily_volume_usd
        return self.slippage_coefficient * (participation ** 0.5) * 10000.0

    def estimate_total_cost(
        self,
        trade_size_usd: float = 500.0,
        daily_volume_usd: float = 1e8,
        volatility: float = 0.004,
        is_taker: bool = True,
    ) -> CostEstimate:
        """Estimate total one-way trading cost."""
        fee = self.taker_bps if is_taker else self.maker_bps
        spread = self.estimate_spread(volatility)
        slippage = self.estimate_slippage(trade_size_usd, daily_volume_usd)

        # One-way: pay half the spread + full slippage + full fee
        one_way = fee + spread / 2.0 + slippage
        return CostEstimate(
            fee_bps=fee,
            spread_bps=spread,
            slippage_bps=slippage,
            total_bps=one_way,
            round_trip_bps=one_way * 2.0,
        )


def generate_freqtrade_config(
    cost_model: CostModel,
    *,
    pairs: List[str],
    exchange: str = "gate",
    timeframe: str = "1h",
    stake_amount: float = 75,
    wallet: float = 1000,
    max_open_trades: int = 4,
    output_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Generate a freqtrade config with realistic fee settings."""
    # Use taker fee as the conservative estimate
    fee_rate = cost_model.taker_bps / 10000.0  # bps to decimal

    # Exchange-specific API URLs
    exchange_urls = {
        "gate": {"public": "https://api.gateio.ws", "private": "https://api.gateio.ws"},
        "kucoin": {
            "public": "https://openapi-v2.kucoin.com",
            "private": "https://openapi-v2.kucoin.com",
            "earn": "https://openapi-v2.kucoin.com",
            "uta": "https://openapi-v2.kucoin.com",
        },
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
        "fee": fee_rate,
        "cancel_open_orders_on_exit": True,
        "unfilledtimeout": {"entry": 15, "exit": 15},
        "exchange": {
            "name": exchange,
            "key": "",
            "secret": "",
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
            "identifier": "agent-market",
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
        "_cost_model": {
            "exchange": cost_model.exchange,
            "taker_bps": cost_model.taker_bps,
            "maker_bps": cost_model.maker_bps,
            "base_spread_bps": cost_model.base_spread_bps,
            "slippage_coefficient": cost_model.slippage_coefficient,
            "fee_rate_used": fee_rate,
        },
    }

    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    return config


__all__ = ["CostModel", "CostEstimate", "generate_freqtrade_config"]

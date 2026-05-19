from __future__ import annotations

import json

from agent_market.strategy_miner._helpers import (
    _freqtrade_market_context,
    _freqtrade_config_defaults,
    _get_leverage_factor,
)


def test_freqtrade_config_defaults_use_shared_loader(tmp_path) -> None:
    cfg = tmp_path / "freqtrade.json"
    cfg.write_text(
        json.dumps({"timeframe": "4h", "trading_mode": "futures"}),
        encoding="utf-8",
    )

    assert _freqtrade_config_defaults(str(cfg)) == ("4h", False)


def test_freqtrade_config_defaults_fallback_for_bad_json(tmp_path) -> None:
    cfg = tmp_path / "freqtrade.json"
    cfg.write_text("{bad json", encoding="utf-8")

    assert _freqtrade_config_defaults(str(cfg)) == ("1h", True)
    assert _get_leverage_factor(str(cfg)) == 1.0


def test_get_leverage_factor_reads_futures_default(tmp_path) -> None:
    cfg = tmp_path / "freqtrade.json"
    cfg.write_text(
        json.dumps({"trading_mode": "futures", "leverage": {"default": 3.0}}),
        encoding="utf-8",
    )

    assert _get_leverage_factor(str(cfg)) == 3.0


def test_freqtrade_market_context_exposes_named_fields_and_tuple_order(tmp_path) -> None:
    cfg = tmp_path / "freqtrade.json"
    cfg.write_text(
        json.dumps(
            {
                "timeframe": "15m",
                "datadir": "user_data/data/binance",
                "exchange": {
                    "name": "binance",
                    "pair_whitelist": ["BTC/USDT", "ETH/USDT"],
                },
            }
        ),
        encoding="utf-8",
    )

    context = _freqtrade_market_context(str(cfg))

    assert context.exchange == "binance"
    assert context.pairs == ["BTC/USDT", "ETH/USDT"]
    assert context.timeframe == "15m"
    assert context.datadir == "user_data/data/binance"
    assert tuple(context) == (
        "binance",
        ["BTC/USDT", "ETH/USDT"],
        "15m",
        "user_data/data/binance",
    )

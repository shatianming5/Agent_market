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


def test_build_market_profile_uses_shared_freqtrade_loader(tmp_path) -> None:
    from agent_market.strategy_miner._generation import _build_market_profile

    cfg = tmp_path / "freqtrade.json"
    cfg.write_text(
        json.dumps(
            {
                "timeframe": "1h",
                "stake_currency": "USDT",
                "trading_mode": "futures",
                "dry_run_wallet": 1000,
                "exchange": {"pair_whitelist": ["BTC/USDT"]},
            }
        ),
        encoding="utf-8",
    )

    profile = _build_market_profile(str(cfg))

    assert profile is not None
    assert "BTC/USDT" in profile
    assert "Stake currency: USDT" in profile
    assert "Trading mode: futures" in profile


def test_build_market_profile_returns_none_for_bad_json(tmp_path) -> None:
    from agent_market.strategy_miner._generation import _build_market_profile

    cfg = tmp_path / "freqtrade.json"
    cfg.write_text("{bad json", encoding="utf-8")

    assert _build_market_profile(str(cfg)) is None

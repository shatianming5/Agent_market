from __future__ import annotations

import json

from agent_market.strategy_miner._helpers import (
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

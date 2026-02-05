from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def _ms(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)


def test_tca_cost_breakdown_and_participation_proxy(tmp_path: Path) -> None:
    from agent_market.tca.report import generate_tca_report

    ex_dir = tmp_path / "user_data" / "data" / "kucoin"
    ex_dir.mkdir(parents=True, exist_ok=True)

    ohlcv = pd.DataFrame(
        {
            "date": [datetime(2024, 1, 1, 0, 0, 0), datetime(2024, 1, 1, 1, 0, 0)],
            "open": [100.0, 100.0],
            "high": [101.0, 101.0],
            "low": [99.0, 99.0],
            "close": [100.0, 100.0],
            "volume": [10.0, 20.0],
        }
    )
    ohlcv.to_feather(ex_dir / "BTC_USDT-1h.feather")

    zip_path = tmp_path / "backtest-result.zip"
    payload = {
        "strategy": {
            "DemoStrategy": {
                "timeframe": "1h",
                "trades": [
                    {
                        "pair": "BTC/USDT",
                        "profit_abs": 1.0,
                        "profit_ratio": 0.01,
                        "fee_open": 0.1,
                        "fee_close": 0.1,
                        "funding_fees": 0.0,
                        "trade_duration": 60,
                        "orders": [
                            {
                                "amount": 1.0,
                                "safe_price": 100.0,
                                "ft_order_side": "buy",
                                "order_filled_timestamp": _ms(datetime(2024, 1, 1, 0, 30, 0)),
                            },
                            {
                                "amount": 1.0,
                                "safe_price": 101.0,
                                "ft_order_side": "sell",
                                "order_filled_timestamp": _ms(datetime(2024, 1, 1, 1, 15, 0)),
                            },
                        ],
                    }
                ],
            }
        },
        "strategy_comparison": [],
    }
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("backtest-result.json", json.dumps(payload))

    out_path = tmp_path / "tca_report.json"
    report = generate_tca_report(zip_path, run_id="abcd1234", out_path=out_path, root=tmp_path)
    assert out_path.exists()

    by_comp = report["costs"]["implementation_shortfall"]["by_component"]
    for key in ("spread", "delay", "market_impact"):
        assert isinstance(by_comp[key]["bps"], (int, float))

    participation = report["diagnostics"]["participation"]
    assert participation["method"] == "ohlcv_volume_proxy_v1"
    assert isinstance(participation["overall"], (int, float))
    expected = 2.0 / 30.0
    assert abs(float(participation["overall"]) - expected) < 1e-12
    assert abs(float(participation["per_symbol"]["BTC/USDT"]) - expected) < 1e-12


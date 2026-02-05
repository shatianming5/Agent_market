from __future__ import annotations

import json
import zipfile
from pathlib import Path


def test_tca_report_extracts_orders_and_fills_when_present(tmp_path: Path) -> None:
    from agent_market.tca.report import generate_tca_report

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
                                "order_filled_timestamp": 1700000000000,
                                "ft_is_entry": True,
                                "ft_order_tag": "entry",
                                "cost": 100.0,
                            },
                            {
                                "amount": 1.0,
                                "safe_price": 101.0,
                                "ft_order_side": "sell",
                                "order_filled_timestamp": 1700003600000,
                                "ft_is_entry": False,
                                "ft_order_tag": "exit",
                                "cost": 101.0,
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
    report = generate_tca_report(zip_path, run_id="abcd1234", out_path=out_path)
    assert out_path.exists()

    assert isinstance(report.get("orders"), list) and report["orders"]
    assert isinstance(report.get("fills"), list) and report["fills"]

    is_total = report["costs"]["implementation_shortfall"]["total"]
    fees_comp = report["costs"]["implementation_shortfall"]["by_component"]["fees"]
    assert isinstance(is_total.get("quote_ccy"), (int, float))
    assert isinstance(fees_comp.get("quote_ccy"), (int, float))


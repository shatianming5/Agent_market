from __future__ import annotations

import json
import zipfile
from pathlib import Path


def test_tca_report_from_minimal_backtest_zip(tmp_path: Path):
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
                    },
                    {
                        "pair": "BTC/USDT",
                        "profit_abs": -2.0,
                        "profit_ratio": -0.02,
                        "fee_open": 0.1,
                        "fee_close": 0.1,
                        "funding_fees": 0.0,
                        "trade_duration": 120,
                    },
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
    loaded = json.loads(out_path.read_text(encoding="utf-8"))
    assert loaded["schema_version"] == "1.0"
    assert loaded["meta"]["run_id"] == "abcd1234"
    assert loaded["meta"]["strategy"]["name"] == "DemoStrategy"
    assert loaded["summary"]["trades"] == 2
    assert loaded["summary"]["fees_total"] is not None
    assert loaded["costs"]["slippage_distribution"]["p50_bps"] is None
    assert loaded["costs"]["fill"]["fill_rate"] is None
    assert "profit_abs" in loaded["distributions"]
    assert report["source"]["strategy"] == "DemoStrategy"

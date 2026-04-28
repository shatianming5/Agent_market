from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_lob_rebuild_from_fixture(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    capture_script = root / "scripts" / "micro_capture.py"
    lob_script = root / "scripts" / "lob_rebuild.py"
    fixture = root / "tests" / "fixtures" / "kucoin_ws_sample.jsonl"
    snapshot = root / "tests" / "fixtures" / "kucoin_lob_snapshot.json"

    capture_dir = tmp_path / "capture"
    subprocess.run(  # noqa: S603,S607
        [
            sys.executable,
            str(capture_script),
            "--exchange",
            "kucoin",
            "--channels",
            "match,level2",
            "--fixture",
            str(fixture),
            "--out-dir",
            str(capture_dir),
        ],
        cwd=str(root),
        check=True,
    )

    out_dir = tmp_path / "lob"
    subprocess.run(  # noqa: S603,S607
        [
            sys.executable,
            str(lob_script),
            "--capture-dir",
            str(capture_dir),
            "--snapshot",
            str(snapshot),
            "--symbol",
            "BTC-USDT",
            "--out-dir",
            str(out_dir),
            "--depth",
            "20",
        ],
        cwd=str(root),
        check=True,
    )

    lob_path = out_dir / "lob_state.parquet"
    report_path = out_dir / "rebuild_report.json"
    preflight_path = out_dir / "preflight.json"
    assert lob_path.exists()
    assert report_path.exists()
    assert preflight_path.exists()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["exchange"] == "kucoin"
    assert report["symbol"] == "BTC-USDT"
    assert report["gaps"]["count"] >= 0

    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    assert preflight["kind"] == "lob_rebuild"
    assert preflight["ok"] is True

    import pandas as pd

    df = pd.read_parquet(lob_path)
    # Basic schema
    for col in [
        "ts",
        "symbol",
        "event",
        "bid1",
        "ask1",
        "mid",
        "spread",
        "rel_spread",
        "depth_bid",
        "depth_ask",
        "imbalance",
        "bid_px",
        "bid_sz",
        "ask_px",
        "ask_sz",
    ]:
        assert col in df.columns
    assert df.shape[0] >= 2  # snapshot + >=1 delta

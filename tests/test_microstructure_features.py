from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_microstructure_features_from_lob_and_match(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]

    capture_script = root / "scripts" / "micro_capture.py"
    lob_script = root / "scripts" / "lob_rebuild.py"
    micro_script = root / "scripts" / "micro_features.py"

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

    lob_dir = tmp_path / "lob"
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
            str(lob_dir),
            "--depth",
            "20",
        ],
        cwd=str(root),
        check=True,
    )

    out_dir = tmp_path / "micro"
    subprocess.run(  # noqa: S603,S607
        [
            sys.executable,
            str(micro_script),
            "--run-id",
            "run1",
            "--out-dir",
            str(out_dir),
            "--lob-state",
            str(lob_dir / "lob_state.parquet"),
            "--match",
            str(capture_dir / "match.ndjson.gz"),
            "--symbol",
            "BTC-USDT",
            "--depth-levels",
            "20",
            "--windows-sec",
            "10",
        ],
        cwd=str(root),
        check=True,
    )

    features_path = out_dir / "features.parquet"
    manifest_path = out_dir / "manifest.json"
    assert features_path.exists()
    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "lob_state_parquet" in (manifest.get("data_sources") or [])
    assert "match_ndjson_gz" in (manifest.get("data_sources") or [])

    import pandas as pd

    df = pd.read_parquet(features_path)
    expected = [
        "ts",
        "symbol",
        "event",
        "mid",
        "spread",
        "rel_spread",
        "microprice",
        "depth_bid_20",
        "depth_ask_20",
        "imbalance_20",
        "trade_sign",
        "vwap_10",
        "ofi_10",
        "arrival_intensity_10",
    ]
    for col in expected:
        assert col in df.columns
    assert df.shape[0] >= 2


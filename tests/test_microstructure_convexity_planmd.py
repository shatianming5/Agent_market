from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_microstructure_convexity_feature_exists(tmp_path: Path) -> None:
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

    import pandas as pd

    df = pd.read_parquet(out_dir / "features.parquet")
    assert "convexity_20" in df.columns


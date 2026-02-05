from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _build_microstructure_features(tmp_path: Path) -> "object":
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

    return pd.read_parquet(out_dir / "features.parquet")


def test_factor_compiler_microstructure_ops_compile_and_eval(tmp_path: Path) -> None:
    df = _build_microstructure_features(tmp_path)

    from agent_market.factor_compiler.dsl.operators import compile_to_expression_engine
    from agent_market.factor_compiler.dsl.parser import parse_formula
    from agent_market.freqai.expression_engine import safe_eval_expression

    cases = {
        "mid(bid1,ask1)": "mid",
        "spread(bid1,ask1)": "spread",
        "microprice(bid1,ask1,bid_sz1,ask_sz1)": "microprice",
        "depth_bid(20)": "depth_bid_20",
        "depth_ask(20)": "depth_ask_20",
        "imbalance(depth_bid(20),depth_ask(20))": "imbalance_20",
        "trade_sign()": "trade_sign",
        "vwap(10)": "vwap_10",
        "ofi(10)": "ofi_10",
        "arrival_intensity(10)": "arrival_intensity_10",
    }

    for formula, expected in cases.items():
        node = parse_formula(formula, allowed_calls=None)
        compiled = compile_to_expression_engine(node)
        assert compiled == expected
        series = safe_eval_expression(compiled, df)
        assert series.shape[0] == df.shape[0]


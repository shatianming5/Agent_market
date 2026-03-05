from __future__ import annotations

from pathlib import Path

import pandas as pd

def test_microstructure_planmd_modules_import_and_minimal_behavior(tmp_path: Path) -> None:
    from agent_market.microstructure.capture.ws_capture import capture_ws
    from agent_market.microstructure.lob.checksum import checksum_capture_session, sha256_file
    from agent_market.microstructure.schemas.lob_parquet import validate_lob_state_df
    from agent_market.microstructure.schemas.trades_parquet import validate_trades_df

    fixture = Path("tests/fixtures/kucoin_ws_sample.jsonl").resolve()
    session_dir = tmp_path / "cap"
    session = capture_ws(out_dir=session_dir, fixture=fixture, exchange="kucoin", channels=("match", "level2"))
    assert session.mode == "fixture"

    # fixture replay writes gzip ndjson files
    assert (session_dir / "match.ndjson.gz").exists()
    assert (session_dir / "level2.ndjson.gz").exists()

    checksums = checksum_capture_session(session_dir)
    assert isinstance(checksums.get("match_ndjson_gz"), str)
    assert isinstance(sha256_file(session_dir / "match.ndjson.gz"), str)

    lob_df = pd.DataFrame(
        {
            "ts": ["2020-01-01T00:00:00Z"],
            "symbol": ["BTC-USDT"],
            "event": ["snapshot"],
            "bid1": [1.0],
            "ask1": [1.1],
            "mid": [1.05],
            "spread": [0.1],
            "rel_spread": [0.1 / 1.05],
            "bid_sz": [[1.0, 2.0]],
            "ask_sz": [[1.0, 2.0]],
        }
    )
    assert validate_lob_state_df(lob_df)["ok"] is True

    trades_df = pd.DataFrame(
        {
            "ts": ["2020-01-01T00:00:00Z"],
            "symbol": ["BTC-USDT"],
            "side": ["buy"],
            "price": [1.0],
            "size": [1.0],
        }
    )
    assert validate_trades_df(trades_df)["ok"] is True




#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _lib import resolve_path  # noqa: E402


def _ensure_imports() -> None:
    return


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Rebuild KuCoin L2 orderbook from snapshot + level2 deltas.")
    parser.add_argument("--capture-dir", required=True, help="Capture session dir containing level2.ndjson.gz")
    parser.add_argument("--snapshot", required=True, help="Snapshot JSON (fixture) with bids/asks + sequence")
    parser.add_argument("--symbol", required=True, help="Symbol, e.g. BTC-USDT")
    parser.add_argument("--depth", type=int, default=20)
    parser.add_argument("--out-dir", required=True, help="Output directory for lob_state.parquet and report")
    args = parser.parse_args(argv)

    _ensure_imports()
    from agent_market.microstructure.lob.rebuild import rebuild_kucoin_lob  # noqa: WPS433

    out = rebuild_kucoin_lob(
        capture_dir=resolve_path(args.capture_dir),
        snapshot_path=resolve_path(args.snapshot),
        symbol=str(args.symbol),
        depth=int(args.depth),
        out_dir=resolve_path(args.out_dir),
    )
    print(f"[lob_rebuild] wrote: {out['lob_state']}")
    print(f"[lob_rebuild] report: {out['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

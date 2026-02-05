#!/usr/bin/env python
from __future__ import annotations

import argparse
import uuid
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (ROOT / p).resolve()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate micro features (OHLCV-based or microstructure from LOB+trades)."
    )
    parser.add_argument("--config", default=None, help="Freqtrade JSON config (OHLCV mode)")
    parser.add_argument("--run-id", default=None, help="Optional run_id (default: generated)")
    parser.add_argument("--out-dir", default=None, help="Output directory (default: artifacts/runs/<run_id>/micro_feature)")
    parser.add_argument("--timeframe", default=None, help="Override timeframe (e.g. 1h)")
    parser.add_argument("--timerange", default=None, help="Optional timerange YYYYMMDD-YYYYMMDD")
    parser.add_argument("--data-dir", default=None, help="Override datadir root (default: from config)")
    parser.add_argument("--pairs", nargs="*", default=None, help="Optional pair override (e.g. BTC/USDT ETH/USDT)")
    parser.add_argument("--lob-state", default=None, help="LOB state parquet (microstructure mode)")
    parser.add_argument("--match", default=None, help="match.ndjson.gz path (microstructure mode)")
    parser.add_argument("--symbol", default=None, help="Optional symbol filter (microstructure mode)")
    parser.add_argument("--depth-levels", type=int, default=20, help="LOB depth levels for depth_* features")
    parser.add_argument("--windows-sec", nargs="*", type=int, default=[10], help="Trade rollup windows in seconds")
    args = parser.parse_args(argv)

    from agent_market.microstructure.micro_feature import (  # noqa: WPS433
        generate_micro_features_from_freqtrade_config,
        generate_microstructure_features_from_lob_and_match,
    )

    run_id = str(args.run_id).strip().lower() if args.run_id else ""
    if not run_id:
        run_id = uuid.uuid4().hex[:12]

    out_dir = _resolve(args.out_dir) if args.out_dir else (ROOT / "artifacts" / "runs" / run_id / "micro_feature")

    if args.lob_state or args.match:
        if not args.lob_state or not args.match:
            raise SystemExit("--lob-state and --match are both required for microstructure mode")
        outputs = generate_microstructure_features_from_lob_and_match(
            root=ROOT,
            run_id=run_id,
            out_dir=out_dir,
            lob_state=_resolve(str(args.lob_state)),
            match_path=_resolve(str(args.match)),
            symbol=str(args.symbol) if args.symbol else None,
            depth_levels=int(args.depth_levels),
            windows_sec=[int(x) for x in (args.windows_sec or []) if int(x) > 0],
        )
    else:
        if not args.config:
            raise SystemExit("--config is required for OHLCV mode (or pass --lob-state/--match for microstructure mode)")
        outputs = generate_micro_features_from_freqtrade_config(
            root=ROOT,
            freqtrade_config=_resolve(args.config),
            run_id=run_id,
            out_dir=out_dir,
            timeframe=args.timeframe,
            timerange=args.timerange,
            pairs=args.pairs,
            data_dir=args.data_dir,
        )
    print(f"[micro_feature] wrote: {outputs.features_parquet}")
    print(f"[micro_feature] manifest: {outputs.manifest_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

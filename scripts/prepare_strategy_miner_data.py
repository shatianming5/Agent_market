#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from agent_market.strategy_miner.data_prep import (  # noqa: E402
    collect_data_requirements,
    download_required_data,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare OHLCV data for strategy miner configs.")
    parser.add_argument(
        "--miner-config",
        dest="miner_configs",
        action="append",
        required=True,
        help="Path to one strategy miner config JSON. Can be passed multiple times.",
    )
    parser.add_argument(
        "--proxy",
        default=None,
        help="Proxy URL, for example http://127.0.0.1:1097",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only print missing data and the resolved download command.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON output.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = []

    for item in args.miner_configs:
        req = collect_data_requirements(item)
        result = download_required_data(
            req,
            proxy=args.proxy,
            check_only=bool(args.check_only),
        )
        result["miner_config"] = str(req.miner_config_path)
        result["freqtrade_config"] = str(req.freqtrade_config_path)
        result["pairs"] = list(req.pairs)
        result["timeframes"] = list(req.timeframes)
        result["timerange"] = req.timerange
        result["trading_mode"] = req.trading_mode
        results.append(result)

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        for result in results:
            print(f"config: {result['miner_config']}")
            print(f"  freqtrade_config: {result['freqtrade_config']}")
            print(f"  trading_mode: {result['trading_mode']}")
            print(f"  timerange: {result['timerange'] or '(none)'}")
            print(f"  timeframes: {', '.join(result['timeframes'])}")
            print(f"  pairs: {', '.join(result['pairs'])}")
            print(f"  missing_before: {len(result['missing_before'])}")
            if result["missing_before"]:
                for path in result["missing_before"][:8]:
                    print(f"    - {path}")
                if len(result["missing_before"]) > 8:
                    print(f"    - ... ({len(result['missing_before']) - 8} more)")
            print(f"  command: {' '.join(result['command'])}")
            if result["executed"]:
                print(f"  returncode: {result['returncode']}")
                print(f"  missing_after: {len(result['missing_after'])}")
            print()

    return 0 if all(int(result["returncode"]) == 0 for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

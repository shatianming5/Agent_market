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

from agent_market import paths  # noqa: E402


def _resolve(path: str) -> Path:
    return paths.resolve_repo_path(path)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a TCA report from a freqtrade backtest result zip.")
    parser.add_argument("--backtest-zip", default=None, help="Path to backtest-result-*.zip (default: latest)")
    parser.add_argument("--results-dir", default="user_data/backtest_results", help="Dir to find latest zip")
    parser.add_argument("--run-id", required=True, help="Flow run_id to attach report to")
    parser.add_argument("--out", default=None, help="Output JSON path (default: artifacts/runs/<run_id>/tca/tca_report.json)")
    parser.add_argument("--html", action="store_true", help="Also write a minimal HTML report next to JSON")
    args = parser.parse_args(argv)

    from agent_market.backtest_results import find_latest_backtest_zip  # noqa: WPS433
    from agent_market.tca import generate_tca_report  # noqa: WPS433

    if args.backtest_zip:
        zip_path = _resolve(str(args.backtest_zip))
    else:
        results_dir = _resolve(str(args.results_dir))
        zip_path = find_latest_backtest_zip(results_dir)
        if zip_path is None:
            raise FileNotFoundError(f"No backtest-result-*.zip found in {results_dir}")

    if args.out:
        out_path = _resolve(str(args.out))
    else:
        out_path = (paths.run_dir(str(args.run_id)) / "tca" / "tca_report.json").resolve()

    html_path = out_path.with_suffix(".html") if args.html else None
    payload = generate_tca_report(zip_path, run_id=str(args.run_id), out_path=out_path, root=ROOT, html_path=html_path)
    print(f"[tca] wrote: {out_path}")
    print(f"[tca] strategy={payload.get('source', {}).get('strategy')} trades={payload.get('summary', {}).get('trades')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

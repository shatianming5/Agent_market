#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent_market.backtest_results import build_backtest_summary, find_latest_backtest_zip  # type: ignore


def main() -> None:
    parser = argparse.ArgumentParser(description='Summarize latest backtest to JSON')
    parser.add_argument('--results-dir', default='user_data/backtest_results/multi_4h')
    parser.add_argument('--out', default='user_data/reports/latest_summary.json')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    zip_path = find_latest_backtest_zip(results_dir)
    if not zip_path:
        raise FileNotFoundError(f'No backtest archives found in {results_dir}')

    summary = build_backtest_summary(zip_path)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Wrote summary to {out}')


if __name__ == '__main__':
    main()

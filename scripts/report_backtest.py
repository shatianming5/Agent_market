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

from agent_market.agent_flow import AgentFlow, AgentFlowConfig  # type: ignore


def find_latest_zip(results_dir: Path) -> Path | None:
    results_dir = results_dir.resolve()
    last = results_dir / '.last_result.json'
    if last.exists():
        try:
            latest = json.loads(last.read_text(encoding='utf-8')).get('latest_backtest')
            if latest:
                p = results_dir / latest
                if p.exists():
                    return p
        except Exception:
            pass
    zips = sorted(results_dir.glob('backtest-result-*.zip'), key=lambda p: p.stat().st_mtime)
    return zips[-1] if zips else None


def main() -> None:
    parser = argparse.ArgumentParser(description='Summarize latest backtest to JSON')
    parser.add_argument('--results-dir', default='user_data/backtest_results/multi_4h')
    parser.add_argument('--out', default='user_data/reports/latest_summary.json')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    zip_path = find_latest_zip(results_dir)
    if not zip_path:
        raise FileNotFoundError(f'No backtest archives found in {results_dir}')

    flow = AgentFlow(AgentFlowConfig())
    summary = flow._build_backtest_summary(zip_path)  # reuse helper
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Wrote summary to {out}')


if __name__ == '__main__':
    main()


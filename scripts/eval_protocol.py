#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT))

from agent_market import paths  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal evaluation protocol (evidence + cost accounting)")
    parser.add_argument("--run-meta", default="artifacts/run_meta.json", help="Path to run_meta.json")
    parser.add_argument("--run-id", default=None, help="Optional run_id override")
    parser.add_argument("--backtest-summary", default=None, help="Optional backtest summary JSON path override")
    parser.add_argument("--tca-report", default=None, help="Optional TCA report JSON path override")
    parser.add_argument("--out-dir", default=None, help="Output directory (default: artifacts/runs/<run_id>/eval_protocol)")
    args = parser.parse_args()

    from agent_market.freqai.training.eval_protocol import build_eval_report, write_eval_report

    meta_path = paths.resolve_repo_path(args.run_meta)
    run_meta = json.loads(meta_path.read_text(encoding="utf-8"))

    run_id = (str(args.run_id).strip().lower() if args.run_id else "") or str(run_meta.get("run_id") or "")
    if not run_id:
        raise SystemExit("run_id missing (provide --run-id or ensure run_meta has run_id)")

    artifacts = run_meta.get("artifacts") or {}
    bt_path_raw = args.backtest_summary or artifacts.get("feedback_summary") or "user_data/llm_feedback/latest_backtest_summary.json"
    bt_path = paths.resolve_repo_path(bt_path_raw)
    backtest_summary = json.loads(bt_path.read_text(encoding="utf-8"))

    tca_report = None
    tca_path_raw = args.tca_report or artifacts.get("tca_report")
    if tca_path_raw:
        tca_path = paths.resolve_repo_path(str(tca_path_raw))
        if tca_path.exists():
            tca_report = json.loads(tca_path.read_text(encoding="utf-8"))

    out_dir = (
        paths.resolve_repo_path(args.out_dir)
        if args.out_dir
        else (paths.run_dir(run_id) / "eval_protocol").resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = build_eval_report(run_meta=run_meta, backtest_summary=backtest_summary, tca_report=tca_report)
    out_path = write_eval_report(out_dir / "eval_report.json", payload)
    print(f"[eval_protocol] wrote: {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _file_check(path: Path) -> dict[str, Any]:
    item: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
    }
    if path.exists() and path.is_file():
        stat = path.stat()
        item["size_bytes"] = int(stat.st_size)
        item["nonempty"] = stat.st_size > 0
    return item


def _candidate_entry(summary_path: Path) -> dict[str, Any]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    candidate = payload.get("candidate") or {}
    training_summary = payload.get("training_summary") or {}
    entry: dict[str, Any] = {
        "candidate": candidate.get("name"),
        "iteration": candidate.get("iteration"),
        "candidate_type": candidate.get("candidate_type"),
        "model_family": candidate.get("model_family"),
        "backtest_summary_path": str(summary_path),
        "training_summary_present": bool(training_summary),
        "train_size": training_summary.get("train_size"),
        "valid_size": training_summary.get("valid_size"),
        "timesteps": training_summary.get("timesteps"),
        "metrics": training_summary.get("metrics"),
        "rolling": training_summary.get("rolling"),
    }
    model_path_raw = training_summary.get("model_path")
    if model_path_raw:
        entry["model_file"] = _file_check(Path(str(model_path_raw)))
    feature_snapshot = training_summary.get("feature_snapshot")
    if feature_snapshot:
        entry["feature_snapshot"] = _file_check(Path(str(feature_snapshot)))
    expressions_snapshot = training_summary.get("expressions_snapshot")
    if expressions_snapshot:
        entry["expressions_snapshot"] = _file_check(Path(str(expressions_snapshot)))

    signals_dir = summary_path.parents[3] / "rl_signals" / f"iter_{int(candidate.get('iteration') or 0):04d}" / str(candidate.get("name") or "")
    if signals_dir.exists():
        files = sorted(signals_dir.rglob("*.feather"))
        entry["signals"] = {
            "dir": str(signals_dir),
            "count": len(files),
            "files": [_file_check(path) for path in files[:20]],
        }

    evidence_path = summary_path.parents[3] / "training" / f"iter_{int(candidate.get('iteration') or 0):04d}" / str(candidate.get("name") or "") / "training_evidence.json"
    if evidence_path.exists():
        entry["training_evidence"] = _file_check(evidence_path)

    model_ok = bool(entry.get("model_file", {}).get("exists")) and bool(entry.get("model_file", {}).get("nonempty"))
    signals_ok = True
    if str(candidate.get("candidate_type")) == "rl":
        signals_ok = bool(entry.get("signals", {}).get("count", 0) > 0)
    entry["trained_ok"] = bool(entry.get("training_summary_present")) and model_ok and signals_ok
    return entry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify that model-miner candidates truly trained and produced artifacts.")
    parser.add_argument("--run-dir", required=True, help="Path to runs/<run_id>/strategy_miner")
    parser.add_argument("--json", action="store_true", help="Print raw JSON only")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir).resolve()
    backtests_dir = run_dir / "backtests"
    if not backtests_dir.exists():
        print(json.dumps({"ok": False, "error": f"backtests dir not found: {backtests_dir}"}, ensure_ascii=False, indent=2))
        return 2

    summaries = sorted(backtests_dir.glob("iter_*/*/summary.json"))
    entries = [_candidate_entry(path) for path in summaries]
    trained = [entry for entry in entries if entry.get("trained_ok")]
    out = {
        "ok": bool(entries) and len(trained) == len(entries),
        "run_dir": str(run_dir),
        "candidate_count": len(entries),
        "trained_ok_count": len(trained),
        "items": entries,
    }
    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

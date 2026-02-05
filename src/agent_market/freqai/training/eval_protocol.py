from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, slots=True)
class WalkForwardSplit:
    train_start: int
    train_end: int  # exclusive
    test_start: int
    test_end: int  # exclusive
    purge: int
    embargo: int


def walkforward_splits(
    n_samples: int,
    *,
    train_size: int,
    test_size: int,
    step_size: int,
    purge: int = 0,
    embargo: int = 0,
) -> List[WalkForwardSplit]:
    """
    Minimal walk-forward splitter with purge/embargo gaps.

    Returns contiguous index splits. Each split ensures:
      - test_start >= train_end + purge
      - next split start >= previous test_end + embargo (best-effort)
    """

    n = int(n_samples)
    if n <= 0:
        return []
    train = int(train_size)
    test = int(test_size)
    step = int(step_size)
    pur = max(0, int(purge))
    emb = max(0, int(embargo))
    if train <= 0 or test <= 0 or step <= 0:
        raise ValueError("train_size/test_size/step_size must be > 0")

    splits: list[WalkForwardSplit] = []
    start = 0
    last_test_end = -1
    while True:
        if last_test_end >= 0:
            start = max(start, last_test_end + emb)
        train_start = start
        train_end = train_start + train
        test_start = train_end + pur
        test_end = test_start + test
        if test_end > n:
            break
        splits.append(
            WalkForwardSplit(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                purge=pur,
                embargo=emb,
            )
        )
        last_test_end = test_end
        start = start + step
    return splits


def compute_cost_adjusted_metrics(
    backtest_summary: Dict[str, Any],
    tca_report: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Minimal cost accounting: profit_total_abs - fees_total."""
    profit_abs = backtest_summary.get("profit_total_abs")
    profit_abs_f = float(profit_abs) if profit_abs is not None else float("nan")

    fees_total = None
    if isinstance(tca_report, dict):
        summary = tca_report.get("summary") or {}
        fees_total = summary.get("fees_total")
    fees_total_f = float(fees_total) if fees_total is not None else float("nan")

    out: Dict[str, Any] = {
        "profit_total_abs": float(profit_abs_f) if np.isfinite(profit_abs_f) else None,
        "fees_total": float(fees_total_f) if np.isfinite(fees_total_f) else None,
        "net_profit_abs": None,
    }
    if np.isfinite(profit_abs_f) and np.isfinite(fees_total_f):
        out["net_profit_abs"] = float(profit_abs_f - fees_total_f)
    return out


def build_eval_report(
    *,
    run_meta: Dict[str, Any],
    backtest_summary: Dict[str, Any],
    tca_report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    run_id = str(run_meta.get("run_id") or "")
    cfg = run_meta.get("config") or {}
    cfg_sha = cfg.get("sha256")
    artifacts = run_meta.get("artifacts") or {}

    metrics = compute_cost_adjusted_metrics(backtest_summary, tca_report)
    return {
        "version": 1,
        "generated_at": utc_now(),
        "run_id": run_id,
        "config_sha256": cfg_sha,
        "sources": {
            "run_meta_path": run_meta.get("paths", {}).get("run_meta"),
            "feedback_summary": artifacts.get("feedback_summary"),
            "tca_report": artifacts.get("tca_report"),
        },
        "metrics": metrics,
        "notes": [
            "v1 minimal eval protocol: cost accounting (fees) + reproducible report artifact.",
            "walk-forward/purge/embargo helpers are available but not yet wired into full training/backtest pipeline.",
        ],
    }


def write_eval_report(path: Path, payload: Dict[str, Any]) -> Path:
    p = Path(path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


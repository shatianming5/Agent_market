"""Experiment tracker — records every run, supports querying history.

Usage:
    from workspace.tracker import record_experiment, query_best, compare, list_experiments
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "workspace" / "results"
EXPERIMENTS_FILE = RESULTS_DIR / "experiments.jsonl"


def record_experiment(
    *,
    backtest_result: Dict[str, Any],
    evaluation: Dict[str, Any],
    strategy_name: str,
    model_name: Optional[str] = None,
    notes: str = "",
    tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Record an experiment to experiments.jsonl. Returns the record."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Auto-increment experiment ID
    existing = list_experiments()
    exp_id = len(existing) + 1

    record = {
        "id": exp_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "strategy_name": strategy_name,
        "strategy_path": backtest_result.get("strategy_path", ""),
        "strategy_sha256": backtest_result.get("strategy_sha256", ""),
        "model_name": model_name,
        "config_path": backtest_result.get("config_path", ""),
        "timerange": backtest_result.get("timerange", ""),
        "backtest_ok": backtest_result.get("ok", False),
        "metrics": {
            "trades": backtest_result.get("trades", 0),
            "profit_pct": backtest_result.get("profit_pct", 0),
            "profit_abs": backtest_result.get("profit_abs", 0),
            "sharpe": backtest_result.get("sharpe", 0),
            "sortino": backtest_result.get("sortino", 0),
            "max_drawdown_pct": backtest_result.get("max_drawdown_pct", 0),
            "win_rate": backtest_result.get("win_rate", 0),
            "profit_factor": backtest_result.get("profit_factor", 0),
            "calmar": backtest_result.get("calmar", 0),
        },
        "evaluation": {
            "total_score": evaluation.get("total_score", 0),
            "grade": evaluation.get("grade", "F"),
        },
        "notes": notes,
        "tags": tags or [],
    }

    with open(EXPERIMENTS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return record


def list_experiments() -> List[Dict[str, Any]]:
    """Load all experiments from JSONL."""
    if not EXPERIMENTS_FILE.exists():
        return []
    records = []
    for line in EXPERIMENTS_FILE.read_text(encoding="utf-8").strip().split("\n"):
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def query_best(metric: str = "sharpe", top_n: int = 5) -> List[Dict[str, Any]]:
    """Return top N experiments by a given metric (descending)."""
    experiments = [e for e in list_experiments() if e.get("backtest_ok")]
    experiments.sort(key=lambda e: e.get("metrics", {}).get(metric, -999), reverse=True)
    return experiments[:top_n]


def compare(id1: int, id2: int) -> Dict[str, Any]:
    """Compare two experiments side-by-side."""
    experiments = {e["id"]: e for e in list_experiments()}
    e1 = experiments.get(id1)
    e2 = experiments.get(id2)
    if e1 is None or e2 is None:
        return {"error": f"experiment {id1 if e1 is None else id2} not found"}

    m1 = e1.get("metrics", {})
    m2 = e2.get("metrics", {})
    diff = {}
    for key in set(list(m1.keys()) + list(m2.keys())):
        v1 = m1.get(key, 0)
        v2 = m2.get(key, 0)
        diff[key] = {"exp1": v1, "exp2": v2, "delta": round(v2 - v1, 6)}

    return {
        "exp1": {"id": id1, "strategy": e1["strategy_name"], "score": e1["evaluation"]["total_score"]},
        "exp2": {"id": id2, "strategy": e2["strategy_name"], "score": e2["evaluation"]["total_score"]},
        "metric_diff": diff,
        "winner": id1 if e1["evaluation"]["total_score"] >= e2["evaluation"]["total_score"] else id2,
    }


__all__ = ["record_experiment", "list_experiments", "query_best", "compare"]

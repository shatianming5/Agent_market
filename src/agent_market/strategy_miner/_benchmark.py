"""Frozen benchmark pack execution for final promotion gating (D5)."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from agent_market import paths

from .dtypes import StrategyCandidate
from .sandbox import validate_strategy_code


def _iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _resolve_suite_path(path_or_dir: str | Path) -> Path:
    raw = str(path_or_dir or "").strip()
    if not raw:
        raise ValueError("benchmark suite path is required")
    path = paths.resolve_repo_path(raw)
    if path.is_dir():
        path = path / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Benchmark suite not found: {path}")
    return path


def load_benchmark_suite(path_or_dir: str | Path) -> Dict[str, Any]:
    """Load a frozen benchmark suite manifest."""
    manifest_path = _resolve_suite_path(path_or_dir)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid benchmark suite manifest: {manifest_path}")
    payload.setdefault("schema_version", "1.0")
    payload.setdefault("suite_id", manifest_path.parent.name)
    payload.setdefault("gates", {})
    payload.setdefault("challenge_suite", [])
    payload["_manifest_path"] = str(manifest_path)
    payload["_manifest_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]
    return payload


def _summary_snapshot(summary: Dict[str, Any]) -> Dict[str, Any]:
    keys = (
        "profit_total_pct",
        "trades",
        "winrate",
        "sharpe",
        "realistic_sharpe",
        "sortino",
        "realistic_sortino",
        "calmar",
        "realistic_calmar",
        "profit_factor",
        "max_drawdown_pct",
        "positive_days_ratio",
        "return_over_drawdown",
        "observation_days",
        "metric_flags",
        "metrics_trusted",
        "results_per_pair",
        "walkforward",
        "daily_profit",
        "starting_balance",
        "final_balance",
    )
    return {k: summary.get(k) for k in keys if k in summary}


def _pair_holdout_check(summary: Dict[str, Any], min_profit_pct: float) -> Dict[str, Any]:
    items = list(summary.get("results_per_pair") or [])
    failing = []
    for row in items:
        if not isinstance(row, dict):
            continue
        pair = str(row.get("key") or row.get("pair") or "")
        profit = row.get("profit_total_pct")
        try:
            profit_f = float(profit)
        except Exception:
            continue
        if profit_f < float(min_profit_pct):
            failing.append({"pair": pair, "profit_total_pct": profit_f})
    return {
        "id": "pair_holdout_floor",
        "passed": not failing,
        "threshold": {"min_pair_profit_pct": float(min_profit_pct)},
        "actual": {"failing_pairs": failing, "checked_pairs": len(items)},
    }


def _metric_integrity_check(summary: Dict[str, Any], rejected_flags: list[str]) -> Dict[str, Any]:
    flags = list(summary.get("metric_flags") or [])
    failing = [flag for flag in flags if flag in rejected_flags]
    metrics_trusted = bool(summary.get("metrics_trusted", not flags))
    return {
        "id": "metric_integrity",
        "passed": metrics_trusted and not failing,
        "threshold": {"rejected_metric_flags": list(rejected_flags)},
        "actual": {"metric_flags": flags, "metrics_trusted": metrics_trusted},
    }


def _holdout_delta_check(holdout_result: Optional[Dict[str, Any]], max_delta_pct: float) -> Dict[str, Any]:
    delta = None
    if isinstance(holdout_result, dict):
        delta = holdout_result.get("delta_pct")
    try:
        delta_f = abs(float(delta)) if delta is not None else None
    except Exception:
        delta_f = None
    passed = delta_f is not None and delta_f <= float(max_delta_pct)
    return {
        "id": "selection_holdout_delta",
        "passed": passed,
        "threshold": {"max_delta_pct": float(max_delta_pct)},
        "actual": {"delta_pct": delta_f},
    }


def _walkforward_check(summary: Dict[str, Any], min_folds: int) -> Dict[str, Any]:
    wf = dict(summary.get("walkforward") or {})
    completed = int(wf.get("folds_completed", 0) or 0)
    return {
        "id": "walkforward_min_folds",
        "passed": completed >= int(min_folds),
        "threshold": {"min_folds_completed": int(min_folds)},
        "actual": {
            "folds_completed": completed,
            "folds_total": int(wf.get("folds_total", 0) or 0),
            "sharpe_mean": wf.get("sharpe_mean"),
            "sharpe_std": wf.get("sharpe_std"),
        },
    }


def _static_challenge_result(challenge: Dict[str, Any], code: str) -> Dict[str, Any]:
    challenge_id = str(challenge.get("id") or "challenge")
    challenge_type = str(challenge.get("type") or "pattern_absent")
    description = str(challenge.get("description") or "")
    pattern = str(challenge.get("pattern") or "")
    passed = True
    evidence: Dict[str, Any] = {}

    if challenge_type == "pattern_absent":
        matched = bool(pattern and pattern in code)
        passed = not matched
        evidence = {"pattern": pattern, "matched": matched}
    elif challenge_type == "regex_absent":
        import re

        matched = bool(pattern and re.search(pattern, code, re.IGNORECASE | re.MULTILINE))
        passed = not matched
        evidence = {"pattern": pattern, "matched": matched}
    else:
        passed = True
        evidence = {"skipped": True, "reason": f"unsupported challenge type: {challenge_type}"}

    return {
        "id": challenge_id,
        "type": challenge_type,
        "description": description,
        "passed": passed,
        "evidence": evidence,
    }


def run_benchmark_suite(
    candidate: StrategyCandidate,
    *,
    suite_path: str | Path,
    holdout_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate a candidate against a frozen benchmark/challenge pack."""
    suite = load_benchmark_suite(suite_path)
    summary = dict(candidate.backtest_summary or {})
    gates = dict(suite.get("gates") or {})

    rejected_flags = list(
        gates.get(
            "reject_metric_flags",
            [
                "native_sharpe_inflated",
                "native_sortino_inflated",
                "native_calmar_inflated",
                "drawdown_mismatch",
            ],
        )
        or []
    )

    selection_holdout = _holdout_delta_check(
        holdout_result,
        float(gates.get("selection_holdout_delta_max_pct", 50.0) or 50.0),
    )
    walkforward = _walkforward_check(
        summary,
        int(gates.get("walkforward_min_folds_completed", 0) or 0),
    )
    pair_holdout = _pair_holdout_check(
        summary,
        float(gates.get("pair_holdout_min_profit_pct", -0.5) or -0.5),
    )
    metric_integrity = _metric_integrity_check(summary, rejected_flags)

    validation_ok, validation_msg = validate_strategy_code(candidate.code)
    validation_check = {
        "id": "strategy_validation",
        "passed": bool(validation_ok),
        "threshold": {"must_validate": True},
        "actual": {"message": validation_msg[:400]},
    }

    observation_days = int(summary.get("observation_days", 0) or 0)
    observation_check = {
        "id": "observation_days",
        "passed": observation_days >= int(gates.get("min_observation_days", 0) or 0),
        "threshold": {"min_observation_days": int(gates.get("min_observation_days", 0) or 0)},
        "actual": {"observation_days": observation_days},
    }

    positive_days_ratio = float(summary.get("positive_days_ratio", 0.0) or 0.0)
    positive_days_check = {
        "id": "positive_days_ratio",
        "passed": positive_days_ratio >= float(gates.get("min_positive_days_ratio", 0.0) or 0.0),
        "threshold": {"min_positive_days_ratio": float(gates.get("min_positive_days_ratio", 0.0) or 0.0)},
        "actual": {"positive_days_ratio": positive_days_ratio},
    }

    checks = [
        selection_holdout,
        walkforward,
        pair_holdout,
        metric_integrity,
        validation_check,
        observation_check,
        positive_days_check,
    ]
    challenges = [
        _static_challenge_result(challenge, candidate.code)
        for challenge in list(suite.get("challenge_suite") or [])
        if isinstance(challenge, dict)
    ]
    passed = all(bool(item.get("passed")) for item in checks + challenges)

    return {
        "evaluated_at": _iso_now(),
        "suite_id": str(suite.get("suite_id") or ""),
        "suite_schema_version": str(suite.get("schema_version") or "1.0"),
        "suite_manifest_path": str(suite.get("_manifest_path") or ""),
        "suite_manifest_sha256": str(suite.get("_manifest_sha256") or ""),
        "candidate": {
            "name": candidate.name,
            "iteration": int(candidate.iteration),
            "stage": str(getattr(candidate, "stage", "") or ""),
            "reward": candidate.reward,
            "constraints_ok": bool(getattr(candidate, "constraints_ok", True)),
        },
        "selection_summary": _summary_snapshot(summary),
        "holdout": holdout_result or None,
        "checks": checks,
        "challenges": challenges,
        "passed": passed,
        "failed_ids": [
            str(item.get("id") or "")
            for item in checks + challenges
            if not bool(item.get("passed"))
        ],
    }


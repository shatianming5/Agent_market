"""Standardized artifacts for Strategy Miner runs.

All artifacts live under:
  runs/<run_id>/strategy_miner/

This module writes stable JSON/code snapshots so that results are reproducible,
API-queryable, and auditable.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from agent_market.utils import sha256_bytes

from .dtypes import MinerConfig, MinerState, StrategyCandidate


def _iso_now() -> str:
    # Keep it simple; avoid timezone complications.
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _atomic_write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))


def config_to_dict(config: MinerConfig) -> dict[str, Any]:
    """Serialize MinerConfig for proposal.json.

    Uses dataclasses.asdict to keep nested structures.
    """
    return asdict(config)


def proposal_path(miner_dir: Path) -> Path:
    return miner_dir / "proposal.json"


def leaderboard_path(miner_dir: Path) -> Path:
    return miner_dir / "leaderboard.json"


def run_manifest_path(miner_dir: Path) -> Path:
    return miner_dir / "manifest.json"


def failure_pareto_path(miner_dir: Path) -> Path:
    return miner_dir / "failure_pareto.json"


def multiagent_summary_path(miner_dir: Path) -> Path:
    return miner_dir / "multiagent_summary.json"


def candidates_dir(miner_dir: Path) -> Path:
    return miner_dir / "candidates"


def backtests_dir(miner_dir: Path) -> Path:
    return miner_dir / "backtests"


def traces_dir(miner_dir: Path) -> Path:
    return miner_dir / "agent_traces"


def trainings_dir(miner_dir: Path) -> Path:
    return miner_dir / "training"


def _file_evidence(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    item: dict[str, Any] = {
        "path": str(resolved),
        "exists": resolved.exists(),
    }
    if not resolved.exists() or not resolved.is_file():
        return item
    payload = resolved.read_bytes()
    stat = resolved.stat()
    item.update(
        {
            "size_bytes": int(stat.st_size),
            "mtime_epoch": float(stat.st_mtime),
            "sha256": sha256_bytes(payload),
        }
    )
    return item


def candidate_verification_status(candidate: StrategyCandidate) -> str:
    """Map Strategy Miner's sealed validation evidence to the shared status vocabulary."""
    stage = str(getattr(candidate, "stage", "") or "").strip().lower()
    funnel = getattr(candidate, "funnel_state", None) or {}
    holdout = funnel.get("holdout") if isinstance(funnel.get("holdout"), dict) else {}
    benchmark = funnel.get("benchmark") if isinstance(funnel.get("benchmark"), dict) else {}

    if holdout:
        if bool(holdout.get("overfitting_flag")):
            return "failed"
        if benchmark and benchmark.get("passed") is False:
            return "failed"
        return "passed" if bool(getattr(candidate, "constraints_ok", True)) else "failed"
    if getattr(candidate, "diagnosis", "") or getattr(candidate, "failure_category", ""):
        return "failed"
    if getattr(candidate, "constraints_ok", True) is False:
        return "failed"
    if getattr(candidate, "backtest_summary", None) is not None:
        return "pending"
    return "pending"


def candidate_promotion_eligible(candidate: StrategyCandidate) -> bool:
    """Strategy Miner never controls final promotion eligibility.

    A miner candidate may pass its own holdout/benchmark gates, but the shared
    promotion flag is reserved for factor_lab.strategy-loop after blind
    verification. Keeping this function fail-closed prevents leaderboard and
    manifest artifacts from advertising pre-blind candidates as promotable.
    """
    return False


def _load_trace_payload(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _trace_body(payload: dict[str, Any]) -> dict[str, Any]:
    body = payload.get("payload") if isinstance(payload.get("payload"), dict) else payload
    return body if isinstance(body, dict) else {}


def classify_candidate_failure(candidate: StrategyCandidate) -> str:
    """Coarse failure taxonomy for run-level Pareto summaries."""
    failure_category = str(getattr(candidate, "failure_category", "") or "").strip().lower()
    diagnosis = str(getattr(candidate, "diagnosis", "") or "").strip().lower()
    violations = [str(v).lower() for v in list(getattr(candidate, "constraint_violations", []) or [])]
    summary = getattr(candidate, "backtest_summary", None) or {}
    quick = getattr(candidate, "quick_backtest_summary", None) or {}
    funnel = getattr(candidate, "funnel_state", None) or {}
    holdout = funnel.get("holdout") if isinstance(funnel.get("holdout"), dict) else {}

    if failure_category.startswith("validation.syntax") or "syntax error" in diagnosis:
        return "syntax_failure"
    if failure_category in {"validation.inheritance", "validation.missing_methods"} or "istrategy" in diagnosis:
        return "syntax_failure"
    if failure_category.startswith("validation."):
        return "validation_failure"
    if failure_category.startswith("train_model."):
        return "model_training_failure"
    if failure_category.startswith("backtest."):
        return "backtest_failure"
    if bool(holdout.get("overfitting_flag")) or any("holdout_overfitting" in v for v in violations):
        return "overfit_failure"
    if any("min_trades" in v or "sample" in v for v in violations):
        return "insufficient_sample"

    fee_drag = summary.get("fee_drag_pct")
    try:
        fee_drag_f = float(fee_drag) if fee_drag is not None else 0.0
    except Exception:
        fee_drag_f = 0.0
    profit = summary.get("profit_total_pct")
    if profit is None:
        profit = quick.get("profit_total_pct") if isinstance(quick, dict) else None
    profit_factor = summary.get("profit_factor")
    if profit_factor is None:
        profit_factor = quick.get("profit_factor") if isinstance(quick, dict) else None
    try:
        profit_f = float(profit) if profit is not None else None
    except Exception:
        profit_f = None
    try:
        pf_f = float(profit_factor) if profit_factor is not None else None
    except Exception:
        pf_f = None

    if fee_drag_f > 0 and profit_f is not None and abs(fee_drag_f) >= max(abs(profit_f), 1.0):
        return "fee_drag_failure"
    if profit_f is not None and profit_f < 0:
        return "unprofitable_failure"
    if pf_f is not None and pf_f < 1.0 and profit_f is not None and profit_f <= 0:
        return "unprofitable_failure"
    if getattr(candidate, "constraints_ok", True) is False:
        return "constraint_failure"
    if getattr(candidate, "reward", None) is None:
        return "unevaluated_failure"
    return "other_failure"


def _norm_trace_path(raw: Any) -> str:
    try:
        return str(Path(str(raw)).resolve())
    except Exception:
        return str(raw)


def _candidate_trace_paths(state: MinerState) -> set[str]:
    out: set[str] = set()
    for candidate in state.candidates:
        for raw in dict(getattr(candidate, "agent_traces", {}) or {}).values():
            if raw:
                out.add(_norm_trace_path(raw))
    return out


def _iter_trace_files(miner_dir: Path | None) -> list[Path]:
    if miner_dir is None:
        return []
    base = traces_dir(Path(miner_dir))
    if not base.exists():
        return []
    try:
        return sorted(path for path in base.glob("iter_*/cand_*/*.json") if path.is_file())
    except Exception:
        return []


def _trace_role(path: Path, payload: dict[str, Any]) -> str:
    role = payload.get("role")
    return str(role or path.stem or "unknown")


def _trace_index(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        if path.parent.name.startswith("cand_"):
            out["candidate_idx"] = int(path.parent.name.split("_", 1)[1])
    except Exception:
        pass
    try:
        iter_part = path.parent.parent.name
        if iter_part.startswith("iter_"):
            out["iteration"] = int(iter_part.split("_", 1)[1])
    except Exception:
        pass
    return out


def _trace_failure_item(path: Path, failure_category: str) -> dict[str, Any]:
    payload = _load_trace_payload(path)
    body = _trace_body(payload)
    item: dict[str, Any] = {
        "name": f"trace:{path.parent.parent.name}/{path.parent.name}/{path.stem}",
        "stage": "generation_trace",
        "verification_status": "failed",
        "promotion_eligible": False,
        "failure_category": failure_category,
        "diagnosis": str(body.get("reason") or body.get("error") or "")[:500],
        "trace_path": str(path.resolve()),
        "trace_role": _trace_role(path, payload),
        "metrics": {},
        "quick_metrics": {},
    }
    item.update(_trace_index(path))
    return item


def build_failure_pareto(state: MinerState, *, miner_dir: Path | None = None) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    promoted: list[dict[str, Any]] = []
    search_passed: list[dict[str, Any]] = []
    seen_trace_paths = _candidate_trace_paths(state)
    for candidate in state.candidates:
        verification_status = candidate_verification_status(candidate)
        promotion_eligible = candidate_promotion_eligible(candidate)
        summary = getattr(candidate, "backtest_summary", None) or {}
        item = {
            "name": candidate.name,
            "iteration": int(getattr(candidate, "iteration", 0) or 0),
            "candidate_type": getattr(candidate, "candidate_type", "rule"),
            "candidate_family": getattr(candidate, "candidate_family", ""),
            "model_family": getattr(candidate, "model_family", ""),
            "stage": getattr(candidate, "stage", ""),
            "verification_status": verification_status,
            "promotion_eligible": promotion_eligible,
            "failure_category": getattr(candidate, "failure_category", ""),
            "diagnosis": str(getattr(candidate, "diagnosis", "") or "")[:500],
            "constraint_violations": list(getattr(candidate, "constraint_violations", []) or []),
            "metrics": {
                "reward": getattr(candidate, "reward", None),
                "profit_total_pct": summary.get("profit_total_pct"),
                "profit_factor": summary.get("profit_factor"),
                "trades": summary.get("trades"),
                "fee_drag_pct": summary.get("fee_drag_pct"),
                "max_drawdown_pct": summary.get("max_drawdown_pct", summary.get("max_drawdown_account")),
            },
            "quick_metrics": {
                "profit_total_pct": (getattr(candidate, "quick_backtest_summary", None) or {}).get("profit_total_pct")
                if isinstance(getattr(candidate, "quick_backtest_summary", None), dict)
                else None,
                "profit_factor": (getattr(candidate, "quick_backtest_summary", None) or {}).get("profit_factor")
                if isinstance(getattr(candidate, "quick_backtest_summary", None), dict)
                else None,
                "trades": (getattr(candidate, "quick_backtest_summary", None) or {}).get("trades")
                if isinstance(getattr(candidate, "quick_backtest_summary", None), dict)
                else None,
            },
        }
        if promotion_eligible:
            promoted.append(item)
            continue
        if verification_status == "passed" and bool(getattr(candidate, "constraints_ok", True)):
            search_passed.append(item)
            continue
        bucket = classify_candidate_failure(candidate)
        buckets.setdefault(bucket, []).append(item)

    orphan_trace_failures = 0
    for trace_path in _iter_trace_files(miner_dir):
        if _norm_trace_path(trace_path) in seen_trace_paths:
            continue
        failure_category = _load_trace_failure_category(trace_path)
        if not failure_category:
            continue
        orphan_trace_failures += 1
        buckets.setdefault(failure_category, []).append(_trace_failure_item(trace_path, failure_category))

    categories = []
    for name, items in buckets.items():
        categories.append(
            {
                "category": name,
                "count": len(items),
                "examples": items[:5],
            }
        )
    categories.sort(key=lambda item: int(item["count"]), reverse=True)
    return {
        "version": "strategy-miner-failure-pareto-v1",
        "run_id": state.run_id,
        "candidate_count": len(state.candidates),
        "promoted_count": len(promoted),
        "search_passed_pending_blind_count": len(search_passed),
        "failure_count": sum(len(items) for items in buckets.values()),
        "orphan_trace_failure_count": orphan_trace_failures,
        "categories": categories,
        "promoted": promoted[:10],
        "search_passed_pending_blind": search_passed[:10],
    }


def write_failure_pareto(miner_dir: Path, state: MinerState) -> Path:
    out = failure_pareto_path(miner_dir)
    _atomic_write_json(out, build_failure_pareto(state, miner_dir=miner_dir))
    return out


def build_run_manifest(
    miner_dir: Path,
    state: MinerState,
    *,
    config: MinerConfig,
    extra_artifacts: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    artifact_names = [
        "proposal.json",
        "goal_contract.json",
        "run_meta.json",
        "checkpoint.json",
        "leaderboard.json",
        "failure_pareto.json",
        "multiagent_summary.json",
        "holdout_gate.json",
        "benchmark_verdict.json",
        "portfolio_plan.json",
        "economics.json",
    ]
    files = {
        name: _file_evidence(miner_dir / name)
        for name in artifact_names
        if (miner_dir / name).exists()
    }
    if extra_artifacts:
        for key, raw in extra_artifacts.items():
            try:
                files[key] = _file_evidence(Path(raw))
            except Exception:
                files[key] = {"path": str(raw), "exists": False}
    trace_files: list[dict[str, Any]] = []
    for trace_path in _iter_trace_files(miner_dir):
        evidence = _file_evidence(trace_path)
        payload = _load_trace_payload(trace_path)
        evidence.update(
            {
                "role": _trace_role(trace_path, payload),
                "failure_category": _load_trace_failure_category(trace_path),
                **_trace_index(trace_path),
            }
        )
        trace_files.append(evidence)

    return {
        "version": "strategy-miner-run-manifest-v1",
        "saved_at": _iso_now(),
        "run_id": state.run_id,
        "promotion_controller": "agent_market.factor_lab.strategy-loop",
        "strategy_miner_role": "candidate_generation_and_sidecar_validation",
        "config": config_to_dict(config),
        "counts": {
            "candidates": len(state.candidates),
            "history": len(state.history),
            "agent_traces": len(trace_files),
            "agent_trace_failures": sum(1 for item in trace_files if item.get("failure_category")),
        },
        "candidates": [
            {
                "name": candidate.name,
                "iteration": int(candidate.iteration),
                "candidate_slot": int(getattr(candidate, "candidate_slot", 0) or 0),
                "candidate_type": getattr(candidate, "candidate_type", "rule"),
                "candidate_family": getattr(candidate, "candidate_family", ""),
                "stage": getattr(candidate, "stage", ""),
                "strategy_path": str(getattr(candidate, "strategy_path", "")),
                "agent_traces": dict(getattr(candidate, "agent_traces", {}) or {}),
                "verification_status": candidate_verification_status(candidate),
                "promotion_eligible": candidate_promotion_eligible(candidate),
                "failure_category": getattr(candidate, "failure_category", ""),
            }
            for candidate in state.candidates
        ],
        "best": {
            "name": state.best_candidate.name if state.best_candidate else None,
            "score": state.best_score,
            "verification_status": candidate_verification_status(state.best_candidate) if state.best_candidate else None,
            "promotion_eligible": candidate_promotion_eligible(state.best_candidate) if state.best_candidate else False,
        },
        "agent_traces": trace_files[:500],
        "files": files,
    }


def write_run_manifest(
    miner_dir: Path,
    state: MinerState,
    *,
    config: MinerConfig,
    extra_artifacts: Optional[dict[str, str]] = None,
) -> Path:
    out = run_manifest_path(miner_dir)
    _atomic_write_json(out, build_run_manifest(miner_dir, state, config=config, extra_artifacts=extra_artifacts))
    return out


def _load_trace_failure_category(path: str | Path) -> str:
    payload = _load_trace_payload(Path(path))
    body = _trace_body(payload)
    return str(body.get("failure_category") or "").strip()


def build_multiagent_summary(
    state: MinerState,
    *,
    config: MinerConfig,
    miner_dir: Path | None = None,
) -> dict[str, Any]:
    """Summarize strategy-side agent roles, traces, and search-only promotion scope."""
    role_counts: dict[str, int] = {}
    failure_taxonomy: dict[str, int] = {}
    candidates: list[dict[str, Any]] = []
    seen_trace_paths = _candidate_trace_paths(state)
    for candidate in state.candidates:
        traces = dict(getattr(candidate, "agent_traces", {}) or {})
        for role, path in traces.items():
            role_counts[str(role)] = role_counts.get(str(role), 0) + 1
            failure_category = _load_trace_failure_category(str(path))
            if failure_category:
                failure_taxonomy[failure_category] = failure_taxonomy.get(failure_category, 0) + 1
        if getattr(candidate, "failure_category", ""):
            category = str(candidate.failure_category)
            failure_taxonomy[category] = failure_taxonomy.get(category, 0) + 1
        candidates.append(
            {
                "name": candidate.name,
                "iteration": int(candidate.iteration),
                "candidate_slot": int(getattr(candidate, "candidate_slot", 0) or 0),
                "candidate_type": getattr(candidate, "candidate_type", "rule"),
                "candidate_family": getattr(candidate, "candidate_family", ""),
                "stage": getattr(candidate, "stage", ""),
                "trace_roles": sorted(traces.keys()),
                "verification_status": candidate_verification_status(candidate),
                "promotion_eligible": candidate_promotion_eligible(candidate),
                "failure_category": getattr(candidate, "failure_category", ""),
            }
        )

    orphan_traces: list[dict[str, Any]] = []
    for trace_path in _iter_trace_files(miner_dir):
        if _norm_trace_path(trace_path) in seen_trace_paths:
            continue
        payload = _load_trace_payload(trace_path)
        role = _trace_role(trace_path, payload)
        role_counts[role] = role_counts.get(role, 0) + 1
        failure_category = _load_trace_failure_category(trace_path)
        if failure_category:
            failure_taxonomy[failure_category] = failure_taxonomy.get(failure_category, 0) + 1
        item = {
            "path": str(trace_path.resolve()),
            "role": role,
            "failure_category": failure_category,
        }
        item.update(_trace_index(trace_path))
        orphan_traces.append(item)

    return {
        "version": "strategy-miner-multiagent-summary-v1",
        "saved_at": _iso_now(),
        "run_id": state.run_id,
        "enabled": bool(getattr(config, "multiagent_enabled", False)),
        "roles": ["planner", "coder", "reviewer", "backtester"],
        "promotion_controller": "agent_market.factor_lab.strategy-loop",
        "promotion_policy": "strategy_miner_outputs_are_candidates_until_blind_verification",
        "config": {
            "candidates_per_iteration": int(getattr(config, "candidates_per_iteration", 1) or 1),
            "max_parallel_candidates": int(getattr(config, "max_parallel_candidates", 0) or 0),
            "max_parallel_roles": int(getattr(config, "max_parallel_roles", 1) or 1),
            "repair_attempts": int(getattr(config, "repair_attempts", 0) or 0),
            "max_iterations": int(getattr(config, "max_iterations", 0) or 0),
        },
        "counts": {
            "candidates": len(state.candidates),
            "traced_roles": sum(role_counts.values()),
            "orphan_traces": len(orphan_traces),
            "orphan_failure_traces": sum(1 for item in orphan_traces if item.get("failure_category")),
            "promoted_by_strategy_miner": sum(1 for c in state.candidates if candidate_promotion_eligible(c)),
        },
        "role_counts": role_counts,
        "failure_taxonomy": failure_taxonomy,
        "orphan_traces": orphan_traces[:200],
        "candidates": candidates,
    }


def write_multiagent_summary(miner_dir: Path, state: MinerState, *, config: MinerConfig) -> Path:
    out = multiagent_summary_path(miner_dir)
    _atomic_write_json(out, build_multiagent_summary(state, config=config, miner_dir=miner_dir))
    return out


def write_agent_trace(
    miner_dir: Path,
    *,
    iteration: int,
    candidate_idx: int,
    role: str,
    payload: Any,
    prompt_meta: Optional[dict] = None,
) -> Path:
    """Write an agent trace JSON under agent_traces/iter_xxxx/cand_xx/.

    Args:
        prompt_meta: D8 context-engineering metadata from ``prompt_metadata()``.
    """
    safe_role = "".join(ch if (ch.isalnum() or ch in "-_@.") else "_" for ch in (role or "role"))
    out_dir = traces_dir(miner_dir) / f"iter_{int(iteration):04d}" / f"cand_{int(candidate_idx):02d}"
    out = out_dir / f"{safe_role}.json"
    wrapped = {
        "saved_at": _iso_now(),
        "iteration": int(iteration),
        "candidate_idx": int(candidate_idx),
        "role": str(role),
        "payload": payload,
    }
    if prompt_meta:
        wrapped["prompt_meta"] = prompt_meta
    _atomic_write_json(out, wrapped)
    return out


def write_proposal(
    miner_dir: Path,
    *,
    run_id: str,
    config: MinerConfig,
    overwrite: bool = False,
) -> Path:
    """Write proposal.json (inputs + constraints + budget/tool policy)."""
    out = proposal_path(miner_dir)
    if out.exists() and not overwrite:
        return out

    payload = {
        "run_id": str(run_id),
        "created_at": _iso_now(),
        "objective": "Generate and validate Freqtrade IStrategy candidates with agentic tool-calls.",
        "config": config_to_dict(config),
    }
    _atomic_write_json(out, payload)
    return out


def write_candidate_snapshot(
    miner_dir: Path,
    candidate: StrategyCandidate,
) -> dict[str, Path]:
    """Write candidate code + metadata under candidates/iter_xxxx/."""
    iter_dir = candidates_dir(miner_dir) / f"iter_{int(candidate.iteration):04d}"
    suffix = candidate.strategy_path.suffix if candidate.strategy_path.suffix in {".py", ".json"} else ".py"
    code_path = iter_dir / f"{candidate.name}{suffix}"
    meta_path = iter_dir / f"{candidate.name}.json"

    _atomic_write_text(code_path, candidate.code)

    meta = {
        "saved_at": _iso_now(),
        "candidate": candidate.to_dict(),
        "artifact": {
            "code_path": str(code_path),
            "meta_path": str(meta_path),
        },
    }
    _atomic_write_json(meta_path, meta)
    return {"code": code_path, "meta": meta_path}


def write_backtest_summary(
    miner_dir: Path,
    candidate: StrategyCandidate,
    *,
    zip_path: Optional[Path] = None,
) -> Optional[Path]:
    """Write backtest summary under backtests/iter_xxxx/<name>/summary.json."""
    if candidate.backtest_summary is None:
        return None

    out_dir = backtests_dir(miner_dir) / f"iter_{int(candidate.iteration):04d}" / str(candidate.name)
    out = out_dir / "summary.json"

    payload = {
        "saved_at": _iso_now(),
        "candidate": {
            "name": candidate.name,
            "iteration": candidate.iteration,
            "reward": candidate.reward,
            "validation_passed": candidate.validation_passed,
            "strategy_path": str(candidate.strategy_path),
            "candidate_type": candidate.candidate_type,
            "model_family": candidate.model_family,
        },
        "backtest_summary": candidate.backtest_summary,
        "training_summary": candidate.training_summary,
        "backtest_zip": str(zip_path) if zip_path is not None else None,
    }
    _atomic_write_json(out, payload)
    return out


def write_training_evidence(
    miner_dir: Path,
    candidate: StrategyCandidate,
    *,
    training_summary_path: Path,
    extra_files: Optional[list[Path]] = None,
    extra_payload: Optional[dict[str, Any]] = None,
) -> Path:
    """Write training evidence under training/iter_xxxx/<name>/training_evidence.json."""
    out_dir = trainings_dir(miner_dir) / f"iter_{int(candidate.iteration):04d}" / str(candidate.name)
    out = out_dir / "training_evidence.json"

    summary = getattr(candidate, "training_summary", None) or {}
    files: list[dict[str, Any]] = [_file_evidence(training_summary_path)]

    model_path_raw = summary.get("model_path") if isinstance(summary, dict) else None
    if model_path_raw:
        files.append(_file_evidence(Path(str(model_path_raw))))

    for key in ("feature_snapshot", "expressions_snapshot"):
        raw = summary.get(key) if isinstance(summary, dict) else None
        if raw:
            files.append(_file_evidence(Path(str(raw))))

    scaler_path = training_summary_path.with_name("scaler.pkl")
    if scaler_path.exists():
        files.append(_file_evidence(scaler_path))

    for extra in extra_files or []:
        files.append(_file_evidence(extra))

    payload = {
        "saved_at": _iso_now(),
        "candidate": {
            "name": candidate.name,
            "iteration": int(candidate.iteration),
            "candidate_type": getattr(candidate, "candidate_type", "rule"),
            "model_family": getattr(candidate, "model_family", ""),
            "strategy_path": str(candidate.strategy_path),
        },
        "training_summary_path": str(training_summary_path.resolve()),
        "training_summary_excerpt": {
            "model": summary.get("model") if isinstance(summary, dict) else None,
            "train_size": summary.get("train_size") if isinstance(summary, dict) else None,
            "valid_size": summary.get("valid_size") if isinstance(summary, dict) else None,
            "timesteps": summary.get("timesteps") if isinstance(summary, dict) else None,
            "metrics": summary.get("metrics") if isinstance(summary, dict) else None,
            "rolling": summary.get("rolling") if isinstance(summary, dict) else None,
            "data": summary.get("data") if isinstance(summary, dict) else None,
        },
        "files": files,
        "extra": extra_payload or {},
    }
    _atomic_write_json(out, payload)
    return out


def write_leaderboard(
    miner_dir: Path,
    state: MinerState,
    *,
    config: MinerConfig,
) -> Path:
    """Write leaderboard.json sorted by Sharpe desc and filtered by constraints."""
    all_items: list[dict[str, Any]] = []
    for c in state.candidates:
        if c.reward is None:
            continue
        # Force no-template: never show template-sourced candidates.
        src_provider = str(getattr(c, 'source_provider', '') or getattr(c, 'agent_provider', '') or '').strip().lower()
        if src_provider == 'template' or c.name == 'TemplateRsiStrategy' or 'TemplateRsiStrategy' in (c.code or ''):
            continue
        summary = c.backtest_summary or {}
        verification_status = candidate_verification_status(c)
        promotion_eligible = candidate_promotion_eligible(c)
        all_items.append(
            {
                "name": c.name,
                "iteration": c.iteration,
                "sharpe": c.reward,
                "score_sharpe": summary.get("realistic_sharpe", summary.get("sharpe")),
                "native_sharpe": summary.get("native_sharpe", summary.get("sharpe")),
                "candidate_type": getattr(c, "candidate_type", "rule"),
                "model_family": getattr(c, "model_family", ""),
                "validation_passed": c.validation_passed,
                "constraints_ok": bool(getattr(c, "constraints_ok", True)),
                "constraint_violations": list(getattr(c, "constraint_violations", []) or []),
                "verification_status": verification_status,
                "promotion_eligible": promotion_eligible,
                "promotion_controller": "agent_market.factor_lab.strategy-loop",
                "strategy_miner_role": "candidate_generation_and_sidecar_validation",
                "stage": getattr(c, "stage", ""),
                "failure_category": getattr(c, "failure_category", ""),
                "trades": summary.get("trades"),
                "winrate": summary.get("winrate"),
                "profit_total_pct": summary.get("profit_total_pct"),
                "max_drawdown_abs": summary.get("max_drawdown_abs"),
                "max_drawdown_pct": summary.get("max_drawdown_pct", summary.get("max_drawdown_account")),
                "positive_days_ratio": summary.get("positive_days_ratio"),
                "return_over_drawdown": summary.get("return_over_drawdown"),
                "sortino": summary.get("sortino"),
                "realistic_sortino": summary.get("realistic_sortino", summary.get("sortino")),
                "calmar": summary.get("calmar"),
                "realistic_calmar": summary.get("realistic_calmar", summary.get("calmar")),
                "profit_factor": summary.get("profit_factor"),
                "sqn": summary.get("sqn"),
                "metric_flags": summary.get("metric_flags", []),
                "training_summary": getattr(c, "training_summary", None),
                "diagnosis": (c.diagnosis or "")[:200],
            }
        )

    eligible = [i for i in all_items if i.get("constraints_ok", True)]
    rejected = [i for i in all_items if not i.get("constraints_ok", True)]

    eligible.sort(key=lambda x: float(x.get("sharpe") or -1e9), reverse=True)
    rejected.sort(key=lambda x: float(x.get("sharpe") or -1e9), reverse=True)

    payload = {
        "run_id": state.run_id,
        "saved_at": _iso_now(),
        "config": {
            "min_trades": config.min_trades,
            "max_abs_drawdown": config.max_abs_drawdown,
            "min_winrate": config.min_winrate,
            "min_profit_factor": config.min_profit_factor,
            "min_profit_pct": config.min_profit_pct,
            "min_positive_days_ratio": config.min_positive_days_ratio,
            "min_return_over_drawdown": config.min_return_over_drawdown,
            "min_pair_profit_pct": config.min_pair_profit_pct,
            "target_trades": config.target_trades,
            "min_acceptable_trades": config.min_acceptable_trades,
            "max_strategy_timeframe": config.max_strategy_timeframe,
            "allowed_informative_timeframes": list(config.allowed_informative_timeframes or []),
        },
        "best": eligible[0] if eligible else None,
        "items": eligible,
        "rejected": rejected,
        "all_count": len(all_items),
    }

    out = leaderboard_path(miner_dir)
    _atomic_write_json(out, payload)
    return out


# ---------------------------------------------------------------------------
# D1: Goal Contract snapshot
# ---------------------------------------------------------------------------

def write_goal_contract(miner_dir: Path, goal_contract: Any) -> Path:
    """Persist an immutable snapshot of the goal contract for this run."""
    out = miner_dir / "goal_contract.json"
    payload = {
        "snapshot_at": _iso_now(),
        "sha256": goal_contract.sha256() if hasattr(goal_contract, "sha256") else "",
        "contract": goal_contract.to_dict() if hasattr(goal_contract, "to_dict") else {},
    }
    _atomic_write_json(out, payload)
    return out


# ---------------------------------------------------------------------------
# D9: Observability — run metadata + event timeline
# ---------------------------------------------------------------------------

def write_run_meta(miner_dir: Path, *, run_id: str, phase: str,
                   iteration: int, extra: Optional[dict] = None) -> Path:
    """Write/overwrite run_meta.json with current run status."""
    out = miner_dir / "run_meta.json"
    payload = {
        "run_id": run_id,
        "updated_at": _iso_now(),
        "phase": phase,
        "iteration": iteration,
    }
    if extra:
        payload.update(extra)
    _atomic_write_json(out, payload)
    return out


def append_event(miner_dir: Path, event_type: str,
                 detail: Optional[dict] = None) -> Path:
    """Append a single JSONL event to events.jsonl for timeline tracking."""
    out = miner_dir / "events.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": _iso_now(),
        "event": event_type,
    }
    if detail:
        entry["detail"] = detail
    line = json.dumps(entry, ensure_ascii=False)
    with open(out, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    return out


# ---------------------------------------------------------------------------
# D10: Promotion chain — holdout gate artifact
# ---------------------------------------------------------------------------

def write_holdout_gate(miner_dir: Path, holdout_result: dict) -> Path:
    """Write holdout_gate.json with pass/fail determination."""
    passed = not holdout_result.get("overfitting_flag", False)
    payload = {
        "evaluated_at": _iso_now(),
        "passed": passed,
        "holdout_profit_pct": holdout_result.get("holdout_profit_pct"),
        "selection_profit_pct": holdout_result.get("selection_profit_pct"),
        "delta_pct": holdout_result.get("delta_pct"),
        "overfitting_flag": holdout_result.get("overfitting_flag", False),
        "holdout_timerange": holdout_result.get("holdout_timerange", ""),
    }
    out = miner_dir / "holdout_gate.json"
    _atomic_write_json(out, payload)
    return out


# ---------------------------------------------------------------------------
# D5: Frozen benchmark verdict
# ---------------------------------------------------------------------------

def write_benchmark_verdict(miner_dir: Path, verdict: dict) -> Path:
    """Write benchmark_verdict.json for frozen benchmark/challenge results."""
    out = miner_dir / "benchmark_verdict.json"
    _atomic_write_json(out, verdict)
    return out


# ---------------------------------------------------------------------------
# D12: Candidate portfolio recommendation
# ---------------------------------------------------------------------------

def write_portfolio_plan(miner_dir: Path, report: dict) -> Path:
    """Write portfolio_plan.json for final candidate allocation."""
    out = miner_dir / "portfolio_plan.json"
    _atomic_write_json(out, report)
    return out


# ---------------------------------------------------------------------------
# D10: Promotion log
# ---------------------------------------------------------------------------

def append_promotion_log(miner_dir: Path, entry: dict) -> Path:
    """Append a promotion-chain decision to promotion_log.jsonl."""
    out = miner_dir / "promotion_log.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": _iso_now(),
        **dict(entry or {}),
    }
    with open(out, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return out


# ---------------------------------------------------------------------------
# D13: Economics — per-run cost rollup
# ---------------------------------------------------------------------------

def write_economics(miner_dir: Path, economics: dict) -> Path:
    """Write or overwrite economics.json rollup."""
    out = miner_dir / "economics.json"
    payload = {
        "updated_at": _iso_now(),
        **economics,
    }
    _atomic_write_json(out, payload)
    return out


def append_candidate_economics(miner_dir: Path, candidate_name: str,
                               iteration: int, econ: dict) -> Path:
    """Append per-candidate economics to economics_per_candidate.jsonl."""
    out = miner_dir / "economics_per_candidate.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": _iso_now(),
        "candidate": candidate_name,
        "iteration": iteration,
        **econ,
    }
    line = json.dumps(entry, ensure_ascii=False)
    with open(out, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    return out

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


def write_agent_trace(
    miner_dir: Path,
    *,
    iteration: int,
    candidate_idx: int,
    role: str,
    payload: Any,
) -> Path:
    """Write an agent trace JSON under agent_traces/iter_xxxx/cand_xx/."""
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

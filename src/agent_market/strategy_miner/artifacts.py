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
    code_path = iter_dir / f"{candidate.name}.py"
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
        },
        "backtest_summary": candidate.backtest_summary,
        "backtest_zip": str(zip_path) if zip_path is not None else None,
    }
    _atomic_write_json(out, payload)
    return out


def write_leaderboard(
    miner_dir: Path,
    state: MinerState,
    *,
    config: MinerConfig,
) -> Path:
    """Write leaderboard.json sorted by reward desc and filtered by constraints."""
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
                "reward": c.reward,
                "validation_passed": c.validation_passed,
                "constraints_ok": bool(getattr(c, "constraints_ok", True)),
                "constraint_violations": list(getattr(c, "constraint_violations", []) or []),
                "trades": summary.get("trades"),
                "winrate": summary.get("winrate"),
                "profit_total_pct": summary.get("profit_total_pct"),
                "max_drawdown_abs": summary.get("max_drawdown_abs"),
                "diagnosis": (c.diagnosis or "")[:200],
            }
        )

    eligible = [i for i in all_items if i.get("constraints_ok", True)]
    rejected = [i for i in all_items if not i.get("constraints_ok", True)]

    eligible.sort(key=lambda x: float(x.get("reward") or -1e9), reverse=True)
    rejected.sort(key=lambda x: float(x.get("reward") or -1e9), reverse=True)

    payload = {
        "run_id": state.run_id,
        "saved_at": _iso_now(),
        "config": {
            "min_trades": config.min_trades,
            "max_abs_drawdown": config.max_abs_drawdown,
            "min_winrate": config.min_winrate,
        },
        "best": eligible[0] if eligible else None,
        "items": eligible,
        "rejected": rejected,
        "all_count": len(all_items),
    }

    out = leaderboard_path(miner_dir)
    _atomic_write_json(out, payload)
    return out

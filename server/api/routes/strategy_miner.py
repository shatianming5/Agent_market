"""API routes for strategy miner: start, status, runs, candidates, approve, backtest."""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Body
from pydantic import BaseModel, Field

from agent_market import paths  # type: ignore

from ..errors import error, not_found, bad_request
from ...runtime import ROOT, jobs

router = APIRouter(prefix="/strategy-miner", tags=["strategy-miner"])

_RUN_ID_RE = re.compile(r"^[0-9a-f]{6,64}$")


def _validate_run_id(run_id: str) -> str | None:
    s = str(run_id or "").strip().lower()
    if not _RUN_ID_RE.fullmatch(s):
        return None
    return s


def _miner_dir(run_id: str) -> Path:
    return paths.run_dir(run_id) / "strategy_miner"


def _checkpoint_path(run_id: str) -> Path:
    return _miner_dir(run_id) / "checkpoint.json"


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_checkpoint(run_id: str) -> dict | None:
    return _load_json(_checkpoint_path(run_id))


def _list_candidate_snapshots(miner_dir: Path) -> list[dict]:
    root = miner_dir / "candidates"
    if not root.exists():
        return []

    items: list[dict] = []
    for meta_path in sorted(root.rglob("*.json")):
        data = _load_json(meta_path)
        if not isinstance(data, dict):
            continue
        cand = data.get("candidate")
        if not isinstance(cand, dict):
            continue
        items.append(
            {
                "name": cand.get("name"),
                "iteration": cand.get("iteration"),
                "reward": cand.get("reward"),
                "validation_passed": cand.get("validation_passed"),
                "diagnosis": (cand.get("diagnosis") or "")[:200],
                "code_path": (data.get("artifact") or {}).get("code_path"),
                "meta_path": str(meta_path),
            }
        )

    items.sort(key=lambda x: (int(x.get("iteration") or 0), str(x.get("name") or "")))
    return items


class StrategyMinerStartReq(BaseModel):
    config: str = Field(
        "configs/strategy_miner_default.json",
        description="Path to strategy miner config JSON",
    )
    max_iterations: Optional[int] = Field(None, description="Override max iterations")
    model: Optional[str] = Field(None, description="Override LLM model")
    resume: Optional[str] = Field(None, description="Path to checkpoint.json for resuming")
    run_id: Optional[str] = Field(None, description="Optional run id (hex)")


def _safe_resolve(user_path: str, allowed_root: Path) -> Path | None:
    """Resolve a user-supplied path and ensure it stays under allowed_root."""
    try:
        resolved = Path(user_path).resolve()
        allowed = allowed_root.resolve()
        if resolved == allowed or str(resolved).startswith(str(allowed) + os.sep):
            return resolved
    except (ValueError, OSError):
        pass
    return None


@router.post("/start")
def start_miner(req: StrategyMinerStartReq = Body(...)):
    """Start a strategy miner job in the background."""
    py = sys.executable
    script = str(ROOT / "scripts" / "strategy_miner.py")

    # Validate config path stays inside project root
    config_path = _safe_resolve(req.config, ROOT)
    if config_path is None or not config_path.exists():
        return bad_request("INVALID_CONFIG", f"Config path invalid or not found: {req.config}")

    # Validate resume path if provided
    if req.resume:
        resume_path = _safe_resolve(req.resume, ROOT)
        if resume_path is None or not resume_path.exists():
            return bad_request("INVALID_RESUME_PATH", f"Resume path invalid or not found: {req.resume}")

    run_id: Optional[str] = None
    if req.resume:
        run_id = None
    else:
        run_id = _validate_run_id(req.run_id or "") or uuid.uuid4().hex[:12]

    cmd = [py, script, "--config", str(config_path)]
    if run_id:
        cmd += ["--run-id", run_id]
    if req.max_iterations is not None:
        cmd += ["--max-iterations", str(req.max_iterations)]
    if req.model:
        cmd += ["--model", req.model]
    if req.resume:
        cmd += ["--resume", req.resume]

    env = os.environ.copy()
    job_id = jobs.start(cmd, cwd=ROOT, env=env, kind="strategy_miner", meta={"run_id": run_id})
    return {
        "status": "started",
        "job_id": job_id,
        "kind": "strategy_miner",
        "run_id": run_id,
        "cmd": cmd,
    }


@router.get("/status/{job_id}")
def miner_status(job_id: str):
    """Get status of a running strategy miner job (log-based best-effort parsing)."""
    res = jobs.logs(job_id, 0)
    if isinstance(res, dict) and res.get("error"):
        return not_found("JOB_NOT_FOUND", str(res.get("error")))
    running = bool(res.get("running"))
    code = res.get("code")
    raw_lines = [str(x) for x in (res.get("logs") or [])]

    # Extract progress from log lines
    current_iteration = 0
    current_phase = "unknown"
    best_score = None
    for line in raw_lines:
        ll = line.lower()
        if "=== iteration" in ll and "| phase" in ll:
            try:
                parts = line.split("Iteration")[1] if "Iteration" in line else ""
                if parts:
                    num = parts.strip().split()[0]
                    current_iteration = int(num)
                phase_part = line.split("Phase")[-1] if "Phase" in line else ""
                if phase_part:
                    current_phase = phase_part.strip().split()[0].lower()
            except (IndexError, ValueError):
                pass
        for marker in ("best_score=", "best_sharpe=", "best_reward="):
            if marker in ll:
                try:
                    val = line.split(marker)[1].split()[0].rstrip(",)")
                    best_score = float(val)
                except (IndexError, ValueError):
                    pass
                break

    return {
        "job_id": job_id,
        "running": running,
        "code": code,
        "iteration": current_iteration,
        "phase": current_phase,
        "best_score": best_score,
        "log_lines": len(raw_lines),
        "last_lines": raw_lines[-20:] if raw_lines else [],
    }


@router.get("/runs")
def miner_runs(limit: int = 10):
    """List strategy miner runs from standardized run storage."""
    runs_root = paths.runs_root()
    if not runs_root.exists():
        return {"items": [], "count": 0}

    cps: list[tuple[float, Path]] = []
    for child in runs_root.iterdir():
        if not child.is_dir():
            continue
        cp = child / "strategy_miner" / "checkpoint.json"
        if not cp.exists():
            continue
        try:
            cps.append((cp.stat().st_mtime, cp))
        except Exception:
            continue

    cps.sort(key=lambda x: x[0], reverse=True)

    items = []
    for _, cp in cps[: max(0, int(limit))]:
        data = _load_json(cp)
        if not isinstance(data, dict):
            continue
        run_id = str(data.get("run_id") or cp.parent.parent.name)
        miner_dir = cp.parent
        items.append(
            {
                "run_id": run_id,
                "phase": data.get("phase"),
                "iteration": data.get("iteration", 0),
                "best_score": data.get("best_score", data.get("best_reward")),
                "best_candidate": (
                    data["best_candidate"].get("name") if data.get("best_candidate") else None
                ),
                "candidates_count": len(data.get("candidates", [])),
                "checkpoint_path": str(cp),
                "has_proposal": (miner_dir / "proposal.json").exists(),
                "has_leaderboard": (miner_dir / "leaderboard.json").exists(),
            }
        )

    return {"items": items, "count": len(items)}


@router.get("/runs/{run_id}")
def miner_run_detail(run_id: str):
    """Get detailed results for a specific strategy miner run."""
    run_id_norm = _validate_run_id(run_id)
    if not run_id_norm:
        return bad_request("INVALID_RUN_ID", f"Invalid run_id: {run_id!r}")

    cp = _checkpoint_path(run_id_norm)
    if not cp.exists():
        return not_found("NOT_FOUND", f"No checkpoint found for run_id={run_id_norm}")

    data = _load_json(cp)
    if not isinstance(data, dict):
        return error("PARSE_ERROR", "Failed to parse checkpoint", status_code=500)

    miner_dir = _miner_dir(run_id_norm)

    kb_path = miner_dir / "knowledge_base.json"
    kb_data = _load_json(kb_path)

    proposal = _load_json(miner_dir / "proposal.json")
    leaderboard = _load_json(miner_dir / "leaderboard.json")
    snapshots = _list_candidate_snapshots(miner_dir)

    return {
        "run_id": data.get("run_id"),
        "phase": data.get("phase"),
        "iteration": data.get("iteration"),
        "best_score": data.get("best_score", data.get("best_reward")),
        "best_candidate": data.get("best_candidate"),
        "candidates": data.get("candidates", []),
        "history": data.get("history", []),
        "knowledge_base": kb_data,
        "proposal": proposal,
        "leaderboard": leaderboard,
        "candidate_snapshots": {"items": snapshots, "count": len(snapshots)},
    }


@router.get("/global-knowledge-base")
def miner_global_knowledge_base():
    path = paths.global_strategy_knowledge_base_path()
    data = _load_json(path)
    if data is None:
        return not_found("NOT_FOUND", "global strategy knowledge base not found")
    return data


@router.get("/runs/{run_id}/proposal")
def miner_run_proposal(run_id: str):
    run_id_norm = _validate_run_id(run_id)
    if not run_id_norm:
        return bad_request("INVALID_RUN_ID", f"Invalid run_id: {run_id!r}")

    path = _miner_dir(run_id_norm) / "proposal.json"
    data = _load_json(path)
    if data is None:
        return not_found("NOT_FOUND", f"proposal not found for run_id={run_id_norm}")
    return data


@router.get("/runs/{run_id}/leaderboard")
def miner_run_leaderboard(run_id: str):
    run_id_norm = _validate_run_id(run_id)
    if not run_id_norm:
        return bad_request("INVALID_RUN_ID", f"Invalid run_id: {run_id!r}")

    path = _miner_dir(run_id_norm) / "leaderboard.json"
    data = _load_json(path)
    if data is None:
        return not_found("NOT_FOUND", f"leaderboard not found for run_id={run_id_norm}")
    return data


@router.get("/runs/{run_id}/status")
def miner_run_status(run_id: str):
    """Checkpoint-based run status (preferred over log parsing)."""
    run_id_norm = _validate_run_id(run_id)
    if not run_id_norm:
        return bad_request("INVALID_RUN_ID", f"Invalid run_id: {run_id!r}")

    data = _load_checkpoint(run_id_norm)
    if data is None:
        return not_found("NOT_FOUND", f"No checkpoint found for run_id={run_id_norm}")

    return {
        "run_id": data.get("run_id"),
        "phase": data.get("phase"),
        "iteration": data.get("iteration"),
        "best_score": data.get("best_score", data.get("best_reward")),
        "candidates_count": len(data.get("candidates", [])),
    }


@router.get("/runs/{run_id}/candidates")
def miner_candidates(run_id: str):
    run_id_norm = _validate_run_id(run_id)
    if not run_id_norm:
        return bad_request("INVALID_RUN_ID", f"Invalid run_id: {run_id!r}")

    data = _load_checkpoint(run_id_norm)
    if data is None:
        return not_found("NOT_FOUND", f"No checkpoint found for run_id={run_id_norm}")

    candidates = data.get("candidates") or []
    items = []
    for c in candidates:
        if not isinstance(c, dict):
            continue
        items.append(
            {
                "name": c.get("name"),
                "iteration": c.get("iteration"),
                "reward": c.get("reward"),
                "validation_passed": c.get("validation_passed"),
                "strategy_path": c.get("strategy_path"),
                "diagnosis": (c.get("diagnosis") or "")[:200],
            }
        )
    return {"run_id": run_id_norm, "items": items, "count": len(items)}


class StrategyMinerApproveReq(BaseModel):
    candidate: Optional[str] = Field(None, description="Candidate strategy name. Defaults to best.")
    overwrite: bool = Field(False, description="Overwrite if destination exists")


@router.post("/runs/{run_id}/approve")
def approve_strategy(run_id: str, req: StrategyMinerApproveReq = Body(...)):
    """Approve a candidate strategy and copy it into user_data/strategies/."""
    run_id_norm = _validate_run_id(run_id)
    if not run_id_norm:
        return bad_request("INVALID_RUN_ID", f"Invalid run_id: {run_id!r}")

    data = _load_checkpoint(run_id_norm)
    if data is None:
        return not_found("NOT_FOUND", f"No checkpoint found for run_id={run_id_norm}")

    target_name = (req.candidate or "").strip() or None
    chosen = None

    if target_name:
        for c in data.get("candidates", []):
            if isinstance(c, dict) and str(c.get("name") or "").strip() == target_name:
                chosen = c
                break
    else:
        chosen = data.get("best_candidate") or (data.get("candidates") or [None])[-1]

    if not isinstance(chosen, dict):
        return error("NO_CANDIDATE", "No candidate strategy available")

    strat_path_raw = str(chosen.get("strategy_path") or "").strip()
    if not strat_path_raw:
        return error("NO_STRATEGY_PATH", "Candidate has no strategy_path")

    strat_path = Path(strat_path_raw)
    # Validate source path stays under runs root (prevent arbitrary file access)
    runs_root = paths.runs_root()
    if _safe_resolve(strat_path_raw, runs_root) is None and _safe_resolve(strat_path_raw, ROOT / "artifacts") is None:
        return bad_request("INVALID_PATH", f"Strategy path not under artifacts/: {strat_path}")
    if not strat_path.exists():
        return not_found("STRATEGY_NOT_FOUND", f"Strategy file not found: {strat_path}")

    dest_dir = paths.user_data_root() / "strategies"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / strat_path.name
    if dest_path.exists() and not req.overwrite:
        return error("DEST_EXISTS", f"Destination exists: {dest_path}")

    try:
        shutil.copy2(str(strat_path), str(dest_path))
    except Exception as exc:
        return error("COPY_FAILED", f"Failed to copy: {exc}")

    return {
        "status": "ok",
        "run_id": run_id_norm,
        "candidate": chosen.get("name"),
        "source": str(strat_path),
        "dest": str(dest_path),
    }


class StrategyMinerBacktestReq(BaseModel):
    candidate: Optional[str] = Field(None, description="Candidate strategy name. Defaults to best.")


@router.post("/runs/{run_id}/backtest")
def backtest_candidate(run_id: str, req: StrategyMinerBacktestReq = Body(...)):
    """Trigger a backtest+summary job for a candidate strategy."""
    run_id_norm = _validate_run_id(run_id)
    if not run_id_norm:
        return bad_request("INVALID_RUN_ID", f"Invalid run_id: {run_id!r}")

    # Validate checkpoint exists early.
    if _load_checkpoint(run_id_norm) is None:
        return not_found("NOT_FOUND", f"No checkpoint found for run_id={run_id_norm}")

    py = sys.executable
    script = str(ROOT / "scripts" / "strategy_miner_backtest.py")
    cmd = [py, script, "--run-id", run_id_norm]
    if req.candidate:
        cmd += ["--candidate", str(req.candidate)]

    env = os.environ.copy()
    job_id = jobs.start(
        cmd,
        cwd=ROOT,
        env=env,
        timeout_sec=7200,
        kind="strategy_miner_backtest",
        meta={"run_id": run_id_norm, "candidate": req.candidate},
    )
    return {
        "status": "started",
        "job_id": job_id,
        "kind": "strategy_miner_backtest",
        "run_id": run_id_norm,
        "cmd": cmd,
    }


# ---------------------------------------------------------------------------
# Backward-compatible aliases
# ---------------------------------------------------------------------------


@router.get("/results")
def miner_results(limit: int = 10):
    return miner_runs(limit=limit)


@router.get("/results/{run_id}")
def miner_result_detail(run_id: str):
    return miner_run_detail(run_id)

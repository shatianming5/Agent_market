"""API routes for strategy miner: start, status, results."""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Body
from pydantic import BaseModel, Field

from ..errors import error
from ...runtime import ROOT, jobs
from agent_market import paths  # type: ignore

router = APIRouter(prefix="/strategy-miner", tags=["strategy-miner"])


class StrategyMinerStartReq(BaseModel):
    config: str = Field(
        "configs/strategy_miner_default.json",
        description="Path to strategy miner config JSON",
    )
    max_iterations: Optional[int] = Field(None, description="Override max iterations")
    model: Optional[str] = Field(None, description="Override LLM model")
    resume: Optional[str] = Field(None, description="Path to checkpoint.json for resuming")


@router.post("/start")
def start_miner(req: StrategyMinerStartReq = Body(...)):
    """Start a strategy miner job in the background."""
    py = sys.executable
    script = str(ROOT / "scripts" / "strategy_miner.py")
    cmd = [py, script, "--config", req.config]
    if req.max_iterations is not None:
        cmd += ["--max-iterations", str(req.max_iterations)]
    if req.model:
        cmd += ["--model", req.model]
    if req.resume:
        cmd += ["--resume", req.resume]

    env = os.environ.copy()
    job_id = jobs.start(cmd, cwd=ROOT, env=env, kind="strategy_miner")
    return {
        "status": "started",
        "job_id": job_id,
        "kind": "strategy_miner",
        "cmd": cmd,
    }


@router.get("/status/{job_id}")
def miner_status(job_id: str):
    """Get status of a running strategy miner job."""
    res = jobs.logs(job_id, 0)
    if isinstance(res, dict) and res.get("error"):
        return error("JOB_NOT_FOUND", str(res.get("error")))
    running = bool(res.get("running"))
    code = res.get("code")
    raw_lines = [str(x) for x in (res.get("logs") or [])]

    # Extract progress from log lines
    current_iteration = 0
    current_phase = "unknown"
    best_reward = None
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
        if "best_reward=" in ll:
            try:
                val = line.split("best_reward=")[1].split()[0].rstrip(",)")
                best_reward = float(val)
            except (IndexError, ValueError):
                pass

    return {
        "job_id": job_id,
        "running": running,
        "code": code,
        "iteration": current_iteration,
        "phase": current_phase,
        "best_reward": best_reward,
        "log_lines": len(raw_lines),
        "last_lines": raw_lines[-20:] if raw_lines else [],
    }


@router.get("/results")
def miner_results(limit: int = 10):
    """List strategy miner run results from artifacts."""
    miner_root = paths.artifacts_root() / "strategy_miner"
    if not miner_root.exists():
        return {"items": [], "count": 0}

    items = []
    for child in sorted(miner_root.iterdir(), reverse=True):
        if not child.is_dir():
            continue
        cp = child / "checkpoint.json"
        if not cp.exists():
            continue
        try:
            data = json.loads(cp.read_text(encoding="utf-8"))
            items.append({
                "run_id": data.get("run_id", child.name),
                "phase": data.get("phase"),
                "iteration": data.get("iteration", 0),
                "best_reward": data.get("best_reward"),
                "best_candidate": (
                    data["best_candidate"].get("name")
                    if data.get("best_candidate")
                    else None
                ),
                "candidates_count": len(data.get("candidates", [])),
                "checkpoint_path": str(cp),
            })
        except Exception:
            continue
        if len(items) >= limit:
            break

    return {"items": items, "count": len(items)}


@router.get("/results/{run_id}")
def miner_result_detail(run_id: str):
    """Get detailed results for a specific strategy miner run."""
    run_id = str(run_id or "").strip()
    if not re.fullmatch(r"[0-9a-f]{6,64}", run_id):
        return error("INVALID_RUN_ID", f"Invalid run_id: {run_id!r}")

    cp = paths.artifacts_root() / "strategy_miner" / run_id / "checkpoint.json"
    if not cp.exists():
        return error("NOT_FOUND", f"No checkpoint found for run_id={run_id}")

    try:
        data = json.loads(cp.read_text(encoding="utf-8"))
    except Exception as e:
        return error("PARSE_ERROR", f"Failed to parse checkpoint: {e}")

    # Load knowledge base if exists
    kb_path = cp.parent / "knowledge_base.json"
    kb_data = None
    if kb_path.exists():
        try:
            kb_data = json.loads(kb_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    return {
        "run_id": data.get("run_id"),
        "phase": data.get("phase"),
        "iteration": data.get("iteration"),
        "best_reward": data.get("best_reward"),
        "best_candidate": data.get("best_candidate"),
        "candidates": data.get("candidates", []),
        "history": data.get("history", []),
        "knowledge_base": kb_data,
    }

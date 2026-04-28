from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Body

from agent_market import paths  # type: ignore

from ..errors import error
from ..models import BacktestReq, HyperoptReq
from ...runtime import ROOT, jobs
from ...job_manager import JobQueueFullError
from ._run_common import resolve_executable

router = APIRouter()


@router.post("/run/backtest")
def run_backtest(req: BacktestReq = Body(...)):
    try:
        cfg_path = paths.safe_resolve(req.config, allow_absolute=True)
    except ValueError as exc:
        return error("INVALID_PATH", str(exc))
    if not cfg_path.exists():
        return error("CONFIG_NOT_FOUND", f"Config file not found: {cfg_path}")

    spath: Optional[Path] = None
    if req.strategy_path:
        try:
            spath = paths.safe_resolve(req.strategy_path, allow_absolute=True)
        except ValueError as exc:
            return error("INVALID_PATH", str(exc))
        if not spath.exists():
            return error("STRATEGY_PATH_NOT_FOUND", f"Strategy path not found: {spath}")

    py = sys.executable
    base_cmd: list[str]
    job_cwd = ROOT
    wrapper = ROOT / "scripts" / "freqtrade_cli.py"
    if wrapper.exists():
        base_cmd = [py, str(wrapper)]
    else:
        binary = resolve_executable("freqtrade")
        if binary:
            base_cmd = [binary]
        else:
            # Avoid `python -m freqtrade` shadowing by running from a cwd that doesn't contain
            # the local `freqtrade/` directory.
            base_cmd = [py, "-m", "freqtrade"]
            user_data_dir = paths.user_data_root()
            if user_data_dir.exists():
                job_cwd = user_data_dir

    cmd = base_cmd + [
        "backtesting",
        "--userdir",
        str(paths.user_data_root()),
        "--config",
        str(cfg_path),
        "--strategy",
        req.strategy,
        "--strategy-path",
        str(spath) if spath else req.strategy_path,
        "--timerange",
        req.timerange,
        "--freqaimodel",
        req.freqaimodel,
    ]
    if req.export:
        cmd += ["--export", "trades", "--export-filename", req.export_filename]
    try:
        if (ROOT / "scripts" / "backtest_wrapper.py").exists():
            cmd = [py, str(ROOT / "scripts" / "backtest_wrapper.py"), "--"] + cmd
    except Exception:
        pass

    env = os.environ.copy()
    try:
        job_id = jobs.start(
            cmd,
            cwd=job_cwd,
            env=env,
            timeout_sec=7200,
            kind="backtest",
            meta={"timerange": req.timerange},
        )
    except JobQueueFullError as exc:
        return error("JOB_QUEUE_FULL", str(exc), status_code=429)
    return {"status": "started", "job_id": job_id, "kind": "backtest", "cmd": cmd}


@router.post("/run/hyperopt")
def run_hyperopt(req: HyperoptReq = Body(...)):
    cmd = [
        "freqtrade",
        "hyperopt",
        "--config",
        req.config,
        "--strategy",
        req.strategy,
        "--strategy-path",
        req.strategy_path,
        "--timerange",
        req.timerange,
        "--hyperopt-loss",
        req.hyperopt_loss,
        "--epochs",
        str(req.epochs),
        "--job-workers",
        str(req.job_workers),
    ]
    if req.spaces:
        cmd += ["--spaces"] + req.spaces.split()
    if req.freqaimodel:
        cmd += ["--freqaimodel", req.freqaimodel]

    env = os.environ.copy()
    try:
        job_id = jobs.start(
            cmd,
            cwd=ROOT,
            env=env,
            timeout_sec=1800,
            kind="hyperopt",
            meta={"timerange": req.timerange},
        )
    except JobQueueFullError as exc:
        return error("JOB_QUEUE_FULL", str(exc), status_code=429)
    return {"status": "started", "job_id": job_id, "kind": "hyperopt", "cmd": cmd}

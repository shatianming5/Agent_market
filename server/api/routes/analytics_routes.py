from __future__ import annotations

import os
import sys
import uuid

from fastapi import APIRouter, Body

from agent_market import paths  # type: ignore

from ..errors import error
from ..models import TCAReq
from ...runtime import ROOT, SRC, jobs
from ._run_common import load_latest_flow_run_id

router = APIRouter()


@router.post("/run/tca")
def run_tca(req: TCAReq = Body(...)):
    run_id = (str(req.run_id).strip().lower() if req.run_id else "") or load_latest_flow_run_id() or uuid.uuid4().hex[:12]

    script = ROOT / "scripts" / "tca_report.py"
    if not script.exists():
        return error("SCRIPT_NOT_FOUND", f"tca_report script not found: {script}")

    try:
        out_path = (
            paths.safe_resolve(req.out, allow_absolute=True)
            if req.out
            else (paths.run_dir(run_id) / "tca" / "tca_report.json").resolve()
        )
    except ValueError as exc:
        return error("INVALID_PATH", str(exc))

    cmd: list[str] = [
        sys.executable,
        str(script),
        "--run-id",
        run_id,
        "--results-dir",
        str(req.results_dir),
        "--out",
        str(out_path),
    ]
    if req.backtest_zip:
        try:
            zp = paths.safe_resolve(req.backtest_zip, allow_absolute=True)
        except ValueError as exc:
            return error("INVALID_PATH", str(exc))
        if not zp.exists():
            return error("BACKTEST_ZIP_NOT_FOUND", f"Backtest zip not found: {zp}")
        cmd += ["--backtest-zip", str(zp)]
    if req.html:
        cmd += ["--html"]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC)
    job_id = jobs.start(
        cmd,
        cwd=ROOT,
        env=env,
        timeout_sec=900,
        kind="tca",
        meta={"run_id": run_id},
    )
    return {"status": "started", "job_id": job_id, "kind": "tca", "cmd": cmd, "run_id": run_id}

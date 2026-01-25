from __future__ import annotations

import asyncio
import json as _jsonmod
import os
import sys
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Body, WebSocket
from starlette.responses import StreamingResponse

from ..errors import error
from ..models import FlowReq
from ...runtime import ROOT, jobs

router = APIRouter()


@router.post("/flow/run")
def run_flow(req: FlowReq = Body(...)):
    py = sys.executable
    script = str(ROOT / "scripts" / "agent_flow.py")
    cmd = [py, script, "--config", req.config]
    if req.steps:
        parts: list[str] = []
        if isinstance(req.steps, str):
            parts = [p for p in req.steps.split(" ") if p]
        elif isinstance(req.steps, list):
            parts = [str(p) for p in req.steps if p]
        if parts:
            cmd += ["--steps"] + parts

    env = os.environ.copy()
    job_id = jobs.start(cmd, cwd=ROOT, env=env, kind="flow")
    return {"status": "started", "job_id": job_id, "kind": "flow", "cmd": cmd}


@router.get("/flow/progress/{job_id}")
def flow_progress(job_id: str, steps: Optional[str] = None):
    """Best-effort progress estimation for flow jobs by scanning logs.

    Returns per-step status: pending/running/ok/failed with fine-grained phase/percent.
    Heuristics:
      - Prefer explicit markers: [FLOW] STEP_START/PHASE/STEP_OK/STEP_FAIL
      - During execute, estimate percent by (a) epoch X/Y, (b) detected progress % tokens,
        (c) log growth compared to expected span per-step as fallback.
    """
    res = jobs.logs(job_id, 0)
    if isinstance(res, dict) and res.get("error"):
        return error("JOB_NOT_FOUND", str(res.get("error")))
    running = bool(res.get("running"))
    code = res.get("code")
    returncode = res.get("returncode")
    raw_lines = [str(x) for x in (res.get("logs") or [])]
    lines = [x.lower() for x in raw_lines]
    steps_list = [
        s
        for s in (steps.split(",") if steps else ["feature", "expression", "ml", "rl", "backtest"])
        if s
    ]

    markers = {"start": set(), "ok": set(), "fail": set()}
    phases: dict[str, str] = {}
    start_idx: dict[str, int] = {}
    expected_span = {
        "feature": 120,
        "expression": 180,
        "ml": 320,
        "rl": 600,
        "backtest": 400,
    }

    import re as _re  # local alias

    r_epoch = _re.compile(r"(?:epoch|iter)\s*(\d+)\s*/\s*(\d+)")
    r_pct = _re.compile(r"(\d{1,3})\s*%")

    for idx, ln in enumerate(lines):
        if "[flow]" in ln and "step_" in ln:
            if "step_start" in ln:
                for nm in ["feature", "expression", "ml", "rl", "backtest"]:
                    if f" {nm}" in ln:
                        markers["start"].add(nm)
                        if nm not in start_idx:
                            start_idx[nm] = idx
            elif "step_ok" in ln:
                for nm in ["feature", "expression", "ml", "rl", "backtest"]:
                    if f" {nm}" in ln:
                        markers["ok"].add(nm)
            elif "step_fail" in ln:
                for nm in ["feature", "expression", "ml", "rl", "backtest"]:
                    if f" {nm}" in ln:
                        markers["fail"].add(nm)
        if "[flow]" in ln and "phase " in ln:
            for nm in ["feature", "expression", "ml", "rl", "backtest"]:
                if f" phase {nm} " in ln:
                    if " prepare" in ln:
                        phases[nm] = "prepare"
                        if nm not in start_idx:
                            start_idx[nm] = idx
                    elif " execute" in ln:
                        phases[nm] = "execute"
                        if nm not in start_idx:
                            start_idx[nm] = idx
                    elif " summarize" in ln:
                        phases[nm] = "summarize"

    kw = {
        "feature": [
            "feature_agent",
            "freqai_feature_agent",
            "--pairs",
            "freqai features",
            "feature file",
        ],
        "expression": [
            "expression_agent",
            "freqai_expression_agent",
            "--llm",
            "llm",
            "expressions",
        ],
        "ml": [
            "training",
            "model",
            "lightgbm",
            "xgboost",
            "catboost",
            "pytorch",
            "training_summary.json",
        ],
        "rl": ["rl", "ppo", "stable-baselines3", "train_rl"],
        "backtest": [
            "backtesting",
            "backtest",
            "backtest-result",
            "results",
            "trades-",
        ],
    }
    last_seen = -1
    total_lines = len(lines)
    for idx, name in enumerate(steps_list):
        keys = kw.get(name, [name])
        if any(any(k in ln for k in keys) for ln in lines):
            last_seen = max(last_seen, idx)

    items: list[dict] = []
    for idx, name in enumerate(steps_list):
        if name in markers["fail"]:
            status = "failed"
            phase = phases.get(name) or "execute"
        elif name in markers["ok"]:
            status = "ok"
            phase = "summarize"
        elif name in markers["start"]:
            status = "running"
            phase = phases.get(name) or "execute"
        else:
            if last_seen < 0:
                status = "running" if running and idx == 0 else "pending"
                phase = "prepare" if status == "running" else None
            else:
                if idx < last_seen:
                    status = "ok"
                    phase = "summarize"
                elif idx == last_seen:
                    if running:
                        status = "running"
                        phase = "execute"
                    else:
                        status = "ok" if (returncode == 0 and code == "OK") else "failed"
                        phase = "summarize" if status == "ok" else "execute"
                else:
                    status = "pending"
                    phase = None

        percent = 0
        if status == "pending":
            percent = 0
        elif status == "running":
            if phase == "prepare":
                percent = 5
            elif phase == "execute":
                base = 20
                sidx = start_idx.get(name, total_lines)
                exec_span = max(1, total_lines - sidx)
                expected = expected_span.get(name, 200)
                ratio = min(1.0, exec_span / max(1, expected))
                epoch_ratio = None
                for raw in raw_lines[sidx:]:
                    m = r_epoch.search(str(raw).lower())
                    if m:
                        cur, tot = m.groups()
                        try:
                            tot_i = max(1, int(tot))
                            cur_i = min(max(0, int(cur)), tot_i)
                            epoch_ratio = cur_i / tot_i
                            break
                        except Exception:
                            pass
                pct_token = None
                for raw in raw_lines[max(0, total_lines - 30) :]:
                    m2 = r_pct.search(str(raw))
                    if m2:
                        try:
                            v = int(m2.group(1))
                            if 0 <= v <= 100:
                                pct_token = v / 100.0
                        except Exception:
                            pass
                dyn = (
                    epoch_ratio
                    if epoch_ratio is not None
                    else (pct_token if pct_token is not None else ratio)
                )
                percent = int(base + 75 * max(0.0, min(1.0, dyn)))
            elif phase == "summarize":
                percent = 98
            else:
                percent = 10
        elif status == "ok":
            percent = 100
        elif status == "failed":
            percent = max(
                20, int(100 * (start_idx.get(name, 0) / max(1, total_lines)))
            )
        items.append({"name": name, "status": status, "phase": phase, "percent": percent})
    return {"job_id": job_id, "running": running, "code": code, "steps": items}


@router.get("/flow/stream/{job_id}")
def flow_stream(job_id: str, steps: Optional[str] = None):
    steps_list = [
        s
        for s in (steps.split(",") if steps else ["feature", "expression", "ml", "rl", "backtest"])
        if s
    ]

    def _gen():
        last_next = 0
        init_logs = jobs.logs(job_id, last_next)
        if isinstance(init_logs, dict) and init_logs.get("error"):
            err = error("JOB_NOT_FOUND", str(init_logs.get("error")))
            yield f"event: progress\\ndata: {_jsonmod.dumps(err, ensure_ascii=False)}\\n\\n"
            return
        last_next = int(init_logs.get("next") or last_next)
        payload = flow_progress(job_id, steps=",".join(steps_list))
        payload["delta"] = init_logs.get("logs") or []
        payload["next"] = last_next
        yield f"event: progress\\ndata: {_jsonmod.dumps(payload, ensure_ascii=False)}\\n\\n"
        while True:
            data = jobs.logs(job_id, last_next)
            if isinstance(data, dict) and data.get("error"):
                err = error("JOB_NOT_FOUND", str(data.get("error")))
                yield f"event: progress\\ndata: {_jsonmod.dumps(err, ensure_ascii=False)}\\n\\n"
                break
            last_next = int(data.get("next") or last_next)
            payload = flow_progress(job_id, steps=",".join(steps_list))
            payload["delta"] = data.get("logs") or []
            payload["next"] = last_next
            yield f"event: progress\\ndata: {_jsonmod.dumps(payload, ensure_ascii=False)}\\n\\n"
            if not data.get("running"):
                break
            time.sleep(1.0)

    return StreamingResponse(_gen(), media_type="text/event-stream")


@router.websocket("/flow/ws/{job_id}")
async def flow_ws(websocket: WebSocket, job_id: str):
    await websocket.accept()
    try:
        last_next = 0
        while True:
            data = jobs.logs(job_id, last_next)
            if isinstance(data, dict) and data.get("error"):
                await websocket.send_json(error("JOB_NOT_FOUND", str(data.get("error"))))
                break
            last_next = int(data.get("next") or last_next)
            payload = flow_progress(job_id)
            await websocket.send_json(payload)
            if not data.get("running"):
                break
            await asyncio.sleep(1.0)
    except Exception:
        try:
            await websocket.close()
        except Exception:
            pass

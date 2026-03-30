from __future__ import annotations

import asyncio
import json as _jsonmod
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Body, WebSocket
from starlette.responses import StreamingResponse

from ..errors import error
from ..models import FlowReq
from ...runtime import ROOT, jobs
from agent_market import paths  # type: ignore

def _validate_config_path(config_path: str, repo_root) -> str:
    """Ensure config path is within the repo root (safe containment check)."""
    from pathlib import Path
    p = Path(config_path).resolve()
    root = Path(repo_root).resolve()
    try:
        p.relative_to(root)  # raises ValueError if not a sub-path
    except ValueError:
        raise ValueError(f"Config path {p} is outside repo root {root}")
    return str(p)


router = APIRouter()


def _resolve_under_root(path: str | Path) -> Optional[Path]:
    try:
        p = Path(path)
        if not p.is_absolute():
            p = (ROOT / p).resolve()
        else:
            p = p.resolve()
        root = ROOT.resolve()
        if p == root or root in p.parents:
            return p
    except Exception:
        return None
    return None


def _check_path(path: Optional[str]) -> dict:
    if not path:
        return {"path": path, "exists": False}
    resolved = _resolve_under_root(path)
    if resolved is None:
        return {"path": path, "exists": False, "error": "path_outside_root"}
    return {"path": str(resolved.relative_to(ROOT.resolve())), "exists": resolved.exists()}


def _iso_to_epoch(value: Optional[str]) -> float:
    if not value:
        return 0.0
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0.0


def _pick_latest_existing_path(paths: list[str]) -> Optional[str]:
    candidates: list[tuple[float, str]] = []
    for path in paths:
        if not path:
            continue
        resolved = _resolve_under_root(path)
        if resolved is None or not resolved.exists():
            continue
        try:
            candidates.append((resolved.stat().st_mtime, path))
        except Exception:
            candidates.append((0.0, path))
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    return None


@router.get("/flow/run-meta/latest")
def flow_run_meta_latest():
    meta_path = paths.run_meta_latest_path()
    if not meta_path.exists():
        return error("NOT_FOUND", f"run_meta not found: {meta_path}")
    try:
        payload = _jsonmod.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return error("PARSE_ERROR", f"Failed to parse {meta_path}: {exc}")

    artifacts = payload.get("artifacts") or {}
    payload["checks"] = {
        "config": _check_path((payload.get("config") or {}).get("path")),
        "feature_output": _check_path(artifacts.get("feature_output")),
        "capture_manifest": _check_path(artifacts.get("capture_manifest")),
        "capture_match_path": _check_path(artifacts.get("capture_match_path")),
        "capture_level2_path": _check_path(artifacts.get("capture_level2_path")),
        "lob_state_parquet": _check_path(artifacts.get("lob_state_parquet")),
        "rebuild_report": _check_path(artifacts.get("rebuild_report")),
        "micro_feature_parquet": _check_path(artifacts.get("micro_feature_parquet")),
        "micro_feature_manifest": _check_path(artifacts.get("micro_feature_manifest")),
        "portfolio_weights": _check_path(artifacts.get("portfolio_weights")),
        "portfolio_report": _check_path(artifacts.get("portfolio_report")),
        "expression_output": _check_path(artifacts.get("expression_output")),
        "factor_spec_json": _check_path(artifacts.get("factor_spec_json")),
        "factor_ast_json": _check_path(artifacts.get("factor_ast_json")),
        "factor_expression_txt": _check_path(artifacts.get("factor_expression_txt")),
        "factor_expression_json": _check_path(artifacts.get("factor_expression_json")),
        "factor_eval_meta": _check_path(artifacts.get("factor_eval_meta")),
        "factor_scores_json": _check_path(artifacts.get("factor_scores_json")),
        "factor_pareto_csv": _check_path(artifacts.get("factor_pareto_csv")),
        "bundle_zip": _check_path(artifacts.get("bundle_zip")),
        "bundle_manifest": _check_path(artifacts.get("bundle_manifest")),
        "feedback_summary": _check_path(artifacts.get("feedback_summary")),
        "training_summaries": [
            _check_path(p) for p in (artifacts.get("training_summaries") or []) if p
        ],
        "backtest_zips": [_check_path(p) for p in (artifacts.get("backtest_zips") or []) if p],
        "tca_report": _check_path(artifacts.get("tca_report")),
        "tca_html": _check_path(artifacts.get("tca_html")),
    }
    payload["run_meta_path"] = str(meta_path.relative_to(ROOT.resolve()))
    return payload


@router.get("/flow/run-meta/{run_id}")
def flow_run_meta(run_id: str):
    run_id = str(run_id or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{8,64}", run_id):
        return error("INVALID_RUN_ID", f"Invalid run_id: {run_id!r}")
    meta_path = paths.run_meta_path(run_id)
    if not meta_path.exists():
        return error("NOT_FOUND", f"run_meta not found: {meta_path}")
    try:
        payload = _jsonmod.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return error("PARSE_ERROR", f"Failed to parse {meta_path}: {exc}")

    artifacts = payload.get("artifacts") or {}
    payload["checks"] = {
        "config": _check_path((payload.get("config") or {}).get("path")),
        "feature_output": _check_path(artifacts.get("feature_output")),
        "capture_manifest": _check_path(artifacts.get("capture_manifest")),
        "capture_match_path": _check_path(artifacts.get("capture_match_path")),
        "capture_level2_path": _check_path(artifacts.get("capture_level2_path")),
        "lob_state_parquet": _check_path(artifacts.get("lob_state_parquet")),
        "rebuild_report": _check_path(artifacts.get("rebuild_report")),
        "micro_feature_parquet": _check_path(artifacts.get("micro_feature_parquet")),
        "micro_feature_manifest": _check_path(artifacts.get("micro_feature_manifest")),
        "portfolio_weights": _check_path(artifacts.get("portfolio_weights")),
        "portfolio_report": _check_path(artifacts.get("portfolio_report")),
        "expression_output": _check_path(artifacts.get("expression_output")),
        "factor_spec_json": _check_path(artifacts.get("factor_spec_json")),
        "factor_ast_json": _check_path(artifacts.get("factor_ast_json")),
        "factor_expression_txt": _check_path(artifacts.get("factor_expression_txt")),
        "factor_expression_json": _check_path(artifacts.get("factor_expression_json")),
        "factor_eval_meta": _check_path(artifacts.get("factor_eval_meta")),
        "factor_scores_json": _check_path(artifacts.get("factor_scores_json")),
        "factor_pareto_csv": _check_path(artifacts.get("factor_pareto_csv")),
        "bundle_zip": _check_path(artifacts.get("bundle_zip")),
        "bundle_manifest": _check_path(artifacts.get("bundle_manifest")),
        "feedback_summary": _check_path(artifacts.get("feedback_summary")),
        "training_summaries": [
            _check_path(p) for p in (artifacts.get("training_summaries") or []) if p
        ],
        "backtest_zips": [_check_path(p) for p in (artifacts.get("backtest_zips") or []) if p],
        "tca_report": _check_path(artifacts.get("tca_report")),
        "tca_html": _check_path(artifacts.get("tca_html")),
    }
    payload["run_meta_path"] = str(meta_path.relative_to(ROOT.resolve()))
    return payload


def _load_portfolio_report_from_run_meta(meta_path: Path) -> dict:
    if not meta_path.exists():
        return error("NOT_FOUND", f"run_meta not found: {meta_path}")
    try:
        meta = _jsonmod.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return error("PARSE_ERROR", f"Failed to parse {meta_path}: {exc}")
    artifacts = meta.get("artifacts") or {}
    report_path = artifacts.get("portfolio_report")
    resolved = _resolve_under_root(str(report_path)) if report_path else None
    if resolved is None or not resolved.exists():
        return error("NOT_FOUND", f"portfolio_report not found for run_meta: {meta_path}")
    try:
        payload = _jsonmod.loads(resolved.read_text(encoding="utf-8"))
    except Exception as exc:
        return error("PARSE_ERROR", f"Failed to parse {resolved}: {exc}")
    payload["_run_id"] = meta.get("run_id")
    try:
        payload["_report_path"] = str(resolved.relative_to(ROOT.resolve()))
    except Exception:
        payload["_report_path"] = str(resolved)
    return payload


def _load_json_artifact_from_run_meta(meta_path: Path, key: str) -> dict:
    if not meta_path.exists():
        return error("NOT_FOUND", f"run_meta not found: {meta_path}")
    try:
        meta = _jsonmod.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return error("PARSE_ERROR", f"Failed to parse {meta_path}: {exc}")
    artifacts = meta.get("artifacts") or {}
    artifact_path = artifacts.get(key)
    resolved = _resolve_under_root(str(artifact_path)) if artifact_path else None
    if resolved is None or not resolved.exists():
        return error("NOT_FOUND", f"{key} not found for run_meta: {meta_path}")
    try:
        payload = _jsonmod.loads(resolved.read_text(encoding="utf-8"))
    except Exception as exc:
        return error("PARSE_ERROR", f"Failed to parse {resolved}: {exc}")
    payload["_run_id"] = meta.get("run_id")
    try:
        payload["_path"] = str(resolved.relative_to(ROOT.resolve()))
    except Exception:
        payload["_path"] = str(resolved)
    return payload


@router.get("/flow/portfolio/latest")
def flow_portfolio_latest():
    meta_path = paths.run_meta_latest_path()
    return _load_portfolio_report_from_run_meta(meta_path)


@router.get("/flow/portfolio/{run_id}")
def flow_portfolio(run_id: str):
    run_id = str(run_id or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{8,64}", run_id):
        return error("INVALID_RUN_ID", f"Invalid run_id: {run_id!r}")
    meta_path = paths.run_meta_path(run_id)
    return _load_portfolio_report_from_run_meta(meta_path)


@router.get("/flow/tca/latest")
def flow_tca_latest():
    meta_path = paths.run_meta_latest_path()
    return _load_json_artifact_from_run_meta(meta_path, "tca_report")


@router.get("/flow/tca/{run_id}")
def flow_tca(run_id: str):
    run_id = str(run_id or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{8,64}", run_id):
        return error("INVALID_RUN_ID", f"Invalid run_id: {run_id!r}")
    meta_path = paths.run_meta_path(run_id)
    return _load_json_artifact_from_run_meta(meta_path, "tca_report")


@router.get("/flow/micro-feature/latest")
def flow_micro_feature_latest():
    meta_path = paths.run_meta_latest_path()
    return _load_json_artifact_from_run_meta(meta_path, "micro_feature_manifest")


@router.get("/flow/micro-feature/{run_id}")
def flow_micro_feature(run_id: str):
    run_id = str(run_id or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{8,64}", run_id):
        return error("INVALID_RUN_ID", f"Invalid run_id: {run_id!r}")
    meta_path = paths.run_meta_path(run_id)
    return _load_json_artifact_from_run_meta(meta_path, "micro_feature_manifest")


@router.get("/flow/factor-scores/latest")
def flow_factor_scores_latest():
    meta_path = paths.run_meta_latest_path()
    return _load_json_artifact_from_run_meta(meta_path, "factor_scores_json")


@router.get("/flow/factor-scores/{run_id}")
def flow_factor_scores(run_id: str):
    run_id = str(run_id or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{8,64}", run_id):
        return error("INVALID_RUN_ID", f"Invalid run_id: {run_id!r}")
    meta_path = paths.run_meta_path(run_id)
    return _load_json_artifact_from_run_meta(meta_path, "factor_scores_json")


@router.get("/flow/runs/list")
def flow_runs_list(limit: int = 20):
    """List recent flow runs from artifacts/runs/**/run_meta.json."""
    try:
        lim = int(limit)
    except Exception:
        lim = 20
    lim = max(1, min(lim, 200))

    runs_dir = paths.runs_root()
    if not runs_dir.exists():
        return {"items": [], "count": 0, "limit": lim}

    root_resolved = ROOT.resolve()
    meta_paths: list[Path] = []
    for child in runs_dir.iterdir():
        if not child.is_dir():
            continue
        meta_path = (child / "run_meta.json").resolve()
        if meta_path.exists():
            meta_paths.append(meta_path)

    rows: list[tuple[float, dict]] = []
    for meta_path in meta_paths:
        try:
            payload = _jsonmod.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as _exc:
            import logging; logging.getLogger(__name__).debug("Skipped: %s", _exc)
            continue

        run_id = str(payload.get("run_id") or meta_path.parent.name).strip().lower()
        status = payload.get("status")
        started_at = payload.get("started_at")
        ended_at = payload.get("ended_at")
        cfg = payload.get("config") or {}
        artifacts = payload.get("artifacts") or {}

        bt_raw = artifacts.get("backtest_zips") or []
        bt_paths = [str(p) for p in bt_raw if p] if isinstance(bt_raw, list) else []
        bt_latest = _pick_latest_existing_path(bt_paths) or (bt_paths[-1] if bt_paths else None)
        bt_latest_name = None
        if bt_latest:
            try:
                bt_latest_name = str(bt_latest).split("/")[-1]
            except Exception:
                bt_latest_name = str(bt_latest)

        ts = _iso_to_epoch(ended_at) or _iso_to_epoch(started_at)
        try:
            ts = ts or meta_path.stat().st_mtime
        except Exception as _exc:
            import logging; logging.getLogger(__name__).debug("Suppressed: %s", _exc)

        try:
            rel_meta_path = str(meta_path.relative_to(root_resolved))
        except Exception:
            rel_meta_path = str(meta_path)

        rows.append(
            (
                float(ts or 0.0),
                {
                    "run_id": run_id,
                    "status": status,
                    "started_at": started_at,
                    "ended_at": ended_at,
                    "config_sha256": cfg.get("sha256"),
                    "config_path": cfg.get("path"),
                    "latest_backtest_zip": _check_path(bt_latest),
                    "latest_backtest_zip_name": bt_latest_name,
                    "run_meta_path": rel_meta_path,
                },
            )
        )

    rows.sort(key=lambda x: x[0], reverse=True)
    items = [row for _, row in rows[:lim]]
    return {"items": items, "count": len(items), "limit": lim}


@router.post("/flow/run")
def run_flow(req: FlowReq = Body(...)):
    py = sys.executable
    script = str(ROOT / "scripts" / "agent_flow.py")
    validated_config = _validate_config_path(req.config, ROOT)
    cmd = [py, script, "--config", validated_config]
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
                        except Exception as _exc:
                            import logging; logging.getLogger(__name__).debug("Suppressed: %s", _exc)
                pct_token = None
                for raw in raw_lines[max(0, total_lines - 30) :]:
                    m2 = r_pct.search(str(raw))
                    if m2:
                        try:
                            v = int(m2.group(1))
                            if 0 <= v <= 100:
                                pct_token = v / 100.0
                        except Exception as _exc:
                            import logging; logging.getLogger(__name__).debug("Suppressed: %s", _exc)
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
        except Exception as _exc:
            import logging; logging.getLogger(__name__).debug("Suppressed: %s", _exc)

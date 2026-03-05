from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Body

from agent_market import paths  # type: ignore

from ..errors import error
from ..models import RLTrainReq, TrainReq
from ..validators import validate_timeframe
from ...runtime import ROOT, jobs

router = APIRouter()


@router.post("/run/rl_train")
def run_rl_train(req: RLTrainReq = Body(...)):
    py = sys.executable
    script = str(ROOT / "scripts" / "train_rl.py")
    cmd = [py, script, "--config", req.config]

    env = os.environ.copy()
    job_id = jobs.start(cmd, cwd=ROOT, env=env, timeout_sec=7200, kind="rl_train")
    return {"status": "started", "job_id": job_id, "kind": "rl_train", "cmd": cmd}


@router.post("/run/train")
def run_train(req: TrainReq = Body(...)):
    py = sys.executable
    script = str(ROOT / "scripts" / "train_pipeline.py")
    cfg_path: Optional[Path] = None
    if req.config:
        try:
            cfg_path = paths.safe_resolve(req.config, allow_absolute=True)
        except ValueError as exc:
            return error("INVALID_PATH", str(exc))
        if not cfg_path.exists():
            return error("CONFIG_NOT_FOUND", f"Config file not found: {cfg_path}")
    elif req.config_obj:
        tmp_dir = (paths.user_data_root() / "tmp").resolve()
        tmp_dir.mkdir(parents=True, exist_ok=True)
        from datetime import datetime  # noqa: PLC0415

        cfg = req.config_obj if isinstance(req.config_obj, dict) else None
        if not cfg:
            return error("INVALID_BODY", "config_obj must be a JSON object")
        required_top = {"data", "model", "training", "output"}
        missing = [k for k in required_top if k not in cfg]
        if missing:
            return error("MISSING_KEYS", f"config_obj missing keys: {missing}")
        data = cfg.get("data") or {}
        model = cfg.get("model") or {}
        training = cfg.get("training") or {}
        output = cfg.get("output") or {}
        for key in ("feature_file", "data_dir", "exchange", "timeframe"):
            if key not in data:
                return error("MISSING_DATA_KEY", f"data.{key} required")
        if not validate_timeframe(str(data.get("timeframe"))):
            return error(
                "INVALID_TIMEFRAME",
                f"Invalid data.timeframe: {data.get('timeframe')}",
            )
        pairs = data.get("pairs")
        if pairs is not None:
            if not isinstance(pairs, list) or not all(isinstance(p, str) for p in pairs):
                return error("INVALID_PAIRS", "data.pairs must be a list of strings")
        try:
            ff = paths.safe_resolve(data.get("feature_file"), allow_absolute=True)
            if not ff.exists():
                return error("FEATURE_FILE_NOT_FOUND", f"Feature file not found: {ff}")
        except ValueError as exc:
            return error("INVALID_PATH", str(exc))
        except Exception:
            return error("FEATURE_FILE_INVALID", f"Invalid feature_file: {data.get('feature_file')}")
        if not model.get("name"):
            return error("MODEL_NAME_REQUIRED", "model.name required")
        vr = float(training.get("validation_ratio", 0.2))
        if not (0.0 <= vr <= 0.9):
            return error(
                "INVALID_VALIDATION_RATIO",
                f"training.validation_ratio out of range: {vr}",
            )
        out_dir = output.get("model_dir") or "artifacts/models/auto"
        try:
            paths.safe_resolve(out_dir, allow_absolute=True).mkdir(parents=True, exist_ok=True)
        except Exception:
            return error("MODEL_DIR_INVALID", f"Cannot create output.model_dir: {out_dir}")
        cfg_path = tmp_dir / f"train_inline_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
        cfg_path.write_text(
            json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    else:
        return error(
            "MISSING_CONFIG",
            "Either 'config' (path) or 'config_obj' (inline JSON) must be provided",
        )
    cmd = [py, script, "--config", str(cfg_path)]

    env = os.environ.copy()
    job_id = jobs.start(cmd, cwd=ROOT, env=env, kind="train")
    return {"status": "started", "job_id": job_id, "kind": "train", "cmd": cmd}

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Body

from ..errors import error
from ..models import (
    BacktestReq,
    ExpressionReq,
    FeatureReq,
    HyperoptReq,
    RLTrainReq,
    TrainReq,
)
from ..validators import validate_pairs_string, validate_timeframe
from ...runtime import ROOT, SRC, jobs

router = APIRouter()


@router.post("/run/expression")
def run_expression(req: ExpressionReq = Body(...)):
    cfg_path = Path(req.config)
    if not cfg_path.is_absolute():
        cfg_path = (ROOT / cfg_path).resolve()
    if not cfg_path.exists():
        return error("CONFIG_NOT_FOUND", f"Config file not found: {cfg_path}")

    ff_path = Path(req.feature_file)
    if not ff_path.is_absolute():
        ff_path = (ROOT / ff_path).resolve()
    if not ff_path.exists():
        return error("FEATURE_FILE_NOT_FOUND", f"Feature file not found: {ff_path}")

    if not validate_timeframe(req.timeframe):
        return error("INVALID_TIMEFRAME", f"Invalid timeframe: {req.timeframe}")
    try:
        if req.llm_count is not None and int(req.llm_count) <= 0:
            return error(
                "INVALID_LLM_COUNT", f"llm_count must be > 0, got {req.llm_count}"
            )
    except Exception:
        return error(
            "INVALID_LLM_COUNT", f"llm_count must be an integer, got {req.llm_count}"
        )
    if not req.llm_model:
        return error("INVALID_LLM_MODEL", "llm_model must be provided")

    py = sys.executable
    wrapper = ROOT / "scripts" / "expr_agent_wrapper.py"
    script_path = ROOT / "scripts" / "freqai_expression_agent.py"
    if wrapper.exists():
        cmd = [py, str(wrapper)]
    elif script_path.exists():
        cmd = [py, str(script_path)]
    else:
        return error(
            "SCRIPT_NOT_FOUND",
            "Expression agent script not found. Expected scripts/expr_agent_wrapper.py or scripts/freqai_expression_agent.py",
        )
    feature_file_arg = str(ff_path)

    cmd += [
        "--config",
        str(cfg_path),
        "--feature-file",
        feature_file_arg,
        "--output",
        req.output,
        "--timeframe",
        req.timeframe,
        "--llm-model",
        req.llm_model,
        "--llm-count",
        str(req.llm_count),
        "--llm-loops",
        str(req.llm_loops),
        "--llm-timeout",
        str(req.llm_timeout),
        "--feedback-top",
        str(req.feedback_top),
    ]
    if req.feedback:
        cmd += ["--feedback", req.feedback]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC)

    llm_key = req.llm_api_key or os.environ.get("LLM_API_KEY")
    if llm_key:
        env["LLM_API_KEY"] = llm_key
    if os.environ.get("LLM_BASE_URL"):
        env["LLM_BASE_URL"] = os.environ["LLM_BASE_URL"]
    if os.environ.get("LLM_MODEL"):
        env["LLM_MODEL"] = os.environ["LLM_MODEL"]
    job_id = jobs.start(
        cmd,
        cwd=ROOT,
        env=env,
        timeout_sec=900,
        kind="expression",
        meta={"timeframe": req.timeframe},
    )
    return {"status": "started", "job_id": job_id, "kind": "expression", "cmd": cmd}


@router.post("/run/backtest")
def run_backtest(req: BacktestReq = Body(...)):
    cfg_path = Path(req.config)
    if not cfg_path.is_absolute():
        cfg_path = (ROOT / cfg_path).resolve()
    if not cfg_path.exists():
        return error("CONFIG_NOT_FOUND", f"Config file not found: {cfg_path}")

    spath: Optional[Path] = None
    if req.strategy_path:
        spath = Path(req.strategy_path)
        if not spath.is_absolute():
            spath = (ROOT / spath).resolve()
        if not spath.exists():
            return error("STRATEGY_PATH_NOT_FOUND", f"Strategy path not found: {spath}")

    py = sys.executable
    binary = "freqtrade"
    if shutil.which(binary):
        cmd = [
            binary,
            "backtesting",
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
    else:
        cmd = [
            py,
            "-m",
            "freqtrade",
            "backtesting",
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
    job_id = jobs.start(
        cmd,
        cwd=ROOT,
        env=env,
        timeout_sec=7200,
        kind="backtest",
        meta={"timerange": req.timerange},
    )
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
    job_id = jobs.start(
        cmd,
        cwd=ROOT,
        env=env,
        timeout_sec=1800,
        kind="hyperopt",
        meta={"timerange": req.timerange},
    )
    return {"status": "started", "job_id": job_id, "kind": "hyperopt", "cmd": cmd}


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
        cfg_path = (ROOT / req.config) if not Path(req.config).is_absolute() else Path(req.config)
        if not cfg_path.exists():
            return error("CONFIG_NOT_FOUND", f"Config file not found: {cfg_path}")
    elif req.config_obj:
        tmp_dir = ROOT / "user_data" / "tmp"
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
            ff = Path(data.get("feature_file"))
            if not ff.is_absolute():
                ff = (ROOT / ff).resolve()
            if not ff.exists():
                return error("FEATURE_FILE_NOT_FOUND", f"Feature file not found: {ff}")
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
            Path(out_dir).mkdir(parents=True, exist_ok=True)
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


@router.post("/run/feature")
def run_feature(req: FeatureReq = Body(...)):
    py = sys.executable
    cfg_path = Path(req.config)
    if not cfg_path.is_absolute():
        cfg_path = (ROOT / cfg_path).resolve()
    if not cfg_path.exists():
        return error("CONFIG_NOT_FOUND", f"Config file not found: {cfg_path}")
    if not validate_timeframe(req.timeframe):
        return error("INVALID_TIMEFRAME", f"Invalid timeframe: {req.timeframe}")
    script_path = ROOT / "scripts" / "freqai_feature_agent.py"
    if not script_path.exists():
        return error(
            "SCRIPT_NOT_FOUND",
            "Feature agent script not found. Expected scripts/freqai_feature_agent.py",
        )
    cmd = [py, str(script_path)]
    cmd += [
        "--config",
        str(cfg_path),
        "--output",
        req.output,
        "--timeframe",
        req.timeframe,
    ]
    if req.pairs:
        ok_pairs, parsed = validate_pairs_string(req.pairs)
        if not ok_pairs:
            return error("INVALID_PAIRS", f"Invalid pairs string: {req.pairs}")
        if parsed:
            cmd += ["--pairs"] + parsed

    env = os.environ.copy()
    job_id = jobs.start(
        cmd,
        cwd=ROOT,
        env=env,
        kind="feature",
        meta={"timeframe": req.timeframe},
    )
    return {"status": "started", "job_id": job_id, "kind": "feature", "cmd": cmd}

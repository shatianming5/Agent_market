from __future__ import annotations
import logging

import json
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Body

from ..errors import error
from ..models import (
    BacktestReq,
    CaptureReq,
    ExpressionReq,
    FactorCompileReq,
    FactorEvalReq,
    FeatureReq,
    HyperoptReq,
    LobRebuildReq,
    MicroFeatureReq,
    RLTrainReq,
    TCAReq,
    TrainReq,
)
from ..validators import validate_pairs_string, validate_timeframe
from ...runtime import ROOT, SRC, jobs

router = APIRouter()
from agent_market import paths  # type: ignore  # noqa: E402


def _ensure_src_on_path() -> None:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))

def _resolve_executable(name: str) -> Optional[str]:
    found = shutil.which(name)
    if found:
        return found
    try:
        sibling = Path(sys.executable).with_name(name)
        if sibling.exists():
            return str(sibling)
    except Exception as _exc:
        logging.getLogger(__name__).debug("Suppressed: %s", _exc)
    for cand in [
        ROOT / ".venv" / "bin" / name,
        ROOT / ".venv" / "Scripts" / (name + ".exe"),
        ROOT / "venv" / "bin" / name,
        ROOT / "venv" / "Scripts" / (name + ".exe"),
    ]:
        if cand.exists():
            return str(cand)
    return None


def _load_latest_flow_run_id() -> Optional[str]:
    meta_path = paths.run_meta_latest_path()
    if not meta_path.exists():
        return None
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    run_id = str(payload.get("run_id") or "").strip().lower()
    if run_id and all(ch in "0123456789abcdef" for ch in run_id) and 8 <= len(run_id) <= 64:
        return run_id
    return None


def _parse_csv(value: Optional[str]) -> list[str]:
    if not value:
        return []
    raw = str(value).replace(",", " ").split()
    return [x.strip() for x in raw if x.strip()]


def _detect_kucoin_level2_sequence_gaps(
    *,
    level2_path: Path,
    snapshot_sequence: int,
    max_updates: int = 5000,
) -> dict:
    """
    Best-effort KuCoin L2 sequence gap precheck.

    This is intentionally lightweight and only scans sequenceStart/sequenceEnd fields.
    """

    import gzip

    last_seq = int(snapshot_sequence) if snapshot_sequence else None
    gaps: list[dict] = []
    updates = 0
    with gzip.open(Path(level2_path), "rt", encoding="utf-8") as fh:
        for line in fh:
            if updates >= int(max_updates):
                break
            s = line.strip()
            if not s:
                continue
            try:
                msg = json.loads(s)
            except Exception:
                continue
            if not isinstance(msg, dict) or msg.get("type") != "message":
                continue
            data = msg.get("data") if isinstance(msg.get("data"), dict) else {}
            seq_start = data.get("sequenceStart")
            seq_end = data.get("sequenceEnd")
            if seq_start is None or seq_end is None:
                continue
            try:
                seq_start_i = int(seq_start)
                seq_end_i = int(seq_end)
            except Exception:
                continue
            if last_seq is not None and seq_start_i != last_seq + 1:
                gaps.append({"expected_start": last_seq + 1, "got_start": seq_start_i, "got_end": seq_end_i})
                if len(gaps) >= 5:
                    break
            last_seq = seq_end_i
            updates += 1
    return {"count": int(len(gaps)), "examples": gaps, "scanned_updates": int(updates)}


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
    cfg_path = paths.resolve_repo_path(req.config)
    if not cfg_path.exists():
        return error("CONFIG_NOT_FOUND", f"Config file not found: {cfg_path}")

    spath: Optional[Path] = None
    if req.strategy_path:
        spath = paths.resolve_repo_path(req.strategy_path)
        if not spath.exists():
            return error("STRATEGY_PATH_NOT_FOUND", f"Strategy path not found: {spath}")

    py = sys.executable
    base_cmd: list[str]
    job_cwd = ROOT
    wrapper = ROOT / "scripts" / "freqtrade_cli.py"
    if wrapper.exists():
        base_cmd = [py, str(wrapper)]
    else:
        binary = _resolve_executable("freqtrade")
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
    except Exception as _exc:
        logging.getLogger(__name__).debug("Suppressed: %s", _exc)

    env = os.environ.copy()
    job_id = jobs.start(
        cmd,
        cwd=job_cwd,
        env=env,
        timeout_sec=7200,
        kind="backtest",
        meta={"timerange": req.timerange},
    )
    return {"status": "started", "job_id": job_id, "kind": "backtest", "cmd": cmd}


@router.post("/run/capture")
def run_capture(req: CaptureReq = Body(...)):
    exchange = str(req.exchange or "").strip().lower()
    if exchange != "kucoin":
        return error("UNSUPPORTED_EXCHANGE", f"Unsupported exchange: {exchange!r}")

    channels = [c.lower() for c in _parse_csv(req.channels)]
    allowed_channels = {"match", "level2"}
    unknown = sorted(set(channels) - allowed_channels)
    if not channels:
        return error("INVALID_CHANNELS", "channels must not be empty")
    if unknown:
        return error(
            "INVALID_CHANNELS",
            f"Unsupported channels: {unknown}. Allowed: {sorted(allowed_channels)}",
        )

    duration_sec = int(req.duration_sec or 0)
    if duration_sec <= 0:
        duration_sec = 60
    duration_sec = max(1, min(duration_sec, 24 * 3600))

    fixture_path = Path(req.fixture) if req.fixture else None
    if fixture_path is not None:
        fixture_path = paths.resolve_repo_path(fixture_path)
    if fixture_path is not None and not fixture_path.exists():
        return error("FIXTURE_NOT_FOUND", f"Fixture not found: {fixture_path}")

    out_dir = Path(req.out_dir) if req.out_dir else None
    if out_dir is not None:
        out_dir = paths.resolve_repo_path(out_dir)

    script = ROOT / "scripts" / "micro_capture.py"
    if not script.exists():
        return error("SCRIPT_NOT_FOUND", f"micro_capture script not found: {script}")

    cmd: list[str] = [
        sys.executable,
        str(script),
        "--exchange",
        exchange,
        "--channels",
        ",".join(channels),
    ]
    if out_dir is not None:
        cmd += ["--out-dir", str(out_dir)]
    if fixture_path is not None:
        cmd += ["--fixture", str(fixture_path)]
        timeout_sec = 120
    else:
        symbols = _parse_csv(req.symbols)
        if not symbols:
            return error("INVALID_SYMBOLS", "symbols must not be empty")
        cmd += ["--symbols", " ".join(symbols), "--duration-sec", str(duration_sec)]
        timeout_sec = int(duration_sec) + 120

    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC)
    job_id = jobs.start(
        cmd,
        cwd=ROOT,
        env=env,
        timeout_sec=timeout_sec,
        kind="capture",
        meta={"exchange": exchange, "channels": channels},
    )
    return {"status": "started", "job_id": job_id, "kind": "capture", "cmd": cmd}


@router.post("/run/lob_rebuild")
def run_lob_rebuild(req: LobRebuildReq = Body(...)):
    exchange = str(req.exchange or "").strip().lower()
    if not exchange:
        exchange = "kucoin"
    if exchange != "kucoin":
        return error("UNSUPPORTED_EXCHANGE", f"Unsupported exchange: {exchange!r}")

    capture_dir = paths.resolve_repo_path(req.capture_dir)
    if not capture_dir.exists():
        return error(
            "DATA_NOT_FOUND",
            f"Capture dir not found: {capture_dir}",
            legacy_code="CAPTURE_DIR_NOT_FOUND",
            details={"resource": "capture_dir", "path": str(capture_dir)},
        )

    snapshot_path = paths.resolve_repo_path(req.snapshot)
    if not snapshot_path.exists():
        return error(
            "DATA_NOT_FOUND",
            f"Snapshot not found: {snapshot_path}",
            legacy_code="SNAPSHOT_NOT_FOUND",
            details={"resource": "snapshot", "path": str(snapshot_path)},
        )

    symbol = str(req.symbol or "").strip()
    if not symbol:
        return error("INVALID_SYMBOL", "symbol must not be empty")

    try:
        depth = int(req.depth or 0)
    except Exception:
        return error("INVALID_DEPTH", f"depth must be an integer, got {req.depth!r}")
    if depth <= 0:
        depth = 20
    depth = max(1, min(depth, 200))

    out_dir = (
        paths.resolve_repo_path(req.out_dir)
        if req.out_dir
        else (paths.artifacts_root() / "lob_rebuild" / uuid.uuid4().hex[:12]).resolve()
    )

    # Precheck: ensure the capture directory contains level2 data and has no sequence gaps.
    level2_path = (capture_dir / "level2.ndjson.gz").resolve()
    if not level2_path.exists():
        return error(
            "DATA_NOT_FOUND",
            f"level2.ndjson.gz not found under capture_dir: {capture_dir}",
            legacy_code="DATA_NOT_FOUND",
            details={"resource": "level2_ndjson_gz", "path": str(level2_path)},
        )
    try:
        snap = json.loads(snapshot_path.read_text(encoding="utf-8-sig"))
        snap_seq = int((snap.get("sequence") if isinstance(snap, dict) else None) or 0)
    except Exception:
        snap_seq = 0
    if snap_seq:
        gaps = _detect_kucoin_level2_sequence_gaps(level2_path=level2_path, snapshot_sequence=snap_seq)
        if int(gaps.get("count") or 0) > 0:
            return error(
                "LOB_SEQUENCE_GAP",
                f"KuCoin level2 sequence gap detected (count={gaps.get('count')})",
                legacy_code="LOB_SEQUENCE_GAP",
                details={"level2_path": str(level2_path), "snapshot_sequence": int(snap_seq), **gaps},
            )

    script = ROOT / "scripts" / "lob_rebuild.py"
    if not script.exists():
        return error("SCRIPT_NOT_FOUND", f"lob_rebuild script not found: {script}")

    cmd: list[str] = [
        sys.executable,
        str(script),
        "--capture-dir",
        str(capture_dir),
        "--snapshot",
        str(snapshot_path),
        "--symbol",
        symbol,
        "--depth",
        str(depth),
        "--out-dir",
        str(out_dir),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC)
    job_id = jobs.start(
        cmd,
        cwd=ROOT,
        env=env,
        timeout_sec=600,
        kind="lob_rebuild",
        meta={"exchange": exchange, "symbol": symbol, "depth": depth},
    )
    return {"status": "started", "job_id": job_id, "kind": "lob_rebuild", "cmd": cmd, "out_dir": str(out_dir)}


@router.post("/run/micro_feature")
def run_micro_feature(req: MicroFeatureReq = Body(...)):
    cfg_path = paths.resolve_repo_path(req.config)
    if not cfg_path.exists():
        return error("CONFIG_NOT_FOUND", f"Config file not found: {cfg_path}")

    if req.timeframe and not validate_timeframe(str(req.timeframe)):
        return error("INVALID_TIMEFRAME", f"Invalid timeframe: {req.timeframe}")

    run_id = (str(req.run_id).strip().lower() if req.run_id else "") or uuid.uuid4().hex[:12]
    out_dir = paths.resolve_repo_path(req.out_dir) if req.out_dir else (paths.run_dir(run_id) / "micro_feature")

    script = ROOT / "scripts" / "micro_features.py"
    if not script.exists():
        return error("SCRIPT_NOT_FOUND", f"micro_features script not found: {script}")

    cmd: list[str] = [
        sys.executable,
        str(script),
        "--config",
        str(cfg_path),
        "--run-id",
        run_id,
        "--out-dir",
        str(out_dir),
    ]
    if req.timeframe:
        cmd += ["--timeframe", str(req.timeframe)]
    if req.timerange:
        cmd += ["--timerange", str(req.timerange)]
    if req.data_dir:
        cmd += ["--data-dir", str(req.data_dir)]
    if req.pairs:
        cmd += ["--pairs"] + [str(p) for p in req.pairs if str(p).strip()]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC)
    job_id = jobs.start(
        cmd,
        cwd=ROOT,
        env=env,
        timeout_sec=900,
        kind="micro_feature",
        meta={"run_id": run_id},
    )
    return {"status": "started", "job_id": job_id, "kind": "micro_feature", "cmd": cmd, "run_id": run_id}


@router.post("/run/tca")
def run_tca(req: TCAReq = Body(...)):
    run_id = (str(req.run_id).strip().lower() if req.run_id else "") or _load_latest_flow_run_id() or uuid.uuid4().hex[:12]

    script = ROOT / "scripts" / "tca_report.py"
    if not script.exists():
        return error("SCRIPT_NOT_FOUND", f"tca_report script not found: {script}")

    out_path = (
        paths.resolve_repo_path(req.out)
        if req.out
        else (paths.run_dir(run_id) / "tca" / "tca_report.json").resolve()
    )

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
        zp = paths.resolve_repo_path(req.backtest_zip)
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


@router.post("/run/factor_compile")
def run_factor_compile(req: FactorCompileReq = Body(...)):
    run_id = (str(req.run_id).strip().lower() if req.run_id else "") or uuid.uuid4().hex[:12]
    out_dir = paths.resolve_repo_path(req.out_dir) if req.out_dir else (paths.run_dir(run_id) / "factor_compile")
    out_dir.mkdir(parents=True, exist_ok=True)

    spec_path: Optional[Path] = None
    if isinstance(req.spec_path, str) and req.spec_path.strip():
        p = paths.resolve_repo_path(req.spec_path)
        if not p.exists():
            return error(
                "DATA_NOT_FOUND",
                f"FactorSpec not found: {p}",
                legacy_code="SPEC_NOT_FOUND",
                details={"resource": "factor_spec", "path": str(p)},
            )
        spec_path = p
    elif isinstance(req.spec, dict):
        spec_path = (out_dir / "input_factor_spec.json").resolve()
        spec_path.write_text(
            json.dumps(req.spec, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    else:
        return error("INVALID_FACTOR_SPEC", "Provide spec (object) or spec_path (file path)")

    # Preflight validation for stable error codes (plan.md 7.2).
    _ensure_src_on_path()
    try:
        from agent_market.factor_compiler import FactorSpec  # type: ignore
        from agent_market.factor_compiler.checks import check_complexity  # type: ignore
        from agent_market.factor_compiler.checks.data_schema import (  # type: ignore
            check_literal_param_ranges,
            check_operator_whitelist,
            collect_var_names,
        )
        from agent_market.factor_compiler.checks.time_safety import check_time_safety  # type: ignore
        from agent_market.factor_compiler.dsl.operators import compile_to_expression_engine  # type: ignore
        from agent_market.factor_compiler.dsl.types import infer_expr_type  # type: ignore
        from agent_market.freqai.expression_engine import (  # type: ignore
            ExpressionValidationError,
            safe_eval_expression,
        )
        import pandas as pd  # type: ignore
    except Exception as exc:  # pragma: no cover
        return error("SCRIPT_FAILED", f"Failed to load Factor Compiler modules: {exc}")

    try:
        spec_payload = req.spec if isinstance(req.spec, dict) else json.loads(spec_path.read_text(encoding="utf-8"))
        spec = FactorSpec.model_validate(spec_payload)
    except Exception as exc:
        return error("INVALID_FACTOR_SPEC", f"FactorSpec validation failed: {exc}")

    op_res = check_operator_whitelist(spec.expr)
    if not op_res.ok:
        return error(
            "UNKNOWN_OPERATOR",
            op_res.message,
            legacy_code=op_res.code,
            details=op_res.to_dict(),
        )
    for r in check_literal_param_ranges(spec.expr):
        if not r.ok:
            return error(
                "INVALID_FACTOR_SPEC",
                r.message,
                legacy_code=r.code,
                details=r.to_dict(),
            )

    inferred = infer_expr_type(spec.expr)
    if getattr(inferred, "kind", None) != "series":
        return error(
            "TYPECHECK_FAILED",
            "Factor expression must evaluate to a time series",
            details={"inferred_kind": getattr(inferred, "kind", None), "inferred_dtype": getattr(inferred, "dtype", None)},
        )

    comp_res = check_complexity(spec.expr, spec.constraints.complexity_budget)
    if not comp_res.ok:
        return error(
            "COMPLEXITY_BUDGET_EXCEEDED",
            comp_res.message,
            legacy_code=comp_res.code,
            details=comp_res.to_dict(),
        )

    for r in check_time_safety(spec.expr, min_delay_ms=spec.constraints.min_delay_ms):
        if not r.ok:
            return error(
                "LOOKAHEAD_DETECTED",
                r.message,
                legacy_code=r.code,
                details=r.to_dict(),
            )

    try:
        compiled_expr = compile_to_expression_engine(spec.expr)
        vars_used = collect_var_names(spec.expr)
        cols = sorted(vars_used) if vars_used else ["close"]
        df = pd.DataFrame({c: [100.0, 101.0, 102.0, 103.0] for c in cols})
        safe_eval_expression(compiled_expr, df)
    except ExpressionValidationError as exc:
        msg = str(exc)
        if "no lookahead" in msg or "shift second argument must be >= 0" in msg:
            return error("LOOKAHEAD_DETECTED", msg, legacy_code="LOOKAHEAD_DETECTED")
        if "Unknown name" in msg:
            return error("DATA_NOT_FOUND", msg, legacy_code="DATA_NOT_FOUND")
        return error("UNKNOWN_OPERATOR", msg, legacy_code="UNKNOWN_OPERATOR")
    except Exception as exc:
        return error("TYPECHECK_FAILED", f"Factor expression validation failed: {exc}")

    script = ROOT / "scripts" / "factor_compile.py"
    if not script.exists():
        return error("SCRIPT_NOT_FOUND", f"factor_compile script not found: {script}")

    cmd: list[str] = [
        sys.executable,
        str(script),
        "--spec",
        str(spec_path),
        "--out-dir",
        str(out_dir),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC)
    job_id = jobs.start(
        cmd,
        cwd=ROOT,
        env=env,
        timeout_sec=900,
        kind="factor_compile",
        meta={"run_id": run_id},
    )
    return {"status": "started", "job_id": job_id, "kind": "factor_compile", "cmd": cmd, "run_id": run_id, "out_dir": str(out_dir)}


@router.post("/run/factor_eval")
def run_factor_eval(req: FactorEvalReq = Body(...)):
    run_id = (str(req.run_id).strip().lower() if req.run_id else "") or uuid.uuid4().hex[:12]
    out_dir = paths.resolve_repo_path(req.out_dir) if req.out_dir else (paths.run_dir(run_id) / "factor_eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    expr_path: Optional[Path] = None
    if isinstance(req.expression_path, str) and req.expression_path.strip():
        p = paths.resolve_repo_path(req.expression_path)
        if not p.exists():
            return error(
                "DATA_NOT_FOUND",
                f"Expression file not found: {p}",
                legacy_code="EXPRESSION_NOT_FOUND",
                details={"resource": "expression", "path": str(p)},
            )
        expr_path = p
    elif isinstance(req.expression, str) and req.expression.strip():
        expr_path = (out_dir / "input_expression.txt").resolve()
        expr_path.write_text(req.expression.strip(), encoding="utf-8")
    else:
        compiled = (paths.run_dir(run_id) / "factor_compile" / "compiled_expression.txt").resolve()
        if compiled.exists():
            expr_path = compiled
        else:
            return error(
                "DATA_NOT_FOUND",
                "Provide expression/expression_path or run factor_compile first",
                legacy_code="EXPRESSION_NOT_FOUND",
                details={"resource": "expression", "path": str(compiled)},
            )

    script = ROOT / "scripts" / "factor_eval.py"
    if not script.exists():
        return error("SCRIPT_NOT_FOUND", f"factor_eval script not found: {script}")

    # Preflight validation for stable error codes (plan.md 7.2).
    _ensure_src_on_path()
    try:
        from agent_market.freqai.expression_engine import (  # type: ignore
            ExpressionValidationError,
            safe_eval_expression,
        )
        import pandas as pd  # type: ignore
    except Exception as exc:  # pragma: no cover
        return error("SCRIPT_FAILED", f"Failed to load ExpressionEngine: {exc}")

    try:
        expr_text = expr_path.read_text(encoding="utf-8").strip()
        df = pd.DataFrame({"close": [100.0, 101.0, 102.0, 103.0]})
        safe_eval_expression(expr_text, df)
    except ExpressionValidationError as exc:
        msg = str(exc)
        if "no lookahead" in msg or "shift second argument must be >= 0" in msg:
            return error("LOOKAHEAD_DETECTED", msg, legacy_code="LOOKAHEAD_DETECTED")
        if "Function call not allowed" in msg or "Operator not allowed" in msg:
            return error("UNKNOWN_OPERATOR", msg, legacy_code="UNKNOWN_OPERATOR")
        if "Unknown name" in msg:
            return error("DATA_NOT_FOUND", msg, legacy_code="DATA_NOT_FOUND")
        return error("TYPECHECK_FAILED", msg, legacy_code="TYPECHECK_FAILED")
    except Exception as exc:
        return error("TYPECHECK_FAILED", f"Expression validation failed: {exc}", legacy_code="TYPECHECK_FAILED")

    cmd: list[str] = [
        sys.executable,
        str(script),
        "--expression-path",
        str(expr_path),
        "--out-dir",
        str(out_dir),
        "--rows",
        str(int(req.rows)),
        "--seed",
        str(int(req.seed)),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC)
    job_id = jobs.start(
        cmd,
        cwd=ROOT,
        env=env,
        timeout_sec=900,
        kind="factor_eval",
        meta={"run_id": run_id},
    )
    return {"status": "started", "job_id": job_id, "kind": "factor_eval", "cmd": cmd, "run_id": run_id, "out_dir": str(out_dir)}


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
        cfg_path = paths.resolve_repo_path(req.config)
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
            ff = Path(data.get("feature_file"))
            if not ff.is_absolute():
                ff = paths.resolve_repo_path(ff)
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
            paths.resolve_repo_path(out_dir).mkdir(parents=True, exist_ok=True)
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
        cfg_path = paths.resolve_repo_path(cfg_path)
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

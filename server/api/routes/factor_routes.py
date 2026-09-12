from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Body

from agent_market import paths  # type: ignore

from ..errors import error, not_found
from ..models import ExpressionReq, FactorCompileReq, FactorEvalReq, FeatureReq
from ..validators import validate_pairs_string, validate_timeframe
from ...runtime import ROOT, SRC, jobs
from ...job_manager import JobQueueFullError
from ._run_common import ensure_src_on_path

router = APIRouter()


@router.post("/run/expression")
def run_expression(req: ExpressionReq = Body(...)):
    try:
        cfg_path = paths.safe_resolve(req.config, allow_absolute=True)
    except ValueError as exc:
        return error("INVALID_PATH", str(exc))
    if not cfg_path.exists():
        return not_found("CONFIG_NOT_FOUND", f"Config file not found: {cfg_path}")

    try:
        ff_path = paths.safe_resolve(req.feature_file, allow_absolute=True)
    except ValueError as exc:
        return error("INVALID_PATH", str(exc))
    if not ff_path.exists():
        return not_found("FEATURE_FILE_NOT_FOUND", f"Feature file not found: {ff_path}")

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
    try:
        job_id = jobs.start(
            cmd,
            cwd=ROOT,
            env=env,
            timeout_sec=900,
            kind="expression",
            meta={"timeframe": req.timeframe},
        )
    except JobQueueFullError as exc:
        return error("JOB_QUEUE_FULL", str(exc), status_code=429)
    return {"status": "started", "job_id": job_id, "kind": "expression", "cmd": cmd}


@router.post("/run/factor_compile")
def run_factor_compile(req: FactorCompileReq = Body(...)):
    run_id = (str(req.run_id).strip().lower() if req.run_id else "") or uuid.uuid4().hex[:12]
    try:
        out_dir = paths.safe_resolve(req.out_dir, allow_absolute=True) if req.out_dir else (paths.run_dir(run_id) / "factor_compile")
    except ValueError as exc:
        return error("INVALID_PATH", str(exc))
    out_dir.mkdir(parents=True, exist_ok=True)

    spec_path: Optional[Path] = None
    if isinstance(req.spec_path, str) and req.spec_path.strip():
        try:
            p = paths.safe_resolve(req.spec_path, allow_absolute=True)
        except ValueError as exc:
            return error("INVALID_PATH", str(exc))
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
    ensure_src_on_path()
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
    try:
        job_id = jobs.start(
            cmd,
            cwd=ROOT,
            env=env,
            timeout_sec=900,
            kind="factor_compile",
            meta={"run_id": run_id},
        )
    except JobQueueFullError as exc:
        return error("JOB_QUEUE_FULL", str(exc), status_code=429)
    return {"status": "started", "job_id": job_id, "kind": "factor_compile", "cmd": cmd, "run_id": run_id, "out_dir": str(out_dir)}


@router.post("/run/factor_eval")
def run_factor_eval(req: FactorEvalReq = Body(...)):
    run_id = (str(req.run_id).strip().lower() if req.run_id else "") or uuid.uuid4().hex[:12]
    try:
        out_dir = paths.safe_resolve(req.out_dir, allow_absolute=True) if req.out_dir else (paths.run_dir(run_id) / "factor_eval")
    except ValueError as exc:
        return error("INVALID_PATH", str(exc))
    out_dir.mkdir(parents=True, exist_ok=True)

    expr_path: Optional[Path] = None
    if isinstance(req.expression_path, str) and req.expression_path.strip():
        try:
            p = paths.safe_resolve(req.expression_path, allow_absolute=True)
        except ValueError as exc:
            return error("INVALID_PATH", str(exc))
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
    ensure_src_on_path()
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
    try:
        job_id = jobs.start(
            cmd,
            cwd=ROOT,
            env=env,
            timeout_sec=900,
            kind="factor_eval",
            meta={"run_id": run_id},
        )
    except JobQueueFullError as exc:
        return error("JOB_QUEUE_FULL", str(exc), status_code=429)
    return {"status": "started", "job_id": job_id, "kind": "factor_eval", "cmd": cmd, "run_id": run_id, "out_dir": str(out_dir)}


@router.post("/run/feature")
def run_feature(req: FeatureReq = Body(...)):
    py = sys.executable
    try:
        cfg_path = paths.safe_resolve(req.config, allow_absolute=True)
    except ValueError as exc:
        return error("INVALID_PATH", str(exc))
    if not cfg_path.exists():
        return not_found("CONFIG_NOT_FOUND", f"Config file not found: {cfg_path}")
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
    try:
        job_id = jobs.start(
            cmd,
            cwd=ROOT,
            env=env,
            kind="feature",
            meta={"timeframe": req.timeframe},
        )
    except JobQueueFullError as exc:
        return error("JOB_QUEUE_FULL", str(exc), status_code=429)
    return {"status": "started", "job_id": job_id, "kind": "feature", "cmd": cmd}

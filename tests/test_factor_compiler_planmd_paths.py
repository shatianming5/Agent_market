from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def test_factor_compiler_planmd_modules_import_and_minimal_behavior() -> None:
    from agent_market.factor_compiler import FactorSpec
    from agent_market.factor_compiler.checks.data_schema import check_data_schema
    from agent_market.factor_compiler.checks.time_safety import check_time_safety
    from agent_market.factor_compiler.checks.unit_test_gen import generate_pytest_for_factor_spec
    from agent_market.factor_compiler.dsl.ast import ExprNode
    from agent_market.factor_compiler.dsl.types import infer_expr_type
    from agent_market.factor_compiler.scoring.novelty import ast_similarity_max, corr_to_library_max
    from agent_market.factor_compiler.scoring.stability import rolling_ic_std

    expr = ExprNode(op="shift", args=[ExprNode(op="var", args=["close"]), 1])
    inferred = infer_expr_type(expr, var_types={"close": infer_expr_type(ExprNode(op="var", args=["close"]))})
    assert inferred.kind in {"scalar", "series"}

    # data schema check
    results = check_data_schema(expr, available_columns={"close"})
    assert any(r.ok for r in results)

    # time safety check (positive shift ok)
    ts = check_time_safety(expr)
    assert ts and all(hasattr(r, "ok") for r in ts)

    # unit test generator returns python snippet
    spec = FactorSpec(
        name="x",
        expr=ExprNode(op="var", args=["close"]),
        meta={"timeframe": "1h", "universe": ["BTC/USDT"], "data_sources": ["ohlcv"]},
    )
    code = generate_pytest_for_factor_spec(spec)
    assert "FactorSpec" in code and "safe_eval_expression" in code

    # novelty/stability helpers
    s = pd.Series([1.0, 2.0, 3.0, 4.0])
    lib = {"a": pd.Series([1.0, 2.0, 2.0, 4.0])}
    corr = corr_to_library_max(s, lib)
    assert corr is None or (0.0 <= corr <= 1.0)
    assert ast_similarity_max("abc", ["def", "abc"]) == 1.0

    y = pd.Series(np.linspace(0.0, 1.0, 200))
    rstd = rolling_ic_std(y, y.shift(1).fillna(0.0), window=50)
    assert rstd is None or np.isfinite(rstd)


def test_factor_compiler_prompts_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    sys_path = root / "src" / "agent_market" / "factor_compiler" / "prompts"
    system_md = sys_path / "factor_spec.system.md"
    fewshot = sys_path / "factor_spec.fewshot.json"

    assert system_md.exists()
    assert fewshot.exists()
    payload = json.loads(fewshot.read_text(encoding="utf-8"))
    assert isinstance(payload.get("examples"), list) and payload["examples"]


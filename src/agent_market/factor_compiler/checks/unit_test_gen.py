from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from agent_market.factor_compiler import FactorSpec
from agent_market.factor_compiler.dsl.operators import compile_to_expression_engine


def generate_pytest_for_factor_spec(spec: FactorSpec, *, test_name: str = "test_factor_spec_compiles") -> str:
    """
    Generate a minimal pytest snippet for a FactorSpec.

    This is intentionally simple: it validates the spec, compiles to an ExpressionEngine
    expression, and ensures it can be evaluated on a tiny synthetic dataframe.
    """

    spec_json = json.dumps(spec.model_dump(mode="json"), ensure_ascii=False, indent=2)
    return "\n".join(
        [
            "import pandas as pd",
            "",
            "from agent_market.factor_compiler import FactorSpec",
            "from agent_market.factor_compiler.dsl.operators import compile_to_expression_engine",
            "from agent_market.freqai.expression_engine import safe_eval_expression",
            "",
            f"def {test_name}():",
            f"    spec = FactorSpec.model_validate({spec_json})",
            "    expr = compile_to_expression_engine(spec.expr)",
            "    df = pd.DataFrame({'close': [100.0, 101.0, 102.0, 103.0]})",
            "    out = safe_eval_expression(expr, df)",
            "    assert len(out) == len(df)",
            "",
        ]
    )


def write_pytest_for_factor_spec(spec: FactorSpec, path: Path) -> Path:
    p = Path(path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(generate_pytest_for_factor_spec(spec), encoding="utf-8")
    return p


__all__ = ["generate_pytest_for_factor_spec", "write_pytest_for_factor_spec"]


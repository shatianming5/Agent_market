#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile FactorSpec into ExpressionEngine expression")
    parser.add_argument("--spec", required=True, help="Path to FactorSpec JSON")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    args = parser.parse_args()

    from agent_market.factor_compiler import FactorSpec
    from agent_market.factor_compiler.dsl.operators import compile_to_expression_engine

    spec_path = Path(args.spec).expanduser()
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    spec = FactorSpec.model_validate(payload)

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    spec_out = out_dir / "factor_spec.json"
    ast_out = out_dir / "factor_ast.json"
    expr_out = out_dir / "compiled_expression.txt"
    expr_meta_out = out_dir / "compiled_expression.json"

    spec_out.write_text(json.dumps(spec.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8")

    ast_payload = {
        "expr": spec.expr.model_dump(mode="json"),
        "canonical_expr_sha256": spec.canonical_expr_sha256(),
    }
    ast_out.write_text(json.dumps(ast_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    compiled = compile_to_expression_engine(spec.expr)
    expr_out.write_text(str(compiled), encoding="utf-8")
    expr_meta_out.write_text(
        json.dumps(
            {
                "version": 1,
                "compiled_expression": compiled,
                "input_spec": str(spec_path.resolve()),
                "artifacts": {
                    "factor_spec_json": str(spec_out),
                    "factor_ast_json": str(ast_out),
                    "compiled_expression_txt": str(expr_out),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[factor_compile] wrote: {spec_out}")
    print(f"[factor_compile] wrote: {ast_out}")
    print(f"[factor_compile] wrote: {expr_out}")


if __name__ == "__main__":
    main()


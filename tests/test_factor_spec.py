from __future__ import annotations

import json
from pathlib import Path


def test_factor_spec_schema_and_canonical_roundtrip(tmp_path: Path):
    from agent_market.factor_compiler.api_models import FactorSpec, write_factor_spec_schema

    payload = {
        "name": "lob_imbalance_z_120_l20",
        "version": "1.0",
        "hypothesis": "订单簿不平衡在短期内影响价格的微小漂移",
        "expr": {
            "op": "zscore",
            "args": [
                {
                    "op": "imbalance",
                    "args": [
                        {"op": "depth_bid", "args": [20]},
                        {"op": "depth_ask", "args": [20]},
                    ],
                },
                120,
            ],
        },
        "constraints": {
            "max_lookback": 2000,
            "min_delay_ms": 0,
            "max_turnover": 8.0,
            "complexity_budget": {"max_nodes": 40, "max_depth": 8},
        },
        "tests": [
            {"type": "no_lookahead", "horizon": 1},
            {"type": "nan_rate", "max_nan_ratio": 0.02},
        ],
        "meta": {
            "timeframe": "1s",
            "universe": ["BTC/USDT", "ETH/USDT"],
            "data_sources": ["trades", "lob_l2"],
        },
    }

    spec = FactorSpec.model_validate(payload)
    dumped = spec.model_dump(mode="json")
    assert dumped["name"] == payload["name"]
    assert dumped["expr"]["op"] == "zscore"

    # JSON schema export
    schema_path = write_factor_spec_schema(tmp_path / "factor_spec.schema.json")
    assert schema_path.exists()
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    props = (schema.get("properties") or {})
    for key in ["name", "version", "expr", "constraints", "tests", "meta"]:
        assert key in props

    # Canonicalization stability
    expr_a = {"op": "depth_bid", "args": [20]}
    expr_b = {"args": [20], "op": "depth_bid"}  # different key order
    spec_a = FactorSpec.model_validate({**payload, "expr": expr_a})
    spec_b = FactorSpec.model_validate({**payload, "expr": expr_b})
    assert spec_a.canonical_expr_sha256() == spec_b.canonical_expr_sha256()


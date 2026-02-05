from __future__ import annotations


def test_time_safety_availability_delay_gate() -> None:
    from agent_market.factor_compiler.api_models import ExprNode
    from agent_market.factor_compiler.checks.time_safety import check_time_safety
    from agent_market.factor_compiler.dsl.types import FactorType

    expr = ExprNode(op="var", args=["close"])
    var_types = {"close": FactorType(kind="series", dtype="float", availability_delay_ms=100)}

    results = check_time_safety(expr, min_delay_ms=0, var_types=var_types)
    availability = next(r for r in results if r.name == "availability_delay")
    assert availability.ok is False
    assert availability.code == "AVAILABILITY_DELAY_EXCEEDED"
    assert availability.details.get("availability_delay_ms") == 100
    assert availability.details.get("min_delay_ms") == 0
    assert not any(r.name == "time_safety" and r.ok for r in results)

    results2 = check_time_safety(expr, min_delay_ms=100, var_types=var_types)
    availability2 = next(r for r in results2 if r.name == "availability_delay")
    assert availability2.ok is True
    assert any(r.name == "time_safety" and r.ok for r in results2)


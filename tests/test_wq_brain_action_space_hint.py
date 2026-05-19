"""Coverage tests for _unexplored_space_hint — drives action space → 100%."""
from __future__ import annotations

import time

from agent_market.wq_brain.prompt_builder import (
    _FIELD_EXAMPLES,
    _OP_EXAMPLES,
    _count_atoms,
    _unexplored_space_hint,
)


def _row(expr: str) -> dict:
    return {"expr": expr, "ts": time.time(), "status": "COMPLETE",
            "sharpe": 1.0, "fitness": 0.5, "turnover": 0.3}


# ── _count_atoms ────────────────────────────────────────────────────────


def test_count_atoms_recognises_ops_and_fields():
    rows = [
        _row("rank(close)"),
        _row("ts_rank(close, 252)"),
        _row("ts_corr(close, volume, 20)"),
    ]
    op_use, fi_use = _count_atoms(rows)
    assert op_use["rank"] == 1
    assert op_use["ts_rank"] == 1
    assert op_use["ts_corr"] == 1
    assert fi_use["close"] == 3
    assert fi_use["volume"] == 1


def test_count_atoms_skips_unknown_fields():
    """`xyz` is not a real field — must NOT show up in the field counter."""
    rows = [_row("rank(xyz)")]
    op_use, fi_use = _count_atoms(rows)
    assert "xyz" not in fi_use
    assert op_use["rank"] == 1


def test_count_atoms_handles_empty_expr():
    rows = [{"expr": "", "ts": time.time(), "status": "COMPLETE"},
            {"expr": None, "ts": time.time(), "status": "COMPLETE"}]
    op_use, fi_use = _count_atoms(rows)
    assert op_use == {}
    assert fi_use == {}


# ── _unexplored_space_hint ──────────────────────────────────────────────


def test_unexplored_hint_empty_when_full_coverage():
    """When every op + field hits min_uses, hint must be empty (silent)."""
    rows = [_row(_OP_EXAMPLES[op]) for op in _OP_EXAMPLES] * 3
    rows += [_row(_FIELD_EXAMPLES[f]) for f in _FIELD_EXAMPLES] * 3
    out = _unexplored_space_hint(rows, min_uses_per_op=1, min_uses_per_field=1)
    # Every atom appears at least once via examples → hint suppressed
    assert out == ""


def test_unexplored_hint_lists_unused_math_operators():
    """With only 'rank(close)' tried, MATH ops should all show up as unused."""
    rows = [_row("rank(close)")] * 5
    out = _unexplored_space_hint(rows, min_uses_per_op=1, min_uses_per_field=1)
    assert "ACTION SPACE COVERAGE" in out
    # MATH bench should appear and include known unused ops
    assert "**MATH**" in out
    for op in ("sqrt", "clamp", "exp", "min", "max", "correlation"):
        assert f"`{op}`" in out


def test_unexplored_hint_lists_unused_fundamentals():
    """No fundamental fields touched → all 14 should be flagged FUND."""
    rows = [_row("rank(close)"), _row("rank(volume)"),
            _row("rank(returns)")] * 3
    out = _unexplored_space_hint(rows, min_uses_per_op=1, min_uses_per_field=1)
    assert "**FUND**" in out
    for f in ("earnings", "net_income", "cash", "fcf", "shares"):
        assert f"`{f}`" in out


def test_unexplored_hint_includes_concrete_examples():
    rows = [_row("rank(close)")] * 3
    out = _unexplored_space_hint(rows)
    assert "Concrete starter expressions" in out
    # Format: `op` → `expr`
    assert "→" in out


def test_unexplored_hint_coverage_percentage_renders():
    rows = [_row("rank(close)")] * 3
    out = _unexplored_space_hint(rows)
    assert "Currently exploring" in out
    assert "%" in out


def test_unexplored_hint_threshold_respected():
    """A given op used 4 times should NOT be flagged when threshold=3."""
    rows = [_row("rank(close)")] * 4
    out_low = _unexplored_space_hint(rows, min_uses_per_op=3,
                                     min_uses_per_field=99)
    # `rank` count = 4 ≥ 3 → not in MATH list
    # but field 'close' is not yet 99 → still shown for fields
    assert "MATH" in out_low or "FUND" in out_low
    # `rank` should NOT appear in the unused-ops list (it's covered)
    # We check by parsing the operators-section lines
    # (loose check: rank not in the leading per-family bullet)
    lines_after_ops = out_low.split("Operators with")[-1].split("Fields with")[0]
    assert "`rank`" not in lines_after_ops


def test_unexplored_hint_empty_tried_returns_full_unused():
    """Empty tried_log → every op + every field flagged."""
    out = _unexplored_space_hint([])
    assert "ACTION SPACE COVERAGE" in out
    # All 34 operators flagged
    for op in ("rank", "ts_corr", "sqrt", "if_else", "hump"):
        assert f"`{op}`" in out
    # All 28 fields flagged
    for f in ("close", "earnings", "sector"):
        assert f"`{f}`" in out


def test_every_op_example_passes_strict_arity_validation():
    """Each starter expr in _OP_EXAMPLES must pass the SAME validator the
    `wq_brain validate` CLI uses (strict mode = arity-checked). A starter
    that parses syntax but violates arity wastes WQ quota when the agent
    pastes it and tries to simulate.

    Production data: reviewer found `hump(_, 0.01)`, `sum(a,b)`,
    `correlation(a,b)`, `covariance(a,b)` all FAIL strict arity (hump max
    1; sum max 1; correlation/covariance min 3). This test pins the
    correct arity for every starter.
    """
    from agent_market.wq_brain.operators import validate_expression
    for op, expr in _OP_EXAMPLES.items():
        errors = validate_expression(expr, strict=True)
        assert not errors, f"_OP_EXAMPLES[{op!r}] failed strict validation: {errors}"


def test_every_field_example_passes_strict_arity_validation():
    """Same for field starters."""
    from agent_market.wq_brain.operators import validate_expression
    for f, expr in _FIELD_EXAMPLES.items():
        errors = validate_expression(expr, strict=True)
        assert not errors, f"_FIELD_EXAMPLES[{f!r}] failed strict validation: {errors}"


def test_unexplored_hint_warns_about_re_using_atoms():
    rows = [_row("rank(close)")] * 5
    out = _unexplored_space_hint(rows)
    # The closing instruction should warn that the hint will keep firing
    assert "diversity hint will keep firing" in out


# ── Integration with _build_prior_knowledge_block ───────────────────────


def test_prior_knowledge_block_includes_coverage_hint(tmp_path, monkeypatch):
    """When the tag's tried_log has only 'rank(close)' usage, the prior-
    knowledge block must include the action-space coverage hint."""
    monkeypatch.setenv("AGENT_MARKET_ARTIFACTS_ROOT", str(tmp_path))
    from agent_market.wq_brain.paths import tried_exprs_path
    from agent_market.wq_brain.prompt_builder import _build_prior_knowledge_block
    from agent_market.wq_brain.tried_log import append_tried

    p = tried_exprs_path("covtag")
    p.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(5):
        append_tried(
            p, expr="rank(close)", sharpe=1.0, fitness=0.5,
            turnover=0.3, alpha_id="A", status="COMPLETE",
        )
    block = _build_prior_knowledge_block("covtag")
    assert "ACTION SPACE COVERAGE" in block
    assert "earnings" in block  # at least one unused fundamental surfaced
    assert "sqrt" in block      # at least one unused math op surfaced

"""submit_gates tests — local jaccard + self-correlation + helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import pytest

from agent_market.wq_brain.submit_gates import (
    _finite_float,
    _finite_positive,
    auto_fill_expr,
    auto_fill_metrics,
    local_jaccard_gate,
    self_correlation_gate,
    summarize_rejection,
)


@pytest.fixture
def isolated_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_MARKET_ARTIFACTS_ROOT", str(tmp_path))
    return tmp_path


# ── Helpers ─────────────────────────────────────────────────────────────


def _seed_active_pool(tag: str, exprs: list[tuple[str, str, float]]) -> None:
    from agent_market.wq_brain.dtypes import AlphaPoolEntry
    from agent_market.wq_brain.paths import alpha_pool_path
    from agent_market.wq_brain.pool import AlphaPool

    pool = AlphaPool(alpha_pool_path(tag))
    for alpha_id, expr, fitness in exprs:
        entry = AlphaPoolEntry(
            alpha_id=alpha_id, expr=expr, settings_dict={},
            sharpe=1.4, fitness=fitness, returns=0.5, turnover=0.3,
            tag=tag, source="agent",
            verified_status="ACTIVE", verified_at=0.0, rejection_reasons=[],
        )
        pool.add(entry)


# ── local_jaccard_gate ──────────────────────────────────────────────────


def test_local_jaccard_no_active_accepts(isolated_artifacts):
    out = local_jaccard_gate("emptytag", "rank(close)")
    assert out["accept"] is True


def test_local_jaccard_token_block(isolated_artifacts):
    _seed_active_pool("jactag", [
        ("a1", "rank(ts_rank(close, 252) * (-ts_delta(close, 3) / close))", 1.5),
    ])
    # Same expression — token jaccard 1.0
    out = local_jaccard_gate(
        "jactag",
        "rank(ts_rank(close, 252) * (-ts_delta(close, 3) / close))",
    )
    assert out["accept"] is False
    assert "token-jaccard" in out["reason"]
    assert out["max_jaccard"] >= 0.99


def test_local_jaccard_semantic_block_on_field_swap(isolated_artifacts):
    """Token jaccard misses the close→vwap swap; semantic jaccard catches it."""
    _seed_active_pool("semtag", [
        ("a1", "rank(ts_rank(close, 252) * (-ts_delta(close, 3) / close))", 1.5),
    ])
    out = local_jaccard_gate(
        "semtag",
        "rank(ts_rank(vwap, 252) * (-ts_delta(vwap, 3) / vwap))",
        threshold=0.99,
        semantic_threshold=0.7,
    )
    assert out["accept"] is False
    assert "semantic-jaccard" in out["reason"]


def test_local_jaccard_disjoint_accepts(isolated_artifacts):
    _seed_active_pool("disjoint_tag", [
        ("a1", "rank(close)", 1.4),
    ])
    out = local_jaccard_gate(
        "disjoint_tag",
        "group_zscore(sales / assets, sector)",
        threshold=0.7, semantic_threshold=0.85,
    )
    assert out["accept"] is True


def test_local_jaccard_skips_when_no_tag():
    out = local_jaccard_gate("", "rank(close)")
    assert out["accept"] is True
    assert "skipped" in out["reason"]


def test_local_jaccard_returns_blocking_alpha_metadata(isolated_artifacts):
    _seed_active_pool("metatag", [
        ("alpha-XYZ", "rank(close)", 1.42),
    ])
    out = local_jaccard_gate("metatag", "rank(close)")
    assert out["accept"] is False
    assert out["vs_alpha_id"] == "alpha-XYZ"
    assert out["vs_alpha_fitness"] == pytest.approx(1.42)


# ── self_correlation_gate ───────────────────────────────────────────────


@dataclass
class _StubMetric:
    sharpe: Optional[float]


class _StubSession:
    def __init__(
        self,
        corrs: list[dict[str, Any]],
        sharpe_map: dict[str, Optional[float]],
    ) -> None:
        self._corrs = corrs
        self._sharpes = sharpe_map

    def get_alpha_correlations(self, alpha_id: str) -> list[dict[str, Any]]:
        return list(self._corrs)

    def fetch_alpha_metrics(self, alpha_id: str) -> _StubMetric:
        return _StubMetric(sharpe=self._sharpes.get(alpha_id))


def test_self_correlation_no_data_accepts():
    sess = _StubSession([], {})
    out = self_correlation_gate(sess, "alpha1")
    assert out["accept"] is True
    assert "no correlation data" in out["reason"]


def test_self_correlation_below_threshold_accepts():
    corrs = [{"alpha": "a", "correlation": 0.5}, {"alpha": "b", "correlation": -0.4}]
    sess = _StubSession(corrs, {})
    out = self_correlation_gate(sess, "alphaX", corr_max=0.7)
    assert out["accept"] is True
    assert out["max_correlation"] == pytest.approx(0.5)
    assert out["high_corr_count"] == 0


def test_self_correlation_blocks_when_correlated_and_short_on_sharpe():
    corrs = [{"alpha": "PEER", "correlation": 0.85}]
    sess = _StubSession(corrs, {"alphaX": 1.10, "PEER": 1.50})
    out = self_correlation_gate(sess, "alphaX", corr_max=0.7, sharpe_margin=0.10)
    assert out["accept"] is False
    assert out["high_corr_count"] == 1
    # required = 1.5 * 1.10 = 1.65 → our 1.10 < required → BLOCK
    assert out["blocking"][0]["status"].startswith("BLOCK")


def test_self_correlation_overrides_when_sharpe_meets_margin():
    corrs = [{"alpha": "PEER", "correlation": 0.85}]
    # PEER sharpe 1.0, margin 0.10 → required 1.10 — our 1.20 ≥ 1.10 → override
    sess = _StubSession(corrs, {"alphaX": 1.20, "PEER": 1.00})
    out = self_correlation_gate(sess, "alphaX", corr_max=0.7, sharpe_margin=0.10)
    assert out["accept"] is True
    assert out["overrides"][0]["status"].startswith("override")


def test_self_correlation_treats_404_as_no_data():
    """WQ returns 404 on /alphas/{id}/correlations for alphas not yet in
    the user's submitted pool. That's `no peer-correlation data stored`,
    NOT an infrastructure failure. Treat as accept (let submit through;
    WQ runs its own self-corr at submit time)."""

    class _Session404:
        def get_alpha_correlations(self, alpha_id):
            raise RuntimeError("404 Client Error: Not Found for url: "
                                "https://api.worldquantbrain.com/alphas/X/correlations")
        def fetch_alpha_metrics(self, alpha_id):
            raise NotImplementedError

    out = self_correlation_gate(_Session404(), "alphaX")
    assert out["accept"] is True
    assert "404" in out["reason"]


def test_self_correlation_raises_infra_error_when_corr_fetch_fails():
    """Network/auth failure during get_alpha_correlations must raise
    GateInfraError, not silently return accept=False (that would conflate
    infra failure with policy reject)."""
    from agent_market.wq_brain.submit_gates import GateInfraError

    class _BrokenSession:
        def get_alpha_correlations(self, alpha_id):
            raise RuntimeError("network reset by peer")
        def fetch_alpha_metrics(self, alpha_id):
            raise NotImplementedError

    with pytest.raises(GateInfraError, match="get_alpha_correlations"):
        self_correlation_gate(_BrokenSession(), "alphaX")


def test_self_correlation_raises_infra_error_when_metrics_fetch_fails():
    from agent_market.wq_brain.submit_gates import GateInfraError

    corrs = [{"alpha": "PEER", "correlation": 0.85}]

    class _PartialSession:
        def __init__(self, corrs):
            self.corrs = corrs
        def get_alpha_correlations(self, alpha_id):
            return self.corrs
        def fetch_alpha_metrics(self, alpha_id):
            raise RuntimeError("502 Bad Gateway")

    with pytest.raises(GateInfraError, match="fetch_alpha_metrics"):
        self_correlation_gate(_PartialSession(corrs), "alphaX")


def test_self_correlation_unknown_other_sharpe_blocks():
    corrs = [{"alpha": "MYSTERY", "correlation": 0.85}]
    sess = _StubSession(corrs, {"alphaX": 1.5})  # MYSTERY sharpe missing
    out = self_correlation_gate(sess, "alphaX")
    assert out["accept"] is False
    assert out["blocking"][0]["reason"] == "unknown sharpe — assumed blocking"


# ── summarize_rejection / auto_fill_expr ────────────────────────────────


def test_summarize_rejection_empty_returns_placeholder():
    assert summarize_rejection([]) == "no specific check failures captured"


def test_summarize_rejection_joins_top_failures():
    reasons = [
        {"name": "fitness", "value": 0.42, "limit": 1.0},
        {"name": "sharpe", "value": 0.9, "limit": 1.25},
    ]
    out = summarize_rejection(reasons)
    assert "fitness=0.42" in out
    assert "sharpe=0.9" in out
    assert "(limit=1.0)" in out


def test_auto_fill_expr_finds_latest_match(isolated_artifacts):
    from agent_market.wq_brain.paths import tried_exprs_path
    from agent_market.wq_brain.tried_log import append_tried

    p = tried_exprs_path("looktag")
    p.parent.mkdir(parents=True, exist_ok=True)
    append_tried(p, expr="old_expr", sharpe=1.0, fitness=0.8, turnover=0.3,
                 alpha_id="A1", status="COMPLETE")
    append_tried(p, expr="new_expr", sharpe=1.4, fitness=1.2, turnover=0.3,
                 alpha_id="A1", status="COMPLETE")
    assert auto_fill_expr("looktag", "A1") == "new_expr"


def test_auto_fill_expr_returns_empty_when_unknown(isolated_artifacts):
    assert auto_fill_expr("", "A1") == ""
    assert auto_fill_expr("notag", "") == ""
    assert auto_fill_expr("missing", "A1") == ""


# ── auto_fill_metrics ───────────────────────────────────────────────────


def test_auto_fill_metrics_returns_latest_metrics(isolated_artifacts):
    from agent_market.wq_brain.paths import tried_exprs_path
    from agent_market.wq_brain.tried_log import append_tried

    p = tried_exprs_path("metricstag")
    p.parent.mkdir(parents=True, exist_ok=True)
    append_tried(p, expr="old_expr", sharpe=1.0, fitness=0.8, turnover=0.3,
                 alpha_id="A1", status="COMPLETE")
    append_tried(p, expr="new_expr", sharpe=1.6, fitness=1.4, turnover=0.32,
                 alpha_id="A1", status="COMPLETE")
    out = auto_fill_metrics("metricstag", "A1")
    assert out["expr"] == "new_expr"
    assert out["sharpe"] == pytest.approx(1.6)
    assert out["fitness"] == pytest.approx(1.4)
    assert out["turnover"] == pytest.approx(0.32)


def test_auto_fill_metrics_returns_empty_when_unknown(isolated_artifacts):
    assert auto_fill_metrics("", "A1") == {}
    assert auto_fill_metrics("notag", "") == {}
    assert auto_fill_metrics("missing", "A1") == {}


# ── local_jaccard_gate sharpe-margin override ──────────────────────────


def _seed_active_with_metrics(
    tag: str, alpha_id: str, expr: str, *, sharpe: float, fitness: float,
) -> None:
    """Variant of _seed_active_pool that pins sharpe + fitness explicitly."""
    from agent_market.wq_brain.dtypes import AlphaPoolEntry
    from agent_market.wq_brain.paths import alpha_pool_path
    from agent_market.wq_brain.pool import AlphaPool

    pool = AlphaPool(alpha_pool_path(tag))
    pool.add(AlphaPoolEntry(
        alpha_id=alpha_id, expr=expr, settings_dict={},
        sharpe=sharpe, fitness=fitness, returns=0.5, turnover=0.3,
        tag=tag, source="agent",
        verified_status="ACTIVE", verified_at=0.0, rejection_reasons=[],
    ))


def test_override_fires_when_sharpe_margin_and_fitness_clear_blocking(
    isolated_artifacts,
):
    """Token-jaccard hits, BUT candidate is strictly better → override accept."""
    _seed_active_with_metrics(
        "ovr1", "ACT-1", "rank(close)", sharpe=1.20, fitness=1.10,
    )
    out = local_jaccard_gate(
        "ovr1", "rank(close)",  # token-jaccard 1.0 — would block
        candidate_sharpe=1.50,   # 1.50 ≥ 1.10 × 1.20 = 1.32 ✓
        candidate_fitness=1.40,  # 1.40 ≥ 1.10 ✓
        sharpe_margin=0.10,
    )
    assert out["accept"] is True
    assert out["override_applied"] is True
    assert "OVERRIDE" in out["reason"]
    assert out["vs_alpha_sharpe"] == pytest.approx(1.20)
    assert out["candidate_sharpe"] == pytest.approx(1.50)


def test_override_declined_when_sharpe_short(isolated_artifacts):
    """Sharpe just barely shy of the margin → override declined, BLOCK."""
    _seed_active_with_metrics(
        "ovr2", "ACT-2", "rank(close)", sharpe=1.30, fitness=1.10,
    )
    out = local_jaccard_gate(
        "ovr2", "rank(close)",
        candidate_sharpe=1.40,    # 1.40 < 1.10 × 1.30 = 1.43 ✗
        candidate_fitness=1.50,   # fitness ok
        sharpe_margin=0.10,
    )
    assert out["accept"] is False
    assert out["override_applied"] is False
    assert "override declined" in out["reason"]
    assert "sh shortfall=0.03" in out["reason"]


def test_override_declined_when_fitness_lower(isolated_artifacts):
    """Sharpe-margin met but fitness lower than blocking → override declined."""
    _seed_active_with_metrics(
        "ovr3", "ACT-3", "rank(close)", sharpe=1.20, fitness=1.50,
    )
    out = local_jaccard_gate(
        "ovr3", "rank(close)",
        candidate_sharpe=1.50,    # 1.50 ≥ 1.32 ✓
        candidate_fitness=1.40,   # 1.40 < 1.50 ✗
        sharpe_margin=0.10,
    )
    assert out["accept"] is False
    assert out["override_applied"] is False
    assert "override declined" in out["reason"]
    assert "fi shortfall=0.10" in out["reason"]


def test_override_disabled_when_no_candidate_metrics(isolated_artifacts):
    """No candidate metrics passed → backward-compat hard-block."""
    _seed_active_with_metrics(
        "ovr4", "ACT-4", "rank(close)", sharpe=1.20, fitness=1.10,
    )
    out = local_jaccard_gate("ovr4", "rank(close)")
    assert out["accept"] is False
    assert out["override_applied"] is False
    assert "OVERRIDE" not in out["reason"]


def test_override_for_semantic_jaccard_block(isolated_artifacts):
    """Override path also reachable on semantic-jaccard block (field-swap dup)."""
    _seed_active_with_metrics(
        "ovr5", "ACT-5",
        "rank(ts_rank(close, 252) * (-ts_delta(close, 3) / close))",
        sharpe=1.10, fitness=1.20,
    )
    out = local_jaccard_gate(
        "ovr5",
        "rank(ts_rank(vwap, 252) * (-ts_delta(vwap, 3) / vwap))",
        threshold=0.99,
        semantic_threshold=0.7,
        candidate_sharpe=1.50,    # 1.50 ≥ 1.10 × 1.10 = 1.21 ✓
        candidate_fitness=1.40,   # 1.40 ≥ 1.20 ✓
        sharpe_margin=0.10,
    )
    assert out["accept"] is True
    assert out["override_applied"] is True
    assert out["vs_alpha_sharpe"] == pytest.approx(1.10)


def test_override_emits_blocking_alpha_sharpe_field(isolated_artifacts):
    """Even on a non-override block, the new vs_alpha_sharpe field is populated."""
    _seed_active_with_metrics(
        "ovr6", "ACT-6", "rank(close)", sharpe=2.00, fitness=1.80,
    )
    out = local_jaccard_gate("ovr6", "rank(close)")
    assert out["accept"] is False
    assert out["vs_alpha_sharpe"] == pytest.approx(2.00)
    assert out["vs_alpha_fitness"] == pytest.approx(1.80)


# ── Codex review #4: defensive metric validation ───────────────────────


def test_finite_float_helper():
    assert _finite_float(1.5) == pytest.approx(1.5)
    assert _finite_float("1.5") == pytest.approx(1.5)  # numeric string
    assert _finite_float(None) is None
    assert _finite_float(float("nan")) is None
    assert _finite_float(float("inf")) is None
    assert _finite_float(float("-inf")) is None
    assert _finite_float("not a number") is None
    assert _finite_float(True) is None  # booleans rejected


def test_finite_positive_helper():
    assert _finite_positive(1.5) == pytest.approx(1.5)
    assert _finite_positive(0.0) is None    # not strictly > 0
    assert _finite_positive(-1.0) is None
    assert _finite_positive(None) is None
    assert _finite_positive(float("nan")) is None


def test_override_fail_closed_when_blocker_sharpe_zero(isolated_artifacts):
    """Codex review #4: a blocker with sharpe=0.0 must NOT yield required_sh=0
    (which would make every candidate pass the override). Override must
    decline when the blocker can't be evaluated."""
    _seed_active_with_metrics(
        "zerosh", "ACT-Z", "rank(close)", sharpe=0.0, fitness=0.0,
    )
    out = local_jaccard_gate(
        "zerosh", "rank(close)",
        candidate_sharpe=1.50, candidate_fitness=1.40,
        sharpe_margin=0.10,
    )
    assert out["accept"] is False
    assert out["override_applied"] is False
    assert "non-positive" in out["reason"] or "non-finite" in out["reason"]


def test_override_fail_closed_when_blocker_sharpe_nan(isolated_artifacts):
    """Same as above but with NaN — must fail-closed, not crash."""
    from agent_market.wq_brain.dtypes import AlphaPoolEntry
    from agent_market.wq_brain.paths import alpha_pool_path
    from agent_market.wq_brain.pool import AlphaPool

    pool = AlphaPool(alpha_pool_path("nanblocker"))
    pool.add(AlphaPoolEntry(
        alpha_id="ACT-NaN", expr="rank(close)", settings_dict={},
        sharpe=float("nan"), fitness=1.20, returns=0.5, turnover=0.3,
        tag="nanblocker", source="agent",
        verified_status="ACTIVE", verified_at=0.0, rejection_reasons=[],
    ))
    out = local_jaccard_gate(
        "nanblocker", "rank(close)",
        candidate_sharpe=1.65, candidate_fitness=1.85,
    )
    assert out["accept"] is False
    assert out["override_applied"] is False


def test_override_handles_nan_candidate_metrics(isolated_artifacts):
    """Candidate sharpe=NaN should be treated as 'no metrics' (override unavailable)
    instead of crashing."""
    _seed_active_with_metrics(
        "nancand", "ACT-NC", "rank(close)", sharpe=1.20, fitness=1.10,
    )
    out = local_jaccard_gate(
        "nancand", "rank(close)",
        candidate_sharpe=float("nan"), candidate_fitness=1.85,
    )
    assert out["accept"] is False
    assert out["override_applied"] is False


# ── Codex review #1: multi-blocker override ────────────────────────────


def test_multi_blocker_override_must_clear_every_blocker(isolated_artifacts):
    """Two ACTIVE alphas both block; candidate clears blocker A but not B → BLOCK.
    The override must clear EVERY blocker, not just the strictest one."""
    from agent_market.wq_brain.dtypes import AlphaPoolEntry
    from agent_market.wq_brain.paths import alpha_pool_path
    from agent_market.wq_brain.pool import AlphaPool

    pool = AlphaPool(alpha_pool_path("multi"))
    # Blocker A: jaccard=1.0 with our expr (same expr); sh=1.20 fi=1.10
    pool.add(AlphaPoolEntry(
        alpha_id="ACT-A", expr="rank(close)", settings_dict={},
        sharpe=1.20, fitness=1.10, returns=0.5, turnover=0.3,
        tag="multi", source="agent",
        verified_status="ACTIVE", verified_at=0.0, rejection_reasons=[],
    ))
    # Blocker B: also jaccard=1.0 (token set is same); sh=1.50 fi=1.40 (stricter)
    pool.add(AlphaPoolEntry(
        alpha_id="ACT-B", expr="rank(close)", settings_dict={},
        sharpe=1.50, fitness=1.40, returns=0.5, turnover=0.3,
        tag="multi", source="agent",
        verified_status="ACTIVE", verified_at=0.0, rejection_reasons=[],
    ))
    # Candidate clears ACT-A (sh=1.40 ≥ 1.10*1.20=1.32) but NOT ACT-B (1.40 < 1.10*1.50=1.65)
    out = local_jaccard_gate(
        "multi", "rank(close)",
        candidate_sharpe=1.40, candidate_fitness=1.50,
        sharpe_margin=0.10,
    )
    assert out["accept"] is False
    assert out["override_applied"] is False
    assert out["blocker_count"] == 2
    assert "shortfall" in out["reason"]


def test_multi_blocker_override_clears_all(isolated_artifacts):
    """Candidate strictly better than EVERY blocker → override fires."""
    from agent_market.wq_brain.dtypes import AlphaPoolEntry
    from agent_market.wq_brain.paths import alpha_pool_path
    from agent_market.wq_brain.pool import AlphaPool

    pool = AlphaPool(alpha_pool_path("multi2"))
    pool.add(AlphaPoolEntry(
        alpha_id="ACT-A", expr="rank(close)", settings_dict={},
        sharpe=1.20, fitness=1.10, returns=0.5, turnover=0.3,
        tag="multi2", source="agent",
        verified_status="ACTIVE", verified_at=0.0, rejection_reasons=[],
    ))
    pool.add(AlphaPoolEntry(
        alpha_id="ACT-B", expr="rank(close)", settings_dict={},
        sharpe=1.40, fitness=1.30, returns=0.5, turnover=0.3,
        tag="multi2", source="agent",
        verified_status="ACTIVE", verified_at=0.0, rejection_reasons=[],
    ))
    # 1.65 ≥ 1.10*1.40=1.54, AND 1.85 ≥ 1.30 → clears both
    out = local_jaccard_gate(
        "multi2", "rank(close)",
        candidate_sharpe=1.65, candidate_fitness=1.85,
    )
    assert out["accept"] is True
    assert out["override_applied"] is True
    assert out["blocker_count"] == 2
    assert "all 2 blocker" in out["reason"]


def test_blocker_count_zero_when_no_match(isolated_artifacts):
    """Disjoint expr → no blockers; blocker_count = 0."""
    _seed_active_with_metrics(
        "disj", "ACT-D", "rank(close)", sharpe=1.20, fitness=1.10,
    )
    out = local_jaccard_gate(
        "disj", "group_zscore(sales/assets, sector)",
        threshold=0.7, semantic_threshold=0.85,
    )
    assert out["accept"] is True
    assert out["blocker_count"] == 0
    assert out["blockers"] == []

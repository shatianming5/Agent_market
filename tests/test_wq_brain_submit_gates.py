"""submit_gates tests — local jaccard + self-correlation + helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import pytest

from agent_market.wq_brain.submit_gates import (
    auto_fill_expr,
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

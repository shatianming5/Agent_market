"""Tests for the bounded TTL+score pheromone cache."""
from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from agent_market.wq_brain.pheromone_cache import (
    CACHE_VERSION,
    CAPACITY,
    PheromoneLink,
    SCORE_ALPHA_DELTA_U,
    SCORE_BETA_SUPPORT,
    SCORE_DELTA_CONFLICT,
    SCORE_GAMMA_RECENCY,
    TTL_SECONDS,
    cache_path,
    read_cache,
    record_conflict,
    score,
    top_links,
    write_links,
)
from agent_market.wq_brain.tried_log import (
    ALTITUDE_L1_REGION_UNIVERSE,
    ALTITUDE_L2_OP_FAMILY,
    ALTITUDE_L3_SLOT_PARAM,
    ALTITUDE_L4_NUMERIC_TWEAK,
)


@pytest.fixture
def isolated_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_MARKET_ARTIFACTS_ROOT", str(tmp_path))
    return tmp_path


def _make(alt, expr, *, delta_u, ts, support=0, conflicts=0, alpha_id=None):
    return PheromoneLink(
        altitude=alt,
        expr=expr,
        alpha_id=alpha_id or f"a_{expr[:6]}",
        region="USA",
        universe="TOP500",
        sharpe=1.4,
        fitness=1.1,
        turnover=0.2,
        delta_U=delta_u,
        evidence_type="op_swap",
        parent_alpha_id=None,
        source_panel_tag="src",
        ts=ts,
        support=support,
        conflicts=conflicts,
    )


def test_capacity_eviction_keeps_highest_score(isolated_artifacts):
    cap = CAPACITY[ALTITUDE_L2_OP_FAMILY]
    now = time.time()
    # Fill bucket past capacity. Low-delta first, then high-delta, then a mid.
    links = []
    for i in range(cap + 5):
        links.append(_make(
            ALTITUDE_L2_OP_FAMILY,
            f"expr_{i}",
            delta_u=0.01 * i,  # later expressions = higher ΔU
            ts=now - i,
            alpha_id=f"a_{i}",
        ))
    kept = write_links("cap_evict_test", links)
    assert kept[ALTITUDE_L2_OP_FAMILY] == cap
    stored = read_cache("cap_evict_test")[ALTITUDE_L2_OP_FAMILY]
    stored_exprs = {l.expr for l in stored}
    # Lowest-ΔU entries (expr_0, expr_1, expr_2, ...) must be evicted.
    assert "expr_0" not in stored_exprs
    assert "expr_1" not in stored_exprs
    # Highest-ΔU entries must survive.
    assert f"expr_{cap+4}" in stored_exprs


def test_ttl_expiry_drops_stale_rows(isolated_artifacts):
    now = time.time()
    ttl = TTL_SECONDS[ALTITUDE_L1_REGION_UNIVERSE]
    fresh = _make(ALTITUDE_L1_REGION_UNIVERSE, "fresh_expr",
                  delta_u=0.5, ts=now - 60, alpha_id="fresh")
    stale = _make(ALTITUDE_L1_REGION_UNIVERSE, "stale_expr",
                  delta_u=0.9, ts=now - ttl - 1, alpha_id="stale")
    write_links("ttl_test", [fresh, stale])

    cached = read_cache("ttl_test")[ALTITUDE_L1_REGION_UNIVERSE]
    exprs = {l.expr for l in cached}
    assert "fresh_expr" in exprs
    assert "stale_expr" not in exprs


def test_score_formula_components_are_summed():
    link = _make(ALTITUDE_L1_REGION_UNIVERSE, "x",
                 delta_u=0.40, ts=time.time(), support=3, conflicts=1)
    s = score(link)
    # Approximate: 0.40 + 0.2*3 + 0.1*~1.0 − 0.5*1 = 0.40 + 0.6 + 0.1 − 0.5 = 0.6
    assert pytest.approx(s, abs=0.05) == 0.60
    # Removing conflicts must raise the score by exactly δ_conflict.
    link_no_conf = _make(ALTITUDE_L1_REGION_UNIVERSE, "y",
                         delta_u=0.40, ts=time.time(), support=3, conflicts=0)
    assert score(link_no_conf) - score(link) == pytest.approx(
        SCORE_DELTA_CONFLICT, abs=0.05
    )


def test_top_links_orders_within_altitude_by_score(isolated_artifacts):
    now = time.time()
    write_links("top_links_test", [
        _make(ALTITUDE_L1_REGION_UNIVERSE, "lo", delta_u=0.1, ts=now, alpha_id="l1_lo"),
        _make(ALTITUDE_L1_REGION_UNIVERSE, "hi", delta_u=0.9, ts=now, alpha_id="l1_hi"),
        _make(ALTITUDE_L2_OP_FAMILY, "m1", delta_u=0.5, ts=now, alpha_id="l2_m"),
    ])
    out = top_links("top_links_test", per_altitude=10)
    # L1 + L2 buckets, L1 hi-first then L1 lo, then L2.
    exprs = [l.expr for l in out]
    assert exprs.index("hi") < exprs.index("lo")
    assert "m1" in exprs


def test_dedupe_on_same_expr_increments_support(isolated_artifacts):
    now = time.time()
    first = _make(ALTITUDE_L2_OP_FAMILY, "shared", delta_u=0.3, ts=now, alpha_id="seed1")
    write_links("dedupe_test", [first])
    second = _make(ALTITUDE_L2_OP_FAMILY, "shared", delta_u=0.4, ts=now + 1, alpha_id="seed2")
    write_links("dedupe_test", [second])
    bucket = read_cache("dedupe_test")[ALTITUDE_L2_OP_FAMILY]
    assert len(bucket) == 1
    only = bucket[0]
    # Support incremented from 0 → 1.
    assert only.support == 1
    # Newest ts wins.
    assert only.ts == pytest.approx(now + 1)


def test_record_conflict_increments_conflicts_field(isolated_artifacts):
    now = time.time()
    write_links("conflict_test", [
        _make(ALTITUDE_L1_REGION_UNIVERSE, "expr_conf", delta_u=0.4, ts=now, alpha_id="rec_a"),
    ])
    assert record_conflict("conflict_test", "rec_a") is True
    cached = read_cache("conflict_test")[ALTITUDE_L1_REGION_UNIVERSE]
    assert cached[0].conflicts == 1
    assert record_conflict("conflict_test", "no_such_alpha") is False


def test_cross_altitude_isolation(isolated_artifacts):
    """Filling L4 to capacity must NOT evict any L1 entry."""
    now = time.time()
    cap_l4 = CAPACITY[ALTITUDE_L4_NUMERIC_TWEAK]
    links = [
        _make(ALTITUDE_L4_NUMERIC_TWEAK, f"l4_{i}", delta_u=0.1 * i, ts=now,
              alpha_id=f"l4_{i}")
        for i in range(cap_l4 + 5)
    ]
    links.append(
        _make(ALTITUDE_L1_REGION_UNIVERSE, "important_l1",
              delta_u=0.05, ts=now, alpha_id="l1_keep")
    )
    write_links("iso_test", links)
    cached = read_cache("iso_test")
    assert len(cached[ALTITUDE_L4_NUMERIC_TWEAK]) == cap_l4
    l1_exprs = {l.expr for l in cached[ALTITUDE_L1_REGION_UNIVERSE]}
    assert "important_l1" in l1_exprs


def test_cache_file_path_matches_artifacts_root(isolated_artifacts):
    write_links("path_test", [
        _make(ALTITUDE_L1_REGION_UNIVERSE, "exp1", delta_u=0.1, ts=time.time())
    ])
    p = cache_path("path_test")
    assert p.exists()
    assert p.is_file()
    assert p.parent.name == "path_test"
    assert p.parent.parent.name == "colony"
    payload = json.loads(p.read_text())
    assert payload["version"] == CACHE_VERSION
    assert "by_altitude" in payload

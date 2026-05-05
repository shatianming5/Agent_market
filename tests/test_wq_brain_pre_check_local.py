"""Tests for _check_local_jaccard_vs_active — token + semantic gate."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from agent_market.wq_brain.dtypes import AlphaPoolEntry
from agent_market.wq_brain.paths import alpha_pool_path
from agent_market.wq_brain.pool import AlphaPool


def _import_wq_brain_script():
    """Load scripts/wq_brain.py as a module so we can call private fn."""
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "scripts" / "wq_brain.py"
    spec = importlib.util.spec_from_file_location("_wqb_script", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_wqb_script"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def wqb_script():
    return _import_wq_brain_script()


@pytest.fixture
def isolated_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_MARKET_ARTIFACTS_ROOT", str(tmp_path))
    return tmp_path


def _seed_pool(tag: str, exprs: list[tuple[str, str]]) -> AlphaPool:
    """Create a tagged pool with each (alpha_id, expr) marked ACTIVE."""
    pool = AlphaPool(alpha_pool_path(tag))
    for i, (aid, expr) in enumerate(exprs):
        pool.add(AlphaPoolEntry(
            alpha_id=aid, expr=expr,
            sharpe=1.30, fitness=1.10, returns=0.20, turnover=0.40,
            settings_dict={}, tag=tag, source="test",
            verified_status="ACTIVE",
        ))
    return pool


def test_check_accepts_when_pool_empty(isolated_artifacts, wqb_script):
    res = wqb_script._check_local_jaccard_vs_active("emptytag", "rank(close)")
    assert res["accept"] is True


def test_check_token_jaccard_blocks_literal_dup(isolated_artifacts, wqb_script):
    expr = "rank(ts_corr(close, volume, 20))"
    _seed_pool("toktest", [("a1", expr)])
    res = wqb_script._check_local_jaccard_vs_active("toktest", expr)
    assert res["accept"] is False
    assert "BLOCK token-jaccard" in res["reason"]
    assert res["max_jaccard"] >= 0.7


def test_check_semantic_jaccard_blocks_field_swap(isolated_artifacts, wqb_script):
    """Field+window swap — operator skeleton identical but token jaccard low.
    Token: shared {rank, ts_rank, ts_delta} → ~0.33 jaccard, below 0.7 gate.
    Semantic: ops identical, fields disjoint → 0.75 weighted similarity, blocks.
    """
    active = "rank(ts_rank(close, 252) * (-ts_delta(close, 3) / close))"
    candidate = "rank(ts_rank(vwap, 144) * (-ts_delta(vwap, 7) / vwap))"
    _seed_pool("semtest", [("a1", active)])
    res = wqb_script._check_local_jaccard_vs_active("semtest", candidate)
    assert res["accept"] is False
    assert "BLOCK semantic-jaccard" in res["reason"], res
    # Token jaccard alone shouldn't have caught this
    assert res["max_jaccard"] < 0.7, res
    # But semantic should
    assert res["max_semantic"] >= 0.65, res


def test_check_accepts_when_genuinely_different(isolated_artifacts, wqb_script):
    active = "rank(ts_rank(close, 252) * (-ts_delta(close, 3) / close))"
    candidate = "rank(group_zscore(close - vwap, sector))"
    _seed_pool("difftest", [("a1", active)])
    res = wqb_script._check_local_jaccard_vs_active("difftest", candidate)
    assert res["accept"] is True
    assert res["max_jaccard"] < 0.7
    assert res["max_semantic"] < 0.65


def test_check_returns_thresholds_in_response(isolated_artifacts, wqb_script):
    _seed_pool("thrtest", [("a1", "rank(close)")])
    res = wqb_script._check_local_jaccard_vs_active(
        "thrtest", "rank(volume)",
        threshold=0.55, semantic_threshold=0.60,
    )
    assert res["jaccard_threshold"] == 0.55
    assert res["semantic_threshold"] == 0.60


def test_check_skips_unverified_pool_entries(isolated_artifacts, wqb_script):
    """Only ACTIVE entries should gate; QUEUED/REJECTED don't count."""
    pool = AlphaPool(alpha_pool_path("verifyonly"))
    pool.add(AlphaPoolEntry(
        alpha_id="q1", expr="rank(close)",
        sharpe=1.30, fitness=1.10, returns=0.20, turnover=0.40,
        settings_dict={}, tag="verifyonly", source="test",
        verified_status="QUEUED",  # NOT ACTIVE
    ))
    res = wqb_script._check_local_jaccard_vs_active("verifyonly", "rank(close)")
    assert res["accept"] is True
    assert "no ACTIVE alphas" in res["reason"]

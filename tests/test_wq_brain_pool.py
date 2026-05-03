"""Tests for wq_brain.pool — no WQ credentials needed."""
from __future__ import annotations

import json
import time

import pytest

from agent_market.wq_brain.dtypes import AlphaPoolEntry
from agent_market.wq_brain.pool import AlphaPool, DuplicateDetector, jaccard, _tokenset


class TestTokenSetAndJaccard:
    def test_identical_expressions(self):
        e = "rank(close / ts_mean(close, 20))"
        ts = _tokenset(e)
        assert jaccard(ts, ts) == pytest.approx(1.0)

    def test_disjoint_expressions(self):
        a = _tokenset("rank(volume)")
        b = _tokenset("ts_mean(returns, 5)")
        j = jaccard(a, b)
        assert j < 0.2

    def test_similar_expressions(self):
        a = _tokenset("rank(close / ts_mean(close, 20))")
        b = _tokenset("rank(close / ts_mean(close, 40))")
        j = jaccard(a, b)
        assert j > 0.8


class TestDuplicateDetector:
    def test_same_expr_is_duplicate(self):
        dd = DuplicateDetector(threshold=0.95)
        expr = "rank(close / ts_mean(close, 20))"
        dd.add(expr)
        assert dd.is_duplicate(expr)

    def test_different_expr_not_duplicate(self):
        dd = DuplicateDetector(threshold=0.95)
        dd.add("rank(volume)")
        assert not dd.is_duplicate("rank(returns(close, 5) / ts_std(returns(close, 5), 20))")

    def test_maxlen_evicts_old(self):
        dd = DuplicateDetector(threshold=0.95, maxlen=2)
        dd.add("rank(close)")
        dd.add("rank(volume)")
        dd.add("rank(vwap)")  # evicts "rank(close)"
        assert not dd.is_duplicate("rank(close)")


class TestAlphaPool:
    def _make_entry(self, alpha_id: str, expr: str, sharpe: float = 1.5) -> AlphaPoolEntry:
        return AlphaPoolEntry(
            alpha_id=alpha_id,
            expr=expr,
            settings_dict={"region": "USA"},
            sharpe=sharpe,
            fitness=sharpe * 0.8,
            returns=0.1,
            turnover=0.4,
        )

    def test_add_and_len(self, tmp_path):
        pool = AlphaPool(tmp_path / "pool.json")
        pool.add(self._make_entry("a1", "rank(close)"))
        pool.add(self._make_entry("a2", "rank(volume)"))
        assert len(pool) == 2

    def test_persistence(self, tmp_path):
        pool_path = tmp_path / "pool.json"
        pool = AlphaPool(pool_path)
        pool.add(self._make_entry("a1", "rank(close)", sharpe=2.0))
        # Reload
        pool2 = AlphaPool(pool_path)
        assert len(pool2) == 1
        assert pool2._entries[0].sharpe == pytest.approx(2.0)

    def test_top_n_by_fitness(self, tmp_path):
        pool = AlphaPool(tmp_path / "pool.json")
        for i, (sharpe, expr) in enumerate(
            [(1.3, "rank(close)"), (2.5, "rank(volume)"), (1.8, "rank(vwap)")]
        ):
            pool.add(self._make_entry(f"a{i}", expr, sharpe=sharpe))
        top = pool.top_n_by_fitness(2)
        assert len(top) == 2
        assert top[0].sharpe == pytest.approx(2.5)

    def test_local_duplicate_detection(self, tmp_path):
        pool = AlphaPool(tmp_path / "pool.json")
        expr = "rank(close / ts_mean(close, 20))"
        pool.add(self._make_entry("a1", expr))
        assert pool.is_local_duplicate(expr)
        assert not pool.is_local_duplicate("rank(ts_std(returns(close, 1), 20))")

    def test_clean(self, tmp_path):
        pool = AlphaPool(tmp_path / "pool.json")
        exprs = [f"rank(close{i})" for i in range(5)]
        sharpes = [1.2, 2.1, 1.9, 1.5, 2.3]
        for i, (expr, sh) in enumerate(zip(exprs, sharpes)):
            pool.add(self._make_entry(f"a{i}", expr, sharpe=sh))
        removed = pool.clean(keep_top_n=3)
        assert removed == 2
        assert len(pool) == 3

    def test_summary_for_prompt_empty(self, tmp_path):
        pool = AlphaPool(tmp_path / "pool.json")
        summary = pool.summary_for_prompt()
        assert "empty" in summary.lower()

    def test_summary_for_prompt_with_entries(self, tmp_path):
        pool = AlphaPool(tmp_path / "pool.json")
        pool.add(self._make_entry("a1", "rank(close)", sharpe=1.8))
        summary = pool.summary_for_prompt()
        assert "1 submitted" in summary or "Pool has" in summary
        assert "rank(close)" in summary

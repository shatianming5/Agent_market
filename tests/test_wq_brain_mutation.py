"""Tests for wq_brain mutation engine."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent_market.wq_brain.mutation import (
    generate_mutations,
    mutate_crossover,
    mutate_field,
    mutate_op,
    mutate_window,
)


@pytest.fixture
def amihud_expr():
    return "rank(-ts_delta(close,1)/close*volume/adv20)"


@pytest.fixture
def reversal_expr():
    return "rank(-ts_mean(returns, 60))"


@pytest.fixture
def sector_expr():
    return "rank(group_zscore(-returns, sector))"


class TestMutateWindow:
    def test_changes_window(self, reversal_expr):
        variants = mutate_window(reversal_expr)
        assert len(variants) > 0
        assert all("ts_mean(returns," in v or "ts_mean(returns, " in v for v in variants)
        # None should have the original window 60
        windows = []
        for v in variants:
            import re
            m = re.search(r"ts_mean\(returns,\s*(\d+)\)", v)
            if m:
                windows.append(int(m.group(1)))
        assert 60 not in windows

    def test_no_ts_in_expr_returns_empty(self):
        variants = mutate_window("rank(close/vwap)")
        assert variants == []

    def test_multiple_windows(self):
        expr = "rank(ts_mean(close, 20) + ts_rank(volume, 60))"
        variants = mutate_window(expr)
        assert len(variants) > 0

    def test_deduplicates(self, reversal_expr):
        variants = mutate_window(reversal_expr)
        assert len(variants) == len(set(variants))

    def test_no_original_in_output(self, reversal_expr):
        variants = mutate_window(reversal_expr)
        assert reversal_expr not in variants


class TestMutateOp:
    def test_ts_mean_alternatives(self, reversal_expr):
        variants = mutate_op(reversal_expr)
        exprs = " ".join(variants)
        assert "ts_rank" in exprs or "ts_zscore" in exprs

    def test_ts_delta_alternatives(self):
        variants = mutate_op("rank(-ts_delta(close, 5) / close)")
        exprs = " ".join(variants)
        assert "ts_delay" in exprs

    def test_no_ts_ops_returns_empty(self, sector_expr):
        variants = mutate_op(sector_expr)
        assert variants == []

    def test_preserves_rank_wrapper(self, reversal_expr):
        variants = mutate_op(reversal_expr)
        for v in variants:
            assert v.startswith("rank(")

    def test_deduplicates(self, reversal_expr):
        variants = mutate_op(reversal_expr)
        assert len(variants) == len(set(variants))


class TestMutateField:
    def test_close_to_vwap(self):
        # reversal_expr uses 'returns', not a price field — use close-based expr
        expr = "rank(-ts_delta(close, 5) / close)"
        variants = mutate_field(expr)
        exprs = " ".join(variants)
        assert "vwap" in exprs or "open" in exprs or "high" in exprs

    def test_no_price_field_returns_empty(self):
        variants = mutate_field("rank(ts_mean(returns, 60))")
        assert variants == []

    def test_does_not_introduce_invalid_fields(self, amihud_expr):
        variants = mutate_field(amihud_expr)
        for v in variants:
            for bad in ["adv60", "adv120", "adv180", "cap"]:
                assert bad not in v

    def test_deduplicates(self):
        expr = "rank(-ts_delta(close, 5) / close)"
        variants = mutate_field(expr)
        assert len(variants) == len(set(variants))


class TestMutateCrossover:
    def test_basic_crossover(self, reversal_expr, sector_expr):
        results = mutate_crossover(reversal_expr, sector_expr)
        assert len(results) > 0
        for r in results:
            assert r.startswith("rank(")

    def test_strips_rank_wrapper(self):
        a = "rank(ts_mean(returns, 60))"
        b = "rank(ts_mean(close, 20))"
        results = mutate_crossover(a, b)
        for r in results:
            # outer wrapper is exactly one rank(; inner parts have no extra rank()
            assert r.startswith("rank(")
            assert not r.startswith("rank(rank(")

    def test_operators(self, reversal_expr, sector_expr):
        results = mutate_crossover(reversal_expr, sector_expr)
        ops_found = set()
        for r in results:
            import re
            m = re.search(r"ts_mean\(returns, 60\)\s*([+\-*])\s*group_zscore", r)
            if m:
                ops_found.add(m.group(1))
        assert len(ops_found) > 0


class TestGenerateMutations:
    def _make_pool_entry(self, expr: str, fitness: float = 1.0):
        entry = MagicMock()
        entry.expr = expr
        entry.fitness = fitness
        return entry

    def _make_settings(self):
        settings = MagicMock()
        return settings

    def test_returns_candidates(self):
        entries = [
            self._make_pool_entry("rank(-ts_mean(returns, 60))", 1.1),
            self._make_pool_entry("rank(group_zscore(-returns, sector))", 1.0),
        ]
        settings = self._make_settings()
        candidates = generate_mutations(entries, settings, top_n=2, variants_per_parent=2)
        assert len(candidates) > 0

    def test_empty_pool_returns_empty(self):
        candidates = generate_mutations([], MagicMock())
        assert candidates == []

    def test_deduplication(self):
        expr = "rank(-ts_mean(returns, 60))"
        entries = [self._make_pool_entry(expr, 1.0)] * 3
        candidates = generate_mutations(entries, MagicMock(), top_n=3, variants_per_parent=4)
        exprs = [c.expr for c in candidates]
        assert len(exprs) == len(set(exprs))

    def test_no_parent_in_output(self):
        entries = [
            self._make_pool_entry("rank(-ts_mean(returns, 60))", 1.1),
            self._make_pool_entry("rank(ts_rank(close, 252))", 1.0),
        ]
        candidates = generate_mutations(entries, MagicMock(), top_n=2)
        parent_exprs = {e.expr for e in entries}
        for c in candidates:
            assert c.expr not in parent_exprs

    def test_crossover_when_multiple_parents(self):
        entries = [
            self._make_pool_entry("rank(-ts_mean(returns, 60))", 1.1),
            self._make_pool_entry("rank(ts_rank(close, 252))", 1.0),
            self._make_pool_entry("rank(group_zscore(-returns, sector))", 0.9),
        ]
        with_cross = generate_mutations(entries, MagicMock(), crossover=True)
        without_cross = generate_mutations(entries, MagicMock(), crossover=False)
        assert len(with_cross) >= len(without_cross)

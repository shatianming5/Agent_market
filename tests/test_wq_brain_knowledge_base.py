"""Tests for wq_brain knowledge base (TF-IDF + Jaccard search)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_market.wq_brain.knowledge_base import KBEntry, KnowledgeBase, _tokenize, _jaccard


class TestTokenize:
    def test_basic(self):
        tokens = _tokenize("rank(-ts_mean(returns, 60))")
        assert "rank" in tokens
        assert "ts_mean" in tokens
        assert "returns" in tokens
        assert "60" in tokens

    def test_lowercase(self):
        tokens = _tokenize("GROUP_ZSCORE(Close, Sector)")
        assert "group_zscore" in tokens
        assert "close" in tokens

    def test_numbers(self):
        tokens = _tokenize("ts_delta(close, 5)")
        assert "5" in tokens


class TestJaccard:
    def test_identical(self):
        a = frozenset(["a", "b", "c"])
        assert _jaccard(a, a) == 1.0

    def test_disjoint(self):
        a = frozenset(["a", "b"])
        b = frozenset(["c", "d"])
        assert _jaccard(a, b) == 0.0

    def test_partial(self):
        a = frozenset(["a", "b", "c"])
        b = frozenset(["b", "c", "d"])
        score = _jaccard(a, b)
        assert 0.0 < score < 1.0

    def test_empty(self):
        assert _jaccard(frozenset(), frozenset()) == 0.0


class TestKBEntry:
    def test_tokens_cached(self):
        entry = KBEntry(source="test", text="rank(ts_mean(returns, 60))")
        t1 = entry.tokens
        t2 = entry.tokens
        assert t1 is t2

    def test_to_dict_roundtrip(self):
        entry = KBEntry(source="seed", text="rank(close)", metadata={"sharpe": 1.5})
        d = entry.to_dict()
        restored = KBEntry.from_dict(d)
        assert restored.source == entry.source
        assert restored.text == entry.text
        assert restored.metadata == entry.metadata


class TestKnowledgeBase:
    def test_seeds_loaded(self, tmp_path):
        kb = KnowledgeBase(tmp_path / "kb.json")
        assert len(kb) >= 10  # at least the seed alphas

    def test_search_returns_results(self, tmp_path):
        kb = KnowledgeBase(tmp_path / "kb.json")
        results = kb.search("ts_mean returns reversal", top_k=3)
        assert len(results) <= 3
        assert all(isinstance(r, KBEntry) for r in results)

    def test_search_relevant(self, tmp_path):
        kb = KnowledgeBase(tmp_path / "kb.json")
        results = kb.search("Amihud illiquidity volume ts_delta close", top_k=5)
        exprs = [r.text for r in results]
        assert any("ts_delta" in e or "volume" in e for e in exprs)

    def test_add_alpha(self, tmp_path):
        kb = KnowledgeBase(tmp_path / "kb.json")
        initial_len = len(kb)
        kb.add_alpha("rank(ts_corr(close, volume, 30))", sharpe=1.5, fitness=1.2, turnover=0.4)
        assert len(kb) == initial_len + 1

    def test_add_alpha_no_duplicate(self, tmp_path):
        kb = KnowledgeBase(tmp_path / "kb.json")
        expr = "rank(ts_corr(close, volume, 30))"
        kb.add_alpha(expr, sharpe=1.5, fitness=1.2, turnover=0.4)
        kb.add_alpha(expr, sharpe=1.6, fitness=1.3, turnover=0.3)
        exprs = [e.text for e in kb._entries]
        assert exprs.count(expr) == 1

    def test_save_and_reload(self, tmp_path):
        kb_path = tmp_path / "kb.json"
        kb = KnowledgeBase(kb_path)
        kb.add_alpha("rank(ts_corr(close, volume, 30))", sharpe=1.5, fitness=1.2, turnover=0.4)
        kb.save()
        assert kb_path.exists()

        kb2 = KnowledgeBase(kb_path)
        pool_entries = [e for e in kb2._entries if e.source == "pool"]
        assert any(e.text == "rank(ts_corr(close, volume, 30))" for e in pool_entries)

    def test_save_no_dirty_flag(self, tmp_path):
        kb_path = tmp_path / "kb.json"
        kb = KnowledgeBase(kb_path)
        kb.save()
        assert not kb_path.exists()  # seeds don't set dirty flag

    def test_search_top_k_limit(self, tmp_path):
        kb = KnowledgeBase(tmp_path / "kb.json")
        results = kb.search("returns sector close volume", top_k=3)
        assert len(results) <= 3

    def test_search_ranks_by_relevance(self, tmp_path):
        kb = KnowledgeBase(tmp_path / "kb.json")
        query = "rank(-ts_mean(returns, 60))"
        results = kb.search(query, top_k=5)
        assert len(results) > 0
        top_result = results[0]
        assert "returns" in top_result.tokens or "ts_mean" in top_result.tokens

    def test_metadata_stored(self, tmp_path):
        kb = KnowledgeBase(tmp_path / "kb.json")
        kb.add_alpha("rank(close/vwap)", sharpe=1.3, fitness=1.05, turnover=0.3)
        entry = next(e for e in kb._entries if e.text == "rank(close/vwap)")
        assert entry.metadata["sharpe"] == 1.3
        assert entry.metadata["fitness"] == 1.05

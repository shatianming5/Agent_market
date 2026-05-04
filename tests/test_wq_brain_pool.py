"""AlphaPool + DuplicateDetector tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_market.wq_brain.dtypes import (
    AlphaCandidate,
    AlphaPoolEntry,
    AlphaSettings,
    SimulationResult,
)
from agent_market.wq_brain.pool import AlphaPool, DuplicateDetector


def test_duplicate_detector_exact_match():
    d = DuplicateDetector()
    d.add("rank(close)")
    assert d.is_duplicate("rank(close)") is True


def test_duplicate_detector_distinguishes_window_numbers():
    d = DuplicateDetector(threshold=0.95)
    d.add("rank(ts_mean(close, 20))")
    # 20 vs 60 should be considered DIFFERENT under tokenizer that captures digits
    assert d.is_duplicate("rank(ts_mean(close, 60))") is False


def test_duplicate_detector_distinguishes_fields():
    d = DuplicateDetector(threshold=0.95)
    d.add("rank(close)")
    assert d.is_duplicate("rank(open)") is False
    assert d.is_duplicate("rank(volume)") is False


def test_duplicate_detector_window_default():
    d = DuplicateDetector(maxlen=3)
    d.add("a")
    d.add("b")
    d.add("c")
    d.add("d")
    # "a" should have rolled out
    assert d.is_duplicate("a") is False


def test_alpha_pool_add_duplicate_alpha_id_returns_false(tmp_path: Path):
    pool = AlphaPool(tmp_path / "pool.json")
    e1 = AlphaPoolEntry(
        alpha_id="A1", expr="rank(close)", settings_dict={}, sharpe=1.5,
        fitness=1.2, returns=0.1, turnover=0.2,
    )
    assert pool.add(e1) is True
    e2 = AlphaPoolEntry(
        alpha_id="A1", expr="rank(open)", settings_dict={}, sharpe=2.0,
        fitness=1.5, returns=0.2, turnover=0.3,
    )
    assert pool.add(e2) is False
    assert len(pool) == 1


def test_alpha_pool_persists_to_disk(tmp_path: Path):
    pool_path = tmp_path / "pool.json"
    pool = AlphaPool(pool_path)
    pool.add(AlphaPoolEntry(
        alpha_id="A1", expr="rank(close)", settings_dict={}, sharpe=1.5,
        fitness=1.2, returns=0.1, turnover=0.2,
    ))
    assert pool_path.exists()
    data = json.loads(pool_path.read_text())
    assert len(data) == 1
    assert data[0]["alpha_id"] == "A1"

    # Reload from disk
    pool2 = AlphaPool(pool_path)
    assert len(pool2) == 1
    assert pool2.entries[0].alpha_id == "A1"


def test_alpha_pool_top_n_by_fitness(tmp_path: Path):
    pool = AlphaPool(tmp_path / "pool.json")
    for i, fi in enumerate([1.1, 1.5, 1.2, 1.8, 1.3]):
        pool.add(AlphaPoolEntry(
            alpha_id=f"A{i}", expr=f"rank(close-{i})", settings_dict={},
            sharpe=1.5, fitness=fi, returns=0.1, turnover=0.2,
        ))
    top3 = pool.top_n_by_fitness(3)
    assert [e.fitness for e in top3] == [1.8, 1.5, 1.3]


def test_alpha_pool_add_from_candidate_with_passing_result(tmp_path: Path):
    pool = AlphaPool(tmp_path / "pool.json")
    settings = AlphaSettings()
    c = AlphaCandidate(expr="rank(close)", settings=settings)
    c.sim_result = SimulationResult(
        sharpe=1.5, fitness=1.2, returns=0.15, turnover=0.18,
        alpha_id="A1", status="COMPLETE",
    )
    entry = pool.add_from_candidate(c, tag="t1")
    assert entry is not None
    assert entry.alpha_id == "A1"
    assert entry.tag == "t1"
    assert len(pool) == 1


def test_alpha_pool_add_from_candidate_skips_no_result(tmp_path: Path):
    pool = AlphaPool(tmp_path / "pool.json")
    c = AlphaCandidate(expr="rank(close)", settings=AlphaSettings())
    # No sim_result set
    assert pool.add_from_candidate(c, tag="t1") is None
    assert len(pool) == 0


def test_alpha_pool_corrupt_json_recovers_empty(tmp_path: Path):
    pool_path = tmp_path / "pool.json"
    pool_path.write_text("not valid json {{")
    pool = AlphaPool(pool_path)
    assert len(pool) == 0

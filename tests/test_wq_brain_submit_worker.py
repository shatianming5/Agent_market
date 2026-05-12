"""Tests for AlphaPool.upsert + cmd_pool_submit_worker.

Submit worker is the Round 3 capability fix: cluster UNSUBMITTED candidates
by operator skeleton, pick top-fi rep per cluster, run WQ pre-check, submit
accepted, persist outcome via upsert (not add — outcomes overwrite stale state).
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


@pytest.fixture
def isolated_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_MARKET_ARTIFACTS_ROOT", str(tmp_path))
    return tmp_path


def _run(cmd_func, **kwargs):
    ns = argparse.Namespace(**kwargs)
    captured = io.StringIO()
    code = {"v": 0}
    real_exit = sys.exit
    def fake_exit(c=0):
        code["v"] = c
        raise SystemExit(c)
    with patch("sys.stdout", captured):
        with patch("sys.exit", fake_exit):
            try:
                cmd_func(ns)
            except SystemExit:
                pass
    out = captured.getvalue()
    return (json.loads(out) if out.strip() else None), code["v"]


# ── AlphaPool.upsert ────────────────────────────────────────────────────


def test_upsert_inserts_new(tmp_path):
    from agent_market.wq_brain.dtypes import AlphaPoolEntry
    from agent_market.wq_brain.pool import AlphaPool
    pool = AlphaPool(tmp_path / "p.json")
    e = AlphaPoolEntry(alpha_id="A", expr="rank(close)", settings_dict={},
                        sharpe=1.5, fitness=1.2, returns=0.1, turnover=0.3,
                        tag="t", source="test",
                        verified_status="UNSUBMITTED",
                        verified_at=0.0, rejection_reasons=[])
    assert pool.upsert(e) == "inserted"
    assert len(pool) == 1


def test_upsert_updates_existing_status(tmp_path):
    from agent_market.wq_brain.dtypes import AlphaPoolEntry
    from agent_market.wq_brain.pool import AlphaPool
    pool = AlphaPool(tmp_path / "p.json")
    e1 = AlphaPoolEntry(alpha_id="A", expr="rank(close)", settings_dict={},
                          sharpe=1.5, fitness=1.2, returns=0.1, turnover=0.3,
                          tag="t", source="salvage",
                          verified_status="UNSUBMITTED",
                          verified_at=0.0, rejection_reasons=[])
    pool.upsert(e1)
    e2 = AlphaPoolEntry(alpha_id="A", expr="rank(close)", settings_dict={},
                          sharpe=1.5, fitness=1.2, returns=0.1, turnover=0.3,
                          tag="t", source="salvage",
                          verified_status="ACTIVE",  # ← changed
                          verified_at=999.0, rejection_reasons=[])
    assert pool.upsert(e2) == "updated"
    assert len(pool) == 1
    [stored] = pool.entries
    assert stored.verified_status == "ACTIVE"


def test_upsert_replaces_in_place_no_duplicate(tmp_path):
    """Two upserts with same alpha_id never produce 2 pool entries."""
    from agent_market.wq_brain.dtypes import AlphaPoolEntry
    from agent_market.wq_brain.pool import AlphaPool
    pool = AlphaPool(tmp_path / "p.json")
    for status in ("UNSUBMITTED", "ACTIVE"):
        pool.upsert(AlphaPoolEntry(
            alpha_id="A", expr="rank(close)", settings_dict={},
            sharpe=1.5, fitness=1.2, returns=0.1, turnover=0.3,
            tag="t", source="test",
            verified_status=status,
            verified_at=0.0, rejection_reasons=[]))
    assert len(pool) == 1
    [stored] = pool.entries
    assert stored.verified_status == "ACTIVE"


# ── cmd_pool_submit_worker — clustering + dry run ───────────────────────


def _seed(tag: str, candidates: list[tuple]):
    """candidates = list of (alpha_id, expr, sh, fi, status)."""
    from agent_market.wq_brain.dtypes import AlphaPoolEntry
    from agent_market.wq_brain.paths import alpha_pool_path
    from agent_market.wq_brain.pool import AlphaPool
    pool = AlphaPool(alpha_pool_path(tag))
    for aid, expr, sh, fi, status in candidates:
        pool.add(AlphaPoolEntry(alpha_id=aid, expr=expr, settings_dict={},
                                  sharpe=sh, fitness=fi, returns=0.1, turnover=0.2,
                                  tag=tag, source="test",
                                  verified_status=status,
                                  verified_at=0.0, rejection_reasons=[]))


def test_submit_worker_dry_run_clusters_by_skeleton(isolated_artifacts):
    """Two `ts_decay_linear(rank(...), 20)` candidates collapse to one cluster
    rep; one `rank(ts_corr(...))` is its own cluster."""
    from wq_brain import cmd_pool_submit_worker
    _seed("sw_t1", [
        ("A1", "ts_decay_linear(rank(close), 20)", 1.6, 1.4, "UNSUBMITTED"),
        ("A2", "ts_decay_linear(rank(volume), 20)", 1.7, 1.5, "UNSUBMITTED"),
        ("B1", "rank(ts_corr(close, volume, 20))", 1.5, 1.2, "UNSUBMITTED"),
        ("ACT1", "rank(close)", 1.5, 1.1, "ACTIVE"),  # filtered out
    ])
    out, _ = _run(cmd_pool_submit_worker, tag="sw_t1", status="UNSUBMITTED",
                   max=20, one_per_cluster=True, corr_max=0.7,
                   sharpe_margin=0.10, verify_after_sec=0.0,
                   continue_on_infra=False, dry_run=True)
    assert out["dry_run"] is True
    assert out["n_targets"] == 2
    ids = [p["alpha_id"] for p in out["preview"]]
    # A2 (fi=1.5) is the rep of the ts_decay_linear|rankx1 cluster
    assert ids == ["A2", "B1"]


def test_submit_worker_no_cluster_returns_all_sorted_by_fitness(isolated_artifacts):
    from wq_brain import cmd_pool_submit_worker
    _seed("sw_t2", [
        ("X", "rank(close)", 1.5, 1.1, "UNSUBMITTED"),
        ("Y", "rank(volume)", 1.6, 1.5, "UNSUBMITTED"),
        ("Z", "rank(open)",   1.5, 1.2, "UNSUBMITTED"),
    ])
    out, _ = _run(cmd_pool_submit_worker, tag="sw_t2", status="UNSUBMITTED",
                   max=20, one_per_cluster=False, corr_max=0.7,
                   sharpe_margin=0.10, verify_after_sec=0.0,
                   continue_on_infra=False, dry_run=True)
    ids = [p["alpha_id"] for p in out["preview"]]
    assert ids == ["Y", "Z", "X"]  # sorted by fi desc


def test_submit_worker_max_caps_targets(isolated_artifacts):
    from wq_brain import cmd_pool_submit_worker
    _seed("sw_t3", [(f"K{i}", f"e{i}", 1.5, 1.0+i*0.1, "UNSUBMITTED") for i in range(5)])
    out, _ = _run(cmd_pool_submit_worker, tag="sw_t3", status="UNSUBMITTED",
                   max=2, one_per_cluster=False, corr_max=0.7,
                   sharpe_margin=0.10, verify_after_sec=0.0,
                   continue_on_infra=False, dry_run=True)
    assert out["n_targets"] == 2


def test_submit_worker_empty_when_no_targets(isolated_artifacts):
    from wq_brain import cmd_pool_submit_worker
    _seed("sw_t4", [("A1", "rank(close)", 1.5, 1.4, "ACTIVE")])
    out, _ = _run(cmd_pool_submit_worker, tag="sw_t4", status="UNSUBMITTED",
                   max=20, one_per_cluster=False, corr_max=0.7,
                   sharpe_margin=0.10, verify_after_sec=0.0,
                   continue_on_infra=False, dry_run=True)
    assert out["n_targets"] == 0


def test_submit_worker_filters_by_status(isolated_artifacts):
    from wq_brain import cmd_pool_submit_worker
    _seed("sw_t5", [
        ("A", "rank(close)", 1.5, 1.0, "ACTIVE"),
        ("B", "rank(volume)", 1.5, 1.1, "UNSUBMITTED"),
        ("C", "rank(open)", 1.5, 1.0, "REJECTED"),
    ])
    out, _ = _run(cmd_pool_submit_worker, tag="sw_t5", status="REJECTED",
                   max=20, one_per_cluster=False, corr_max=0.7,
                   sharpe_margin=0.10, verify_after_sec=0.0,
                   continue_on_infra=False, dry_run=True)
    assert out["n_targets"] == 1
    assert out["preview"][0]["alpha_id"] == "C"

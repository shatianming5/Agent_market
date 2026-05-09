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


def test_cluster_key_uses_family_skeleton_kinds_not_just_skeleton(isolated_artifacts):
    """Codex review R3-#2: rank(close) and rank(sales/assets) share the
    skeleton `rankx1` but differ in family AND field-kinds — they must
    NOT collapse to the same cluster, otherwise the worker would suppress
    exactly the family/field diversity the prompt encourages.
    """
    from wq_brain import cmd_pool_submit_worker
    _seed("clk", [
        ("A", "rank(close)",                          1.6, 1.5, "UNSUBMITTED"),
        ("B", "rank(sales / assets)",                 1.55, 1.45, "UNSUBMITTED"),
        ("C", "rank((high - low) / close)",           1.50, 1.40, "UNSUBMITTED"),
    ])
    out, _ = _run(cmd_pool_submit_worker, tag="clk", status="UNSUBMITTED",
                   max=20, one_per_cluster=True, corr_max=0.7,
                   sharpe_margin=0.10, verify_after_sec=0.0,
                   continue_on_infra=False, dry_run=True)
    # All 3 are distinct clusters: ('other','rankx1',{P/V}),
    # ('fundamental_ratio','rankx1',{FUND}), ('intraday_range','rankx1',{P/V}).
    assert out["n_targets"] == 3
    ids = [p["alpha_id"] for p in out["preview"]]
    assert set(ids) == {"A", "B", "C"}


def test_explicit_scan_limit_honored_even_when_max_large(isolated_artifacts):
    """Codex review R3-#3: --scan-limit 50 --max 200 must scan 50, not 1000."""
    from wq_brain import cmd_pool_submit_worker
    _seed("scl", [(f"K{i}", f"rank(field{i})", 1.5, 1.0, "UNSUBMITTED") for i in range(80)])
    out, _ = _run(cmd_pool_submit_worker, tag="scl", status="UNSUBMITTED",
                   max=200, scan_limit=10, one_per_cluster=False, corr_max=0.7,
                   sharpe_margin=0.10, verify_after_sec=0.0,
                   continue_on_infra=False, dry_run=True)
    assert out["n_targets"] == 10
    assert out["scan_limit"] == 10


def test_dry_run_aggregate_reflects_all_scanned_not_preview_cap(isolated_artifacts):
    """Codex review R3-#4: would_local_block / projected_submit reflect ALL
    scanned targets; preview list still capped by --dry-run-limit."""
    from wq_brain import cmd_pool_submit_worker
    # Seed an ACTIVE alpha to force local jaccard blocks
    _seed("dra", [("ACT", "rank(close)", 1.5, 1.4, "ACTIVE")])
    # 30 candidates that all hit jaccard=1.0 with the ACTIVE alpha
    rows = [(f"K{i}", "rank(close)", 1.6, 1.5, "UNSUBMITTED") for i in range(30)]
    _seed("dra", rows)
    out, _ = _run(cmd_pool_submit_worker, tag="dra", status="UNSUBMITTED",
                   max=20, scan_limit=200, dry_run_limit=5, one_per_cluster=False,
                   corr_max=0.7, sharpe_margin=0.10, verify_after_sec=0.0,
                   continue_on_infra=False, dry_run=True)
    assert out["n_targets"] == 30
    # Aggregate: every candidate's sh=1.6 ≥ 1.10*1.5=1.65 fails (just barely)
    # Wait — 1.6 < 1.65 → all 30 BLOCKED → would_local_block = 30
    assert out["would_local_block"] == 30
    # Preview cap: only 5 entries rendered
    assert len(out["preview"]) == 5
    assert out["dry_run_limit"] == 5


def test_pool_upsert_persists_in_place_mutations_to_disk(tmp_path):
    """Codex review (latent bug surfaced by integration test): when caller
    mutates an entry returned by ``pool.entries`` in place and calls
    ``upsert``, the prior code's ``e.to_dict() == entry.to_dict()`` check
    returned True (same object, post-mutation) so the change was silently
    dropped. Fix: identity check + always save when same object."""
    from agent_market.wq_brain.dtypes import AlphaPoolEntry
    from agent_market.wq_brain.pool import AlphaPool
    pool = AlphaPool(tmp_path / "p.json")
    e = AlphaPoolEntry(alpha_id="A", expr="rank(close)", settings_dict={},
                        sharpe=1.5, fitness=1.2, returns=0.1, turnover=0.3,
                        tag="t", source="test",
                        verified_status="UNSUBMITTED",
                        verified_at=0.0, rejection_reasons=[])
    pool.upsert(e)
    # Mutate the ENTRY THE POOL HOLDS in place (same path the worker takes)
    in_pool = pool.entries[0]
    in_pool.verified_status = "ACTIVE"
    in_pool.rejection_reasons = []
    result = pool.upsert(in_pool)
    assert result == "updated"      # was "unchanged" pre-fix
    # Reload from disk and verify persistence
    pool2 = AlphaPool(tmp_path / "p.json")
    assert pool2.entries[0].verified_status == "ACTIVE"


def test_pool_upsert_unchanged_when_distinct_object_equal(tmp_path):
    """Backward compat: passing a *fresh* AlphaPoolEntry that fully matches
    an existing one (including submitted_at timestamp) still returns
    'unchanged' (no spurious save). Note AlphaPoolEntry auto-sets
    submitted_at=time.time() so we have to copy it explicitly to test
    the equality path."""
    from agent_market.wq_brain.dtypes import AlphaPoolEntry
    from agent_market.wq_brain.pool import AlphaPool
    pool = AlphaPool(tmp_path / "p2.json")
    e1 = AlphaPoolEntry(alpha_id="B", expr="rank(close)", settings_dict={},
                          sharpe=1.5, fitness=1.2, returns=0.1, turnover=0.3,
                          tag="t", source="test",
                          verified_status="ACTIVE",
                          verified_at=0.0, rejection_reasons=[])
    pool.upsert(e1)
    # Reload from disk to get a distinct object whose submitted_at matches
    pool2 = AlphaPool(tmp_path / "p2.json")
    e2 = pool2.entries[0]
    assert e2 is not e1
    assert pool.upsert(e2) == "unchanged"


def test_pool_replace_all_does_not_resurrect_deleted_entries(tmp_path):
    """Codex review R2-CRIT: my R1-#3 fcntl+merge introduced a critical
    regression — _save's merge-from-disk re-added entries that the caller
    intentionally dropped (e.g. pool dedup). replace_all is the
    intent-explicit destructive-replace path."""
    from agent_market.wq_brain.dtypes import AlphaPoolEntry
    from agent_market.wq_brain.pool import AlphaPool
    pool = AlphaPool(tmp_path / "pdedup.json")
    for aid in ("A", "B", "C"):
        pool.upsert(AlphaPoolEntry(alpha_id=aid, expr=f"rank({aid.lower()})",
                                       settings_dict={}, sharpe=1.5, fitness=1.0,
                                       returns=0.1, turnover=0.3, tag="t",
                                       source="test", verified_status="UNSUBMITTED",
                                       verified_at=0.0, rejection_reasons=[]))
    # Caller decides to keep only A — drops B + C
    keep = [e for e in pool.entries if e.alpha_id == "A"]
    pool.replace_all(keep)
    # Reload from disk: B and C must NOT come back
    reloaded = AlphaPool(tmp_path / "pdedup.json")
    ids = {e.alpha_id for e in reloaded.entries}
    assert ids == {"A"}


def test_pool_save_authoritative_ids_allows_intentional_demotion(tmp_path):
    """Codex review R3-CRIT: when caller deliberately demotes a status
    (e.g. sync-status --reset-local-blocks moving LOCAL_BLOCKED →
    UNSUBMITTED), passing authoritative_ids must bypass the precedence
    merge so the demotion persists."""
    from agent_market.wq_brain.dtypes import AlphaPoolEntry
    from agent_market.wq_brain.pool import AlphaPool
    pool_path = tmp_path / "pauth.json"
    p = AlphaPool(pool_path)
    p.upsert(AlphaPoolEntry(alpha_id="X", expr="rank(close)", settings_dict={},
                              sharpe=1.5, fitness=1.4, returns=0.1, turnover=0.3,
                              tag="t", source="test",
                              verified_status="LOCAL_BLOCKED",  # precedence 60
                              verified_at=0.0, rejection_reasons=[]))
    # Simulate authoritative demotion: caller observed UNSUBMITTED from WQ,
    # mutates in-memory, calls _save with authoritative_ids={X}.
    p._entries[0].verified_status = "UNSUBMITTED"  # precedence 20
    p._save(authoritative_ids={"X"})
    reloaded = AlphaPool(pool_path)
    # Without authoritative_ids: precedence 60 > 20 → would keep LOCAL_BLOCKED.
    # With authoritative_ids: caller's UNSUBMITTED wins.
    assert reloaded.entries[0].verified_status == "UNSUBMITTED"


def test_pool_save_richness_tiebreak_keeps_rejection_reasons(tmp_path):
    """Codex review R3-#4: same-status tie → prefer disk row with more
    rejection_reasons. Protects against losing WQ-probed failure details
    when an unrelated stale writer's memory snapshot clobbers them."""
    from agent_market.wq_brain.dtypes import AlphaPoolEntry
    from agent_market.wq_brain.pool import AlphaPool
    pool_path = tmp_path / "prich.json"
    # Process A writes X=UNSUBMITTED with detailed rejection_reasons
    pa = AlphaPool(pool_path)
    pa.upsert(AlphaPoolEntry(alpha_id="X", expr="rank(close)", settings_dict={},
                                sharpe=1.5, fitness=0.9, returns=0.1, turnover=0.3,
                                tag="t", source="test",
                                verified_status="UNSUBMITTED",
                                verified_at=100.0,
                                rejection_reasons=[{"name": "fitness", "value": 0.9}]))
    # Process B has stale UNSUBMITTED-no-reasons in memory; saves something else.
    pb = AlphaPool(pool_path)
    pb._entries[0].rejection_reasons = []  # mimic stale state
    pb._entries[0].verified_at = 0.0  # older timestamp
    pb.upsert(AlphaPoolEntry(alpha_id="Y", expr="rank(volume)", settings_dict={},
                                sharpe=1.6, fitness=1.5, returns=0.1, turnover=0.3,
                                tag="t", source="test",
                                verified_status="UNSUBMITTED",
                                verified_at=200.0, rejection_reasons=[]))
    final = AlphaPool(pool_path)
    by_id = {e.alpha_id: e for e in final.entries}
    # X must keep the disk's richer state (rejection_reasons present)
    assert len(by_id["X"].rejection_reasons or []) == 1
    assert by_id["X"].rejection_reasons[0]["name"] == "fitness"


def test_pool_save_status_precedence_protects_active_from_demotion(tmp_path):
    """Codex review R2-#2: when in-memory has UNSUBMITTED but disk has
    ACTIVE for the same alpha (concurrent writer set it ACTIVE), the
    merge must keep ACTIVE — not demote it via stale in-memory state."""
    from agent_market.wq_brain.dtypes import AlphaPoolEntry
    from agent_market.wq_brain.pool import AlphaPool
    pool_path = tmp_path / "pprec.json"
    # Process A: upserts X as ACTIVE
    pa = AlphaPool(pool_path)
    pa.upsert(AlphaPoolEntry(alpha_id="X", expr="rank(close)", settings_dict={},
                                sharpe=1.5, fitness=1.4, returns=0.1, turnover=0.3,
                                tag="t", source="test",
                                verified_status="ACTIVE",
                                verified_at=0.0, rejection_reasons=[]))
    # Process B: loaded BEFORE A's update; thinks X is UNSUBMITTED.
    # Now B saves something else and the merge sees disk has ACTIVE for X.
    pb = AlphaPool(pool_path)
    # Force-load stale state: pretend B started with X=UNSUBMITTED
    pb._entries[0].verified_status = "UNSUBMITTED"
    # (in real life, B's local mutation would be of a different alpha)
    pb.upsert(AlphaPoolEntry(alpha_id="Y", expr="rank(volume)", settings_dict={},
                                sharpe=1.6, fitness=1.5, returns=0.1, turnover=0.3,
                                tag="t", source="test",
                                verified_status="UNSUBMITTED",
                                verified_at=0.0, rejection_reasons=[]))
    # Reload final state — X must still be ACTIVE, Y is UNSUBMITTED
    final = AlphaPool(pool_path)
    by_id = {e.alpha_id: e for e in final.entries}
    assert by_id["X"].verified_status == "ACTIVE"
    assert by_id["Y"].verified_status == "UNSUBMITTED"


def test_pool_save_merges_concurrent_writer_inserts(tmp_path):
    """Codex review R1-#3: when two processes load the pool simultaneously,
    one adds entry X, saves; the other adds Y and saves — Y's save must
    NOT delete X. The fcntl + re-read-on-save logic merges X back in."""
    from agent_market.wq_brain.dtypes import AlphaPoolEntry
    from agent_market.wq_brain.pool import AlphaPool

    # Simulate two processes by instantiating two pools on the same path
    p1 = AlphaPool(tmp_path / "p3.json")
    p2 = AlphaPool(tmp_path / "p3.json")
    # P1 adds X, persists
    p1.upsert(AlphaPoolEntry(alpha_id="X", expr="rank(close)", settings_dict={},
                                sharpe=1.5, fitness=1.2, returns=0.1, turnover=0.3,
                                tag="t", source="test",
                                verified_status="UNSUBMITTED",
                                verified_at=0.0, rejection_reasons=[]))
    # P2 (which doesn't know about X) adds Y, persists
    p2.upsert(AlphaPoolEntry(alpha_id="Y", expr="rank(volume)", settings_dict={},
                                sharpe=1.6, fitness=1.3, returns=0.1, turnover=0.3,
                                tag="t", source="test",
                                verified_status="UNSUBMITTED",
                                verified_at=0.0, rejection_reasons=[]))
    # Reload from disk: BOTH X and Y must be present (no lost update)
    final = AlphaPool(tmp_path / "p3.json")
    ids = {e.alpha_id for e in final.entries}
    assert ids == {"X", "Y"}


def test_submit_worker_full_flow_with_mocked_session(isolated_artifacts, monkeypatch):
    """Codex review R4-#4: end-to-end non-dry-run integration.

    Pool seeded with:
      ACT — verified ACTIVE alpha (sh=1.20 fi=1.10) used as local jaccard anchor
      L1  — local-block target (jaccard 1.0, sh=1.30 < 1.10×1.20=1.32 → no override)
      O1  — local-override target (jaccard 1.0 sh=1.65 fi=1.50 → override fires)
      P1  — passes local; mocked self-corr returns 0.8 (no peer sharpe → BLOCK)
      S1  — passes local; mocked self-corr returns clean; submit_alpha → ACTIVE
      S2  — passes local; submit_alpha → ACTIVE (would be 3rd submit; capped by --max=2)

    Asserts: --max=2 stops the loop after 2 successful submits regardless of how
    many candidates were scanned. local_overrides counts override hits. Terminal
    states (LOCAL_BLOCKED / SELF_CORR_BLOCKED / ACTIVE / UNSUBMITTED) persist
    correctly to the pool.
    """
    from wq_brain import cmd_pool_submit_worker
    # Fitness ranks chosen so the desc sort visits all 5 UNSUBMITTED in order
    # L1 → O1 → P1 → S1 → S2. With --max=2, the loop stops after 2 successful
    # submits (O1 + S1); L1 still gets local-block scan, P1 still gets
    # self-corr-block scan.
    _seed("intg", [
        ("ACT", "rank(close)",        1.20, 1.10, "ACTIVE"),
        # Same expr as ACTIVE → token jaccard 1.0; sh=1.30 < 1.10×1.20=1.32 → block
        ("L1",  "rank(close)",        1.30, 1.80, "UNSUBMITTED"),
        # Override: sh=1.65 ≥ 1.32, fi=1.50 ≥ 1.10
        ("O1",  "rank(close)",        1.65, 1.70, "UNSUBMITTED"),
        # Different family → not blocked locally; self-corr returns blocking peer
        ("P1",  "rank(sales / debt)", 1.50, 1.60, "UNSUBMITTED"),
        # Different family → no self-corr peers; would submit
        ("S1",  "rank(volume / cap)", 1.55, 1.50, "UNSUBMITTED"),
        ("S2",  "rank(equity / fcf)", 1.60, 1.40, "UNSUBMITTED"),
    ])

    class _MockSession:
        def __init__(self):
            self.submitted: list[str] = []
        def get_alpha_correlations(self, alpha_id):
            # P1: high-corr peer with no sharpe → forces BLOCK
            if alpha_id == "P1":
                return [{"alpha": "MYSTERY", "correlation": 0.80}]
            return []
        def fetch_alpha_metrics(self, alpha_id):
            from dataclasses import dataclass
            @dataclass
            class _M: sharpe: object = None
            return _M(sharpe=1.50 if alpha_id == "P1" else None)
        def submit_alpha(self, alpha_id, verify_after_sec=30.0):
            self.submitted.append(alpha_id)
            return {"verified_status": "ACTIVE", "rejection_reasons": []}

    sess = _MockSession()
    monkeypatch.setattr(
        "agent_market.wq_brain.client.session_from_env",
        lambda: sess,
    )
    out, code = _run(
        cmd_pool_submit_worker,
        tag="intg", status="UNSUBMITTED",
        max=2, scan_limit=200, one_per_cluster=False,
        jaccard_max=0.7, semantic_max=0.85,
        corr_max=0.7, sharpe_margin=0.10,
        override_mode="sharpe_and_fitness",
        absolute_fitness_floor=1.0,
        verify_after_sec=0.0, continue_on_infra=False,
        dry_run=False,
    )
    assert out["ok"] is True
    # --max=2 enforced: only 2 actual WQ submit calls
    assert out["submitted"] == 2
    assert len(sess.submitted) == 2
    # L1 should be local-blocked, O1 should NOT (override)
    assert out["local_blocked"] == 1
    assert out["local_overrides"] == 1
    # P1 hits self-corr policy block
    assert out["policy_blocked"] == 1
    # All 2 submits returned ACTIVE
    assert out["active"] == 2

    # Verify terminal pool states
    from agent_market.wq_brain.paths import alpha_pool_path
    from agent_market.wq_brain.pool import AlphaPool
    pool = AlphaPool(alpha_pool_path("intg"))
    by_id = {e.alpha_id: e for e in pool.entries}
    assert by_id["L1"].verified_status == "LOCAL_BLOCKED"
    assert by_id["P1"].verified_status == "SELF_CORR_BLOCKED"
    # First two scanned-and-passed entries (O1, then S1) become ACTIVE
    # via the mocked submit_alpha. S2 never reaches the loop body.
    assert by_id["O1"].verified_status == "ACTIVE"
    assert by_id["S1"].verified_status == "ACTIVE"
    assert by_id["S2"].verified_status == "UNSUBMITTED"  # not yet attempted


def test_submit_worker_max_is_submit_budget_not_scan_cap(isolated_artifacts):
    """Codex review R2-#1: --max bounds successful submissions, not scanned
    candidates. n_targets shows the total scanned (capped by scan_limit)."""
    from wq_brain import cmd_pool_submit_worker
    _seed("sw_t3", [(f"K{i}", f"e{i}", 1.5, 1.0+i*0.1, "UNSUBMITTED") for i in range(5)])
    out, _ = _run(cmd_pool_submit_worker, tag="sw_t3", status="UNSUBMITTED",
                   max=2, scan_limit=200, one_per_cluster=False, corr_max=0.7,
                   sharpe_margin=0.10, verify_after_sec=0.0,
                   continue_on_infra=False, dry_run=True)
    assert out["n_targets"] == 5            # all scanned in dry-run
    assert out["max_submit"] == 2            # submit budget surfaces in output
    assert out["projected_submit_attempts"] <= 2


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

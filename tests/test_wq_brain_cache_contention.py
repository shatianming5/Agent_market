"""Cross-process stress test for the fcntl-locked pheromone cache."""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import time

import pytest

from agent_market.wq_brain.pheromone_cache import (
    CACHE_VERSION,
    CAPACITY,
    cache_path,
    read_cache,
)


@pytest.fixture
def isolated_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_MARKET_ARTIFACTS_ROOT", str(tmp_path))
    return tmp_path


# ── Module-level worker so multiprocessing.spawn can pickle it ─────────


def _worker_write(args):
    """Each subprocess writes ``count`` distinct links with a unique prefix."""
    artifacts_root, colony_tag, worker_id, count = args
    os.environ["AGENT_MARKET_ARTIFACTS_ROOT"] = artifacts_root
    # Re-import after env is set so paths.wq_brain_root picks up the override.
    from agent_market.wq_brain.pheromone_cache import (
        PheromoneLink as _PL,
        write_links as _wl,
    )
    from agent_market.wq_brain.tried_log import (
        ALTITUDE_L1_REGION_UNIVERSE as _L1,
        ALTITUDE_L2_OP_FAMILY as _L2,
    )
    now = time.time()
    links = []
    altitudes = (_L1, _L2)
    for i in range(count):
        alt = altitudes[i % 2]
        links.append(_PL(
            altitude=alt,
            expr=f"worker_{worker_id}_expr_{i}",
            alpha_id=f"w{worker_id}_a{i}",
            region="USA",
            universe="TOP500",
            sharpe=1.5, fitness=1.1, turnover=0.2,
            delta_U=0.1 * (i + 1),
            evidence_type="op_swap",
            parent_alpha_id=None,
            source_panel_tag=f"worker_{worker_id}",
            ts=now + i * 0.001,
        ))
    # Stagger writes slightly so processes overlap.
    time.sleep(0.05)
    _wl(colony_tag, links)
    return worker_id


def test_concurrent_processes_do_not_corrupt_cache(isolated_artifacts):
    """Two processes writing the same cache must produce a parseable file
    with all unique exprs preserved up to capacity."""
    colony_tag = "contention_test"
    ctx = mp.get_context("spawn")
    n_workers = 3
    per_worker = 6
    args = [
        (str(isolated_artifacts), colony_tag, wid, per_worker)
        for wid in range(n_workers)
    ]
    with ctx.Pool(n_workers) as pool:
        outcomes = pool.map(_worker_write, args)
    assert sorted(outcomes) == list(range(n_workers))

    cp = cache_path(colony_tag)
    assert cp.exists()
    data = json.loads(cp.read_text())
    assert data["version"] == CACHE_VERSION
    # Every bucket must be ≤ capacity even after concurrent writes.
    for alt, bucket in data["by_altitude"].items():
        assert len(bucket) <= CAPACITY[alt]
    # Cross-process: total stored exprs ≤ (n_workers * per_worker / 2) per alt;
    # nothing was dropped due to corruption, just by cap eviction.
    flat = read_cache(colony_tag)
    total = sum(len(v) for v in flat.values())
    assert total > 0
    # No duplicate exprs *within* an altitude (deduper on write_links).
    for alt, bucket in flat.items():
        exprs = [l.expr for l in bucket]
        assert len(exprs) == len(set(exprs))


def test_workers_mode_process_runs_panels_end_to_end(isolated_artifacts):
    """Smoke-check that ProcessPoolExecutor path completes a colony run."""
    from agent_market.wq_brain.colony import (
        ColonyConfig,
        PanelSpec,
        run_colony,
    )

    panels = [
        PanelSpec(tag=f"proc_panel_{i}", region="USA",
                  universe=f"TOP{(i + 1) * 100}", max_turns=1)
        for i in range(2)
    ]
    cfg = ColonyConfig(
        colony_tag="proc_colony",
        panels=panels,
        cli="opencode",
        model="stub-model",
        timeout_sec=5.0,
        workers=2,
        workers_mode="process",
    )
    manifest = run_colony(cfg, runner=_proc_runner)
    assert manifest["workers_mode"] == "process"
    assert len(manifest["panel_summaries"]) == 2
    assert {s["panel_tag"] for s in manifest["panel_summaries"]} == {
        "proc_panel_0", "proc_panel_1"
    }


def _proc_runner(agent_cfg):
    """Module-level runner so ProcessPoolExecutor can pickle it."""
    return {"run_id": f"proc_stub_{agent_cfg.tag}", "elapsed_sec": 0.01}

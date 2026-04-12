from __future__ import annotations

import json
import os
import time
from pathlib import Path


def _write_run_meta(run_dir: Path, *, run_id: str, started_at: str, ended_at: str, backtest_zips: list[str] | None = None) -> None:
    payload = {
        "run_id": run_id,
        "started_at": started_at,
        "ended_at": ended_at,
        "artifacts": {
            "backtest_zips": backtest_zips or [],
        },
    }
    (run_dir / "run_meta.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_gc_runs_uses_run_dir_mtime_when_meta_missing(tmp_path: Path, monkeypatch) -> None:
    artifacts_root = (tmp_path / "artifacts").resolve()
    runs_root = (artifacts_root / "runs").resolve()
    user_data_root = (tmp_path / "user_data").resolve()
    runs_root.mkdir(parents=True, exist_ok=True)
    (user_data_root / "backtest_results").mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("AGENT_MARKET_ARTIFACTS_ROOT", str(artifacts_root))
    monkeypatch.setenv("AGENT_MARKET_USER_DATA_ROOT", str(user_data_root))

    # Create 3 runs:
    # - new_no_meta: no run_meta.json (simulate in-progress), very recent mtime
    # - old_no_meta: no run_meta.json, old mtime (should be removed)
    # - meta_run: has ended_at (should be kept)
    new_no_meta = runs_root / "new_no_meta"
    old_no_meta = runs_root / "old_no_meta"
    meta_run = runs_root / "meta_run"
    new_no_meta.mkdir()
    old_no_meta.mkdir()
    meta_run.mkdir()

    now = time.time()
    os.utime(old_no_meta, (now - 100_000, now - 100_000))
    os.utime(new_no_meta, (now - 10, now - 10))

    _write_run_meta(
        meta_run,
        run_id="meta_run",
        started_at="2026-04-12T00:00:00+00:00",
        ended_at="2026-04-12T01:00:00+00:00",
    )

    from server.janitor import gc_runs

    res = gc_runs(keep=2, prune_backtests=False, dry_run=False)
    assert res["removed"] == 1
    assert new_no_meta.exists()
    assert meta_run.exists()
    assert not old_no_meta.exists()


def test_gc_jobs_keeps_recent_and_removes_old(tmp_path: Path, monkeypatch) -> None:
    artifacts_root = (tmp_path / "artifacts").resolve()
    user_data_root = (tmp_path / "user_data").resolve()
    reg_dir = (user_data_root / "job_registry").resolve()
    log_dir = (user_data_root / "job_logs").resolve()
    reg_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("AGENT_MARKET_ARTIFACTS_ROOT", str(artifacts_root))
    monkeypatch.setenv("AGENT_MARKET_USER_DATA_ROOT", str(user_data_root))

    old_job = "job_old"
    new_job = "job_new"
    (reg_dir / f"{old_job}.json").write_text(
        json.dumps({"id": old_job, "started_at": "2026-04-01T00:00:00+00:00", "ended_at": "2026-04-01T00:01:00+00:00"}, indent=2),
        encoding="utf-8",
    )
    (log_dir / f"{old_job}.log").write_text("old\n", encoding="utf-8")
    (reg_dir / f"{new_job}.json").write_text(
        json.dumps({"id": new_job, "started_at": "2026-04-12T00:00:00+00:00", "ended_at": "2026-04-12T00:01:00+00:00"}, indent=2),
        encoding="utf-8",
    )
    (log_dir / f"{new_job}.log").write_text("new\n", encoding="utf-8")

    from server.janitor import gc_jobs

    res = gc_jobs(keep_recent=1, keep_days=0, dry_run=False)
    assert res["removed"] == 1
    assert not (reg_dir / f"{old_job}.json").exists()
    assert not (log_dir / f"{old_job}.log").exists()
    assert (reg_dir / f"{new_job}.json").exists()
    assert (log_dir / f"{new_job}.log").exists()


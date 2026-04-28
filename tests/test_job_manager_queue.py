from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from server.job_manager import JobManager, JobQueueFullError


def _wait_status(mgr: JobManager, job_id: str, *, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + float(timeout)
    last: dict = {}
    while time.monotonic() < deadline:
        last = mgr.status(job_id)
        if last.get("returncode") is not None:
            return last
        time.sleep(0.05)
    raise AssertionError(f"timeout waiting for job {job_id}: {last}")


def test_job_manager_queues_and_drains(tmp_path: Path) -> None:
    log_dir = tmp_path / "job_logs"
    reg_dir = tmp_path / "job_registry"
    mgr = JobManager(log_dir=log_dir, registry_dir=reg_dir, keep_recent=10, max_running=1, max_queued=1)

    job1 = mgr.start(
        [sys.executable, "-c", "import time; print('job1'); time.sleep(0.25)"],
        cwd=None,
        env=None,
        timeout_sec=5,
        kind="t1",
    )
    st1 = mgr.status(job1)
    assert st1["status"] in {"running", "queued"}

    job2 = mgr.start(
        [sys.executable, "-c", "import time; print('job2'); time.sleep(0.05)"],
        cwd=None,
        env=None,
        timeout_sec=5,
        kind="t2",
    )
    st2 = mgr.status(job2)
    assert st2["status"] == "queued"
    assert (log_dir / f"{job2}.log").exists()
    assert (reg_dir / f"{job2}.json").exists()

    with pytest.raises(JobQueueFullError):
        mgr.start(
            [sys.executable, "-c", "print('job3')"],
            cwd=None,
            env=None,
            timeout_sec=5,
            kind="t3",
        )

    # job1 finishes, then the manager should drain the queue and run job2.
    s1_done = _wait_status(mgr, job1, timeout=5.0)
    assert s1_done["returncode"] == 0
    s2_done = _wait_status(mgr, job2, timeout=5.0)
    assert s2_done["returncode"] == 0


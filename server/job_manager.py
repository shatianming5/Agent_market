from __future__ import annotations

import subprocess
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class Job:
    id: str
    cmd: List[str]
    cwd: Optional[Path] = None
    env: Optional[dict] = None
    process: Optional[subprocess.Popen] = None
    logs: List[str] = field(default_factory=list)
    returncode: Optional[int] = None
    running: bool = False


class JobManager:
    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    def start(self, cmd: List[str], cwd: Optional[Path] = None, env: Optional[dict] = None) -> str:
        job_id = uuid.uuid4().hex[:12]
        job = Job(id=job_id, cmd=cmd, cwd=cwd, env=env)
        self._jobs[job_id] = job
        self._spawn(job)
        return job_id

    def _spawn(self, job: Job) -> None:
        job.running = True
        proc = subprocess.Popen(
            job.cmd,
            cwd=str(job.cwd) if job.cwd else None,
            env=job.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        job.process = proc

        def _reader() -> None:
            try:
                assert proc.stdout is not None
                for line in proc.stdout:
                    with self._lock:
                        job.logs.append(line.rstrip("\n"))
            finally:
                proc.wait()
                with self._lock:
                    job.returncode = proc.returncode
                    job.running = False

        t = threading.Thread(target=_reader, daemon=True)
        t.start()

    def status(self, job_id: str) -> dict:
        job = self._jobs.get(job_id)
        if not job:
            return {"error": "job not found"}
        with self._lock:
            return {
                "id": job.id,
                "cmd": job.cmd,
                "running": job.running,
                "returncode": job.returncode,
                "lines": len(job.logs),
            }

    def logs(self, job_id: str, offset: int = 0) -> dict:
        job = self._jobs.get(job_id)
        if not job:
            return {"error": "job not found"}
        with self._lock:
            total = len(job.logs)
            start = max(0, min(offset, total))
            chunk = job.logs[start:]
            return {
                "id": job.id,
                "offset": start,
                "next": start + len(chunk),
                "running": job.running,
                "returncode": job.returncode,
                "logs": chunk,
            }


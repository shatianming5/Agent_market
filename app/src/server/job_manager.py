from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable


@dataclass
class Job:
    id: str
    cmd: List[str]
    cwd: Optional[Path] = None
    env: Optional[dict] = None
    process: Optional[subprocess.Popen] = None
    logs: List[Dict[str, Any]] = field(default_factory=list)
    returncode: Optional[int] = None
    running: bool = False
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    finished_at: Optional[str] = None
    pid: Optional[int] = None


class JobManager:
    def __init__(self, max_seconds: int | None = None, on_step: Optional[Callable[[str, int, int, str, str], None]] = None) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()
        self._max_seconds = max_seconds
        self._on_step = on_step

    def start(self, cmd: List[str], cwd: Optional[Path] = None, env: Optional[dict] = None) -> str:
        job_id = uuid.uuid4().hex[:12]
        job = Job(id=job_id, cmd=cmd, cwd=cwd, env=env)
        self._jobs[job_id] = job
        self._spawn(job)
        return job_id

    def _spawn(self, job: Job) -> None:
        job.running = True
        creationflags = 0
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        proc = subprocess.Popen(
            job.cmd,
            cwd=str(job.cwd) if job.cwd else None,
            env=job.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            creationflags=creationflags,
        )
        job.process = proc
        job.pid = proc.pid
        start_mono = time.monotonic()

        def _reader() -> None:
            try:
                assert proc.stdout is not None
                for line in proc.stdout:
                    raw = line.rstrip("\n")
                    event = self._parse_event(raw)
                    if event is None:
                        event = {
                            "event": "log",
                            "message": raw,
                            "ts": datetime.now(timezone.utc).isoformat(),
                        }
                    else:
                        event.setdefault("ts", datetime.now(timezone.utc).isoformat())
                    self._append_event(job, event)
                    if event.get("event") == "progress":
                        self._handle_progress_event(job, event)
            finally:
                proc.wait()
                with self._lock:
                    job.returncode = proc.returncode
                    job.running = False
                    job.finished_at = datetime.now(timezone.utc).isoformat()

        t = threading.Thread(target=_reader, daemon=True)
        t.start()

        def _watchdog() -> None:
            if not self._max_seconds or self._max_seconds <= 0:
                return
            while True:
                with self._lock:
                    running = job.running
                if not running:
                    break
                if time.monotonic() - start_mono > float(self._max_seconds):
                    try:
                        if os.name == "nt":
                            subprocess.run([
                                "taskkill", "/PID", str(proc.pid), "/T", "/F"
                            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
                        else:
                            try:
                                os.killpg(proc.pid, signal.SIGTERM)
                            except Exception:
                                proc.terminate()
                    except Exception:
                        try:
                            proc.kill()
                        except Exception:
                            pass
                    break
                time.sleep(0.5)

        tw = threading.Thread(target=_watchdog, daemon=True)
        tw.start()

    def status(self, job_id: str) -> dict:
        job = self._jobs.get(job_id)
        if not job:
            return {"error": "job not found"}
        with self._lock:
            return {
                "id": job.id,
                "pid": job.pid,
                "cmd": job.cmd,
                "running": job.running,
                "returncode": job.returncode,
                "lines": len(job.logs),
                "started_at": job.started_at,
                "finished_at": job.finished_at,
            }

    def logs(self, job_id: str, offset: int = 0) -> dict:
        job = self._jobs.get(job_id)
        if not job:
            return {"error": "job not found"}
        with self._lock:
            total = len(job.logs)
            start = max(0, min(offset, total))
            chunk = [dict(event) for event in job.logs[start:]]
            return {
                "id": job.id,
                "pid": job.pid,
                "offset": start,
                "next": start + len(chunk),
                "running": job.running,
                "returncode": job.returncode,
                "logs": [self._event_to_text(event) for event in chunk],
                "events": chunk,
            }

    def list(self) -> List[dict]:
        with self._lock:
            return [
                {
                    "id": j.id,
                    "pid": j.pid,
                    "cmd": j.cmd,
                    "running": j.running,
                    "returncode": j.returncode,
                    "lines": len(j.logs),
                    "started_at": j.started_at,
                    "finished_at": j.finished_at,
                }
                for j in self._jobs.values()
            ]

    def terminate(self, job_id: str) -> dict:
        job = self._jobs.get(job_id)
        if not job:
            return {"error": "job not found"}
        proc = job.process
        if proc and job.running:
            try:
                if os.name == "nt":
                    subprocess.run([
                        "taskkill", "/PID", str(proc.pid), "/T", "/F"
                    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
                else:
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                    except Exception:
                        proc.terminate()
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    return {"ok": False, "message": "failed to terminate"}
            with self._lock:
                job.running = False
                job.finished_at = datetime.now(timezone.utc).isoformat()
            return {"ok": True}
        return {"ok": False, "message": "not running"}

    def _append_event(self, job: Job, event: Dict[str, Any]) -> None:
        with self._lock:
            job.logs.append(dict(event))

    def _handle_progress_event(self, job: Job, event: Dict[str, Any]) -> None:
        if not self._on_step:
            return
        try:
            current = int(event.get("current") or 0)
            total = int(event.get("total") or 0)
            label = str(event.get("label") or "")
            ts = str(event.get("ts") or datetime.now(timezone.utc).isoformat())
        except Exception:
            return
        try:
            self._on_step(job.id, current, total, label, ts)
        except Exception:
            pass

    @staticmethod
    def _parse_event(raw: str) -> Optional[Dict[str, Any]]:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict) or "event" not in data:
            return None
        return data

    @staticmethod
    def _event_to_text(event: Dict[str, Any]) -> str:
        kind = event.get("event")
        if kind == "log":
            return str(event.get("message", ""))
        if kind == "progress":
            current = event.get("current")
            total = event.get("total")
            label = event.get("label", "")
            status = event.get("status", "")
            pct = event.get("percent")
            parts = []
            if total is not None:
                parts.append(f"[progress] {current}/{total}")
            if label:
                parts.append(str(label))
            if status:
                parts.append(f"status={status}")
            if pct is not None:
                parts.append(f"{pct}%")
            return " ".join(parts) if parts else "[progress]"
        return json.dumps(event, ensure_ascii=False)


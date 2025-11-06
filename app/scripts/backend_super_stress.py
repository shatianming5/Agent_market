#!/usr/bin/env python
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import requests

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if SRC.exists():
    sp = str(SRC)
    if sp not in sys.path:
        sys.path.append(sp)
from agent_market.paths import USER_DATA_DIR  # type: ignore
PORT = 8022
BASE = f"http://127.0.0.1:{PORT}"


def start_server() -> subprocess.Popen:
    py = str(Path(sys.executable))
    args = [py, '-m', 'uvicorn', 'server.main:app', '--host', '127.0.0.1', '--port', str(PORT), '--log-level', 'warning']
    proc = subprocess.Popen(args, cwd=str(ROOT))
    # wait for health
    t0 = time.time()
    for _ in range(60):
        try:
            r = requests.get(f"{BASE}/health", timeout=0.5)
            if r.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(0.2)
    else:
        raise RuntimeError('server failed to start')
    return proc


def stop_server(proc: subprocess.Popen) -> None:
    try:
        proc.terminate()
        try:
            proc.wait(3)
        except Exception:
            proc.kill()
    except Exception:
        pass


def write_flow_config(tmp: Path, py_cmd: List[str]) -> Path:
    cfg = {
        'backtest': {
            'command': py_cmd,
        }
    }
    path = tmp / 'flow_cmd.json'
    path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf-8')
    return path


def run_job(flow_cfg: Path) -> Tuple[str, float]:
    # start job
    t0 = time.time()
    resp = requests.post(f"{BASE}/flow/run", json={
        'config': str(flow_cfg),
        'steps': 'backtest'
    }, timeout=3)
    j = resp.json()
    if 'job_id' not in j:
        raise RuntimeError(f"flow.run failed: {j}")
    job_id = j['job_id']
    offset = 0
    while True:
        r = requests.get(f"{BASE}/jobs/{job_id}/logs", params={'offset': offset}, timeout=3)
        jj = r.json()
        offset = jj.get('next') or offset
        if not jj.get('running'):
            break
        time.sleep(0.05)
    return job_id, time.time() - t0


def main() -> None:
    # prepare python command printing logs
    py = 'python'
    # prints 200 lines quickly and exits
    code = "import time;\nfor i in range(200):\n print(f'[S] line {i}'); time.sleep(0.001)\n"
    py_cmd = [py, '-c', code]

    tmp = USER_DATA_DIR / 'tmp' / 'backend_super_stress'
    tmp.mkdir(parents=True, exist_ok=True)
    flow_cfg = write_flow_config(tmp, py_cmd)

    proc = start_server()
    try:
        N = 80
        print(f'Starting {N} concurrent flow jobs...')
        t0 = time.time()
        durations: List[float] = []
        ok = 0
        with ThreadPoolExecutor(max_workers=16) as ex:
            futs = [ex.submit(run_job, flow_cfg) for _ in range(N)]
            for fut in as_completed(futs):
                try:
                    jid, dt = fut.result()
                    durations.append(dt)
                    ok += 1
                except Exception as exc:
                    print('job failed:', exc)
        t1 = time.time()
        if ok != N:
            raise SystemExit(f'Only {ok}/{N} jobs finished')
        durations.sort()
        p50 = durations[len(durations)//2]
        p95 = durations[int(len(durations)*0.95)-1]
        print(f'All {ok} jobs completed. total={t1-t0:.2f}s, p50={p50:.3f}s, p95={p95:.3f}s')
    finally:
        stop_server(proc)


if __name__ == '__main__':
    main()

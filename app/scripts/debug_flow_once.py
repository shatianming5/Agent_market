from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:8034"


def start_server() -> subprocess.Popen[bytes]:
    py = str(Path(sys.executable))
    args = [py, "-m", "uvicorn", "server.main:app", "--host", "127.0.0.1", "--port", BASE.rsplit(":", 1)[1]]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ROOT / "freqtrade"), env.get("PYTHONPATH", "")])
    proc = subprocess.Popen(args, cwd=str(ROOT), env=env)
    for _ in range(200):
        try:
            r = requests.get(f"{BASE}/health", timeout=1)
            if r.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(0.2)
    else:
        raise RuntimeError("server failed to start")
    return proc


def stop_server(proc: subprocess.Popen[bytes]) -> None:
    try:
        proc.terminate(); proc.wait(3)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def main() -> None:
    proc = start_server()
    try:
        payload = {"config": "configs/agent_flow_example.json", "steps": "backtest"}
        r = requests.post(f"{BASE}/flow/run", json=payload, timeout=20)
        j = r.json(); jid = j.get("job_id"); print("job:", j)
        off = 0
        while True:
            rr = requests.get(f"{BASE}/jobs/{jid}/logs", params={"offset": off}, timeout=30)
            jj = rr.json()
            off = jj.get("next", off)
            if not jj.get("running"):
                print("returncode:", jj.get("returncode"))
                break
            time.sleep(0.5)
    finally:
        stop_server(proc)


if __name__ == "__main__":
    main()


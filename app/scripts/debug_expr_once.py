from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:8032"


def start_server() -> subprocess.Popen[bytes]:
    py = str(Path(sys.executable))
    args = [py, "-m", "uvicorn", "server.main:app", "--host", "127.0.0.1", "--port", BASE.rsplit(":", 1)[1]]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ROOT / "freqtrade"), env.get("PYTHONPATH", "")])
    proc = subprocess.Popen(args, cwd=str(ROOT), env=env)
    for _ in range(100):
        try:
            r = requests.get(f"{BASE}/health", timeout=0.8)
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
    user_data = ROOT / "user_data"
    user_data.mkdir(exist_ok=True)
    feature = user_data / "freqai_features.json"
    if not feature.exists():
        payload = {
            "timeframe": "1h",
            "exchange": "binanceus",
            "pairs": ["BTC/USDT"],
            "features": [{"name": "feat_demo", "type": "sma_pct", "period": 5}],
        }
        feature.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    proc = start_server()
    try:
        expr_payload = {
            "config": "configs/config_freqai_multi.json",
            "feature_file": str(feature.relative_to(ROOT)),
            "output": "user_data/freqai_expressions.json",
            "timeframe": "1h",
            "llm_model": "gpt-3.5-turbo",
            "llm_count": 2,
            "llm_loops": 1,
            "llm_timeout": 10,
            "feedback_top": 0,
        }
        r = requests.post(f"{BASE}/run/expression", json=expr_payload, timeout=60)
        j = r.json(); jid = j.get("job_id"); print("job:", j)
        off = 0
        while True:
            rr = requests.get(f"{BASE}/jobs/{jid}/logs", params={"offset": off}, timeout=10)
            jj = rr.json()
            logs = jj.get("logs", [])
            for line in logs:
                print(line)
            off = jj.get("next", off)
            if not jj.get("running"):
                print("returncode:", jj.get("returncode"))
                break
            time.sleep(0.5)
    finally:
        stop_server(proc)


if __name__ == "__main__":
    main()


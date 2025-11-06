from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

import requests


ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:8031"


def start_server() -> subprocess.Popen[bytes]:
    py = str(Path(sys.executable))
    args = [py, "-m", "uvicorn", "server.main:app", "--host", "127.0.0.1", "--port", BASE.rsplit(":", 1)[1]]
    env = os.environ.copy()
    # Ensure src/ and freqtrade/ are importable by the server process
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ROOT / "freqtrade"), env.get("PYTHONPATH", "")])
    proc = subprocess.Popen(args, cwd=str(ROOT), env=env)
    # wait for health
    for _ in range(150):
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
        proc.terminate()
        try:
            proc.wait(3)
        except Exception:
            proc.kill()
    except Exception:
        pass


def poll_logs(job_id: str) -> Dict[str, Any]:
    off = 0
    last = {}
    for _ in range(600):
        r = requests.get(f"{BASE}/jobs/{job_id}/logs", params={"offset": off}, timeout=5)
        j = r.json()
        last = j
        off = j.get("next", off)
        if not j.get("running"):
            break
        time.sleep(0.2)
    return last


def main() -> None:
    # Ensure minimal feature file exists
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
        print("[1] GET /health")
        r = requests.get(f"{BASE}/health", timeout=5)
        print("status=", r.status_code, r.json())

        print("\n[2] POST /run/expression (no LLM)")
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
        print("status=", r.status_code)
        j = r.json()
        print("resp keys:", list(j.keys()))
        jid = j.get("job_id")
        if not jid:
            print("error: ", j)
            raise SystemExit(1)
        last = poll_logs(jid)
        print("expr finished: returncode=", last.get("returncode"), "lines=", last.get("next", 0))

        print("\n[3] POST /run/backtest (short timerange)")
        bt_payload = {
            "config": "configs/config_freqai_multi.json",
            "strategy": "ExpressionLongStrategy",
            "strategy_path": "resources/user_data/strategies",
            "timerange": "20210101-20210131",
            "freqaimodel": "LightGBMRegressor",
            "export": False,
        }
        r = requests.post(f"{BASE}/run/backtest", json=bt_payload, timeout=60)
        print("status=", r.status_code)
        j = r.json()
        print("resp keys:", list(j.keys()))
        jid = j.get("job_id")
        if not jid:
            print("error: ", j)
            raise SystemExit(1)
        last = poll_logs(jid)
        print("backtest finished: returncode=", last.get("returncode"), "lines=", last.get("next", 0))

        print("\n[4] POST /flow/run (steps=backtest)")
        r = requests.post(f"{BASE}/flow/run", json={"config": "configs/agent_flow_example.json", "steps": "backtest"}, timeout=30)
        print("status=", r.status_code)
        j = r.json()
        print("resp keys:", list(j.keys()))
        jid = j.get("job_id")
        if not jid:
            print("error: ", j)
            raise SystemExit(1)
        last = poll_logs(jid)
        print("flow backtest finished: returncode=", last.get("returncode"), "lines=", last.get("next", 0))

        print("\n[OK] All real calls finished.")
    finally:
        stop_server(proc)


if __name__ == "__main__":
    main()


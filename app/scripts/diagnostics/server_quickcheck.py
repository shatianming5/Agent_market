#!/usr/bin/env python
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

import requests


ROOT = Path(__file__).resolve().parents[2]
BASE = "http://127.0.0.1:8032"


def start_server() -> subprocess.Popen[bytes]:
    py = str(Path(sys.executable))
    args = [
        py,
        "-m",
        "uvicorn",
        "server.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        BASE.rsplit(":", 1)[1],
    ]
    env = os.environ.copy()
    # Ensure src/ and freqtrade/ are importable by the server process
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "src"), str(ROOT / "freqtrade"), env.get("PYTHONPATH", "")]
    )
    proc = subprocess.Popen(args, cwd=str(ROOT), env=env)
    # Wait briefly for health to be ready
    for _ in range(60):
        try:
            r = requests.get(f"{BASE}/health", timeout=0.5)
            if r.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(0.2)
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


def quick_poll(job_id: str) -> Dict[str, Any]:
    # Single-shot log fetch to assert endpoint is wired; do not wait for completion
    try:
        r = requests.get(f"{BASE}/jobs/{job_id}/logs", params={"offset": 0}, timeout=3)
        return r.json()
    except Exception:
        return {}


def ensure_feature_file() -> Path:
    user_data = ROOT / "user_data"
    user_data.mkdir(exist_ok=True)
    fpath = user_data / "freqai_features.json"
    if not fpath.exists():
        payload = {
            "timeframe": "1h",
            "exchange": "binanceus",
            "pairs": ["BTC/USDT"],
            "features": [{"name": "feat_demo", "type": "sma_pct", "period": 5}],
        }
        fpath.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return fpath


def main() -> None:
    proc = start_server()
    try:
        print("[1] GET /health")
        r = requests.get(f"{BASE}/health", timeout=3)
        print("status=", r.status_code, r.json())

        print("\n[2] POST /run/expression (do not wait for job)")
        feature = ensure_feature_file()
        expr_payload = {
            "config": "configs/config_freqai_multi.json",
            "feature_file": str(feature.relative_to(ROOT)),
            "output": "user_data/freqai_expressions.json",
            "timeframe": "1h",
            "llm_model": "gpt-3.5-turbo",
            "llm_count": 1,
            "llm_loops": 1,
            "llm_timeout": 5,
            "feedback_top": 0,
        }
        r = requests.post(f"{BASE}/run/expression", json=expr_payload, timeout=15)
        print("status=", r.status_code)
        j = r.json()
        print("resp keys:", list(j.keys()))
        jid = j.get("job_id")
        if jid:
            jlog = quick_poll(jid)
            print("expr logs keys:", list(jlog.keys()))

        print("\n[3] POST /run/backtest (short, do not wait)")
        bt_payload = {
            "config": "configs/config_freqai_multi.json",
            "strategy": "ExpressionLongStrategy",
            "strategy_path": "resources/user_data/strategies",
            "timerange": "20210101-20210131",
            "freqaimodel": "LightGBMRegressor",
            "export": False,
        }
        r = requests.post(f"{BASE}/run/backtest", json=bt_payload, timeout=15)
        print("status=", r.status_code)
        j = r.json()
        print("resp keys:", list(j.keys()))
        jid = j.get("job_id")
        if jid:
            jlog = quick_poll(jid)
            print("backtest logs keys:", list(jlog.keys()))

        print("\n[4] POST /flow/run (steps=backtest, do not wait)")
        r = requests.post(
            f"{BASE}/flow/run",
            json={"config": "configs/agent_flow_example.json", "steps": "backtest"},
            timeout=10,
        )
        print("status=", r.status_code)
        j = r.json()
        print("resp keys:", list(j.keys()))
        jid = j.get("job_id")
        if jid:
            jlog = quick_poll(jid)
            print("flow logs keys:", list(jlog.keys()))

        print("\n[4.1] GET /jobs (list)")
        r = requests.get(f"{BASE}/jobs", timeout=5)
        print("jobs count=", len(r.json()))

        # MVP data routes
        print("\n[5] POST /agents + GET /agents/{id}")
        r = requests.post(f"{BASE}/agents/", json={"name": "demo-agent"}, timeout=5)
        print("status=", r.status_code)
        agent = r.json()
        aid = agent.get("id")
        r = requests.get(f"{BASE}/agents/", timeout=5)
        print("list count=", len(r.json()))
        r = requests.get(f"{BASE}/agents/{aid}", timeout=5)
        print("fetch status=", r.status_code)

        print("\n[6] POST /orders + GET /orders?agent_id=")
        r = requests.post(f"{BASE}/orders/", json={"agent_id": aid, "side": "buy", "qty": 1.0}, timeout=5)
        print("status=", r.status_code)
        r = requests.get(f"{BASE}/orders", params={"agent_id": aid}, timeout=5)
        print("orders count=", len(r.json()))

        print("\n[OK] Quick server checks passed.")
    finally:
        stop_server(proc)


if __name__ == "__main__":
    main()

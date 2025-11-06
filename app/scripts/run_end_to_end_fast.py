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
BASE = "http://127.0.0.1:8035"


def start_server() -> subprocess.Popen[bytes]:
    py = str(Path(sys.executable))
    args = [py, "-m", "uvicorn", "server.main:app", "--host", "127.0.0.1", "--port", BASE.rsplit(":", 1)[1]]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ROOT / "freqtrade"), env.get("PYTHONPATH", "")])
    proc = subprocess.Popen(args, cwd=str(ROOT), env=env)
    for _ in range(180):
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


def poll_until_done(job_id: str, timeout: float = 900.0) -> Dict[str, Any]:
    start = time.time()
    off = 0
    last: Dict[str, Any] = {}
    while time.time() - start < timeout:
        r = requests.get(f"{BASE}/jobs/{job_id}/logs", params={"offset": off}, timeout=10)
        j = r.json()
        last = j
        off = j.get("next", off)
        if not j.get("running"):
            break
        time.sleep(0.3)
    last["duration_sec"] = round(time.time() - start, 2)
    return last


def ensure_feature_file() -> Path:
    """Create a small feature file for fast runs, regardless of existing heavy file."""
    user_data = ROOT / "user_data"
    user_data.mkdir(exist_ok=True)
    feature = user_data / "freqai_features_fast.json"
    payload = {
        "timeframe": "1h",
        "exchange": "binanceus",
        "pairs": ["BTC/USDT"],
        "features": [
            {"name": "feat_rsi_14", "type": "rsi", "period": 14},
            {"name": "feat_return_z_5", "type": "return_zscore", "period": 5},
            {"name": "feat_kama_pct_10", "type": "kama_pct", "period": 10},
        ],
    }
    feature.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return feature


def main() -> None:
    feature = ensure_feature_file()
    proc = start_server()
    try:
        # 1) expression
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
            "fast": True,
            "top": 40,
            "combo_top": 0,
            "stability_windows": 1,
            "stability_min_samples": 50,
        }
        t0 = time.time()
        r = requests.post(f"{BASE}/run/expression", json=expr_payload, timeout=60)
        j = r.json(); jid = j.get("job_id")
        expr_res = poll_until_done(jid)
        expr_res["queue_latency_sec"] = round(expr_res.get("duration_sec", 0) - (time.time() - t0), 2)
        print("[expression] rc=", expr_res.get("returncode"), "duration=", expr_res.get("duration_sec"), "lines=", len(expr_res.get("logs", [])))

        # 2) backtest
        bt_payload = {
            "config": "configs/config_freqai_multi.json",
            "strategy": "ExpressionLongStrategy",
            "strategy_path": "resources/user_data/strategies",
            "timerange": "20210101-20210115",
            "freqaimodel": "LightGBMRegressor",
            "export": False,
            "fast": True,
        }
        t0 = time.time()
        r = requests.post(f"{BASE}/run/backtest", json=bt_payload, timeout=60)
        j = r.json(); jid = j.get("job_id")
        bt_res = poll_until_done(jid)
        bt_res["queue_latency_sec"] = round(bt_res.get("duration_sec", 0) - (time.time() - t0), 2)
        print("[backtest] rc=", bt_res.get("returncode"), "duration=", bt_res.get("duration_sec"), "lines=", len(bt_res.get("logs", [])))

        # 3) flow (backtest only)
        t0 = time.time()
        r = requests.post(f"{BASE}/flow/run", json={"config": "configs/agent_flow_example.json", "steps": "backtest"}, timeout=30)
        j = r.json(); jid = j.get("job_id")
        flow_res = poll_until_done(jid)
        flow_res["queue_latency_sec"] = round(flow_res.get("duration_sec", 0) - (time.time() - t0), 2)
        print("[flow/backtest] rc=", flow_res.get("returncode"), "duration=", flow_res.get("duration_sec"), "lines=", len(flow_res.get("logs", [])))

    finally:
        stop_server(proc)


if __name__ == "__main__":
    main()

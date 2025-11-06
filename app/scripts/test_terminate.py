from __future__ import annotations

import time
from pathlib import Path

import requests


BASE = "http://127.0.0.1:8032"


def main() -> None:
    # Start a backtest job (likely long enough)
    payload = {
        "config": "configs/config_freqai_multi.json",
        "strategy": "ExpressionLongStrategy",
        "strategy_path": "resources/user_data/strategies",
        "timerange": "20210101-20211231",
        "freqaimodel": "LightGBMRegressor",
        "export": False,
        "fast": False,
    }
    r = requests.post(f"{BASE}/run/backtest", json=payload, timeout=15)
    r.raise_for_status()
    jid = r.json().get("job_id")
    assert jid, f"no job_id: {r.text}"
    print("started job:", jid)
    time.sleep(1)
    # Terminate it
    r = requests.post(f"{BASE}/jobs/{jid}/terminate", timeout=10)
    print("terminate:", r.status_code, r.json())
    # Poll status until not running
    for _ in range(30):
        s = requests.get(f"{BASE}/jobs/{jid}/status", timeout=5).json()
        print("status:", s.get("running"), s.get("returncode"))
        if not s.get("running"):
            break
        time.sleep(0.5)


if __name__ == "__main__":
    main()


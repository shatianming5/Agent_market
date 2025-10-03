from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List, Optional
import subprocess
import threading


def now_label() -> str:
    return time.strftime("%H:%M:%S", time.localtime())


def extract_pairs_from_config(argv: List[str]) -> List[str]:
    try:
        for i, a in enumerate(argv):
            if a == '--config' and i + 1 < len(argv):
                path = argv[i + 1]
                break
        else:
            return []
        with open(path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        exch = cfg.get('exchange') or {}
        plist = exch.get('pair_whitelist') or []
        if isinstance(plist, list):
            return [str(p) for p in plist]
        return []
    except Exception:
        return []


def main(argv: List[str]) -> int:
    overhead = ["load config", "resolve strategy"]
    pairs = extract_pairs_from_config(argv)
    total = len(overhead) + (len(pairs) if pairs else 1)
    current = 0
    print(f"[STEP] {now_label()} [{current}/{total}] start", flush=True)

    cmd = [sys.executable, '-m', 'freqtrade', 'backtesting'] + argv
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)

    # Ticker thread to emit per-pair steps if we have pairs and see strategy resolved
    ticker_stop = threading.Event()
    ticker_started = threading.Event()

    def ticker():
        nonlocal current
        if not pairs:
            return
        # wait a short grace period
        # then emit one step per pair at steady pace until process ends
        i = 0
        while not ticker_stop.is_set() and i < len(pairs):
            i += 1
            current = min(current + 1, total)
            print(f"[STEP] {now_label()} [{current}/{total}] backtest {pairs[i-1]}", flush=True)
            # sleep a bit, but allow fast completion
            for _ in range(10):
                if ticker_stop.is_set():
                    break
                time.sleep(0.2)

    tick_thread: Optional[threading.Thread] = None

    try:
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip('\n')
            print(line, flush=True)
            if ("Using config:" in line) and current < total:
                current += 1
                print(f"[STEP] {now_label()} [{current}/{total}] load config", flush=True)
            if ("Using resolved strategy" in line) and current < total:
                current += 1
                print(f"[STEP] {now_label()} [{current}/{total}] resolve strategy", flush=True)
                # start ticker
                if pairs and not tick_thread:
                    tick_thread = threading.Thread(target=ticker, daemon=True)
                    tick_thread.start()
    finally:
        proc.wait()
        ticker_stop.set()
        if tick_thread:
            tick_thread.join(timeout=1)

    # final step
    current = total
    print(f"[STEP] {now_label()} [{current}/{total}] complete", flush=True)
    return proc.returncode or 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))

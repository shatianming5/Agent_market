from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List
import subprocess


def now_label() -> str:
    return time.strftime("%H:%M:%S", time.localtime())


def main(argv: List[str]) -> int:
    steps = [
        ("start", lambda line: False),
        ("load config", lambda line: "Using config:" in line),
        ("resolve strategy", lambda line: "Using resolved strategy" in line),
        ("run backtest", lambda line: "Backtested" in line or "STRATEGY SUMMARY" in line),
        ("complete", lambda line: False),
    ]
    total = len(steps) - 1  # exclude final complete until process end
    current = 0
    print(f"[STEP] {now_label()} [{current}/{total}] start", flush=True)

    cmd = [sys.executable, '-m', 'freqtrade', 'backtesting'] + argv
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
    try:
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip('\n')
            print(line, flush=True)
            # detect step changes
            if current < total and steps[current + 1][1](line):
                current += 1
                print(f"[STEP] {now_label()} [{current}/{total}] {steps[current][0]}", flush=True)
    finally:
        proc.wait()

    # final step
    print(f"[STEP] {now_label()} [{total}/{total}] complete", flush=True)
    return proc.returncode or 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))


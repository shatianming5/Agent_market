#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path


def _stream(proc: subprocess.Popen) -> None:
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            try:
                sys.stdout.write(line)
            except Exception:
                pass
    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description='Wrapper to run backtest command with coarse progress markers')
    parser.add_argument('cmd', nargs=argparse.REMAINDER, help='Command to run (use -- to separate)')
    args = parser.parse_args()

    # Support invocation like: python backtest_wrapper.py -- freqtrade backtesting ...
    cmd = args.cmd
    if cmd and cmd[0] == '--':
        cmd = cmd[1:]
    if not cmd:
        print('[wrapper] no command provided', file=sys.stderr)
        return 2

    # Start subprocess
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    t = threading.Thread(target=_stream, args=(proc,), daemon=True)
    t.start()

    # Emit coarse progress markers based on elapsed time
    start = time.time()
    expected = float(os.environ.get('BT_EXPECTED_SECS', '300'))  # default 5 min heuristic
    last_pct = -1
    try:
        while True:
            if proc.poll() is not None:
                break
            elapsed = time.time() - start
            pct = int(min(95.0, max(0.0, (elapsed / expected) * 95.0)))
            if pct != last_pct and pct in (5, 10, 20, 30, 40, 50, 60, 70, 80, 90):
                print(f"[FLOW] PROGRESS {pct}%")
                last_pct = pct
            time.sleep(3.0)
    except KeyboardInterrupt:
        try:
            proc.terminate()
        except Exception:
            pass
        return 130

    rc = proc.wait()
    print("[FLOW] PROGRESS 100%")
    return rc


if __name__ == '__main__':
    raise SystemExit(main())


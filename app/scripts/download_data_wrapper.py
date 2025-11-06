from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List
import subprocess


def now_label() -> str:
    return time.strftime("%H:%M:%S", time.localtime())


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    # we don't validate, just collect known flags to estimate total work
    parser.add_argument('--config')
    parser.add_argument('--pairs-file')
    parser.add_argument('--timerange')
    parser.add_argument('--exchange')
    parser.add_argument('--erase', action='store_true')
    parser.add_argument('--new-pairs', action='store_true')
    parser.add_argument('--dl-trades', action='store_true')
    parser.add_argument('--prepend', action='store_true')
    parser.add_argument('-t', dest='timeframes', action='append', default=[])
    parser.add_argument('-p', dest='pairs', action='append', default=[])
    # parse only known, ignore the rest
    ns, _ = parser.parse_known_args(argv)
    return ns


def load_pairs_from_file(pairs_file: str) -> List[str]:
    try:
        p = Path(pairs_file)
        if not p.exists():
            return []
        return [line.strip() for line in p.read_text(encoding='utf-8').splitlines() if line.strip() and not line.strip().startswith('#')]
    except Exception:
        return []


def main(argv: List[str]) -> int:
    ns = parse_args(argv)
    timeframes = list(dict.fromkeys(ns.timeframes or []))
    pairs = list(dict.fromkeys(ns.pairs or []))
    if ns.pairs_file:
        pairs_from_file = load_pairs_from_file(ns.pairs_file)
        for p in pairs_from_file:
            if p not in pairs:
                pairs.append(p)
    total = max(1, (len(timeframes) or 1) * (len(pairs) or 1))
    current = 0

    print(f"[STEP] {now_label()} [0/{total}] start download", flush=True)

    cmd = [sys.executable, '-m', 'freqtrade', 'download-data'] + argv
    # stream and proxy output
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            # proxy original line
            print(line.rstrip('\n'), flush=True)
            # heuristic: count completed items
            if 'Downloaded data for ' in line:
                current += 1
                label = line.strip()
                print(f"[STEP] {now_label()} [{min(current,total)}/{total}] {label}", flush=True)
    finally:
        proc.wait()

    # final step
    print(f"[STEP] {now_label()} [{total}/{total}] complete", flush=True)
    return proc.returncode or 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))


from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def run(cmd, timeout=None):
    print("$", " ".join(cmd))
    return subprocess.run(cmd, cwd=str(ROOT), env=os.environ.copy(), timeout=timeout, check=True)


def main() -> None:
    py = sys.executable
    # 1) fast unit tests
    run([py, "-m", "pytest", "-q", "-rs"], timeout=300)
    # 2) server quick endpoint checks
    run([py, "scripts/server_quickcheck.py"], timeout=180)
    print("\n[ACCEPT] All checks passed under time limits.")


if __name__ == "__main__":
    main()


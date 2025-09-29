#!/usr/bin/env python
from __future__ import annotations

import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from server.job_manager import JobManager  # type: ignore


def main() -> None:
    jm = JobManager()
    # Simple echo job
    py = sys.executable
    code = "print('smoke-ok')"
    job_id = jm.start([py, '-c', code], cwd=ROOT, env=os.environ.copy())
    print('job_id:', job_id)
    off = 0
    while True:
        st = jm.status(job_id)
        chunk = jm.logs(job_id, off)
        off = chunk.get('next', off)
        for line in chunk.get('logs', []):
            print('> ', line)
        if not st.get('running'):
            print('exit:', st.get('returncode'))
            break
        time.sleep(0.2)


if __name__ == '__main__':
    main()


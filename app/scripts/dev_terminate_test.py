from __future__ import annotations

import os
import time
import requests


BASE = os.environ.get('BASE', 'http://127.0.0.1:8032')


def main() -> None:
    r = requests.post(f'{BASE}/jobs/dev/sleep', params={'secs': 10}, timeout=10)
    print('start dev sleep:', r.status_code, r.text)
    jid = r.json().get('job_id')
    assert jid
    time.sleep(1)
    r = requests.post(f'{BASE}/jobs/{jid}/terminate', timeout=10)
    print('terminate:', r.status_code, r.text)
    for _ in range(10):
        s = requests.get(f'{BASE}/jobs/{jid}/status', timeout=5).json()
        print('status:', s.get('running'), s.get('returncode'))
        if not s.get('running'):
            break
        time.sleep(0.5)


if __name__ == '__main__':
    main()


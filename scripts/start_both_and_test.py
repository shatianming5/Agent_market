from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]


def wait_http(url: str, timeout: float = 60.0) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = requests.get(url, timeout=1.5)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.5)
    raise TimeoutError(f"timeout waiting for {url}")


def main() -> None:
    py = sys.executable
    env = os.environ.copy()
    env.setdefault('PYTHONPATH', str(ROOT / 'src'))

    # 1) Start backend
    api = subprocess.Popen([py, '-m', 'uvicorn', 'server.main:app', '--host', '127.0.0.1', '--port', '8032'], cwd=str(ROOT), env=env)

    try:
        wait_http('http://127.0.0.1:8032/health', timeout=30)
        print('[ok] backend ready')

        # 2) Install and start frontend (vite)
        webdir = ROOT / 'web'
        npm = 'npm.cmd' if os.name == 'nt' else 'npm'
        subprocess.run([npm, 'i'], cwd=str(webdir), check=True)

        web_env = os.environ.copy()
        web_env['VITE_API_BASE'] = 'http://127.0.0.1:8032'
        web = subprocess.Popen([npm, 'run', 'dev', '--', '--host', '127.0.0.1', '--port', '5173'], cwd=str(webdir), env=web_env)

        try:
            wait_http('http://127.0.0.1:5173', timeout=45)
            print('[ok] frontend ready')

            # 3) Connectivity test via vite proxy -> backend /health
            r = requests.get('http://127.0.0.1:5173/api/health', timeout=5)
            print('[GET] /api/health status=', r.status_code, 'body=', r.text[:120])
            if r.status_code != 200 or 'ok' not in r.text:
                raise SystemExit(2)

            print('\n[PASS] Frontend <-> Backend connectivity verified.')
        finally:
            try:
                web.terminate()
                web.wait(timeout=5)
            except Exception:
                try:
                    web.kill()
                except Exception:
                    pass
    finally:
        try:
            api.terminate()
            api.wait(timeout=5)
        except Exception:
            try:
                api.kill()
            except Exception:
                pass


if __name__ == '__main__':
    main()

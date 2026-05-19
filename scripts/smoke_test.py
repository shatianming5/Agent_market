#!/usr/bin/env python
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Callable, Optional

from fastapi.testclient import TestClient


def main() -> None:
    # Ensure repo root and src are importable when running as a script.
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    # Import app
    import server.main as srv  # type: ignore

    app = srv.app
    client = TestClient(app)
    api_key = os.environ.get("AGENT_MARKET_API_KEY", "").strip()
    if api_key:
        client.headers.update({"X-API-Key": api_key})

    results: list[tuple[str, bool, str]] = []

    def check(name: str, fn: Callable[[], Any]) -> None:
        ok = False
        msg = ""
        try:
            fn()
            ok = True
        except AssertionError as e:  # noqa: PERF203
            msg = str(e)
        except Exception as e:  # pragma: no cover
            msg = f"unexpected: {e}"
        results.append((name, ok, msg))

    def request_json(
        method: str,
        path: str,
        *,
        payload: Optional[dict[str, Any]] = None,
        expected_status: int = 200,
    ) -> dict[str, Any]:
        method_u = str(method or "").strip().upper()
        if method_u == "GET":
            resp = client.get(path)
        elif method_u == "POST":
            resp = client.post(path, json=payload or {})
        else:
            raise ValueError(f"Unsupported method: {method!r}")
        assert resp.status_code == expected_status, f"{method_u} {path} -> {resp.status_code}"
        body = resp.json()
        assert isinstance(body, dict), f"{method_u} {path} response must be dict, got {type(body)}"
        return body

    # ---------- Basic endpoints ----------
    def _check_health() -> None:
        j = request_json("GET", "/health")
        assert_eq(j.get("status"), "ok")

    check("GET /health", _check_health)

    def _check_root(path: str) -> None:
        j = request_json("GET", path)
        assert_in("message", j)
        assert_in("docs", j)
        assert_in("health", j)

    check("GET /", lambda: _check_root("/"))
    check("GET /index", lambda: _check_root("/index"))

    # ---------- Settings ----------
    def _check_settings_get() -> None:
        j = request_json("GET", "/settings")
        assert_in("default_timeframe", j)

    check("GET /settings", _check_settings_get)

    def _check_settings_invalid_timeframe() -> None:
        j = request_json(
            "POST",
            "/settings",
            payload={"default_timeframe": "4hours"},
            expected_status=400,
        )
        assert_eq(j.get("status"), "error")
        assert_eq(j.get("code"), "INVALID_TIMEFRAME")

    check("POST /settings invalid timeframe", _check_settings_invalid_timeframe)

    def _check_settings_ok() -> None:
        j = request_json("POST", "/settings", payload={"default_timeframe": "4h"})
        assert_eq(j.get("status"), "ok")
        assert_in("settings", j)

    check("POST /settings ok", _check_settings_ok)

    # ---------- Files & results endpoints ----------
    def _check_features_top_missing() -> None:
        resp = client.get("/features/top", params={"file": "user_data/missing.json"})
        assert resp.status_code == 404
        j = resp.json()
        assert_eq(j.get("status"), "error")
        assert_eq(j.get("code"), "FEATURE_FILE_NOT_FOUND")

    check("GET /features/top missing", _check_features_top_missing)

    def _check_results_list_missing_dir() -> None:
        resp = client.get("/results/list", params={"results_dir": "user_data/does_not_exist"})
        assert resp.status_code == 404
        j = resp.json()
        assert_eq(j.get("status"), "error")
        assert_eq(j.get("code"), "RESULTS_DIR_NOT_FOUND")

    check("GET /results/list (missing dir -> error)", _check_results_list_missing_dir)

    # Accept success (when example backtests exist) or NO_ARCHIVES error otherwise
    def _check_results_latest_summary() -> None:
        resp = client.get("/results/latest-summary")
        assert resp.status_code == 200
        j = resp.json()
        if j.get("status") != "error":
            assert_in("profit_total_pct", j)

    check("GET /results/latest-summary", _check_results_latest_summary)

    # ---------- Jobs error wrapping ----------
    for path in ["/jobs/unknown/status", "/jobs/unknown/logs", "/jobs/unknown/cancel"]:
        def _check_job_not_found(p: str = path) -> None:
            resp = client.get(p) if p.endswith("status") or p.endswith("logs") else client.post(p)
            assert resp.status_code == 404
            j = resp.json()
            assert_eq(j.get("status"), "error")
            assert_eq(j.get("code"), "JOB_NOT_FOUND")

        check(f"{path} -> JOB_NOT_FOUND", _check_job_not_found)

    # ---------- run/* validation-only checks (no heavy exec) ----------
    cfg_dir = Path("configs")
    cfg_path = cfg_dir / "config_freqai_multi.json"

    def _check_run_feature_config_not_found() -> None:
        resp = client.post("/run/feature", json={"config": "configs/missing.json", "timeframe": "4h"})
        assert resp.status_code == 404
        j = resp.json()
        assert_eq(j.get("status"), "error")
        assert_eq(j.get("code"), "CONFIG_NOT_FOUND")

    check("POST /run/feature -> CONFIG_NOT_FOUND", _check_run_feature_config_not_found)

    if cfg_path.exists():
        def _check_run_feature_invalid_timeframe() -> None:
            resp = client.post("/run/feature", json={"config": str(cfg_path), "timeframe": "4hours"})
            assert resp.status_code == 400
            j = resp.json()
            assert_eq(j.get("status"), "error")
            assert_eq(j.get("code"), "INVALID_TIMEFRAME")

        check("POST /run/feature -> INVALID_TIMEFRAME", _check_run_feature_invalid_timeframe)

    def _check_run_expression_config_not_found() -> None:
        resp = client.post(
            "/run/expression",
            json={"config": "configs/missing.json", "feature_file": "user_data/missing.json"},
        )
        assert resp.status_code == 404
        j = resp.json()
        assert_eq(j.get("status"), "error")
        assert_eq(j.get("code"), "CONFIG_NOT_FOUND")

    check("POST /run/expression -> CONFIG_NOT_FOUND", _check_run_expression_config_not_found)

    def _check_run_backtest_config_not_found() -> None:
        resp = client.post("/run/backtest", json={"config": "configs/missing.json"})
        assert resp.status_code == 404
        j = resp.json()
        assert_eq(j.get("status"), "error")
        assert_eq(j.get("code"), "CONFIG_NOT_FOUND")

    check("POST /run/backtest -> CONFIG_NOT_FOUND", _check_run_backtest_config_not_found)

    # ---------- flow progress ----------
    def _check_flow_progress_unknown_job() -> None:
        resp = client.get("/flow/progress/unknown")
        assert resp.status_code == 404
        j = resp.json()
        assert_eq(j.get("status"), "error")
        assert_eq(j.get("code"), "JOB_NOT_FOUND")

    check("GET /flow/progress unknown job", _check_flow_progress_unknown_job)

    # ---------- web assets quick sanity ----------
    index_html = Path("web/index.html").read_text(encoding="utf-8", errors="replace")
    assert "Agent Market" in index_html
    assert 'id="root"' in index_html
    assert 'data-tab="miner"' in index_html

    # ---------- Report ----------
    width = max(len(n) for n, _, _ in results)
    ok_count = 0
    for name, ok, msg in results:
        ok_count += 1 if ok else 0
        print(f"[{'OK' if ok else 'FAIL'}] {name:<{width}}  {'' if ok else msg}")
    print(f"passed {ok_count}/{len(results)} checks")
    if ok_count != len(results):
        sys.exit(1)


# helpers
def assert_eq(a: Any, b: Any) -> None:
    assert a == b, f"expected {b!r}, got {a!r}"


def assert_in(k: str, obj: dict[str, Any]) -> None:
    assert k in obj, f"missing key: {k}"


if __name__ == "__main__":
    # Allow running without installing server/extra deps
    os.environ.setdefault("PYTHONPATH", str(Path(__file__).resolve().parents[1] / 'src'))
    main()

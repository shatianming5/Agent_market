#!/usr/bin/env python
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

from fastapi.testclient import TestClient


def main() -> None:
    # Import app
    import server.main as srv  # type: ignore

    app = srv.app
    client = TestClient(app)

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

    # ---------- Basic endpoints ----------
    check("GET /health", lambda: (
        lambda j: (
            assert_eq(j.get("status"), "ok")
        )
    )(client.get("/health").json()))

    check("GET /", lambda: (
        lambda j: (
            assert_in("message", j),
            assert_in("docs", j),
            assert_in("health", j)
        )
    )(client.get("/").json()))

    check("GET /index", lambda: (
        lambda j: (
            assert_in("message", j),
            assert_in("docs", j),
            assert_in("health", j)
        )
    )(client.get("/index").json()))

    # ---------- Settings ----------
    check("GET /settings", lambda: (
        lambda j: (
            assert_in("default_timeframe", j)
        )
    )(client.get("/settings").json()))

    check("POST /settings invalid timeframe", lambda: (
        lambda j: (
            assert_eq(j.get("status"), "error"),
            assert_eq(j.get("code"), "INVALID_TIMEFRAME")
        )
    )(client.post("/settings", json={"default_timeframe": "4hours"}).json()))

    check("POST /settings ok", lambda: (
        lambda j: (
            assert_eq(j.get("status"), "ok"),
            assert_in("settings", j)
        )
    )(client.post("/settings", json={"default_timeframe": "4h"}).json()))

    # ---------- Files & results endpoints ----------
    check("GET /features/top missing", lambda: (
        lambda j: (
            assert_eq(j.get("status"), "error"),
            assert_eq(j.get("code"), "FEATURE_FILE_NOT_FOUND")
        )
    )(client.get("/features/top", params={"file": "user_data/missing.json"}).json()))

    # If fallback freqtrade/user_data/backtest_results exists, this should return ok (items list),
    # otherwise return standardized error. Accept both.
    check("GET /results/list (fallback aware)", lambda: (
        lambda j: (
            None if (j.get("status") == "error") else assert_in("items", j)
        )
    )(client.get("/results/list", params={"results_dir": "user_data/does_not_exist"}).json()))

    # Accept success (when example backtests exist under freqtrade/) or NO_ARCHIVES error otherwise
    check("GET /results/latest-summary (fallback aware)", lambda: (
        lambda j: (
            None if (j.get("status") == "error") else assert_in("profit_total_pct", j)
        )
    )(client.get("/results/latest-summary").json()))

    # ---------- Jobs error wrapping ----------
    for path in ["/jobs/unknown/status", "/jobs/unknown/logs", "/jobs/unknown/cancel"]:
        check(f"{path} -> JOB_NOT_FOUND", lambda p=path: (
            lambda j: (
                assert_eq(j.get("status"), "error"),
                assert_eq(j.get("code"), "JOB_NOT_FOUND")
            )
        )(client.get(p).json() if p.endswith("status") or p.endswith("logs") else client.post(p).json()))

    # ---------- run/* validation-only checks (no heavy exec) ----------
    cfg_dir = Path("configs")
    cfg_path = cfg_dir / "config_freqai_multi.json"

    check("POST /run/feature -> CONFIG_NOT_FOUND", lambda: (
        lambda j: (
            assert_eq(j.get("status"), "error"),
            assert_eq(j.get("code"), "CONFIG_NOT_FOUND")
        )
    )(client.post("/run/feature", json={"config": "configs/missing.json", "timeframe": "4h"}).json()))

    if cfg_path.exists():
        check("POST /run/feature -> INVALID_TIMEFRAME", lambda: (
            lambda j: (
                assert_eq(j.get("status"), "error"),
                assert_eq(j.get("code"), "INVALID_TIMEFRAME")
            )
        )(client.post("/run/feature", json={"config": str(cfg_path), "timeframe": "4hours"}).json()))

    check("POST /run/expression -> CONFIG_NOT_FOUND", lambda: (
        lambda j: (
            assert_eq(j.get("status"), "error"),
            assert_eq(j.get("code"), "CONFIG_NOT_FOUND")
        )
    )(client.post("/run/expression", json={"config": "configs/missing.json", "feature_file": "user_data/missing.json"}).json()))

    check("POST /run/backtest -> CONFIG_NOT_FOUND", lambda: (
        lambda j: (
            assert_eq(j.get("status"), "error"),
            assert_eq(j.get("code"), "CONFIG_NOT_FOUND")
        )
    )(client.post("/run/backtest", json={"config": "configs/missing.json"}).json()))

    # ---------- flow progress ----------
    check("GET /flow/progress unknown job", lambda: (
        lambda j: (
            assert_eq(j.get("status"), "error"),
            assert_eq(j.get("code"), "JOB_NOT_FOUND")
        )
    )(client.get("/flow/progress/unknown").json()))

    # ---------- web assets quick sanity ----------
    index_html = Path("web/index.html").read_text(encoding="utf-8")
    assert "节点面板" in index_html or "Agent Market Flow" in index_html

    # ---------- Report ----------
    width = max(len(n) for n, _, _ in results)
    ok_count = 0
    for name, ok, msg in results:
        ok_count += 1 if ok else 0
        print(f"[{'OK' if ok else 'FAIL'}] {name:<{width}}  {'' if ok else msg}")
    print(f"passed {ok_count}/{len(results)} checks")


# helpers
def assert_eq(a: Any, b: Any) -> None:
    assert a == b, f"expected {b!r}, got {a!r}"


def assert_in(k: str, obj: dict[str, Any]) -> None:
    assert k in obj, f"missing key: {k}"


if __name__ == "__main__":
    # Allow running without installing server/extra deps
    os.environ.setdefault("PYTHONPATH", str(Path(__file__).resolve().parents[1] / 'src'))
    main()

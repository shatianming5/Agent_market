from __future__ import annotations

import gzip
import json
from pathlib import Path

from fastapi.testclient import TestClient


def _client() -> TestClient:
    import server.main as srv  # type: ignore

    return TestClient(srv.app)


def test_factor_compile_returns_planmd_error_codes() -> None:
    client = _client()

    # 1) missing spec -> INVALID_FACTOR_SPEC
    res = client.post("/run/factor_compile", json={})
    body = res.json()
    assert body.get("status") == "error"
    assert body.get("code") == "INVALID_FACTOR_SPEC"

    base_meta = {"timeframe": "1h", "universe": ["BTC/USDT"], "data_sources": ["ohlcv"]}

    # 2) lookahead -> LOOKAHEAD_DETECTED
    spec_lookahead = {
        "name": "x",
        "expr": {"op": "shift", "args": [{"op": "var", "args": ["close"]}, -1]},
        "meta": base_meta,
    }
    res = client.post("/run/factor_compile", json={"spec": spec_lookahead})
    body = res.json()
    assert body.get("status") == "error"
    assert body.get("code") == "LOOKAHEAD_DETECTED"

    # 3) unknown operator -> UNKNOWN_OPERATOR
    spec_unknown = {
        "name": "x",
        "expr": {"op": "unknown_op", "args": [{"op": "var", "args": ["close"]}]},
        "meta": base_meta,
    }
    res = client.post("/run/factor_compile", json={"spec": spec_unknown})
    body = res.json()
    assert body.get("status") == "error"
    assert body.get("code") == "UNKNOWN_OPERATOR"

    # 4) scalar output -> TYPECHECK_FAILED
    spec_scalar = {"name": "x", "expr": {"op": "const", "args": [1.0]}, "meta": base_meta}
    res = client.post("/run/factor_compile", json={"spec": spec_scalar})
    body = res.json()
    assert body.get("status") == "error"
    assert body.get("code") == "TYPECHECK_FAILED"

    # 5) complexity budget -> COMPLEXITY_BUDGET_EXCEEDED
    spec_budget = {
        "name": "x",
        "expr": {"op": "var", "args": ["close"]},
        "constraints": {"complexity_budget": {"max_nodes": 1, "max_depth": 8}},
        "meta": base_meta,
    }
    res = client.post("/run/factor_compile", json={"spec": spec_budget})
    body = res.json()
    assert body.get("status") == "error"
    assert body.get("code") == "COMPLEXITY_BUDGET_EXCEEDED"


def test_lob_rebuild_returns_data_not_found_and_sequence_gap(tmp_path: Path) -> None:
    client = _client()

    # Missing capture dir -> DATA_NOT_FOUND
    res = client.post(
        "/run/lob_rebuild",
        json={
            "exchange": "kucoin",
            "capture_dir": "artifacts/__does_not_exist__",
            "snapshot": "tests/fixtures/kucoin_lob_snapshot.json",
            "symbol": "BTC-USDT",
            "depth": 20,
        },
    )
    body = res.json()
    assert body.get("status") == "error"
    assert body.get("code") == "DATA_NOT_FOUND"

    # Sequence gap -> LOB_SEQUENCE_GAP
    cap_dir = tmp_path / "cap"
    cap_dir.mkdir(parents=True, exist_ok=True)
    level2_path = cap_dir / "level2.ndjson.gz"
    with gzip.open(level2_path, "wt", encoding="utf-8") as fh:
        fh.write(json.dumps({"type": "message", "data": {"sequenceStart": 200, "sequenceEnd": 200}}) + "\n")

    snap_path = tmp_path / "snapshot.json"
    snap_path.write_text(json.dumps({"symbol": "BTC-USDT", "sequence": 99, "bids": [], "asks": []}), encoding="utf-8")

    res = client.post(
        "/run/lob_rebuild",
        json={
            "exchange": "kucoin",
            "capture_dir": str(cap_dir),
            "snapshot": str(snap_path),
            "symbol": "BTC-USDT",
            "depth": 20,
            "out_dir": str(tmp_path / "out"),
        },
    )
    body = res.json()
    assert body.get("status") == "error"
    assert body.get("code") == "LOB_SEQUENCE_GAP"


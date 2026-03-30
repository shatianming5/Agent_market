"""Regression tests for auth middleware and AST-safe gate parsing."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path


def test_auth_middleware_blocks_without_key():
    """401 when AGENT_MARKET_API_KEY is set but no header provided."""
    os.environ["AGENT_MARKET_API_KEY"] = "test-secret-123"
    try:
        from server.app import create_app
        from starlette.testclient import TestClient

        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/run/feature", json={})
        assert resp.status_code == 401, f"expected 401, got {resp.status_code}"
    finally:
        os.environ.pop("AGENT_MARKET_API_KEY", None)


def test_auth_middleware_allows_with_key():
    """200/422 when correct X-API-Key header is provided."""
    os.environ["AGENT_MARKET_API_KEY"] = "test-secret-123"
    try:
        from server.app import create_app
        from starlette.testclient import TestClient

        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/run/feature", json={}, headers={"X-API-Key": "test-secret-123"})
        # May get 422 (validation error) but NOT 401
        assert resp.status_code != 401, f"got 401 despite valid key"
    finally:
        os.environ.pop("AGENT_MARKET_API_KEY", None)


def test_auth_options_preflight_passes():
    """OPTIONS requests bypass auth (CORS preflight)."""
    os.environ["AGENT_MARKET_API_KEY"] = "test-secret-123"
    try:
        from server.app import create_app
        from starlette.testclient import TestClient

        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.options("/run/feature")
        assert resp.status_code != 401, f"preflight blocked: {resp.status_code}"
    finally:
        os.environ.pop("AGENT_MARKET_API_KEY", None)


def test_detect_pairs_config_ast_only():
    """_detect_pairs_config extracts constants without exec()."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from workspace.gate_pipeline import GatePipeline

    gp = GatePipeline()

    # Valid strategy file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write('PAIR_A = "DOGE/USDT"\nPAIR_B = "SOL/USDT"\nLOOKBACK = 80\nENTRY_Z = 3.0\nEXIT_Z = 0.5\n')
        f.flush()
        config = gp._detect_pairs_config(Path(f.name))
    os.unlink(f.name)

    assert config is not None
    assert config["pair_a"] == "DOGE/USDT"
    assert config["pair_b"] == "SOL/USDT"
    assert config["lookback"] == 80
    assert config["entry_z"] == 3.0


def test_detect_pairs_config_rejects_dynamic():
    """_detect_pairs_config returns None for non-constant assignments."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from workspace.gate_pipeline import GatePipeline

    gp = GatePipeline()

    # Dynamic assignment — should NOT be extracted
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write('import os\nPAIR_A = os.environ.get("PAIR")\nPAIR_B = "SOL/USDT"\n')
        f.flush()
        config = gp._detect_pairs_config(Path(f.name))
    os.unlink(f.name)

    assert config is None, "should reject dynamic PAIR_A"


def test_detect_pairs_config_handles_syntax_error():
    """_detect_pairs_config returns None on broken Python."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from workspace.gate_pipeline import GatePipeline

    gp = GatePipeline()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write('this is not valid python {{{{')
        f.flush()
        config = gp._detect_pairs_config(Path(f.name))
    os.unlink(f.name)

    assert config is None

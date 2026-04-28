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


def test_detect_strategy_timeframe_ast_only():
    """_detect_strategy_timeframe extracts static class timeframe values."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from workspace.gate_pipeline import GatePipeline

    gp = GatePipeline()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("class Demo:\n    timeframe = '5m'\n")
        f.flush()
        timeframe = gp._detect_strategy_timeframe(Path(f.name))
    os.unlink(f.name)

    assert timeframe == "5m"


def test_run_gates_reports_failure_when_not_stopping(monkeypatch):
    """stop_on_fail=False should still produce a rejected final report."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from workspace.gate_pipeline import GatePipeline, GateResult

    gp = GatePipeline()
    monkeypatch.setattr(gp, "_detect_pairs_config", lambda path: None)
    monkeypatch.setattr(gp, "_save_report", lambda report, name: None)
    monkeypatch.setattr(
        gp,
        "_gate_2_backtest",
        lambda *args, **kwargs: GateResult("gate_2", False, {}, ["sharpe too low"]),
    )
    monkeypatch.setattr(
        gp,
        "_gate_3_robustness",
        lambda *args, **kwargs: GateResult("gate_3", False, {}, ["insufficient windows"]),
    )

    report = gp.run_gates("workspace/strategies/strategy_template.py", stop_on_fail=False)

    assert report["final_gate_passed"] == "gate_1"
    assert report["recommendation"].startswith("REJECTED at Gate 2")


def test_gate_2_backtest_passes_runtime_overrides(monkeypatch):
    """Gate 2 should propagate exchange/pair/timeframe into backtest execution."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from workspace.gate_pipeline import GatePipeline

    captured = {}

    def fake_run_backtest(strategy_path, strategy_name, **kwargs):
        captured["strategy_path"] = strategy_path
        captured["strategy_name"] = strategy_name
        captured.update(kwargs)
        return {
            "ok": True,
            "sharpe": 1.2,
            "max_drawdown_pct": 5.0,
            "trades": 45,
            "win_rate": 0.55,
            "profit_factor": 1.4,
            "profit_pct": 8.0,
        }

    gp = GatePipeline()
    monkeypatch.setattr("workspace.backtest_api.run_backtest", fake_run_backtest)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("class Demo:\n    timeframe = '5m'\n")
        f.flush()
        result = gp._gate_2_backtest(
            Path(f.name),
            "Demo",
            "gate",
            "BTC/USDT",
            "20260301-20260329",
        )
    os.unlink(f.name)

    assert result.passed is True
    assert captured["pairs"] == ["BTC/USDT"]
    assert captured["exchange_name"] == "gate"
    assert captured["timeframe"] == "5m"

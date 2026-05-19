from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from agent_market.strategy_miner.dtypes import MinerConfig, Phase


def test_strategy_miner_preflight_import_defers_optional_modules() -> None:
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import agent_market.strategy_miner.preflight; "
                "assert 'agent_market.agents.executor' not in sys.modules; "
                "assert 'agent_market.strategy_miner.research' not in sys.modules; "
                "print('ok')"
            ),
        ],
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    assert proc.stdout.strip() == "ok"


def _prepare_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AGENT_MARKET_ARTIFACTS_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("AGENT_MARKET_USER_DATA_ROOT", str(tmp_path / "user_data"))
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_preflight_sets_dummy_key_for_internal_proxy(monkeypatch, tmp_path: Path) -> None:
    from agent_market.strategy_miner import preflight
    from agent_market import runtime_preflight

    _prepare_env(monkeypatch, tmp_path)
    ft_cfg = tmp_path / "freqtrade.json"
    ft_cfg.write_text(
        '{"datadir":"user_data/data","exchange":{"pair_whitelist":["BTC/USDT"]},"timeframe":"5m"}',
        encoding="utf-8",
    )

    monkeypatch.setattr(
        preflight,
        "_check_opencli",
        lambda: {"name": "system.opencli", "ok": False, "severity": "warning", "detail": "missing"},
    )
    monkeypatch.setattr(
        preflight,
        "_check_freqtrade_cli",
        lambda: {"name": "system.freqtrade", "ok": True, "severity": "info", "detail": "freqtrade 1.0"},
    )

    calls: list[tuple[str, str]] = []

    def _fake_request_json(url: str, *, api_key: str, timeout: float = 5.0):  # noqa: ARG001
        calls.append((url, api_key))
        return True, {"data": [{"id": "gpt-5.2"}]}, ""

    monkeypatch.setattr(runtime_preflight, "_request_json", _fake_request_json)

    report = preflight.run_startup_preflight(
        MinerConfig(
            provider="openai_compatible",
            model="gpt-5.2",
            base_url="http://127.0.0.1:4141/v1",
            freqtrade_config=str(ft_cfg),
        ),
        miner_dir=tmp_path / "artifacts" / "runs" / "abc" / "strategy_miner",
        phase=Phase.STRATEGY_GEN,
        raise_on_error=False,
    )

    assert report["ok"] is True
    assert report["applied_env"] == {"LLM_API_KEY": "_"}
    assert calls == [("http://127.0.0.1:4141/v1/models", "_")]
    assert preflight.os.environ["LLM_API_KEY"] == "_"
    assert (tmp_path / "artifacts" / "runs" / "abc" / "strategy_miner" / "preflight.json").exists()


def test_preflight_reports_missing_freqtrade_config(monkeypatch, tmp_path: Path) -> None:
    from agent_market.strategy_miner import preflight

    _prepare_env(monkeypatch, tmp_path)
    monkeypatch.setattr(
        preflight,
        "_check_provider",
        lambda *_args, **_kwargs: [{"name": "llm.openai_compatible", "ok": True, "severity": "info", "detail": "ok"}],
    )
    monkeypatch.setattr(
        preflight,
        "_check_opencli",
        lambda: {"name": "system.opencli", "ok": True, "severity": "info", "detail": "ok"},
    )
    monkeypatch.setattr(
        preflight,
        "_check_freqtrade_cli",
        lambda: {"name": "system.freqtrade", "ok": True, "severity": "info", "detail": "ok"},
    )

    report = preflight.run_startup_preflight(
        MinerConfig(
            provider="openai_compatible",
            model="gpt-5.2",
            base_url="http://127.0.0.1:4141/v1",
            freqtrade_config=str(tmp_path / "missing.json"),
        ),
        miner_dir=tmp_path / "artifacts" / "runs" / "missing" / "strategy_miner",
        phase=Phase.STRATEGY_GEN,
        raise_on_error=False,
    )

    assert report["ok"] is False
    assert any(item["name"] == "config.freqtrade" and item["severity"] == "error" for item in report["checks"])


def test_preflight_skips_complete_phase(monkeypatch, tmp_path: Path) -> None:
    from agent_market.strategy_miner import preflight

    _prepare_env(monkeypatch, tmp_path)
    miner_dir = tmp_path / "artifacts" / "runs" / "done" / "strategy_miner"

    report = preflight.run_startup_preflight(
        MinerConfig(),
        miner_dir=miner_dir,
        phase=Phase.COMPLETE,
        raise_on_error=False,
    )

    assert report["ok"] is True
    assert report["skipped"] is True
    assert (miner_dir / "preflight.json").exists()

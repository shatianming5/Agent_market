from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

from agent_market.strategy_miner._holdout import run_sealed_holdout
from agent_market.strategy_miner.dtypes import MinerConfig, StrategyCandidate


def test_run_sealed_holdout_uses_freqtrade_wrapper(monkeypatch, tmp_path: Path) -> None:
    sandbox = tmp_path / "sandbox"
    strategy_path = sandbox / "user_data" / "strategies" / "Demo.py"
    strategy_path.parent.mkdir(parents=True, exist_ok=True)
    strategy_path.write_text("class Demo: pass\n", encoding="utf-8")

    candidate = StrategyCandidate(
        name="Demo",
        code="class Demo: pass\n",
        strategy_path=strategy_path,
    )
    candidate.backtest_summary = {"profit_total_pct": 1.5}

    config = MinerConfig(
        freqtrade_config="user_data/config_freqai_gate_spot_core_alpha.json",
        holdout_timerange="20260319-20260321",
        backtest_timeout=123,
    )

    captured: dict[str, object] = {}

    def _fake_run_sandboxed(cmd, *, cwd, timeout, cpu_seconds, mem_mb):  # noqa: ANN001
        captured["cmd"] = list(cmd)
        captured["cwd"] = cwd
        captured["timeout"] = timeout
        captured["cpu_seconds"] = cpu_seconds
        captured["mem_mb"] = mem_mb
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    fake_zip = sandbox / "user_data" / "backtest_results" / "backtest-result-demo.zip"
    fake_zip.parent.mkdir(parents=True, exist_ok=True)
    fake_zip.write_text("zip", encoding="utf-8")

    monkeypatch.setattr("agent_market.strategy_miner._sandbox_exec.run_sandboxed", _fake_run_sandboxed)
    monkeypatch.setattr("agent_market.strategy_miner._holdout.find_latest_backtest_zip", lambda _results: fake_zip)
    monkeypatch.setattr(
        "agent_market.strategy_miner._holdout.build_backtest_summary",
        lambda _zip: {"profit_total_pct": 1.0, "trades": 8},
    )

    result = run_sealed_holdout(candidate, config, tmp_path)

    wrapper = Path("scripts/freqtrade_cli.py").resolve()
    assert result is not None
    assert captured["cmd"][0] == sys.executable
    assert captured["cmd"][1] == str(wrapper)
    assert captured["cmd"][2] == "backtesting"
    assert "--userdir" in captured["cmd"]


def test_run_sealed_holdout_respects_holdout_delta_max_pct(monkeypatch, tmp_path: Path) -> None:
    sandbox = tmp_path / "sandbox"
    strategy_path = sandbox / "user_data" / "strategies" / "Demo.py"
    strategy_path.parent.mkdir(parents=True, exist_ok=True)
    strategy_path.write_text("class Demo: pass\n", encoding="utf-8")

    candidate = StrategyCandidate(
        name="Demo",
        code="class Demo: pass\n",
        strategy_path=strategy_path,
    )
    candidate.backtest_summary = {"profit_total_pct": 1.5}

    config = MinerConfig(
        freqtrade_config="user_data/config_freqai_gate_spot_core_alpha.json",
        holdout_timerange="20260319-20260321",
        backtest_timeout=123,
        holdout_delta_max_pct=0.4,
    )

    def _fake_run_sandboxed(cmd, *, cwd, timeout, cpu_seconds, mem_mb):  # noqa: ANN001
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    fake_zip = sandbox / "user_data" / "backtest_results" / "backtest-result-demo.zip"
    fake_zip.parent.mkdir(parents=True, exist_ok=True)
    fake_zip.write_text("zip", encoding="utf-8")

    monkeypatch.setattr("agent_market.strategy_miner._sandbox_exec.run_sandboxed", _fake_run_sandboxed)
    monkeypatch.setattr("agent_market.strategy_miner._holdout.find_latest_backtest_zip", lambda _results: fake_zip)
    monkeypatch.setattr(
        "agent_market.strategy_miner._holdout.build_backtest_summary",
        lambda _zip: {"profit_total_pct": 1.0, "trades": 8},
    )

    result = run_sealed_holdout(candidate, config, tmp_path)
    assert result is not None
    assert abs(float(result["delta_pct"]) - 0.5) < 1e-9
    assert abs(float(result["holdout_delta_max_pct"]) - 0.4) < 1e-9
    assert bool(result["overfitting_flag"]) is True

from __future__ import annotations

from pathlib import Path

from agent_market.strategy_miner.dtypes import StrategyCandidate


def test_classify_backtest_failure_filelock():
    from agent_market.strategy_miner._backtest import _classify_backtest_failure

    category, detail = _classify_backtest_failure(
        "No module named 'filelock'. Please ensure that the hyperopt dependencies are installed.",
        "",
    )

    assert category == "backtest.dependency_missing.filelock"
    assert "filelock" in detail


def test_classify_backtest_failure_thread_allocation():
    from agent_market.strategy_miner._backtest import _classify_backtest_failure

    category, detail = _classify_backtest_failure(
        "cannot allocate memory for thread-local data: ABORT",
        "",
    )

    assert category == "backtest.runtime_thread_allocation"
    assert "AGENT_MARKET_SANDBOX_THREADS" in detail


def test_classify_train_failure_stable_baselines3_underscore():
    from agent_market.strategy_miner._backtest import _classify_train_failure

    category, detail = _classify_train_failure(
        ModuleNotFoundError("No module named 'stable_baselines3'"),
        "rl",
    )

    assert category == "train_model.dependency_missing.stable_baselines3"
    assert "stable_baselines3" in detail


def test_classify_train_failure_unregistered_dl_adapter_maps_to_torch():
    from agent_market.strategy_miner._backtest import _classify_train_failure

    category, detail = _classify_train_failure(
        KeyError("model adapter 'pytorch_mlp' not registered"),
        "dl",
    )

    assert category == "train_model.dependency_missing.torch"
    assert "pytorch_mlp" in detail


def test_candidate_runtime_mem_mb_increases_for_model_candidates():
    from agent_market.strategy_miner._backtest import _candidate_runtime_mem_mb

    rule = StrategyCandidate("Rule", "", Path("/tmp/rule.py"))
    rule.candidate_type = "rule"
    dl = StrategyCandidate("DL", "", Path("/tmp/dl.py"))
    dl.candidate_type = "dl"
    ml = StrategyCandidate("ML", "", Path("/tmp/ml.py"))
    ml.candidate_type = "ml"

    assert _candidate_runtime_mem_mb(rule) == 4096
    assert _candidate_runtime_mem_mb(ml) == 6144
    assert _candidate_runtime_mem_mb(dl) == 12288

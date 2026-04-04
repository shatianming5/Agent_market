"""Tests for walk-forward timerange splitting and evaluation integration."""
from __future__ import annotations


def test_walkforward_timeranges_basic():
    from agent_market.strategy_miner._helpers import _walkforward_timeranges

    folds = _walkforward_timeranges("20250101-20250401", folds=3, train_ratio=0.6)
    assert len(folds) == 3
    for train_range, test_range in folds:
        assert "-" in train_range and "-" in test_range
        # Train always starts at 20250101
        assert train_range.startswith("20250101-")


def test_walkforward_timeranges_no_overlap():
    from agent_market.strategy_miner._helpers import _walkforward_timeranges

    folds = _walkforward_timeranges("20250101-20250701", folds=4, train_ratio=0.5)
    test_ranges = [test for _, test in folds]
    # Test ranges should not overlap
    for i in range(len(test_ranges) - 1):
        end_i = test_ranges[i].split("-")[1]
        start_next = test_ranges[i + 1].split("-")[0]
        assert end_i <= start_next, f"Fold {i} overlaps fold {i+1}"


def test_walkforward_short_timerange_returns_empty():
    from agent_market.strategy_miner._helpers import _walkforward_timeranges

    folds = _walkforward_timeranges("20250101-20250107", folds=3, train_ratio=0.6)
    assert folds == []


def test_walkforward_invalid_format():
    from agent_market.strategy_miner._helpers import _walkforward_timeranges

    assert _walkforward_timeranges("invalid") == []
    assert _walkforward_timeranges("") == []


def test_walkforward_config_fields():
    from agent_market.strategy_miner.dtypes import MinerConfig

    config = MinerConfig.from_dict({
        "evaluation": {
            "walkforward_enabled": True,
            "walkforward_folds": 5,
            "walkforward_train_ratio": 0.7,
        }
    })
    assert config.walkforward_enabled is True
    assert config.walkforward_folds == 5
    assert config.walkforward_train_ratio == 0.7


def test_walkforward_disabled_by_default():
    from agent_market.strategy_miner.dtypes import MinerConfig

    config = MinerConfig()
    assert config.walkforward_enabled is False

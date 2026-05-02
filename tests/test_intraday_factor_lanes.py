from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_market.factor_lab.rank_portfolio import RiskConfig
from agent_market.factor_lab.strategy_loop import StrategyLoopConfig, _rank_kwargs, validate_candidate
from agent_market.factor_lab.timeframes import (
    lane_manifest,
    normalize_lane,
    parse_label_horizons,
    primary_label_horizon,
)


def test_lane_defaults_select_intraday_horizons() -> None:
    lane = normalize_lane("auto", timeframe="15m")
    horizons = parse_label_horizons(None, default=lane.label_horizons)

    assert lane.lane == "15m_intraday"
    assert horizons == (1, 2, 4)
    assert primary_label_horizon(horizons, default=lane.label_horizons) == 1


def test_one_minute_ohlcv_only_manifest_blocks_formal_promotion() -> None:
    manifest = lane_manifest(
        lane="1m_micro",
        timeframe="1m",
        data_venue="okx",
        label_horizons=(1, 3, 5, 15),
        fee_bps=8.0,
        slippage_bps=3.0,
        embargo_bars=30,
        micro_data_quality="ohlcv_only",
    )

    assert manifest["promotion_eligible"] is False
    assert manifest["cost_model"]["round_trip_cost"] == pytest.approx(0.0022)


def test_rank_config_converts_rebalance_minutes_to_intraday_bars() -> None:
    cfg = RiskConfig.from_profile(
        "aggressive",
        timeframe="15m",
        rebalance_minutes=60,
    )

    assert cfg.timeframe == "15m"
    assert cfg.rebalance_minutes == 60
    assert cfg.rebalance_hours == 4


def test_strategy_loop_rank_kwargs_inherit_intraday_config() -> None:
    cfg = StrategyLoopConfig.from_args(
        tag="unit",
        timeframe="5m",
        data_venue="okx",
        evaluation_lane="auto",
    )
    kwargs = _rank_kwargs({"top_k": 3}, cfg, candidate_state=None, tag="unit")

    assert cfg.evaluation_lane == "5m_micro"
    assert kwargs["timeframe"] == "5m"
    assert kwargs["data_venue"] == "okx"


def test_strategy_loop_lane_sets_default_timeframe_when_not_overridden() -> None:
    cfg = StrategyLoopConfig.from_args(tag="unit", evaluation_lane="15m_intraday")

    assert cfg.timeframe == "15m"
    assert cfg.evaluation_lane == "15m_intraday"


def test_candidate_state_timeframe_mismatch_is_rejected(tmp_path: Path) -> None:
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"timeframe": "15m", "evaluation_lane": "15m_intraday", "survivors": []}), encoding="utf-8")
    candidate = tmp_path / "candidate.json"
    candidate.write_text(
        json.dumps(
            {
                "candidate_type": "rank_profile",
                "rank_profile": {
                    "timeframe": "5m",
                    "evaluation_lane": "5m_micro",
                    "candidate_state": str(state),
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="timeframe=15m"):
        validate_candidate(candidate)

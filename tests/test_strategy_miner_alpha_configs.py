from __future__ import annotations

import json
import random
from pathlib import Path


def test_miner_config_parses_search_families_and_objective_weights() -> None:
    from agent_market.strategy_miner.dtypes import MinerConfig

    raw = {
        "backtest": {
            "freqtrade_config": "user_data/config_freqai_gate_spot_core_alpha.json",
            "timerange": "20260310-20260324",
        },
        "evaluation": {
            "target_trades": 80,
            "objective_weights": {
                "sharpe": 0.18,
                "sortino": 0.10,
                "profit_factor": 0.15,
                "calmar": 0.05,
                "profit_pct": 0.27,
                "winrate": 0.05,
                "return_over_drawdown": 0.20,
            },
            "discovery_quick_min_trades": 4,
            "discovery_quick_min_profit_factor": 0.75,
            "discovery_quick_min_profit_pct": -2.0,
            "discovery_quick_max_drawdown_pct": 45.0,
        },
        "search_space": {
            "search_families": [
                "rule/trend-following",
                "ml/lightgbm",
                "dl/pytorch_mlp",
                "rl/ppo",
            ],
            "family_weight_schedule": [
                {
                    "iteration_lte": 12,
                    "weights": {
                        "rule/trend-following": 2.5,
                        "ml/lightgbm": 2.0,
                        "dl/pytorch_mlp": 0.2,
                        "rl/ppo": 0.1,
                    },
                }
            ],
        },
    }

    cfg = MinerConfig.from_dict(raw)

    assert cfg.target_trades == 80
    assert cfg.search_families == [
        "rule/trend-following",
        "ml/lightgbm",
        "dl/pytorch_mlp",
        "rl/ppo",
    ]
    assert cfg.objective_weights["profit_pct"] == 0.27
    assert cfg.objective_weights["return_over_drawdown"] == 0.20
    assert cfg.discovery_quick_min_trades == 4
    assert cfg.discovery_quick_min_profit_factor == 0.75
    assert cfg.discovery_quick_min_profit_pct == -2.0
    assert cfg.discovery_quick_max_drawdown_pct == 45.0
    assert cfg.family_weight_schedule[0]["weights"]["rl/ppo"] == 0.1


def test_goal_contract_uses_config_objective_weights() -> None:
    from agent_market.strategy_miner.dtypes import MinerConfig
    from agent_market.strategy_miner.goal_contract import GoalContract

    cfg = MinerConfig(
        objective_weights={
            "sharpe": 0.20,
            "sortino": 0.10,
            "profit_factor": 0.15,
            "calmar": 0.05,
            "profit_pct": 0.25,
            "winrate": 0.05,
            "return_over_drawdown": 0.20,
        }
    )

    gc = GoalContract.from_miner_config(cfg)
    assert gc.objective_weights["profit_pct"] == 0.25
    assert gc.objective_weights["return_over_drawdown"] == 0.20


def test_bandit_scheduler_selects_unique_families_first() -> None:
    from agent_market.strategy_miner._scheduler import BanditScheduler

    random.seed(7)
    sched = BanditScheduler(
        [
            "rule/trend-following",
            "rule/breakout",
            "ml/lightgbm",
            "rl/ppo",
        ],
        min_exploration_slots=1,
    )

    selected = sched.select_families(4)
    assert len(selected) == 4
    assert len(set(selected)) == 4


def test_bandit_scheduler_honors_family_weights() -> None:
    from agent_market.strategy_miner._scheduler import BanditScheduler, resolve_family_weights

    random.seed(11)
    sched = BanditScheduler(
        [
            "rule/breakout",
            "ml/lightgbm",
            "dl/pytorch_mlp",
            "rl/ppo",
        ],
        min_exploration_slots=0,
    )
    weights = resolve_family_weights(
        [
            {
                "iteration_lte": 8,
                "weights": {
                    "rule/breakout": 2.5,
                    "ml/lightgbm": 2.0,
                    "dl/pytorch_mlp": 0.0,
                    "rl/ppo": 0.0,
                },
            }
        ],
        iteration=3,
    )
    draws = [sched.select_families(1, family_weights=weights)[0] for _ in range(40)]
    assert "dl/pytorch_mlp" not in draws
    assert "rl/ppo" not in draws
    assert set(draws).issubset({"rule/breakout", "ml/lightgbm"})


def test_resolve_quick_gate_thresholds_uses_futures_discovery_overrides(tmp_path: Path) -> None:
    from agent_market.strategy_miner._backtest import _resolve_quick_gate_thresholds
    from agent_market.strategy_miner.dtypes import MinerConfig

    ft_config = tmp_path / "ft.json"
    ft_config.write_text(
        json.dumps({"trading_mode": "futures", "timeframe": "5m"}),
        encoding="utf-8",
    )
    cfg = MinerConfig(
        freqtrade_config=str(ft_config),
        quick_min_trades=8,
        quick_min_profit_factor=0.9,
        quick_min_profit_pct=-1.0,
        quick_max_drawdown_pct=35.0,
        discovery_quick_min_trades=4,
        discovery_quick_min_profit_factor=0.75,
        discovery_quick_min_profit_pct=-2.0,
        discovery_quick_max_drawdown_pct=45.0,
    )

    thresholds = _resolve_quick_gate_thresholds(cfg)
    assert thresholds == {
        "min_trades": 4.0,
        "min_profit_factor": 0.75,
        "min_profit_pct": -2.0,
        "max_drawdown_pct": 45.0,
    }


def test_data_prep_collects_union_timerange_and_builds_check_only(tmp_path: Path) -> None:
    from agent_market.strategy_miner.data_prep import (
        collect_data_requirements,
        download_required_data,
        missing_ohlcv_paths,
    )

    ft_config = tmp_path / "ft.json"
    ft_config.write_text(
        json.dumps(
            {
                "trading_mode": "futures",
                "timeframe": "5m",
                "datadir": str(tmp_path / "user_data" / "data"),
                "exchange": {
                    "name": "gate",
                    "pair_whitelist": ["BTC/USDT:USDT", "ETH/USDT:USDT"],
                },
                "freqai": {
                    "feature_parameters": {
                        "include_timeframes": ["5m", "15m", "1h"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    miner_config = tmp_path / "miner.json"
    miner_config.write_text(
        json.dumps(
            {
                "backtest": {
                    "freqtrade_config": str(ft_config),
                    "timerange": "20260312-20260328",
                    "holdout_timerange": "20260329-20260403",
                    "allowed_informative_timeframes": ["15m", "1h"],
                },
                "model_mining": {
                    "train_timerange": "20260228-20260311",
                },
            }
        ),
        encoding="utf-8",
    )

    req = collect_data_requirements(miner_config)
    result = download_required_data(req, check_only=True)

    assert req.timerange == "20260228-20260403"
    assert req.timeframes == ["5m", "15m", "1h"]
    assert req.trading_mode == "futures"
    assert len(missing_ohlcv_paths(req)) == 6
    assert result["executed"] is False
    assert "--trading-mode" in result["command"]
    assert "futures" in result["command"]
    assert "--timerange" in result["command"]

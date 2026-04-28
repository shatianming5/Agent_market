from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_market import paths as repo_paths
from agent_market.factor_lab.strategy_loop import (
    PHASE_BACKTEST,
    StrategyLoopConfig,
    StrategyLoopState,
    load_checkpoint,
    promote_candidate,
    render_agent_prompt,
    save_checkpoint,
    score_backtest_result,
    validate_candidate,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_strategy_loop_checkpoint_roundtrip() -> None:
    cfg = StrategyLoopConfig.from_args(tag="unit", run_id="unit_checkpoint", max_iterations=2)
    state = StrategyLoopState(run_id=cfg.run_id, iteration=2, phase=PHASE_BACKTEST)
    state.best_score = 12.5
    state.candidate_paths.append("artifacts/factor_strategy_loop/unit/iter_01/candidate.json")

    path = save_checkpoint(cfg, state)
    loaded_cfg, loaded_state = load_checkpoint("unit_checkpoint")

    assert path.exists()
    assert loaded_cfg.tag == "unit"
    assert loaded_state.iteration == 2
    assert loaded_state.phase == PHASE_BACKTEST
    assert loaded_state.best_score == 12.5
    assert loaded_state.candidate_paths == state.candidate_paths


def test_candidate_schema_rejects_out_of_bounds_leverage(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.json"
    _write_json(
        candidate,
        {
            "candidate_type": "rank_profile",
            "name": "too_hot",
            "params": {"top_k": 2, "leverage_cap": 25.0},
        },
    )

    with pytest.raises(ValueError, match="leverage_cap"):
        validate_candidate(candidate)


def test_candidate_schema_rejects_strategy_path_escape(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.json"
    _write_json(
        candidate,
        {
            "candidate_type": "freqtrade_strategy",
            "name": "escape",
            "strategy_path": "../strategy.py",
        },
    )

    with pytest.raises(ValueError, match="escapes"):
        validate_candidate(candidate)


def test_candidate_schema_accepts_freqtrade_strategy(tmp_path: Path) -> None:
    (tmp_path / "strategy.py").write_text(
        "\n".join(
            [
                "import os",
                "import pandas as pd",
                "from freqtrade.strategy import IStrategy",
                "",
                "class UnitRankStrategy(IStrategy):",
                "    def _signals(self):",
                "        path = os.environ.get('RP_SIGNAL_DIR') or os.environ.get('RP_TAG', 'tag')",
                "        df = pd.read_feather(path)",
                "        return df[['rp_target_weight', 'rp_side']]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    candidate = tmp_path / "candidate.json"
    _write_json(
        candidate,
        {
            "candidate_type": "freqtrade_strategy",
            "name": "unit_strategy",
            "strategy_path": "strategy.py",
            "rank_profile": {"top_k": 2, "gross_cap": 2.0, "leverage_cap": 5.0},
        },
    )

    normalized = validate_candidate(candidate)

    assert normalized["candidate_type"] == "freqtrade_strategy"
    assert normalized["strategy_validation"]["istrategy_classes"] == ["UnitRankStrategy"]
    assert normalized["rank_profile"]["leverage_cap"] == 5.0


def test_candidate_schema_rejects_hardcoded_strategy_paths(tmp_path: Path) -> None:
    (tmp_path / "strategy.py").write_text(
        "\n".join(
            [
                "import pandas as pd",
                "from freqtrade.strategy import IStrategy",
                "",
                "class BadPathStrategy(IStrategy):",
                "    def _signals(self):",
                "        df = pd.read_feather('/Users/shatianming/Downloads/Agent_market/artifacts/x.feather')",
                "        return df[['rp_target_weight', 'rp_side']]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    candidate = tmp_path / "candidate.json"
    _write_json(
        candidate,
        {
            "candidate_type": "freqtrade_strategy",
            "name": "bad_path",
            "strategy_path": "strategy.py",
        },
    )

    with pytest.raises(ValueError, match="hard-code absolute"):
        validate_candidate(candidate)


def test_prompt_can_force_freqtrade_candidate_type(tmp_path: Path) -> None:
    prompt = render_agent_prompt(tmp_path / "context" / "prepare.json", candidate_type="freqtrade_strategy")

    assert "must create a `freqtrade_strategy` candidate" in prompt
    assert "Do not put signal-column mappings" in prompt
    assert "RP_SIGNAL_DIR" in prompt


def test_config_accepts_cli_opencode_mode() -> None:
    cfg = StrategyLoopConfig.from_args(tag="unit", opencode_mode="cli", candidate_type="freqtrade_strategy")

    assert cfg.opencode_mode == "cli"
    assert cfg.candidate_type == "freqtrade_strategy"


def test_scorer_enforces_hard_gates_and_orders_candidates() -> None:
    strong = score_backtest_result(
        {
            "total_return_pct": 38.0,
            "max_drawdown_pct": 12.0,
            "profit_over_max_drawdown": 3.17,
            "trades": 140,
            "simulated_liquidations": 0,
            "liquidation_rejects": 0,
            "avg_turnover": 0.8,
        }
    )
    weak = score_backtest_result(
        {
            "total_return_pct": 30.0,
            "max_drawdown_pct": 31.0,
            "profit_over_max_drawdown": 0.97,
            "trades": 40,
            "simulated_liquidations": 1,
            "liquidation_rejects": 2,
            "avg_turnover": 4.0,
        }
    )

    assert strong["constraints_ok"] is True
    assert weak["constraints_ok"] is False
    assert "simulated_liquidations" in ";".join(weak["violations"])
    assert strong["score"] > weak["score"]


def test_promotion_requires_passing_constraints(tmp_path: Path) -> None:
    cfg = StrategyLoopConfig.from_args(tag="unit_promote", run_id="unit_promote_run", promote=True)
    iter_dir = tmp_path / "iter_01"
    iter_dir.mkdir()
    _write_json(iter_dir / "candidate.json", {"candidate_type": "rank_profile", "rank_profile": {"top_k": 2}})
    candidate = validate_candidate(iter_dir / "candidate.json")

    failed = promote_candidate(
        candidate,
        {"constraints_ok": False, "promotion_reason": "trades=0 < 80"},
        cfg,
        iter_dir=iter_dir,
    )
    out = repo_paths.artifacts_root() / "rank_portfolio" / "unit_promote" / "optimized_profile.json"

    assert failed["promoted"] is False
    assert not out.exists()

    passed = promote_candidate(
        candidate,
        {"constraints_ok": True, "score": 123.0, "promotion_reason": "passes hard gates"},
        cfg,
        iter_dir=iter_dir,
    )

    assert passed["promoted"] is True
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["rank_profile"]["top_k"] == 2

"""Integration tests for strategy_miner runner loop."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from agent_market import paths
from agent_market.strategy_miner.dtypes import MinerConfig, MinerState, Phase, StrategyCandidate
from agent_market.strategy_miner.knowledge_base import KnowledgeBase


# ---------------------------------------------------------------------------
# Checkpoint save/load
# ---------------------------------------------------------------------------


def test_checkpoint_roundtrip():
    from agent_market.strategy_miner.runner import _load_checkpoint, _save_checkpoint

    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td)
        state = MinerState()
        state.phase = Phase.EVALUATION
        state.iteration = 3
        state.best_score = 0.75

        _save_checkpoint(state, run_dir)

        cp = run_dir / "checkpoint.json"
        assert cp.exists()

        loaded = _load_checkpoint(cp)
        assert loaded.run_id == state.run_id
        assert loaded.phase == Phase.EVALUATION
        assert loaded.iteration == 3
        assert loaded.best_score == 0.75


def test_checkpoint_atomic_write():
    """Verify no .tmp file remains after save."""
    from agent_market.strategy_miner.runner import _save_checkpoint

    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td)
        state = MinerState()
        _save_checkpoint(state, run_dir)

        assert not (run_dir / "checkpoint.tmp").exists()
        assert (run_dir / "checkpoint.json").exists()


def test_checkpoint_recovery_from_tmp():
    """When main checkpoint is corrupt, recover from .tmp file."""
    from agent_market.strategy_miner.runner import _load_checkpoint, _save_checkpoint

    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td)

        # Save a valid checkpoint
        state = MinerState()
        state.phase = Phase.EVALUATION
        state.iteration = 5
        state.best_score = 1.23
        _save_checkpoint(state, run_dir)

        cp = run_dir / "checkpoint.json"
        tmp = cp.with_suffix(".tmp")

        # Simulate crash: write newer state to .tmp, corrupt main
        state2 = MinerState(run_id=state.run_id)
        state2.phase = Phase.ANALYSIS
        state2.iteration = 6
        state2.best_score = 2.0
        tmp.write_text(json.dumps(state2.to_dict(), ensure_ascii=False, indent=2))
        cp.write_text("CORRUPTED")

        # Should recover from .tmp
        loaded = _load_checkpoint(cp)
        assert loaded.iteration == 6
        assert loaded.best_score == 2.0
        assert loaded.phase == Phase.ANALYSIS
        # .tmp should have been promoted
        assert cp.exists()
        assert not tmp.exists()


def test_checkpoint_gen_retries_persisted():
    """gen_retries field should survive checkpoint roundtrip."""
    from agent_market.strategy_miner.runner import _load_checkpoint, _save_checkpoint

    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td)
        state = MinerState()
        state.gen_retries = 2
        _save_checkpoint(state, run_dir)

        loaded = _load_checkpoint(run_dir / "checkpoint.json")
        assert loaded.gen_retries == 2


# ---------------------------------------------------------------------------
# _update_knowledge_base
# ---------------------------------------------------------------------------


def test_update_kb_with_successful_candidate():
    from agent_market.strategy_miner.runner import _update_knowledge_base

    with tempfile.TemporaryDirectory() as td:
        kb = KnowledgeBase(Path(td) / "kb.json")
        state = MinerState()
        c = StrategyCandidate("Good", "code", Path("/tmp/x.py"))
        c.reward = 0.8
        c.backtest_summary = {"profit_total_pct": 10, "trades": 50}
        c.validation_passed = True
        state.candidates.append(c)

        _update_knowledge_base(kb, state)
        assert len(kb.elites) == 1
        assert kb.elites[0]["name"] == "Good"


def test_update_kb_strategy_card_uses_market_context_order():
    from agent_market.strategy_miner.runner import _update_knowledge_base

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        ft_cfg = root / "freqtrade.json"
        ft_cfg.write_text(
            json.dumps(
                {
                    "timeframe": "4h",
                    "exchange": {
                        "name": "binance",
                        "pair_whitelist": ["BTC/USDT", "ETH/USDT"],
                    },
                }
            ),
            encoding="utf-8",
        )
        kb = KnowledgeBase(root / "kb.json")
        state = MinerState(run_id="run123", iteration=3)
        c = StrategyCandidate("Good", "class Good: pass", root / "Good.py")
        c.reward = 0.8
        c.backtest_summary = {"profit_total_pct": 10, "trades": 50}
        c.validation_passed = True
        state.candidates.append(c)

        _update_knowledge_base(kb, state, config=MinerConfig(freqtrade_config=str(ft_cfg)))

        assert kb.strategy_cards[0]["timeframe"] == "4h"
        assert kb.strategy_cards[0]["universe"] == ["BTC/USDT", "ETH/USDT"]


def test_update_kb_with_failed_candidate():
    from agent_market.strategy_miner.runner import _update_knowledge_base

    with tempfile.TemporaryDirectory() as td:
        kb = KnowledgeBase(Path(td) / "kb.json")
        state = MinerState()
        c = StrategyCandidate("Bad", "code", Path("/tmp/x.py"))
        c.validation_passed = False
        c.diagnosis = "Missing IStrategy"
        state.candidates.append(c)

        _update_knowledge_base(kb, state)
        assert len(kb.failures) == 1
        assert "validation" in kb.failures[0]["failure_type"]


def test_update_kb_no_candidates():
    from agent_market.strategy_miner.runner import _update_knowledge_base

    with tempfile.TemporaryDirectory() as td:
        kb = KnowledgeBase(Path(td) / "kb.json")
        state = MinerState()
        _update_knowledge_base(kb, state)
        assert len(kb.elites) == 0
        assert len(kb.failures) == 0


def test_update_kb_constraint_violated_goes_to_failures():
    """Candidates that violate constraints should go to failures, not elites."""
    from agent_market.strategy_miner.runner import _update_knowledge_base

    with tempfile.TemporaryDirectory() as td:
        kb = KnowledgeBase(Path(td) / "kb.json")
        state = MinerState()
        c = StrategyCandidate("Constrained", "code", Path("/tmp/x.py"))
        c.reward = 0.5
        c.backtest_summary = {"profit_total_pct": 5, "trades": 3}
        c.constraints_ok = False
        c.constraint_violations = ["min_trades:3<10"]
        state.candidates.append(c)

        _update_knowledge_base(kb, state)
        assert len(kb.elites) == 0
        assert len(kb.failures) == 1
        assert kb.failures[0]["failure_type"] == "constraint_violation"


def test_backfill_global_strategy_knowledge_base_converts_legacy_elites() -> None:
    from agent_market.strategy_miner.runner import backfill_global_strategy_knowledge_base

    with tempfile.TemporaryDirectory() as td:
        env = {
            "AGENT_MARKET_ARTIFACTS_ROOT": str(Path(td) / "artifacts"),
            "AGENT_MARKET_RUNS_ROOT": str(Path(td) / "artifacts" / "runs"),
        }
        with patch.dict(os.environ, env, clear=False):
            miner_dir = paths.run_dir("legacyrun01") / "strategy_miner"
            miner_dir.mkdir(parents=True, exist_ok=True)
            (miner_dir / "knowledge_base.json").write_text(
                json.dumps(
                    {
                        "elites": [
                            {
                                "name": "LegacyWinner",
                                "reward": 0.91,
                                "iteration": 2,
                                "profit_pct": 3.2,
                                "trades": 48,
                                "winrate": 0.62,
                                "max_drawdown": 5.0,
                            }
                        ],
                        "failures": [
                            {
                                "name": "LegacyLoser",
                                "iteration": 1,
                                "failure_type": "constraint_violation",
                                "detail": "min_trades:2<8",
                            }
                        ],
                        "strategy_cards": [],
                        "failure_cards": [],
                        "edges": [],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            (miner_dir / "strategy_miner_summary.json").write_text(
                json.dumps(
                    {
                        "candidates": [
                            {
                                "name": "LegacyWinner",
                                "candidate_type": "rule",
                                "candidate_family": "rule/breakout",
                                "code": 'class LegacyWinner:\n    timeframe = "5m"\n',
                                "candidate_payload": {"trace_grade": {"overall_grade": 0.9}},
                            },
                            {
                                "name": "LegacyLoser",
                                "candidate_type": "rule",
                                "candidate_family": "rule/trend-pullback",
                                "code": 'class LegacyLoser:\n    timeframe = "15m"\n',
                            },
                        ]
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            (miner_dir / "proposal.json").write_text(
                json.dumps(
                    {"config": {"model_training_pairs": ["BTC/USDT", "ETH/USDT"]}},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            global_kb = KnowledgeBase(paths.global_strategy_knowledge_base_path())
            stats = backfill_global_strategy_knowledge_base(global_kb, current_run_id="current123")

            assert stats["strategy_cards_added"] == 1
            assert stats["failure_cards_added"] == 1

            strategy_cards = global_kb.strategy_cards
            assert len(strategy_cards) == 1
            assert strategy_cards[0]["card_id"] == "legacyrun01:LegacyWinner:2"
            assert strategy_cards[0]["run_id"] == "legacyrun01"
            assert strategy_cards[0]["candidate_family"] == "rule/breakout"
            assert strategy_cards[0]["timeframe"] == "5m"
            assert strategy_cards[0]["universe"] == ["BTC/USDT", "ETH/USDT"]
            assert strategy_cards[0]["metrics"]["trace_grade"] == 0.9

            failure_cards = global_kb.failure_cards
            assert len(failure_cards) == 1
            assert failure_cards[0]["failure_id"] == "legacyrun01:LegacyLoser:1:constraint_violation"
            assert failure_cards[0]["candidate_family"] == "rule/trend-pullback"
            assert failure_cards[0]["timeframe"] == "15m"


def test_backfill_global_strategy_knowledge_base_is_idempotent() -> None:
    from agent_market.strategy_miner.runner import backfill_global_strategy_knowledge_base

    with tempfile.TemporaryDirectory() as td:
        env = {
            "AGENT_MARKET_ARTIFACTS_ROOT": str(Path(td) / "artifacts"),
            "AGENT_MARKET_RUNS_ROOT": str(Path(td) / "artifacts" / "runs"),
        }
        with patch.dict(os.environ, env, clear=False):
            miner_dir = paths.run_dir("legacyrun02") / "strategy_miner"
            miner_dir.mkdir(parents=True, exist_ok=True)
            (miner_dir / "knowledge_base.json").write_text(
                json.dumps(
                    {
                        "elites": [],
                        "failures": [],
                        "strategy_cards": [
                            {
                                "name": "LegacyCardOnly",
                                "iteration": 0,
                                "candidate_type": "rule",
                                "candidate_family": "rule/breakout",
                                "metrics": {"sharpe": 1.2, "trades": 30},
                            }
                        ],
                        "failure_cards": [],
                        "edges": [],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            global_kb = KnowledgeBase(paths.global_strategy_knowledge_base_path())
            first = backfill_global_strategy_knowledge_base(global_kb, current_run_id="current123")
            second = backfill_global_strategy_knowledge_base(global_kb, current_run_id="current123")

            assert first["strategy_cards_added"] == 1
            assert second["strategy_cards_added"] == 0
            assert len(global_kb.strategy_cards) == 1
            assert global_kb.strategy_cards[0]["card_id"] == "legacyrun02:LegacyCardOnly:0"


# ---------------------------------------------------------------------------
# max_retries propagation
# ---------------------------------------------------------------------------


def test_build_strategy_agent_passes_max_retries_to_agent():
    """Agent factory should pass config.max_retries into StrategyAgent."""
    from agent_market.strategy_miner.agent_factory import build_strategy_agent

    with tempfile.TemporaryDirectory() as td:
        with patch("agent_market.strategy_miner.agent_factory.StrategyAgent") as MockAgent:
            mock_instance = MagicMock()
            MockAgent.return_value = mock_instance

            config = MinerConfig(model="test-model", max_retries=7)
            build_strategy_agent(config, Path(td))

            _, kwargs = MockAgent.call_args
            assert kwargs["max_retries"] == 7


def test_runner_resume_finalizes_holdout_benchmark_and_portfolio() -> None:
    from agent_market.strategy_miner.runner import run_strategy_miner

    with tempfile.TemporaryDirectory() as td:
        with patch.dict(os.environ, {"AGENT_MARKET_RUNS_ROOT": td}, clear=False):
            run_id = "runnerbench1"
            miner_dir = paths.run_dir(run_id) / "strategy_miner"
            miner_dir.mkdir(parents=True, exist_ok=True)

            best = StrategyCandidate(
                name="Best",
                code="class X: pass\n",
                strategy_path=miner_dir / "sandbox" / "user_data" / "strategies" / "Best.py",
                iteration=2,
                stage="evaluated",
            )
            best.reward = 1.2
            best.constraints_ok = True
            best.candidate_payload = {
                "factor_retrieval": {
                    "factor_cards": [{"card_id": "flowrun:spec:breakout_card"}],
                    "query": {"family": "rule/breakout"},
                }
            }
            best.backtest_summary = {
                "profit_total_pct": 11.0,
                "daily_profit": [
                    ["2026-03-12", 10.0],
                    ["2026-03-13", -2.0],
                    ["2026-03-14", 6.0],
                ],
                "starting_balance": 1000.0,
                "walkforward": {"folds_completed": 3, "folds_total": 3},
                "observation_days": 40,
            }

            peer = StrategyCandidate(
                name="Peer",
                code="class Y: pass\n",
                strategy_path=miner_dir / "sandbox" / "user_data" / "strategies" / "Peer.py",
                iteration=2,
                stage="evaluated",
            )
            peer.reward = 1.0
            peer.constraints_ok = True
            peer.backtest_summary = {
                "profit_total_pct": 8.0,
                "daily_profit": [
                    ["2026-03-12", -5.0],
                    ["2026-03-13", 3.0],
                    ["2026-03-14", -1.0],
                ],
                "starting_balance": 1000.0,
            }

            state = MinerState(run_id=run_id, phase=Phase.COMPLETE, iteration=2)
            state.best_score = 1.2
            state.best_candidate = best
            state.candidates = [best, peer]
            (miner_dir / "checkpoint.json").write_text(
                json.dumps(state.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            cfg = MinerConfig(
                model="test-model",
                benchmark_suite="benchmark_pack/default",
                portfolio_enabled=True,
            )

            with (
                patch(
                    "agent_market.strategy_miner._holdout.run_sealed_holdout",
                    return_value={"delta_pct": 8.0, "overfitting_flag": False, "holdout_timerange": "20260324-20260329"},
                ),
                patch(
                    "agent_market.strategy_miner._benchmark.run_benchmark_suite",
                    return_value={"passed": True, "failed_ids": [], "suite_id": "strategy_miner_default"},
                ),
                patch(
                    "agent_market.strategy_miner._portfolio.build_candidate_portfolio_report",
                    return_value={"passed": True, "method": "hrp", "weights": {"Best": 0.6, "Peer": 0.4}},
                ),
            ):
                final_state = run_strategy_miner(cfg, resume=miner_dir / "checkpoint.json")

            assert final_state.best_candidate is not None
            assert final_state.best_candidate.stage == "holdout_tested"
            assert (miner_dir / "holdout_gate.json").exists()
            assert (miner_dir / "benchmark_verdict.json").exists()
            assert (miner_dir / "portfolio_plan.json").exists()
            run_meta = json.loads((miner_dir / "run_meta.json").read_text(encoding="utf-8"))
            assert run_meta["benchmark_passed"] is True
            assert run_meta["promotion_status"] == "strategy_loop_required"
            assert run_meta["portfolio_method"] == "hrp"
            promotion_rows = [
                json.loads(line)
                for line in (miner_dir / "promotion_log.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            assert promotion_rows[-1]["factor_references"] == ["flowrun:spec:breakout_card"]
            assert promotion_rows[-1]["candidate_gate_passed"] is True
            assert promotion_rows[-1]["promotion_ok"] is False


def test_holdout_failure_writes_demoted_promotion_log() -> None:
    from agent_market.strategy_miner.runner import run_strategy_miner

    with tempfile.TemporaryDirectory() as td:
        with patch.dict(os.environ, {"AGENT_MARKET_RUNS_ROOT": td}, clear=False):
            run_id = "runnerholdoutfail"
            miner_dir = paths.run_dir(run_id) / "strategy_miner"
            miner_dir.mkdir(parents=True, exist_ok=True)
            best = StrategyCandidate(
                name="Best",
                code="class X: pass\n",
                strategy_path=miner_dir / "sandbox" / "user_data" / "strategies" / "Best.py",
                iteration=1,
                stage="evaluated",
            )
            best.reward = 1.0
            best.constraints_ok = True
            best.candidate_payload = {
                "factor_retrieval": {
                    "factor_cards": [{"card_id": "flowrun:spec:breakout_card"}],
                    "query": {"family": "rule/breakout"},
                }
            }
            best.backtest_summary = {"profit_total_pct": 10.0, "daily_profit": [["2026-03-12", 10.0]]}

            state = MinerState(run_id=run_id, phase=Phase.COMPLETE, iteration=1)
            state.best_score = 1.0
            state.best_candidate = best
            state.candidates = [best]
            (miner_dir / "checkpoint.json").write_text(
                json.dumps(state.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            with patch(
                "agent_market.strategy_miner._holdout.run_sealed_holdout",
                return_value={"delta_pct": -50.0, "overfitting_flag": True, "holdout_timerange": "20260324-20260329"},
            ):
                final_state = run_strategy_miner(MinerConfig(), resume=miner_dir / "checkpoint.json")

            promotion_rows = [
                json.loads(line)
                for line in (miner_dir / "promotion_log.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            assert final_state.best_candidate is None
            assert promotion_rows[-1]["candidate"] == "Best"
            assert promotion_rows[-1]["promotion_ok"] is False
            assert promotion_rows[-1]["promotion_status"] == "holdout_failed"
            assert promotion_rows[-1]["factor_references"] == ["flowrun:spec:breakout_card"]

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from agent_market import paths
from agent_market.agent_flow import AgentFlow, AgentFlowConfig
from agent_market.run_artifacts import RunArtifacts
from agent_market.strategy_factory import finalize_strategy_factory_artifacts


def test_finalize_strategy_factory_artifacts_writes_registry_replay_and_lineage() -> None:
    with tempfile.TemporaryDirectory() as td:
        env = {
            "AGENT_MARKET_ARTIFACTS_ROOT": str(Path(td) / "artifacts"),
            "AGENT_MARKET_RUNS_ROOT": str(Path(td) / "artifacts" / "runs"),
        }
        with patch.dict("os.environ", env, clear=False):
            run_id = "facade123456"
            run_dir = paths.run_dir(run_id)
            run_dir.mkdir(parents=True, exist_ok=True)

            factor_dir = run_dir / "factor_memory"
            factor_dir.mkdir(parents=True, exist_ok=True)
            factor_cards = {
                "items": [
                    {
                        "card_id": "flowrun:spec:breakout_card",
                        "name": "breakout_card",
                        "timeframe": "5m",
                        "gate_pass": True,
                        "pareto": True,
                        "metrics": {"weighted_score": 0.9},
                    }
                ]
            }
            factor_memory = {
                "schema_version": "1.0",
                "factor_cards": [
                    {
                        "card_id": "flowrun:spec:breakout_card",
                        "name": "breakout_card",
                        "timeframe": "5m",
                        "gate_pass": True,
                        "pareto": True,
                        "metrics": {"weighted_score": 0.9},
                    }
                ],
                "failure_cards": [],
                "edges": [{"parent": "spec", "child": "breakout_card", "edge_type": "factor_eval"}],
            }
            factor_memory_json = factor_dir / "factor_memory.json"
            factor_cards_json = factor_dir / "factor_cards.json"
            factor_failures_json = factor_dir / "factor_failure_cards.json"
            factor_lineage_json = factor_dir / "factor_lineage.json"
            factor_memory_json.write_text(json.dumps(factor_memory, ensure_ascii=False, indent=2), encoding="utf-8")
            factor_cards_json.write_text(json.dumps(factor_cards, ensure_ascii=False, indent=2), encoding="utf-8")
            factor_failures_json.write_text(json.dumps({"items": []}, ensure_ascii=False, indent=2), encoding="utf-8")
            factor_lineage_json.write_text(json.dumps({"edges": factor_memory["edges"]}, ensure_ascii=False, indent=2), encoding="utf-8")

            miner_dir = run_dir / "strategy_miner"
            miner_dir.mkdir(parents=True, exist_ok=True)
            summary_path = miner_dir / "strategy_miner_summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "candidates": [
                            {
                                "name": "CandidateA",
                                "candidate_payload": {
                                    "factor_retrieval": {
                                        "factor_cards": [{"card_id": "flowrun:spec:breakout_card"}],
                                        "query": {"family": "rule/breakout"},
                                    }
                                },
                            }
                        ],
                        "best_candidate": {
                            "name": "CandidateA",
                            "candidate_payload": {
                                "factor_retrieval": {
                                    "factor_cards": [{"card_id": "flowrun:spec:breakout_card"}],
                                    "query": {"family": "rule/breakout"},
                                }
                            },
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            (miner_dir / "leaderboard.json").write_text(
                json.dumps({"items": [{"name": "CandidateA"}], "rejected": []}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (miner_dir / "economics.json").write_text(
                json.dumps({"total_cost_usd": 12.0, "total_tokens": 1000}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (miner_dir / "portfolio_plan.json").write_text(
                json.dumps({"weights": {"CandidateA": 0.6}}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (miner_dir / "holdout_gate.json").write_text(json.dumps({"passed": True}, ensure_ascii=False, indent=2), encoding="utf-8")
            (miner_dir / "benchmark_verdict.json").write_text(json.dumps({"passed": True}, ensure_ascii=False, indent=2), encoding="utf-8")
            (miner_dir / "promotion_log.jsonl").write_text(
                json.dumps({"promotion_ok": True, "candidate": "CandidateA"}) + "\n",
                encoding="utf-8",
            )

            arts = RunArtifacts(
                factor_memory_json=str(factor_memory_json),
                factor_cards_json=str(factor_cards_json),
                factor_failure_cards_json=str(factor_failures_json),
                factor_lineage_json=str(factor_lineage_json),
                strategy_miner_summary=str(summary_path),
                strategy_miner_dir=str(miner_dir),
            )
            outputs = finalize_strategy_factory_artifacts(
                run_id=run_id,
                run_dir=run_dir,
                experiment_cfg={"hypothesis": "demo", "owner": "tester"},
                status="success",
                started_at="2026-04-07T00:00:00Z",
                steps_meta=[{"name": "factor_eval", "status": "ok"}, {"name": "strategy_miner", "status": "ok"}],
                error_info=None,
                arts=arts,
            )

            registry_path = paths.experiment_registry_path()
            assert registry_path.exists()
            replay = json.loads((run_dir / "replay_manifest.json").read_text(encoding="utf-8"))
            assert replay["retrieval_snapshot"]["factor_retrieval"]["factor_cards"][0]["card_id"] == "flowrun:spec:breakout_card"
            lineage = json.loads((run_dir / "lineage_graph.json").read_text(encoding="utf-8"))
            assert any(edge["edge_type"] == "factor_to_strategy_candidate" for edge in lineage["edges"])
            promotion = json.loads((run_dir / "promotion_chain.json").read_text(encoding="utf-8"))
            assert promotion["factor_references"] == ["flowrun:spec:breakout_card"]
            dashboard = json.loads((run_dir / "resource_dashboard.json").read_text(encoding="utf-8"))
            assert dashboard["totals"]["promoted_strategies"] == 1
            assert outputs["replay_manifest_json"].endswith("replay_manifest.json")


def test_agent_flow_returns_run_id_and_writes_strategy_factory_artifacts() -> None:
    with tempfile.TemporaryDirectory() as td:
        env = {
            "AGENT_MARKET_ARTIFACTS_ROOT": str(Path(td) / "artifacts"),
            "AGENT_MARKET_RUNS_ROOT": str(Path(td) / "artifacts" / "runs"),
            "AGENT_MARKET_RUN_META_LATEST_PATH": str(Path(td) / "artifacts" / "run_meta.json"),
        }
        with patch.dict("os.environ", env, clear=False):
            flow = AgentFlow(AgentFlowConfig(experiment={"hypothesis": "unified contract", "owner": "ops"}))
            run_id = flow.run()
            meta = json.loads(paths.run_meta_path(run_id).read_text(encoding="utf-8"))
            artifacts = meta["artifacts"]

            assert isinstance(run_id, str) and len(run_id) == 12
            assert artifacts["experiment_registry"]
            assert artifacts["budget_plan_json"]
            assert artifacts["replay_manifest_json"]
            assert artifacts["resource_dashboard_json"]


def test_finalize_strategy_factory_artifacts_for_factor_only_flow() -> None:
    with tempfile.TemporaryDirectory() as td:
        env = {
            "AGENT_MARKET_ARTIFACTS_ROOT": str(Path(td) / "artifacts"),
            "AGENT_MARKET_RUNS_ROOT": str(Path(td) / "artifacts" / "runs"),
        }
        with patch.dict("os.environ", env, clear=False):
            run_id = "factoronly123"
            run_dir = paths.run_dir(run_id)
            run_dir.mkdir(parents=True, exist_ok=True)

            expression_path = run_dir / "expressions.json"
            expression_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "generated_at": "2026-04-07T00:00:00Z",
                        "exchange": "kucoin",
                        "pairs": ["BTC/USDT", "ETH/USDT"],
                        "timeframe": "1h",
                        "label_period": 12,
                        "llm_usage": {"total_tokens": 1234, "cost_usd": 0.42},
                        "expressions": [
                            {
                                "name": "f001",
                                "expression": "rolling_min(close, 12)",
                                "description": "breakout factor",
                                "category": "breakout",
                                "score": 0.11,
                            }
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            scored_path = run_dir / "expressions_scored.json"
            scored_path.write_text(
                json.dumps(
                    {
                        "candidates": [
                            {
                                "expression": "rolling_min(close, 12)",
                                "description": "breakout factor",
                                "category": "breakout",
                                "score": 0.11,
                                "metric_abs_ic": 0.13,
                                "complexity": 3,
                            }
                        ]
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            factor_dir = run_dir / "factor_memory"
            factor_dir.mkdir(parents=True, exist_ok=True)
            factor_memory = {
                "schema_version": "1.0",
                "factor_cards": [
                    {
                        "card_id": "factoronly123:expression:f001",
                        "name": "f001",
                        "timeframe": "1h",
                        "gate_pass": True,
                        "pareto": True,
                        "metrics": {"weighted_score": 0.11},
                    }
                ],
                "failure_cards": [],
                "edges": [{"parent": "legacy_expression_mining", "child": "f001", "edge_type": "factor_eval"}],
            }
            factor_memory_json = factor_dir / "factor_memory.json"
            factor_cards_json = factor_dir / "factor_cards.json"
            factor_failures_json = factor_dir / "factor_failure_cards.json"
            factor_lineage_json = factor_dir / "factor_lineage.json"
            factor_memory_json.write_text(json.dumps(factor_memory, ensure_ascii=False, indent=2), encoding="utf-8")
            factor_cards_json.write_text(json.dumps({"items": factor_memory["factor_cards"]}, ensure_ascii=False, indent=2), encoding="utf-8")
            factor_failures_json.write_text(json.dumps({"items": []}, ensure_ascii=False, indent=2), encoding="utf-8")
            factor_lineage_json.write_text(json.dumps({"edges": factor_memory["edges"]}, ensure_ascii=False, indent=2), encoding="utf-8")

            training_summary = run_dir / "training_summary.json"
            training_summary.write_text(json.dumps({"model": "lightgbm"}, ensure_ascii=False, indent=2), encoding="utf-8")
            feedback_summary = run_dir / "latest_backtest_summary.json"
            feedback_summary.write_text(
                json.dumps({"profit_total_pct": -0.51, "trades": 200}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            arts = RunArtifacts(
                expression_output=str(expression_path),
                expression_scored_output=str(scored_path),
                factor_memory_json=str(factor_memory_json),
                factor_cards_json=str(factor_cards_json),
                factor_failure_cards_json=str(factor_failures_json),
                factor_lineage_json=str(factor_lineage_json),
                training_summary_json=str(training_summary),
                feedback_summary_json=str(feedback_summary),
            )
            outputs = finalize_strategy_factory_artifacts(
                run_id=run_id,
                run_dir=run_dir,
                experiment_cfg={"hypothesis": "legacy factor flow", "owner": "tester"},
                status="success",
                started_at="2026-04-07T00:00:00Z",
                steps_meta=[{"name": "expression", "status": "ok"}, {"name": "ml", "status": "ok"}, {"name": "backtest", "status": "ok"}],
                error_info=None,
                arts=arts,
            )

            replay = json.loads((run_dir / "replay_manifest.json").read_text(encoding="utf-8"))
            assert replay["factor_inputs"]["expression_output"] == str(expression_path)
            assert replay["factor_inputs"]["factor_memory_json"] == str(factor_memory_json)
            promotion = json.loads((run_dir / "promotion_chain.json").read_text(encoding="utf-8"))
            assert promotion["best_factor"] == "factoronly123:expression:f001"
            dashboard = json.loads((run_dir / "resource_dashboard.json").read_text(encoding="utf-8"))
            assert dashboard["totals"]["factor_cards"] == 1
            assert dashboard["totals"]["total_tokens"] == 1234
            assert dashboard["totals"]["total_cost_usd"] == 0.42
            assert outputs["promotion_chain_json"].endswith("promotion_chain.json")


def test_step_strategy_miner_injects_factor_memory_path() -> None:
    from agent_market.flow_ext.step_dispatch import StepContext, _step_strategy_miner

    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td)
        arts = RunArtifacts(factor_memory_json="artifacts/runs/demo/factor_memory/factor_memory.json")
        ctx = StepContext(run_id="deadbeef1234", run_dir=run_dir, feedback_path=run_dir / "feedback.json", full_config=None)

        captured: dict[str, object] = {}

        def _fake_run_strategy_miner_step(cfg, *, run_id, out_dir):
            captured["cfg"] = cfg
            return {
                "strategy_miner_summary": str(out_dir / "strategy_miner_summary.json"),
                "strategy_miner_dir": str(out_dir),
            }

        with patch("agent_market.flow_ext.step_dispatch.flow_steps.run_strategy_miner_step", side_effect=_fake_run_strategy_miner_step):
            _step_strategy_miner({}, arts, ctx)

        assert captured["cfg"]["factor_memory_path"] == arts.factor_memory_json
        assert captured["cfg"]["global_factor_memory_path"]
        assert captured["cfg"]["global_strategy_knowledge_base_path"]
        assert arts.strategy_miner_summary.endswith("strategy_miner_summary.json")


def test_step_expression_builds_factor_memory_artifacts() -> None:
    from agent_market.flow_ext.step_dispatch import StepContext, _step_expression

    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td) / "run"
        run_dir.mkdir(parents=True, exist_ok=True)
        expr_out = Path(td) / "freqai_expressions_selected.json"
        scored_out = Path(td) / "freqai_expressions_selected_scored_all.json"

        def _fake_run_expression_generation(cfg, feedback_path):
            expr_out.write_text(
                json.dumps(
                    {
                        "generated_at": "2026-04-07T00:00:00Z",
                        "exchange": "kucoin",
                        "pairs": ["BTC/USDT"],
                        "timeframe": "1h",
                        "label_period": 12,
                        "llm_usage": {"total_tokens": 88, "cost_usd": 0.01},
                        "feature_file": "user_data/freqai_features_real.json",
                        "expressions": [
                            {
                                "name": "f001",
                                "expression": "rolling_min(close, 12)",
                                "description": "breakout factor",
                                "category": "breakout",
                                "score": 0.11,
                                "metric_abs_ic": 0.13,
                                "complexity": 3,
                            }
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            scored_out.write_text(
                json.dumps(
                    {
                        "candidates": [
                            {
                                "expression": "rolling_min(close, 12)",
                                "description": "breakout factor",
                                "category": "breakout",
                                "score": 0.11,
                                "metric_abs_ic": 0.13,
                                "complexity": 3,
                            },
                            {
                                "expression": "ema(close, 6)",
                                "description": "weak trend factor",
                                "category": "trend",
                                "score": -0.02,
                                "metric_abs_ic": 0.01,
                                "complexity": 2,
                            }
                        ]
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            prefix = expr_out.stem + "_"
            (expr_out.parent / f"{prefix}factor_agent_traces.json").write_text(
                json.dumps({"enabled": True, "events": []}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (expr_out.parent / f"{prefix}factor_transfer_audit.json").write_text(
                json.dumps({"items": []}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (expr_out.parent / f"{prefix}multiagent_summary.json").write_text(
                json.dumps({"enabled": True, "promotion_eligible": False}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (expr_out.parent / f"{prefix}manifest.json").write_text(
                json.dumps({"promotion_eligible": False}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        arts = RunArtifacts()
        ctx = StepContext(run_id="deadbeef1234", run_dir=run_dir, feedback_path=run_dir / "feedback.json", full_config=None)
        cfg = {"args": ["--output", str(expr_out)]}
        with patch("agent_market.flow_ext.step_dispatch.flow_steps.run_expression_generation", side_effect=_fake_run_expression_generation):
            _step_expression(cfg, arts, ctx)

        # Expression mining outputs are copied into the run directory so the run
        # is self-contained and can be archived/retained independently.
        expected_expr = (run_dir / "expression" / expr_out.name).resolve()
        expected_scored = (run_dir / "expression" / scored_out.name).resolve()
        assert arts.expression_output == str(expected_expr)
        assert arts.expression_scored_output == str(expected_scored)
        assert expected_expr.exists()
        assert expected_scored.exists()
        assert arts.factor_agent_traces_json
        assert arts.factor_transfer_audit_json
        assert arts.multiagent_summary_json
        assert arts.factor_manifest_json
        assert arts.factor_memory_json
        memory = json.loads(Path(arts.factor_memory_json).read_text(encoding="utf-8"))
        assert memory["factor_cards"][0]["name"] == "f001"
        assert arts.global_factor_memory_json
        global_memory = json.loads(Path(arts.global_factor_memory_json).read_text(encoding="utf-8"))
        assert global_memory["factor_cards"][0]["name"] == "f001"
        assert arts.factor_eval_meta
        assert arts.factor_scores_json

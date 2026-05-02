from __future__ import annotations

import json
from pathlib import Path

from agent_market.factor_memory import (
    FactorMemoryStore,
    audit_factor_memory_path,
    build_factor_memory_artifacts,
    build_factor_memory_artifacts_from_expression_output,
    merge_factor_memory_artifacts,
)
from agent_market.factor_multiagent import (
    critic_audit_expression,
    run_factor_multiagent_review,
    write_factor_multiagent_artifacts,
)


def test_build_factor_memory_artifacts_writes_cards_failures_and_edges(tmp_path: Path) -> None:
    spec_path = tmp_path / "factor_spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "name": "demo_factor",
                "hypothesis": "demo hypothesis",
                "meta": {"timeframe": "1h", "universe": ["BTC/USDT"], "data_sources": ["ohlcv"]},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    eval_meta_path = tmp_path / "factor_eval_meta.json"
    eval_meta_path.write_text(
        json.dumps({"expression": "ts_z(close, 4)"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    scores_path = tmp_path / "factor_scores.json"
    scores_path.write_text(
        json.dumps(
            {
                "generated_at": "2026-04-07T00:00:00+08:00",
                "target_col": "y",
                "items": [
                    {
                        "name": "factor_good",
                        "weighted_score": 0.7,
                        "gate_pass": True,
                        "pareto": True,
                        "ic": 0.2,
                        "turnover": 0.1,
                        "nan_ratio": 0.0,
                        "corr_to_library_max": 0.2,
                        "gates": {"max_nan_ratio": 0.5, "max_turnover": 1.0, "max_corr_to_library": 0.9},
                    },
                    {
                        "name": "factor_bad",
                        "weighted_score": -0.1,
                        "gate_pass": False,
                        "pareto": False,
                        "ic": -0.1,
                        "turnover": 2.0,
                        "nan_ratio": 0.7,
                        "corr_to_library_max": 0.95,
                        "gates": {"max_nan_ratio": 0.5, "max_turnover": 1.0, "max_corr_to_library": 0.9},
                    },
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    out = build_factor_memory_artifacts(
        run_id="deadbeef1234",
        memory_dir=tmp_path / "factor_memory",
        factor_spec_path=spec_path,
        factor_eval_meta_path=eval_meta_path,
        factor_scores_path=scores_path,
    )

    memory = json.loads(Path(out.factor_memory_json).read_text(encoding="utf-8"))
    cards = json.loads(Path(out.factor_cards_json).read_text(encoding="utf-8"))
    failures = json.loads(Path(out.factor_failure_cards_json).read_text(encoding="utf-8"))
    lineage = json.loads(Path(out.factor_lineage_json).read_text(encoding="utf-8"))

    assert len(memory["factor_cards"]) == 2
    assert len(cards["items"]) == 2
    assert len(failures["items"]) == 1
    assert failures["items"][0]["subcategory"] == "nan_ratio,turnover,corr_to_library"
    assert failures["items"][0]["repair_recipe"]["mutation_hints"]
    assert lineage["edges"][0]["parent"] == "demo_factor"
    assert memory["factor_cards"][0]["source_run_id"] == "deadbeef1234"
    assert memory["factor_cards"][0]["target"] == "y"
    assert "memory_status" in memory["factor_cards"][0]


def test_factor_multiagent_review_tags_memory_without_promotion(tmp_path: Path) -> None:
    expressions = [
        {
            "name": "valid_breakout",
            "expression": "ts_z(close, 4)",
            "category": "trend",
            "score": 0.12,
            "metric_abs_ic": 0.14,
            "complexity": 3,
        },
        {
            "name": "dup_breakout",
            "expression": "ts_z(close, 4)",
            "category": "trend",
            "score": 0.11,
        },
        {
            "name": "bad_factor",
            "expression": "__import__('os').system('x')",
            "category": "other",
        },
    ]
    curated, traces, transfer, summary = run_factor_multiagent_review(
        expressions=expressions,
        feature_cols=["close"],
        enabled=True,
        roles=["discoverer", "critic", "transfer_auditor", "curator"],
        parallelism=4,
    )

    assert len(curated) == 1
    assert curated[0]["promotion_eligible"] is False
    assert curated[0]["memory_scope"] == "pending_review"
    assert transfer["summary"]["rejected_count"] == 2
    assert summary["counts"]["duplicates_removed"] == 1
    assert traces["failure_taxonomy"]["duplicate_expression"] == 1

    expressions_path = tmp_path / "expressions.json"
    expressions_path.write_text(
        json.dumps(
            {
                "generated_at": "2026-04-07T00:00:00Z",
                "exchange": "kucoin",
                "pairs": ["BTC/USDT"],
                "timeframe": "1h",
                "label_period": 12,
                "feature_file": "user_data/freqai_features_real.json",
                "multiagent": {"enabled": True, "promotion_eligible": False},
                "multiagent_failures": summary["failures"],
                "expressions": curated,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    out = build_factor_memory_artifacts_from_expression_output(
        run_id="magent123456",
        memory_dir=tmp_path / "memory",
        expressions_path=expressions_path,
    )
    memory = json.loads(Path(out.factor_memory_json).read_text(encoding="utf-8"))
    card = memory["factor_cards"][0]
    assert card["agent_tags"]
    assert card["memory_scope"] == "pending_review"
    assert card["promotion_eligible"] is False
    assert card["metrics"]["rank_portfolio_transfer_score"] is not None
    assert any(item["category"] == "multiagent_review" for item in memory["failure_cards"])


def test_factor_multiagent_critic_uses_engine_functions_and_semantics() -> None:
    for expr in (
        "rolling_std(close, 3) + impact_proxy(3) + queue_pos_proxy()",
        "zscore_xs(close)",
        "neutralize(close, volume)",
        "fill_prob(0.01, 5)",
    ):
        audit = critic_audit_expression(expr, ["feat_momentum_24"])
        assert audit["ok"], audit

    leak = critic_audit_expression("shift(close, -1)", ["close"])
    assert leak["ok"] is False
    assert any("shift second argument must be >= 0" in reason for reason in leak["reasons"])

    grouped = critic_audit_expression("rank_xs(close, date) + zscore_xs(volume, ts)", ["close", "volume"])
    assert grouped["ok"], grouped

    for expr in ("date", "ts_z(date, 3)", "neutralize(close, ts)"):
        audit = critic_audit_expression(expr, ["close"])
        assert audit["ok"] is False
        assert any(reason.startswith("time_key_not_factor:") for reason in audit["reasons"])


def test_factor_multiagent_roles_control_transfer_and_curator() -> None:
    expressions = [
        {"name": "a", "expression": "ts_z(close, 4)", "category": "trend", "score": 0.1},
        {"name": "b", "expression": "ts_z(close, 4)", "category": "trend", "score": 0.09},
    ]

    curated, _traces, transfer, summary = run_factor_multiagent_review(
        expressions=expressions,
        feature_cols=["close"],
        enabled=True,
        roles=["discoverer", "critic"],
        parallelism=2,
    )

    assert len(curated) == 2
    assert "transfer_audit" not in curated[0]
    assert "agent_tags" not in curated[0]
    assert transfer["items"][0]["transfer_audit"]["status"] == "not_run"
    assert summary["role_execution"]["transfer_auditor"] is False
    assert summary["role_execution"]["curator"] is False
    assert summary["counts"]["duplicates_removed"] == 0


def test_factor_multiagent_artifacts_are_stem_scoped(tmp_path: Path) -> None:
    payload = {"enabled": True, "promotion_eligible": False}
    first = write_factor_multiagent_artifacts(
        output_dir=tmp_path,
        output_stem="expr_a",
        traces=payload,
        transfer_audit={"items": []},
        summary=payload,
    )
    second = write_factor_multiagent_artifacts(
        output_dir=tmp_path,
        output_stem="expr_b",
        traces=payload,
        transfer_audit={"items": []},
        summary=payload,
    )

    assert Path(first["manifest"]).name == "expr_a_manifest.json"
    assert Path(second["manifest"]).name == "expr_b_manifest.json"
    assert Path(first["manifest"]).exists()
    assert Path(second["manifest"]).exists()
    assert not (tmp_path / "manifest.json").exists()


def test_factor_memory_query_and_strategy_reference_roundtrip(tmp_path: Path) -> None:
    memory_path = tmp_path / "factor_memory.json"
    store = FactorMemoryStore(memory_path)
    store.ingest_evaluation(
        run_id="run123456789",
        factor_spec={
            "name": "breakout_factor",
            "hypothesis": "Breakout + trend confirmation",
            "meta": {"timeframe": "5m", "universe": ["BTC/USDT", "ETH/USDT"], "data_sources": ["ohlcv"]},
            "expression": "ts_max(close, 20) - ts_min(close, 20)",
        },
        factor_eval_meta={"expression": "ts_max(close, 20) - ts_min(close, 20)"},
        score_payload={
            "generated_at": "2026-04-07T00:00:00Z",
            "target_col": "future_return",
            "items": [
                {
                    "name": "breakout_card",
                    "weighted_score": 0.8,
                    "gate_pass": True,
                    "pareto": True,
                    "turnover": 0.2,
                    "nan_ratio": 0.0,
                    "corr_to_library_max": 0.2,
                    "gates": {"max_nan_ratio": 0.4, "max_turnover": 1.0, "max_corr_to_library": 0.9},
                }
            ],
        },
        source_artifacts={"factor_scores_json": "scores.json"},
    )
    store.save()

    retrieval = store.retrieve_for_strategy(
        family="rule/breakout",
        timeframe="5m",
        universe=["BTC/USDT"],
        top_n=2,
    )
    assert retrieval.factor_cards
    assert retrieval.factor_cards[0]["card_id"].endswith("breakout_card")
    assert "Factor Memory Retrieval" in retrieval.context

    store.register_strategy_references(
        card_ids=[retrieval.factor_cards[0]["card_id"]],
        strategy_name="BreakoutStrategyCandidate",
        strategy_run_id="stratrun123456",
        candidate_family="rule/breakout",
    )
    reloaded = FactorMemoryStore(memory_path)
    card = reloaded.factor_cards[0]
    assert card["reuse_count"] == 1
    assert card["downstream_strategy_references"][0]["strategy_name"] == "BreakoutStrategyCandidate"


def test_register_strategy_references_keeps_export_sidecars_in_sync(tmp_path: Path) -> None:
    spec_path = tmp_path / "factor_spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "name": "demo_factor",
                "hypothesis": "demo hypothesis",
                "meta": {"timeframe": "5m", "universe": ["BTC/USDT"], "data_sources": ["ohlcv"]},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    eval_meta_path = tmp_path / "factor_eval_meta.json"
    eval_meta_path.write_text(
        json.dumps({"expression": "ts_z(close, 4)"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    scores_path = tmp_path / "factor_scores.json"
    scores_path.write_text(
        json.dumps(
            {
                "generated_at": "2026-04-07T00:00:00+08:00",
                "target_col": "y",
                "items": [
                    {
                        "name": "factor_good",
                        "weighted_score": 0.7,
                        "gate_pass": True,
                        "pareto": True,
                        "ic": 0.2,
                        "turnover": 0.1,
                        "nan_ratio": 0.0,
                        "corr_to_library_max": 0.2,
                        "gates": {"max_nan_ratio": 0.5, "max_turnover": 1.0, "max_corr_to_library": 0.9},
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    out = build_factor_memory_artifacts(
        run_id="cafefeed1234",
        memory_dir=tmp_path / "factor_memory",
        factor_spec_path=spec_path,
        factor_eval_meta_path=eval_meta_path,
        factor_scores_path=scores_path,
    )

    store = FactorMemoryStore(Path(out.factor_memory_json))
    store.register_strategy_references(
        card_ids=["cafefeed1234:demo_factor:factor_good"],
        strategy_name="BreakoutStrategyCandidate",
        strategy_run_id="stratrun123456",
        candidate_family="rule/breakout",
    )

    cards = json.loads(Path(out.factor_cards_json).read_text(encoding="utf-8"))
    lineage = json.loads(Path(out.factor_lineage_json).read_text(encoding="utf-8"))

    assert cards["items"][0]["reuse_count"] == 1
    assert cards["items"][0]["downstream_strategy_references"][0]["strategy_name"] == "BreakoutStrategyCandidate"
    assert any(edge["edge_type"] == "strategy_reference" for edge in lineage["edges"])


def test_merge_factor_memory_artifacts_accumulates_across_runs(tmp_path: Path) -> None:
    def _write_inputs(root: Path, *, name: str, score: float) -> tuple[Path, Path, Path]:
        spec_path = root / f"{name}_spec.json"
        spec_path.write_text(
            json.dumps(
                {
                    "name": name,
                    "hypothesis": f"{name} hypothesis",
                    "meta": {"timeframe": "5m", "universe": ["BTC/USDT"], "data_sources": ["ohlcv"]},
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        eval_meta_path = root / f"{name}_eval.json"
        eval_meta_path.write_text(
            json.dumps({"expression": f"ts_z(close, {int(score * 10) + 2})"}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        scores_path = root / f"{name}_scores.json"
        scores_path.write_text(
            json.dumps(
                {
                    "generated_at": "2026-04-07T00:00:00Z",
                    "target_col": "future_return",
                    "items": [
                        {
                            "name": name,
                            "weighted_score": score,
                            "gate_pass": True,
                            "pareto": True,
                            "turnover": 0.1,
                            "nan_ratio": 0.0,
                            "corr_to_library_max": 0.2,
                            "gates": {"max_nan_ratio": 0.4, "max_turnover": 1.0, "max_corr_to_library": 0.9},
                        }
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return spec_path, eval_meta_path, scores_path

    spec1, eval1, scores1 = _write_inputs(tmp_path, name="factor_one", score=0.4)
    spec2, eval2, scores2 = _write_inputs(tmp_path, name="factor_two", score=0.7)

    local1 = build_factor_memory_artifacts(
        run_id="runfactor1111",
        memory_dir=tmp_path / "local1",
        factor_spec_path=spec1,
        factor_eval_meta_path=eval1,
        factor_scores_path=scores1,
    )
    local2 = build_factor_memory_artifacts(
        run_id="runfactor2222",
        memory_dir=tmp_path / "local2",
        factor_spec_path=spec2,
        factor_eval_meta_path=eval2,
        factor_scores_path=scores2,
    )

    global1 = merge_factor_memory_artifacts(
        source_memory_path=Path(local1.factor_memory_json),
        target_memory_dir=tmp_path / "global",
    )
    global2 = merge_factor_memory_artifacts(
        source_memory_path=Path(local2.factor_memory_json),
        target_memory_dir=tmp_path / "global",
    )
    merge_factor_memory_artifacts(
        source_memory_path=Path(local1.factor_memory_json),
        target_memory_dir=tmp_path / "global",
    )

    global_memory = json.loads(Path(global2.factor_memory_json).read_text(encoding="utf-8"))
    global_cards = json.loads(Path(global1.factor_cards_json).read_text(encoding="utf-8"))

    assert len(global_memory["factor_cards"]) == 2
    assert {item["name"] for item in global_cards["items"]} == {"factor_one", "factor_two"}


def test_factor_memory_audit_reports_coverage_duplicates_and_write_tags(tmp_path: Path) -> None:
    memory_path = tmp_path / "factor_memory.json"
    payload = {
        "schema_version": "1.0",
        "factor_cards": [
            {
                "card_id": "c1",
                "signature": "same",
                "source_run_id": "r1",
                "timeframe": "1h",
                "universe": ["BTC/USDT"],
                "target": "future_return",
                "gate_pass": True,
                "regime_tags": ["trend"],
                "snoop_level": "clean",
                "capacity_slippage_proxy": 0.4,
                "metrics": {
                    "train_ic": 0.03,
                    "validation_ic": 0.025,
                    "blind_ic": 0.02,
                    "turnover": 0.8,
                    "rank_ic": 0.02,
                    "corr_to_library_max": 0.2,
                    "strategy_transfer_score": 0.1,
                },
            },
            {
                "card_id": "c2",
                "signature": "same",
                "run_id": "legacy",
                "gate_pass": True,
                "metrics": {"ic": 0.1},
            },
        ],
        "failure_cards": [],
        "edges": [],
    }
    memory_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    audit = audit_factor_memory_path(memory_path, write_tags=True)

    assert audit["status_counts"]["tradeable_candidate"] == 1
    assert audit["status_counts"]["legacy"] == 1
    assert audit["duplicate_cluster_count"] == 1
    assert audit["coverage"]["source_run_id"]["missing"] == 1
    reloaded = json.loads(memory_path.read_text(encoding="utf-8"))
    assert reloaded["factor_cards"][0]["memory_status"] == "tradeable_candidate"
    assert "audit_missing_fields" in reloaded["factor_cards"][1]

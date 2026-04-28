from __future__ import annotations

import json
from pathlib import Path

from agent_market.factor_memory import (
    FactorMemoryStore,
    build_factor_memory_artifacts,
    merge_factor_memory_artifacts,
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

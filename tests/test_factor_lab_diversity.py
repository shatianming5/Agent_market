from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from agent_market.factor_lab import mining


def _record(expr: str, score: float, family_expr: str | None = None) -> mining.CandidateRecord:
    expression = family_expr or expr
    return mining.CandidateRecord(
        expression=expression,
        origin="test",
        oos_ic=score,
        sign_agree=8,
        combined=score,
        fitness=score,
        stability_ic=score / 2,
    )


def _ranks(values: list[float]) -> np.ndarray:
    return mining._series_to_ranks(np.asarray(values, dtype=float))  # noqa: SLF001


def test_family_tags_and_canonical_signature_collapse_monotone_variants() -> None:
    signatures = {
        mining.canonical_signature("z(close)"),
        mining.canonical_signature("ema(close, 12)"),
        mining.canonical_signature("sign(close)"),
        mining.canonical_signature("-(close)"),
    }
    assert signatures == {"close"}

    assert mining.primary_family("funding_z_200 - z(close)") == "funding"
    assert mining.primary_family("rank_xs(btc_rel_strength)") == "cross_sectional"
    tags = mining.infer_family_tags("ifelse(mtf4h_rsi_14 > 60, roll_std(close, 24), close)")
    assert {"mtf", "regime", "volatility", "trend"}.issubset(set(tags))


def test_abs_spearman_clusters_positive_and_negative_variants() -> None:
    base = np.arange(300, dtype=float)
    unrelated = np.random.default_rng(7).permutation(base)
    candidates = [_record("close", 0.2), _record("-(close)", 0.18), _record("funding_z_200", 0.15)]
    rank_cache = {
        "close": _ranks(base.tolist()),
        "-(close)": _ranks((-base).tolist()),
        "funding_z_200": _ranks(unrelated.tolist()),
    }

    clusters, pairs = mining.cluster_by_abs_corr(candidates, rank_cache, corr_gate=0.65)

    close_cluster = candidates[0].cluster_id
    assert candidates[1].cluster_id == close_cluster
    assert candidates[2].cluster_id != close_cluster
    assert len(clusters) == 2
    assert max(row["abs_corr"] for row in pairs) == 1.0


def test_diverse_selector_enforces_family_and_signature_limits() -> None:
    base = np.arange(300, dtype=float)
    unrelated_a = np.random.default_rng(11).permutation(base)
    unrelated_b = np.random.default_rng(13).permutation(base)
    candidates = [
        _record("z(close)", 0.30),
        _record("ema(close, 12)", 0.29),
        _record("sign(close)", 0.28),
        _record("funding_z_200", 0.20),
        _record("ofi_imbalance", 0.19),
    ]
    rank_cache = {
        "z(close)": _ranks(base.tolist()),
        "ema(close, 12)": _ranks((base + 1).tolist()),
        "sign(close)": _ranks((base + 2).tolist()),
        "funding_z_200": _ranks(unrelated_a.tolist()),
        "ofi_imbalance": _ranks(unrelated_b.tolist()),
    }

    selected, rejected = mining.select_diverse_candidates(
        candidates,
        rank_cache,
        top_n=5,
        hard_corr_gate=0.85,
        soft_corr_penalty_start=0.55,
        max_same_family=2,
        max_same_signature=2,
        score_mode="portfolio",
    )

    selected_exprs = {c.expression for c in selected}
    assert len(selected_exprs & {"z(close)", "ema(close, 12)", "sign(close)"}) == 1
    assert {"funding_z_200", "ofi_imbalance"}.issubset(selected_exprs)
    assert any(row["reason"] in {"corr_gate", "signature_quota"} for row in rejected)


def test_family_quota_scales_with_top_k() -> None:
    assert mining._family_limit_for_top_n(8, 20) == 4  # noqa: SLF001
    assert mining._family_limit_for_top_n(8, 40) == 8  # noqa: SLF001
    assert mining._family_limit_for_top_n(8, 200) == 40  # noqa: SLF001


def test_low_coverage_feature_filter_rejects_sparse_refs() -> None:
    panel = pd.DataFrame(
        {
            "good_alpha": np.linspace(0.0, 1.0, 10),
            "sparse_l2": [np.nan] * 9 + [1.0],
            "constant_alpha": [5.0] * 10,
            "close": np.linspace(100.0, 110.0, 10),
        }
    )
    cfg = mining.MiningConfig(
        llm_filter_low_coverage=True,
        llm_min_feature_coverage=0.6,
        llm_min_feature_rows=3,
    )
    stats = mining._feature_quality_stats(panel, ["good_alpha", "sparse_l2", "constant_alpha"])  # noqa: SLF001

    usable = mining._llm_usable_feature_cols(["good_alpha", "sparse_l2", "constant_alpha"], stats, cfg)  # noqa: SLF001
    assert usable == ["good_alpha"]

    reason, detail = mining._feature_rejection_detail(  # noqa: SLF001
        "z(sparse_l2) + z(good_alpha)",
        allowed_columns=set(panel.columns),
        feature_stats=stats,
        cfg=cfg,
    )
    assert reason == "low_feature_coverage"
    assert "sparse_l2" in detail

    reason, detail = mining._feature_rejection_detail(  # noqa: SLF001
        "z(unknown_feature)",
        allowed_columns=set(panel.columns),
        feature_stats=stats,
        cfg=cfg,
    )
    assert reason == "invalid_expr"
    assert "unknown_feature" in detail


def test_diverse_export_writes_expression_file_and_report(tmp_path: Path, monkeypatch) -> None:
    survivors = [
        _record("z(close)", 0.30),
        _record("ema(close, 12)", 0.29),
        _record("funding_z_200", 0.22),
    ]
    series = {
        "z(close)": np.arange(300, dtype=float),
        "ema(close, 12)": np.arange(300, dtype=float) + 1,
        "funding_z_200": np.random.default_rng(17).permutation(np.arange(300, dtype=float)),
    }

    def fake_eval_ic(_big, expr: str, _cfg, return_oos_series: bool = False):  # noqa: ANN001
        return {
            "status": "ok",
            "passes": True,
            "train_ic": 0.1,
            "oos_ic": 0.1,
            "sign_agree": 8,
            "combined": 0.1,
            "fitness": 0.1,
            "oos_series": series[expr],
        }

    from agent_market.factor_lab import paths as factor_paths

    monkeypatch.setattr(mining, "load_state", lambda _tag: (1, survivors, set()))
    monkeypatch.setattr(mining, "build_big", lambda **_kwargs: (pd.DataFrame(), []))
    monkeypatch.setattr(mining, "eval_ic", fake_eval_ic)
    monkeypatch.setattr(factor_paths, "USER_DATA", tmp_path)

    out = mining.export_top("unit", n=2, diverse=True, corr_gate=0.65, score_mode="portfolio")

    payload = json.loads(out.read_text(encoding="utf-8"))
    report = json.loads((tmp_path / "factor_diversity_report_unit.json").read_text(encoding="utf-8"))
    assert out.name == "freqai_expressions_unit_diverse.json"
    assert len(payload["expressions"]) == 2
    assert {row["expression"] for row in payload["expressions"]} == {"z(close)", "funding_z_200"}
    assert report["selected_n"] == 2
    assert report["cluster_count"] == 2
    assert report["max_pairwise_abs_corr"]["abs_corr"] < 0.65

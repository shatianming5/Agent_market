from __future__ import annotations

import json

import numpy as np
import pandas as pd

from agent_market.factor_lab import mining, purification, reporting
from agent_market.freqai.llm_miner_v2 import FactorExample, build_generation_prompt_v2, build_system_prompt


def _panel(n_dates: int = 40, pairs: tuple[str, ...] = ("A/USDT", "B/USDT", "C/USDT", "D/USDT", "E/USDT")) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=n_dates, freq="1h", tz="UTC")
    rows = []
    rng = np.random.default_rng(123)
    pair_load = {pair: i - 2 for i, pair in enumerate(pairs)}
    for t, date in enumerate(dates):
        shock = np.sin(t / 5.0)
        for pair in pairs:
            exposure = pair_load[pair] + 0.1 * shock
            alpha = rng.normal(0.0, 0.3)
            rows.append(
                {
                    "date": date,
                    "__pair__": pair,
                    "close": 100.0 + t + pair_load[pair],
                    "volume": 1000.0 + 10.0 * pair_load[pair],
                    "pair_beta_72_btc": exposure,
                    "alpha": alpha,
                    "__fwd_ret__": alpha,
                }
            )
    return pd.DataFrame(rows)


def test_winsor_zscore_neutralize_removes_linear_exposure() -> None:
    panel = _panel()
    raw = panel["pair_beta_72_btc"].copy()
    raw.iloc[0] = 10_000.0

    clipped = purification.winsorize_cross_section(panel, raw, "mad")
    assert float(clipped.iloc[0]) < 10_000.0

    z = purification.standardize_cross_section(panel, clipped, "zscore")
    grouped = z.groupby(panel["date"])
    assert abs(float(grouped.mean().abs().max())) < 1e-9
    assert abs(float(grouped.std(ddof=0).dropna().median()) - 1.0) < 1e-6

    cfg = purification.PurifyConfig(
        mode="neutralized",
        winsor="none",
        standardize="zscore",
        neutralize="ridge",
        exposures=("pair",),
        ridge_alpha=1e-6,
    )
    pur = purification.apply_purification(panel, panel["pair_beta_72_btc"], cfg)
    assert pur.diagnostics["exposure_count"] >= 1
    assert float(pur.neutralized.abs().mean()) < 1e-3
    assert float(pur.diagnostics["max_exposure_corr"]) < 0.05


def test_neutralized_ic_preserves_independent_alpha_component() -> None:
    panel = _panel(n_dates=720)
    panel["factor"] = panel["pair_beta_72_btc"] + panel["alpha"]
    cfg = mining.MiningConfig(
        train=("2025-01-01", "2025-01-16"),
        oos=("2025-01-16", "2025-01-31"),
        ic_gate=0.01,
        sign_gate=1,
        purify_mode="blend",
        purify_neutralize="ridge",
        purify_exposures="pair",
    )

    metrics = mining.eval_ic(panel, "factor", cfg)

    assert metrics["status"] == "ok"
    assert "raw_ic" in metrics
    assert "clean_ic" in metrics
    assert "neutralized_ic" in metrics
    assert metrics["residual_ic_ratio"] > 0.05
    assert metrics["exposure_count"] >= 1


def test_cached_and_uncached_purification_metrics_match(tmp_path) -> None:
    panel = _panel(n_dates=720)
    panel["factor"] = panel["pair_beta_72_btc"] + panel["alpha"]
    common = dict(
        train=("2025-01-01", "2025-01-16"),
        oos=("2025-01-16", "2025-01-31"),
        ic_gate=0.01,
        sign_gate=1,
        purify_mode="blend",
        purify_neutralize="ridge",
        purify_exposures="pair",
    )
    uncached = mining.eval_ic(panel, "factor", mining.MiningConfig(**common, no_cache=True))
    cached = mining.eval_ic(panel, "factor", mining.MiningConfig(**common, cache_dir=str(tmp_path / "cache")))

    assert cached["status"] == uncached["status"] == "ok"
    for key in ("raw_ic", "clean_ic", "neutralized_ic", "residual_ic_ratio", "exposure_r2"):
        assert np.isclose(float(cached[key]), float(uncached[key]), atol=1e-12)


def test_cached_eval_hit_skips_recompute(tmp_path, monkeypatch) -> None:
    panel = _panel(n_dates=20)
    cfg = mining.MiningConfig(cache_dir=str(tmp_path / "cache"))
    calls = {"n": 0}

    def fake_eval(_big, _expr, _cfg, _return_oos_series):  # noqa: ANN001
        calls["n"] += 1
        return {
            "status": "ok",
            "train_ic": 0.1,
            "oos_ic": 0.2,
            "sign_agree": 3,
            "n_pairs": 5,
            "combined": 0.06,
            "fitness": 0.06,
            "passes": True,
            "eval_mode": "legacy",
        }

    monkeypatch.setattr(mining, "_eval_legacy", fake_eval)
    first = mining.eval_ic(panel, "alpha", cfg)
    second = mining.eval_ic(panel, "alpha", cfg)

    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert calls["n"] == 1


def test_panel_and_exposure_cache_keys_invalidate(tmp_path, monkeypatch) -> None:
    dates = pd.date_range("2025-01-01", periods=80, freq="1h", tz="UTC")
    for pair, close in {
        "BTC_USDT": np.linspace(100.0, 110.0, len(dates)),
        "ETH_USDT": np.linspace(200.0, 230.0, len(dates)),
    }.items():
        pd.DataFrame(
            {
                "date": dates,
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": 1000.0,
            }
        ).to_feather(tmp_path / f"{pair}-1h.feather")
    feat_file = tmp_path / "freqai_features_real.json"
    feat_file.write_text(json.dumps({"features": []}), encoding="utf-8")
    monkeypatch.setattr(mining, "FEATURE_FILE", feat_file)
    monkeypatch.setattr(mining, "apply_configured_features", lambda df, _cfg: df)

    cache_dir = tmp_path / "cache"
    big_a, _ = mining.build_big(
        timeframe="1h",
        label_bars=3,
        label_mode="forward_return",
        data_dir=tmp_path,
        pairs="auto",
        cache_dir=cache_dir,
    )
    big_b, _ = mining.build_big(
        timeframe="1h",
        label_bars=3,
        label_mode="pair_spread_btc",
        pair_reference="BTC/USDT",
        data_dir=tmp_path,
        pairs="auto",
        cache_dir=cache_dir,
    )
    feat_file.write_text(json.dumps({"features": [{"name": "noop"}]}), encoding="utf-8")
    big_c, _ = mining.build_big(
        timeframe="1h",
        label_bars=3,
        label_mode="forward_return",
        data_dir=tmp_path,
        pairs="auto",
        cache_dir=cache_dir,
    )

    assert big_a.attrs["factor_lab_panel_key"] != big_b.attrs["factor_lab_panel_key"]
    assert big_a.attrs["factor_lab_panel_key"] != big_c.attrs["factor_lab_panel_key"]

    exp_a, cols_a = purification.build_exposure_frame(
        big_a,
        "pair",
        cache_dir=str(cache_dir),
        panel_fingerprint_hint="panel-a",
    )
    exp_b, cols_b = purification.build_exposure_frame(
        big_a,
        "market",
        cache_dir=str(cache_dir),
        panel_fingerprint_hint="panel-a",
    )
    assert len(exp_a) == len(exp_b) == len(big_a)
    assert cols_a != cols_b


def test_pure_residual_rejects_exposure_only_factor() -> None:
    panel = _panel(n_dates=720)
    panel["__fwd_ret__"] = panel["pair_beta_72_btc"]
    cfg = mining.MiningConfig(
        train=("2025-01-01", "2025-01-16"),
        oos=("2025-01-16", "2025-01-31"),
        purify_mode="neutralized",
        purify_neutralize="ridge",
        purify_exposures="pair",
        alpha_objective="pure_residual",
    )

    metrics = mining.eval_ic(panel, "pair_beta_72_btc", cfg)

    assert not metrics.get("passes")
    assert abs(float(metrics.get("raw_ic", 0.0))) > 0.5
    assert abs(float(metrics.get("neutralized_ic", 0.0))) < 0.008


def test_pure_residual_accepts_exposure_plus_independent_alpha() -> None:
    pairs = tuple(f"P{i}/USDT" for i in range(10))
    panel = _panel(n_dates=720, pairs=pairs)
    panel["factor"] = 0.1 * panel["pair_beta_72_btc"] + panel["alpha"]
    cfg = mining.MiningConfig(
        train=("2025-01-01", "2025-01-16"),
        oos=("2025-01-16", "2025-01-31"),
        purify_mode="neutralized",
        purify_neutralize="ridge",
        purify_exposures="pair",
        alpha_objective="pure_residual",
    )

    metrics = mining.eval_ic(panel, "factor", cfg)

    assert metrics["status"] == "ok"
    assert metrics["passes"]
    assert abs(metrics["neutralized_ic"]) >= 0.008
    assert metrics["residual_ic_ratio"] >= 0.15
    assert metrics["exposure_r2"] <= 0.90
    assert metrics["combined"] == metrics["fitness"]


def test_pure_residual_gate_rejects_high_exposure_and_low_residual_ratio() -> None:
    cfg = mining.MiningConfig(alpha_objective="pure_residual")
    low_ratio = {
        "status": "ok",
        "neutralized_ic": 0.02,
        "sign_agree": 10,
        "n_pairs": 10,
        "cost_mult": 1.0,
        "residual_ic_ratio": 0.05,
        "exposure_r2": 0.1,
        "max_exposure_corr": 0.1,
    }
    high_exposure = dict(low_ratio, residual_ic_ratio=0.4, exposure_r2=0.95)

    low_ratio_scored = mining._apply_pure_residual_objective(low_ratio, cfg)  # noqa: SLF001
    high_exposure_scored = mining._apply_pure_residual_objective(high_exposure, cfg)  # noqa: SLF001

    assert not low_ratio_scored["passes"]
    assert "low_residual_ratio" in low_ratio_scored["reject_reasons"]
    assert not high_exposure_scored["passes"]
    assert "high_exposure_r2" in high_exposure_scored["reject_reasons"]


def test_residual_alpha_prompt_profile_includes_purity_feedback() -> None:
    sys_prompt = build_system_prompt(prompt_profile="residual_alpha_v2")
    prompt = build_generation_prompt_v2(
        feature_glossary="alpha\npair_beta_72_btc\nfunding_z_200",
        functions_doc="z(x), ifelse(cond, a, b)",
        success_examples=[
            FactorExample(
                name="pure_1",
                expression="z(alpha)",
                category="micro",
                abs_ic=0.02,
                oos_ic=0.02,
                raw_ic=0.03,
                clean_ic=0.025,
                neutralized_ic=0.018,
                residual_ic_ratio=0.72,
                exposure_r2=0.12,
                max_exposure_corr=0.21,
            )
        ],
        avoid_examples=[],
        failure_patterns=[],
        round_idx=3,
        request_count=2,
        label_period=12,
        prompt_profile="residual_alpha_v2",
        rejection_summary={
            "counts": {"low_neutralized_ic": 4, "high_exposure_r2": 2},
            "recent": [{"reason": "high_exposure_r2", "expression": "pair_beta_72_btc"}],
        },
    )

    assert "neutralized_ic" in sys_prompt
    assert "neutralized_ic=+0.0180" in prompt
    assert "residual_ic_ratio=0.72" in prompt
    assert "exposure_r2=0.12" in prompt
    assert "orthogonality_claim" in prompt
    assert "Recent rejection summary" in prompt
    assert "pair_beta_*" in prompt


def test_pure_residual_checkpoint_saves_gates_and_rejection_summary(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(mining, "LAB_STATE", tmp_path)
    cfg = mining.MiningConfig(
        alpha_objective="pure_residual",
        prompt_profile="residual_alpha_v2",
        cache_dir=str(tmp_path / "cache"),
    )
    summary = mining._new_rejection_summary()  # noqa: SLF001
    loop_counts = {reason: 0 for reason in mining.REJECTION_REASON_ORDER}
    mining._record_rejection(  # noqa: SLF001
        summary,
        loop_counts,
        "low_neutralized_ic",
        expression="pair_beta_72_btc",
    )

    mining.save_state("unit", 1, [], set(), cfg, rejection_summary=summary)

    payload = json.loads((tmp_path / "mining" / "unit" / "latest.json").read_text(encoding="utf-8"))
    assert payload["alpha_objective"] == "pure_residual"
    assert payload["pure_residual_gates"]["min_abs_neutralized_ic"] == 0.008
    assert payload["cache_stats"]["enabled"] is True
    assert payload["rejection_summary"]["total"]["low_neutralized_ic"] == 1


def test_factor_and_exposure_reports_write_machine_readable_outputs(tmp_path, monkeypatch) -> None:
    panel = _panel(n_dates=30)
    panel["factor"] = panel["alpha"] + 0.2 * panel["pair_beta_72_btc"]
    survivors = [
        mining.CandidateRecord(
            expression="factor",
            origin="unit",
            combined=0.1,
            fitness=0.1,
            sign_agree=3,
        )
    ]

    monkeypatch.setattr(reporting, "LAB_STATE", tmp_path / "factor_lab")
    monkeypatch.setattr(mining, "load_state", lambda _tag: (0, survivors, set()))
    state_dir = tmp_path / "factor_lab" / "mining" / "unit"
    state_dir.mkdir(parents=True)
    (state_dir / "latest.json").write_text(
        json.dumps(
            {
                "config": {
                    "timeframe": "4h",
                    "label_period": 3,
                    "label_horizons": [3],
                    "label_mode": "pair_beta_resid_btc",
                    "pair_reference": "ETH/USDT",
                    "data_venue": "binance",
                    "pairs": "default",
                }
            }
        ),
        encoding="utf-8",
    )
    build_calls = []

    def fake_build_big(**kwargs):
        build_calls.append(dict(kwargs))
        return panel.copy(), ["factor", "pair_beta_72_btc"]

    monkeypatch.setattr(mining, "build_big", fake_build_big)

    cache_dir = tmp_path / "cache"
    factor_paths = reporting.factor_report(
        tag="unit",
        n=1,
        purify_mode="blend",
        purify_exposures="pair",
        cache_dir=cache_dir,
    )
    exposure_paths = reporting.exposure_report(
        tag="unit",
        n=1,
        purify_mode="blend",
        purify_exposures="pair",
        cache_dir=cache_dir,
        attribution_mode="fast",
        attribution_max_dates=8,
    )
    fast_json = json.loads((tmp_path / "factor_lab" / "reports" / "unit" / "exposure_report.json").read_text())
    exact_paths = reporting.exposure_report(
        tag="unit",
        n=1,
        purify_mode="blend",
        purify_exposures="pair",
        cache_dir=cache_dir,
        attribution_mode="exact",
    )

    factor_json = json.loads((tmp_path / "factor_lab" / "reports" / "unit" / "factor_report.json").read_text())
    exposure_json = json.loads((tmp_path / "factor_lab" / "reports" / "unit" / "exposure_report.json").read_text())
    assert factor_paths["n_reported"] == 1
    assert exposure_paths["n_reported"] == 1
    assert exact_paths["n_reported"] == 1
    assert build_calls[0]["timeframe"] == "4h"
    assert build_calls[0]["label_bars"] == 3
    assert build_calls[0]["label_mode"] == "pair_beta_resid_btc"
    assert build_calls[0]["pair_reference"] == "ETH/USDT"
    assert build_calls[0]["data_venue"] == "binance"
    assert build_calls[0]["pairs"] == "default"
    assert factor_json["timeframe"] == "4h"
    assert factor_json["label_bars"] == 3
    assert factor_json["data_venue"] == "binance"
    assert factor_json["pairs"] == "default"
    assert exposure_json["timeframe"] == "4h"
    assert exposure_json["label_bars"] == 3
    assert exposure_json["data_venue"] == "binance"
    assert exposure_json["pairs"] == "default"
    assert fast_json["attribution_mode"] == "fast"
    assert fast_json["rows"][0]["attribution_mode"] == "fast"
    assert "mean_contrib_pair" in fast_json["rows"][0]
    assert factor_json["rows"][0]["turnover"] >= 0.0
    assert factor_json["rows"][0]["turnover"] <= 1.0
    assert exposure_json["attribution_mode"] == "exact"
    assert exposure_json["rows"][0]["attribution_mode"] == "exact"
    assert abs(exposure_json["rows"][0]["mean_abs_reconstruction_error"]) < 1e-12

    monkeypatch.setattr(mining, "_eval_factor_by_pair", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("recomputed")))
    cached_again = reporting.factor_report(
        tag="unit",
        n=1,
        purify_mode="blend",
        purify_exposures="pair",
        cache_dir=cache_dir,
    )
    assert cached_again["n_reported"] == 1

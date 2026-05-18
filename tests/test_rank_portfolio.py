from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from agent_market.factor_lab import mining, rank_portfolio as rp


def _record(expr: str, ic: float, *, family_hint: str = "close", sign_agree: int = 7) -> mining.CandidateRecord:
    return mining.annotate_diversity(
        mining.CandidateRecord(
            expression=expr if expr else family_hint,
            origin="unit",
            oos_ic=ic,
            neutralized_ic=ic,
            residual_ic_ratio=1.0,
            sign_agree=sign_agree,
            combined=abs(ic),
            fitness=abs(ic),
        )
    )


def _ranks(values: np.ndarray) -> np.ndarray:
    ranks = mining._series_to_ranks(values.astype(float))  # noqa: SLF001
    assert ranks is not None
    return ranks


def _write_futures_feather(root, pair: str, timeframe: str = "1h", *, close: float = 100.0) -> None:
    root.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range("2026-01-01", periods=3, freq=timeframe, tz="UTC")
    frame = pd.DataFrame(
        {
            "date": dates,
            "open": [close] * len(dates),
            "high": [close + 1.0] * len(dates),
            "low": [close - 1.0] * len(dates),
            "close": [close] * len(dates),
            "volume": [1000.0] * len(dates),
        }
    )
    frame.to_feather(root / f"{rp._pair_file_token(pair)}-{timeframe}-futures.feather")  # noqa: SLF001


def test_load_venue_ohlcv_uses_target_futures_root_without_okx_fallback(tmp_path, monkeypatch) -> None:
    user_data = tmp_path / "user_data"
    monkeypatch.setenv("AGENT_MARKET_USER_DATA_ROOT", str(user_data))
    _write_futures_feather(user_data / "data" / "okx" / "futures", "BTC/USDT", close=100.0)

    with pytest.raises(FileNotFoundError, match="no bybit 1h futures"):
        rp.load_venue_ohlcv(venue="bybit", pairs=["BTC/USDT"], timeframe="1h")

    _write_futures_feather(user_data / "data" / "bybit" / "futures", "BTC/USDT", close=200.0)
    panel = rp.load_venue_ohlcv(venue="bybit", pairs=["BTC/USDT"], timeframe="1h")

    assert panel["__pair__"].unique().tolist() == ["BTC/USDT"]
    assert panel["close"].tolist() == [200.0, 200.0, 200.0]


def test_load_candidates_can_freeze_specific_mining_state(tmp_path) -> None:
    state_path = tmp_path / "state_0001.json"
    state_path.write_text(
        json.dumps(
            {
                "loop": 1,
                "survivors": [
                    {
                        "expression": "close",
                        "origin": "unit_state",
                        "neutralized_ic": 0.02,
                        "oos_ic": 0.02,
                        "sign_agree": 7,
                        "residual_ic_ratio": 1.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    candidates, source = rp.load_candidates(candidate_state=state_path)

    assert source == str(state_path)
    assert len(candidates) == 1
    assert candidates[0].expression == "close"
    assert candidates[0].origin == "unit_state"


def test_rank_pair_universe_inherits_auto_futures_pairs_from_candidate_state(tmp_path) -> None:
    data_root = tmp_path / "binance_futures"
    for pair in ("BTC/USDT", "ETH/USDT", "SOL/USDT"):
        _write_futures_feather(data_root, pair, timeframe="4h")
    state_path = tmp_path / "state_0001.json"
    state_path.write_text(
        json.dumps(
            {
                "config": {
                    "timeframe": "4h",
                    "data_venue": "binance",
                    "data_dir": str(data_root),
                    "pairs": "auto",
                },
                "survivors": [],
            }
        ),
        encoding="utf-8",
    )

    pairs, report = rp._resolve_rank_pair_universe(  # noqa: SLF001
        tag="unit",
        candidate_state=state_path,
        timeframe="4h",
        feature_venue="binance",
    )

    assert report["source"] == "mining_config"
    assert report["count"] == 3
    assert pairs == ["BTC/USDT", "ETH/USDT", "SOL/USDT"]


def test_negative_ic_factor_is_reversed_in_ensemble_score() -> None:
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01"] * 3 + ["2026-01-02"] * 3, utc=True),
            "__pair__": ["A/USDT", "B/USDT", "C/USDT"] * 2,
            "alpha": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        }
    )
    factors = [
        rp.SelectedFactor(
            name="neg",
            expression="alpha",
            direction=-1.0,
            weight=1.0,
            ensemble_score=1.0,
            neutralized_ic=-0.02,
            oos_ic=-0.02,
            residual_ic_ratio=1.0,
            sign_agree=8,
            primary_family="trend",
        )
    ]

    scores, report = rp.compute_ensemble_scores(panel, factors)

    first_date = scores.loc[scores["date"] == pd.Timestamp("2026-01-01", tz="UTC")]
    assert report["used_factor_count"] == 1
    assert first_date.sort_values("rp_score", ascending=False)["__pair__"].tolist()[0] == "A/USDT"


def test_selector_uses_fallback_and_respects_family_cap_and_corr_gate() -> None:
    base = np.arange(300, dtype=float)
    records = [
        _record("close", 0.011),
        _record("ema(close, 12)", 0.0105),
        _record("funding_z_200", 0.010),
        _record("ofi_10", 0.0095),
    ]
    rank_cache = {
        "close": _ranks(base),
        "ema(close, 12)": _ranks(base + 1.0),
        "funding_z_200": _ranks(np.random.default_rng(2).permutation(base)),
        "ofi_10": _ranks(np.random.default_rng(3).permutation(base)),
    }
    cfg = rp.SelectionConfig(
        n=3,
        strict_abs_ic=0.012,
        fallback_abs_ic=0.008,
        min_before_fallback=3,
        family_cap=1,
        strict_corr_gate=0.65,
        fallback_corr_gate=0.75,
    )

    selected, report = rp.select_factor_records(records, config=cfg, rank_cache=rank_cache)

    expressions = {factor.expression for factor in selected}
    assert report["mode"] == "fallback_relaxed"
    assert len(selected) == 3
    assert len(expressions & {"close", "ema(close, 12)"}) == 1
    assert "funding_z_200" in expressions
    assert "ofi_10" in expressions


def test_rank_signals_enforce_counts_and_exposure_caps() -> None:
    pairs = [f"P{i}/USDT" for i in range(10)]
    date = pd.Timestamp("2026-01-01", tz="UTC")
    score_frame = pd.DataFrame(
        {
            "date": [date] * 10,
            "__pair__": pairs,
            "rp_score": np.arange(10, dtype=float),
            "rp_score_z": np.linspace(-2.0, 2.0, 10),
        }
    )
    venue = pd.DataFrame(
        {
            "date": [date] * 10,
            "__pair__": pairs,
            "open": [100.0] * 10,
            "high": [101.0] * 10,
            "low": [99.0] * 10,
            "close": [100.0] * 10,
            "volume": [1000.0] * 10,
        }
    )
    cfg = rp.RiskConfig(gross_cap=10.0, net_cap=2.5, single_pair_cap=2.0, top_k=3)

    signals, diag = rp.build_rank_signals(score_frame, venue, cfg)

    assert int((signals["rp_side"] == 1).sum()) == 3
    assert int((signals["rp_side"] == -1).sum()) == 3
    assert signals["rp_target_weight"].abs().sum() <= cfg.gross_cap + 1e-9
    assert abs(signals["rp_target_weight"].sum()) <= cfg.net_cap + 1e-9
    assert signals["rp_target_weight"].abs().max() <= cfg.single_pair_cap + 1e-9
    assert diag["liquidation_rejects"] == 0


def test_rolling_edge_direction_is_causal() -> None:
    pairs = [f"P{i}/USDT" for i in range(4)]
    dates = pd.date_range("2026-01-01", periods=3, freq="1h", tz="UTC")
    score_frame = pd.DataFrame(
        {
            "date": np.repeat(dates, len(pairs)),
            "__pair__": pairs * len(dates),
            "rp_score": [0.0, 1.0, 2.0, 3.0] * len(dates),
            "rp_score_z": [-1.2, -0.4, 0.4, 1.2] * len(dates),
        }
    )
    closes = {
        dates[0]: [100.0, 100.0, 100.0, 100.0],
        # The first forward return is positively aligned with score.
        dates[1]: [100.0, 101.0, 102.0, 103.0],
        # The second forward return is negatively aligned with score.
        dates[2]: [103.0, 103.02, 103.02, 103.0],
    }
    rows = []
    for date in dates:
        for pair, close in zip(pairs, closes[date]):
            rows.append({
                "date": date,
                "__pair__": pair,
                "open": close,
                "high": close * 1.001,
                "low": close * 0.999,
                "close": close,
                "volume": 1000.0,
            })
    venue = pd.DataFrame(rows)
    cfg = rp.RiskConfig(
        gross_cap=1.0,
        net_cap=1.0,
        single_pair_cap=1.0,
        risk_per_trade=0.02,
        top_k=1,
        min_pairs_for_top_k=4,
        low_pair_top_k=1,
        side_mode="short",
        min_abs_score_z=0.0,
        rebalance_hours=1,
        edge_mode="rolling_ic",
        edge_lookback_hours=4,
        edge_min_periods=1,
        edge_deadband=0.0,
    )

    signals, _ = rp.build_rank_signals(score_frame, venue, cfg)

    first = signals.loc[signals["date"] == dates[0]]
    second = signals.loc[signals["date"] == dates[1]]
    assert int((first["rp_side"] != 0).sum()) == 0
    assert float(second["rp_edge_sign"].iloc[0]) == 1.0
    assert second.loc[second["pair"] == "P0/USDT", "rp_side"].iloc[0] == -1
    assert second.loc[second["pair"] == "P3/USDT", "rp_side"].iloc[0] == 0


def test_rank_signal_trading_start_preserves_edge_warmup_without_carrying_positions() -> None:
    pairs = [f"P{i}/USDT" for i in range(4)]
    dates = pd.date_range("2026-01-01", periods=3, freq="1h", tz="UTC")
    score_frame = pd.DataFrame(
        {
            "date": np.repeat(dates, len(pairs)),
            "__pair__": pairs * len(dates),
            "rp_score": [0.0, 1.0, 2.0, 3.0] * len(dates),
            "rp_score_z": [-1.2, -0.4, 0.4, 1.2] * len(dates),
        }
    )
    closes = {
        dates[0]: [100.0, 100.0, 100.0, 100.0],
        dates[1]: [100.0, 101.0, 102.0, 103.0],
        dates[2]: [103.0, 103.02, 103.02, 103.0],
    }
    venue_rows = []
    for date in dates:
        for pair, close in zip(pairs, closes[date]):
            venue_rows.append({
                "date": date,
                "__pair__": pair,
                "open": close,
                "high": close * 1.001,
                "low": close * 0.999,
                "close": close,
                "volume": 1000.0,
            })
    cfg = rp.RiskConfig(
        gross_cap=1.0,
        net_cap=1.0,
        single_pair_cap=1.0,
        risk_per_trade=0.02,
        top_k=1,
        min_pairs_for_top_k=4,
        low_pair_top_k=1,
        side_mode="short",
        min_abs_score_z=0.0,
        rebalance_hours=2,
        edge_mode="rolling_ic",
        edge_lookback_hours=4,
        edge_min_periods=1,
        edge_deadband=0.0,
    )

    signals, _ = rp.build_rank_signals(
        score_frame,
        pd.DataFrame(venue_rows),
        cfg,
        trading_start=dates[1],
    )

    assert signals["date"].min() == dates[1]
    first_trading_date = signals.loc[signals["date"] == dates[1]]
    assert bool(first_trading_date["rp_rebalance"].all())
    assert float(first_trading_date["rp_edge_sign"].iloc[0]) == 1.0
    assert first_trading_date.loc[first_trading_date["pair"] == "P0/USDT", "rp_side"].iloc[0] == -1


def test_rolling_edge_rank_export_uses_pre_start_warmup_window() -> None:
    cfg = rp.RiskConfig(edge_mode="rolling_ic", edge_lookback_hours=336, edge_min_periods=168)

    load_start, report = rp._warmup_start_for_rank_signals("2026-04-01", cfg)  # noqa: SLF001

    assert report["enabled"] is True
    assert report["warmup_hours"] == 336
    assert load_start == "2026-03-18 00:00:00"


def test_short_momentum_filter_blocks_strong_rebound_entries() -> None:
    pairs = [f"P{i}/USDT" for i in range(4)]
    dates = pd.date_range("2026-01-01", periods=80, freq="1h", tz="UTC")
    score_rows = []
    venue_rows = []
    for i, date in enumerate(dates):
        for j, pair in enumerate(pairs):
            score_rows.append({
                "date": date,
                "__pair__": pair,
                "rp_score": float(j),
                "rp_score_z": float(j - 1.5),
            })
            close = 100.0
            if pair == "P0/USDT" and i >= 72:
                close = 110.0
            venue_rows.append({
                "date": date,
                "__pair__": pair,
                "open": close,
                "high": close * 1.001,
                "low": close * 0.999,
                "close": close,
                "volume": 1000.0,
            })
    cfg = rp.RiskConfig(
        gross_cap=1.0,
        net_cap=1.0,
        single_pair_cap=1.0,
        risk_per_trade=0.02,
        top_k=1,
        min_pairs_for_top_k=4,
        low_pair_top_k=1,
        side_mode="short",
        min_abs_score_z=0.0,
        rebalance_hours=1,
        short_max_mom_72h=0.05,
    )

    signals, _ = rp.build_rank_signals(pd.DataFrame(score_rows), pd.DataFrame(venue_rows), cfg)
    last = signals.loc[signals["date"] == dates[-1]]

    assert last.loc[last["pair"] == "P0/USDT", "rp_side"].iloc[0] == 0


def test_market_regime_filter_blocks_short_entries_in_strong_market() -> None:
    pairs = [f"P{i}/USDT" for i in range(4)]
    dates = pd.date_range("2026-01-01", periods=80, freq="1h", tz="UTC")
    score_rows = []
    venue_rows = []
    for i, date in enumerate(dates):
        for j, pair in enumerate(pairs):
            score_rows.append({
                "date": date,
                "__pair__": pair,
                "rp_score": float(j),
                "rp_score_z": float(j - 1.5),
            })
            close = 100.0 if i < 72 else 110.0
            venue_rows.append({
                "date": date,
                "__pair__": pair,
                "open": close,
                "high": close * 1.001,
                "low": close * 0.999,
                "close": close,
                "volume": 1000.0,
            })
    cfg = rp.RiskConfig(
        gross_cap=1.0,
        net_cap=1.0,
        single_pair_cap=1.0,
        risk_per_trade=0.02,
        top_k=1,
        min_pairs_for_top_k=4,
        low_pair_top_k=1,
        side_mode="short",
        min_abs_score_z=0.0,
        rebalance_hours=1,
        short_max_market_mom_72h=0.05,
    )

    signals, _ = rp.build_rank_signals(pd.DataFrame(score_rows), pd.DataFrame(venue_rows), cfg)
    last = signals.loc[signals["date"] == dates[-1]]

    assert last["rp_market_mom_72h"].median() > 0.05
    assert last.loc[last["pair"] == "P0/USDT", "rp_side"].iloc[0] == 0


def test_exclude_pairs_blocks_known_bad_pair_entries() -> None:
    pairs = [f"P{i}/USDT" for i in range(4)]
    date = pd.Timestamp("2026-01-01", tz="UTC")
    score_frame = pd.DataFrame(
        {
            "date": [date] * 4,
            "__pair__": pairs,
            "rp_score": [0.0, 1.0, 2.0, 3.0],
            "rp_score_z": [-1.5, -0.5, 0.5, 1.5],
        }
    )
    venue = pd.DataFrame(
        {
            "date": [date] * 4,
            "__pair__": pairs,
            "open": [100.0] * 4,
            "high": [101.0] * 4,
            "low": [99.0] * 4,
            "close": [100.0] * 4,
            "volume": [1000.0] * 4,
        }
    )
    cfg = rp.RiskConfig(
        gross_cap=1.0,
        net_cap=1.0,
        single_pair_cap=1.0,
        risk_per_trade=0.02,
        top_k=1,
        min_pairs_for_top_k=4,
        low_pair_top_k=1,
        side_mode="short",
        min_abs_score_z=0.0,
        exclude_pairs=("P0/USDT",),
    )

    signals, _ = rp.build_rank_signals(score_frame, venue, cfg)

    assert signals.loc[signals["pair"] == "P0/USDT", "rp_side"].iloc[0] == 0


def test_short_exit_momentum_closes_held_position_between_rebalances() -> None:
    pairs = [f"P{i}/USDT" for i in range(4)]
    dates = pd.date_range("2026-01-01", periods=30, freq="1h", tz="UTC")
    score_rows = []
    venue_rows = []
    for i, date in enumerate(dates):
        for j, pair in enumerate(pairs):
            score_rows.append({
                "date": date,
                "__pair__": pair,
                "rp_score": float(j),
                "rp_score_z": float(j - 1.5),
            })
            close = 100.0
            if pair == "P0/USDT" and i >= 25:
                close = 120.0
            venue_rows.append({
                "date": date,
                "__pair__": pair,
                "open": close,
                "high": close * 1.001,
                "low": close * 0.999,
                "close": close,
                "volume": 1000.0,
            })
    cfg = rp.RiskConfig(
        gross_cap=1.0,
        net_cap=1.0,
        single_pair_cap=1.0,
        risk_per_trade=0.02,
        top_k=1,
        min_pairs_for_top_k=4,
        low_pair_top_k=1,
        side_mode="short",
        min_abs_score_z=0.0,
        rebalance_hours=8,
        short_exit_mom_24h=0.05,
    )

    signals, _ = rp.build_rank_signals(pd.DataFrame(score_rows), pd.DataFrame(venue_rows), cfg)
    entry = signals.loc[(signals["date"] == dates[24]) & (signals["pair"] == "P0/USDT")].iloc[0]
    exited = signals.loc[(signals["date"] == dates[25]) & (signals["pair"] == "P0/USDT")].iloc[0]

    assert entry["rp_side"] == -1
    assert exited["rp_side"] == 0
    assert bool(exited["rp_exit_short"])


def test_liquidation_reducer_lowers_ten_x_when_stop_is_too_wide() -> None:
    lev, dist, rejected = rp.reduce_leverage_for_liq(10, 0.03, ten_x_requested=True)

    assert not rejected
    assert lev < 10
    assert dist >= 3 * 0.03


def test_pair_edge_dynamic_leverage_caps_misaligned_pair() -> None:
    cfg = rp.RiskConfig(
        edge_mode="rolling_ic",
        leverage_cap=10.0,
        pair_edge_leverage=True,
        pair_edge_deadband=0.01,
        pair_edge_strong_ic=0.05,
        pair_edge_very_strong_ic=0.10,
        pair_edge_weak_cap=2.0,
    )

    lev_aligned, _, _ = rp.choose_dynamic_leverage(
        side_rank=1,
        score_z=2.2,
        atr_pct=0.02,
        volume_ratio=1.2,
        stop_pct=0.01,
        pair_edge_ic=0.12,
        edge_sign=1.0,
        cfg=cfg,
    )
    lev_misaligned, _, _ = rp.choose_dynamic_leverage(
        side_rank=1,
        score_z=2.2,
        atr_pct=0.02,
        volume_ratio=1.2,
        stop_pct=0.01,
        pair_edge_ic=-0.12,
        edge_sign=1.0,
        cfg=cfg,
    )

    assert lev_aligned > lev_misaligned
    assert lev_misaligned <= cfg.pair_edge_weak_cap + 1e-9


def test_hq_regime_gate_blocks_entries_when_edge_threshold_not_met() -> None:
    pairs = [f"P{i}/USDT" for i in range(4)]
    dates = pd.date_range("2026-01-01", periods=5, freq="1h", tz="UTC")
    score_rows = []
    venue_rows = []
    for i, date in enumerate(dates):
        for j, pair in enumerate(pairs):
            score_rows.append({
                "date": date,
                "__pair__": pair,
                "rp_score": float(j),
                "rp_score_z": float(j - 1.5),
            })
            close = 100.0 + float(i) + float(j) * 0.01
            venue_rows.append({
                "date": date,
                "__pair__": pair,
                "open": close,
                "high": close * 1.001,
                "low": close * 0.999,
                "close": close,
                "volume": 1000.0,
            })
    cfg = rp.RiskConfig(
        gross_cap=1.0,
        net_cap=1.0,
        single_pair_cap=1.0,
        risk_per_trade=0.02,
        top_k=1,
        min_pairs_for_top_k=4,
        low_pair_top_k=1,
        side_mode="short",
        min_abs_score_z=0.0,
        rebalance_hours=1,
        edge_mode="rolling_ic",
        edge_lookback_hours=4,
        edge_min_periods=1,
        edge_deadband=0.0,
        regime_mode="hq",
        regime_min_edge_ic=2.0,
    )

    signals, diag = rp.build_rank_signals(pd.DataFrame(score_rows), pd.DataFrame(venue_rows), cfg)

    assert int((signals["rp_side"] != 0).sum()) == 0
    assert int((signals["rp_kill_mode"] == "regime_hq").sum()) > 0
    assert int(diag["regime_blocks"]) > 0


def test_account_risk_controller_modes_are_reproducible() -> None:
    cfg = rp.RiskConfig(daily_loss_limit=0.04, weekly_loss_limit=0.08, drawdown_safe_mode=0.12, consecutive_loss_limit=2)
    controller = rp.AccountRiskController(cfg)

    assert controller.update("2026-01-01T00:00:00Z", 1.0).mode == "normal"
    assert controller.update("2026-01-01T01:00:00Z", 0.95).mode == "daily_halt"

    controller = rp.AccountRiskController(cfg)
    assert controller.update("2026-01-01T00:00:00Z", 1.0, realized_pnl=-0.01).mode == "normal"
    assert controller.update("2026-01-01T01:00:00Z", 0.995, realized_pnl=-0.01).mode == "loss_pause"

    controller = rp.AccountRiskController(cfg)
    assert controller.update("2026-01-01T00:00:00Z", 1.0).mode == "normal"
    assert controller.update("2026-01-02T00:00:00Z", 0.90).mode == "weekly_safe"

    controller = rp.AccountRiskController(cfg)
    assert controller.update("2026-01-01T00:00:00Z", 1.0).mode == "normal"
    assert controller.update("2026-01-08T00:00:00Z", 0.87).mode == "drawdown_safe"

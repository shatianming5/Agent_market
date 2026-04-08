from __future__ import annotations

import json

import pandas as pd


def _frame(dates: list[str], price_a: list[float], price_b: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.to_datetime(dates, utc=True),
        "price_a": price_a,
        "vol_a": [1.0] * len(dates),
        "price_b": price_b,
        "vol_b": [1.0] * len(dates),
    })


def test_pairs_paper_cycle_initializes_without_backfill(tmp_path) -> None:
    from workspace.paper_cycle import PairsPaperCycle

    cycle = PairsPaperCycle(paper_dir=tmp_path, min_refresh_interval_sec=999999)
    frame = _frame(
        ["2026-04-08T00:00:00Z", "2026-04-08T01:00:00Z", "2026-04-08T02:00:00Z"],
        [100.0, 101.0, 102.0],
        [200.0, 201.0, 202.0],
    )
    cycle.sync_market_data = lambda *args, **kwargs: {"ok": True}  # type: ignore[method-assign]
    cycle._load_merged_frame = lambda *args, **kwargs: frame.copy()  # type: ignore[method-assign]

    result = cycle.run_pairs_strategy("pairs_A_B", {"pair_a": "A/USDT", "pair_b": "B/USDT"})

    assert result["ok"] is True
    assert result["initialized"] is True
    assert result["new_bars"] == 0
    state = json.loads((tmp_path / "pairs_A_B.json").read_text(encoding="utf-8"))
    assert state["last_processed_at"] == "2026-04-08T02:00:00+00:00"
    assert state["daily_equity"] == {}


def test_pairs_paper_cycle_processes_only_new_bars(tmp_path) -> None:
    from workspace.paper_cycle import PairsPaperCycle

    cycle = PairsPaperCycle(paper_dir=tmp_path, min_refresh_interval_sec=999999, position_pct=0.5)
    initial = _frame(
        [
            "2026-04-08T00:00:00Z",
            "2026-04-08T01:00:00Z",
            "2026-04-08T02:00:00Z",
            "2026-04-08T03:00:00Z",
            "2026-04-08T04:00:00Z",
        ],
        [100.0, 101.0, 102.0, 103.0, 104.0],
        [200.0, 201.0, 202.0, 203.0, 204.0],
    )
    expanded = _frame(
        [
            "2026-04-08T00:00:00Z",
            "2026-04-08T01:00:00Z",
            "2026-04-08T02:00:00Z",
            "2026-04-08T03:00:00Z",
            "2026-04-08T04:00:00Z",
            "2026-04-08T05:00:00Z",
            "2026-04-09T05:00:00Z",
        ],
        [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 108.0],
        [200.0, 201.0, 202.0, 203.0, 204.0, 205.0, 206.0],
    )
    cycle.sync_market_data = lambda *args, **kwargs: {"ok": True}  # type: ignore[method-assign]

    frames = [initial.copy(), expanded.copy()]
    cycle._load_merged_frame = lambda *args, **kwargs: frames.pop(0)  # type: ignore[method-assign]

    config = {
        "pair_a": "A/USDT",
        "pair_b": "B/USDT",
        "params": {"lookback": 0, "entry_z": 2.0, "exit_z": 0.5},
    }
    cycle.run_pairs_strategy("pairs_A_B", config)

    signals = iter(["enter_long_spread", "hold"])
    cycle._signal_for_history = lambda *args, **kwargs: next(signals)  # type: ignore[method-assign]
    result = cycle.run_pairs_strategy("pairs_A_B", config)

    assert result["ok"] is True
    assert result["initialized"] is False
    assert result["new_bars"] == 2
    assert result["orders_added"] == 1
    assert result["daily_equity"]["2026-04-08"] > 0
    assert result["daily_equity"]["2026-04-09"] >= result["daily_equity"]["2026-04-08"]

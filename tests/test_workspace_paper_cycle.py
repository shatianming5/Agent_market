from __future__ import annotations

import json
from datetime import datetime, timezone

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
    assert result["paper_days"] == 0
    assert result["cumulative_return_pct"] == 0.0
    assert result["latest_day_return_pct"] is None
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

    cycle._signals_for_frame = lambda *args, **kwargs: pd.DataFrame({  # type: ignore[method-assign]
        "date": pd.to_datetime(
            [
                "2026-04-08T00:00:00Z",
                "2026-04-08T01:00:00Z",
                "2026-04-08T02:00:00Z",
                "2026-04-08T03:00:00Z",
                "2026-04-08T04:00:00Z",
                "2026-04-08T05:00:00Z",
                "2026-04-09T05:00:00Z",
            ],
            utc=True,
        ),
        "signal": ["hold", "hold", "hold", "hold", "hold", "enter_long_spread", "hold"],
    })  # type: ignore[method-assign]
    result = cycle.run_pairs_strategy("pairs_A_B", config)

    assert result["ok"] is True
    assert result["initialized"] is False
    assert result["new_bars"] == 2
    assert result["orders_added"] == 1
    assert result["paper_days"] == 2
    assert result["cumulative_return_pct"] > 0
    assert result["latest_day_return_pct"] is not None
    assert result["daily_equity"]["2026-04-08"] > 0
    assert result["daily_equity"]["2026-04-09"] >= result["daily_equity"]["2026-04-08"]


def test_sync_pair_data_fetches_when_candles_stale_despite_fresh_cache(tmp_path) -> None:
    from workspace.paper_cycle import PairsPaperCycle, _closed_candle_start, _timeframe_delta

    cycle = PairsPaperCycle(min_refresh_interval_sec=999999, paper_dir=tmp_path / "paper")
    data_path = tmp_path / "user_data" / "data" / "gate" / "A_USDT-1h.feather"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    stale = pd.DataFrame({
        "date": pd.to_datetime(["2025-04-14T08:00:00Z", "2025-04-14T09:00:00Z"], utc=True),
        "open": [1.0, 1.0],
        "high": [1.0, 1.0],
        "low": [1.0, 1.0],
        "close": [1.0, 1.0],
        "volume": [1.0, 1.0],
    })
    stale.to_feather(data_path)

    cycle._data_path = lambda pair: data_path  # type: ignore[method-assign]
    captured: dict[str, object] = {}

    def _fake_fetch(*, pair: str, since_ms: int | None, max_batches: int) -> pd.DataFrame:
        captured["pair"] = pair
        captured["since_ms"] = since_ms
        captured["max_batches"] = max_batches
        latest = _closed_candle_start(datetime.now(timezone.utc), "1h") - _timeframe_delta("1h")
        return pd.DataFrame({
            "date": pd.to_datetime([latest - _timeframe_delta("1h"), latest], utc=True),
            "open": [1.0, 1.0],
            "high": [1.0, 1.0],
            "low": [1.0, 1.0],
            "close": [1.0, 1.0],
            "volume": [1.0, 1.0],
        })

    cycle._fetch_pair_data = _fake_fetch  # type: ignore[method-assign]

    result = cycle._sync_pair_data("A/USDT")

    assert result["ok"] is True
    assert captured["pair"] == "A/USDT"
    assert int(captured["max_batches"]) > 3
    assert result["stale_bars_before"] > 0
    assert result["latest_candle"] is not None


def test_required_batches_scales_with_stale_bar_gap(tmp_path) -> None:
    from workspace.paper_cycle import PairsPaperCycle

    cycle = PairsPaperCycle(paper_dir=tmp_path / "paper")

    assert cycle._required_batches(0) == 3
    assert cycle._required_batches(500) == 3
    assert cycle._required_batches(3000) == 4
    assert cycle._required_batches(9000) > 3


def test_missing_state_bootstraps_from_paper_start(tmp_path) -> None:
    from workspace.paper_cycle import PairsPaperCycle

    cycle = PairsPaperCycle(paper_dir=tmp_path, min_refresh_interval_sec=999999, position_pct=0.5)
    frame = _frame(
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
    cycle._load_merged_frame = lambda *args, **kwargs: frame.copy()  # type: ignore[method-assign]
    cycle._signals_for_frame = lambda *args, **kwargs: pd.DataFrame({  # type: ignore[method-assign]
        "date": pd.to_datetime(
            [
                "2026-04-08T00:00:00Z",
                "2026-04-08T01:00:00Z",
                "2026-04-08T02:00:00Z",
                "2026-04-08T03:00:00Z",
                "2026-04-08T04:00:00Z",
                "2026-04-08T05:00:00Z",
                "2026-04-09T05:00:00Z",
            ],
            utc=True,
        ),
        "signal": ["hold", "hold", "hold", "hold", "hold", "enter_long_spread", "hold"],
    })  # type: ignore[method-assign]

    result = cycle.run_pairs_strategy(
        "pairs_A_B",
        {
            "pair_a": "A/USDT",
            "pair_b": "B/USDT",
            "paper_started_at": "2026-04-08T04:00:00+00:00",
            "params": {"lookback": 0, "entry_z": 2.0, "exit_z": 0.5},
        },
    )

    assert result["ok"] is True
    assert result["initialized"] is False
    assert result["bootstrapped"] is True
    assert result["new_bars"] == 3
    assert result["paper_days"] == 2
    assert result["cumulative_return_pct"] > 0

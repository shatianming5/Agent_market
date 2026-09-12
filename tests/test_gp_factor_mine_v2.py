from __future__ import annotations

import numpy as np
import pandas as pd

from scripts import gp_factor_mine_v2 as miner


def test_build_big_purges_forward_labels_at_each_window_boundary(tmp_path, monkeypatch):
    monkeypatch.setattr(miner, "ROOT", tmp_path)
    monkeypatch.setattr(miner, "load_expression_file", lambda path: [])
    monkeypatch.setattr(miner, "apply_configured_features", lambda frame, config: frame)
    monkeypatch.setattr(miner, "apply_expressions", lambda frame, specs, **kwargs: (frame, []))
    config = tmp_path / "features.json"
    config.write_text("{}", encoding="utf-8")
    ends = [end for _, end in miner.TRAIN_WINDOWS] + ["2025-12-01"]
    dates = pd.DatetimeIndex(
        [
            date
            for end in ends
            for date in pd.date_range(
                pd.Timestamp(end, tz="UTC") - pd.Timedelta(days=10),
                pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1),
                freq="h",
            )
        ]
    )
    frame = pd.DataFrame({"date": dates, "close": np.arange(len(dates)) + 100.0})
    data_dir = tmp_path / "user_data/data/kucoin"
    data_dir.mkdir(parents=True)
    frame.to_feather(data_dir / "BTC_USDT-1h.feather")

    big, train_masks, val_mask, columns = miner.build_big(
        ["BTC/USDT"], config, tmp_path / "expressions.json",
        "2024-01-01", "2025-12-02", label_period=12,
    )

    assert len(train_masks) == 3
    for mask, end in zip([*train_masks, val_mask], ends):
        cutoff = pd.Timestamp(end, tz="UTC")
        boundary = (big["date"] >= cutoff - pd.Timedelta(hours=12)) & (big["date"] < cutoff)
        assert boundary.sum() == 12
        assert not mask[boundary].any()
        assert mask.sum() > 200
        assert big.loc[mask, "__label_end__"].lt(cutoff).all()
    assert "__label_end__" not in columns

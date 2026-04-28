from __future__ import annotations

import pandas as pd


def align_to_ohlcv(
    ohlcv_dates: pd.Series,
    external_df: pd.DataFrame,
    date_col: str = "timestamp",
    value_cols: list[str] | None = None,
    shift_periods: int = 1,
) -> pd.DataFrame:
    """
    将外部数据按 backward-looking merge_asof 对齐到 OHLCV 时间轴，
    然后整体向右 shift_periods，确保 t 时刻只能看到 t 之前的外部数据。

    Args:
        ohlcv_dates:   OHLCV 的 date 列（UTC Timestamp Series），可乱序
        external_df:   外部数据，含 date_col 及若干指标列
        date_col:      外部数据的时间列名
        value_cols:    要保留的指标列，None 表示全取（除 date_col 外）
        shift_periods: 向右偏移几个 bar（默认 1，防止当 bar 泄露）

    Returns:
        与 ohlcv_dates 等长、index 对齐的 DataFrame，列为外部特征
    """
    if value_cols is None:
        value_cols = [c for c in external_df.columns if c != date_col]

    ext = external_df[[date_col] + value_cols].copy()
    ext[date_col] = pd.to_datetime(ext[date_col], utc=True)
    ext = ext.sort_values(date_col).reset_index(drop=True)

    ohlcv_sorted = pd.Series(
        pd.to_datetime(ohlcv_dates.values, utc=True), name="date"
    ).sort_values().reset_index(drop=True)

    # backward merge_asof: 每个 OHLCV bar 取最近一条 <= bar 时间的外部数据
    merged = pd.merge_asof(
        ohlcv_sorted.to_frame(),
        ext.rename(columns={date_col: "date"}),
        on="date",
        direction="backward",
    )
    merged = merged.set_index("date")

    # 还原原始 OHLCV 顺序
    result = merged.reindex(pd.to_datetime(ohlcv_dates.values, utc=True))
    result.index = ohlcv_dates.index

    # 向右 shift，使 t 时刻的特征值来自 t-1（或更早），彻底消除前向偏差
    if shift_periods > 0:
        result = result.shift(shift_periods)

    return result[value_cols]

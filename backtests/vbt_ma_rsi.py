#!/usr/bin/env python
"""Run a vectorbt moving-average + RSI strategy backtest on cleaned OHLCV data."""
import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import pandas_ta as ta
import vectorbt as vbt
import yaml

CLEAN_DIR = Path("data/clean")
RESULTS_DIR = Path("backtests/results")


@dataclass
class SeriesSpec:
    exchange: str
    symbol: str
    timeframe: str

    @property
    def symbol_slug(self) -> str:
        return self.symbol.replace("/", "-")


TF_FREQ_MAP = {
    "1m": "1min",
    "3m": "3min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "6h": "6h",
    "8h": "8h",
    "12h": "12h",
    "1d": "1D",
}


DEFAULT_PARAMS = {
    "fast_windows": [10, 20],
    "slow_windows": [50, 100],
    "rsi_windows": [14, 21],
    "rsi_entries": [25, 30],
    "rsi_exits": [65, 70],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vectorbt MA+RSI backtest")
    parser.add_argument("--conf", type=Path, required=True)
    parser.add_argument("--fees", type=float, default=0.001, help="Trading fee per trade (e.g. 0.001 for 0.1%)")
    parser.add_argument("--slippage", type=float, default=0.0005, help="Slippage ratio per trade")
    parser.add_argument("--stop-loss", type=float, default=0.02, help="Stop loss as ratio (e.g. 0.02 for 2%)")
    parser.add_argument("--take-profit", type=float, default=0.04, help="Take profit as ratio")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Fraction of data used for training")
    parser.add_argument("--cash", type=float, default=10_000.0, help="Initial cash for portfolio")
    return parser.parse_args()


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_close_series(spec: SeriesSpec) -> pd.Series:
    base = CLEAN_DIR / f"exchange={spec.exchange}" / f"symbol={spec.symbol_slug}" / f"tf={spec.timeframe}"
    if not base.exists():
        raise FileNotFoundError(f"Clean data not found for {spec.exchange}/{spec.symbol}/{spec.timeframe}")
    frames: List[pd.DataFrame] = []
    for parquet in sorted(base.rglob("*.parquet")):
        df = pd.read_parquet(parquet)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No parquet files in {base}")
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    combined["datetime"] = pd.to_datetime(combined["timestamp"], unit="ms", utc=True)
    combined = combined.set_index("datetime")
    close = combined["close"].astype(float)
    close.name = spec.symbol
    return close


def ensure_freq(timeframe: str) -> str:
    if timeframe not in TF_FREQ_MAP:
        raise ValueError(f"Timeframe {timeframe} not mapped to pandas freq")
    return TF_FREQ_MAP[timeframe]


def build_signals(close: pd.Series, fast: int, slow: int, rsi_window: int, rsi_entry: float, rsi_exit: float) -> Tuple[pd.Series, pd.Series]:
    fast_ma = close.rolling(fast).mean()
    slow_ma = close.rolling(slow).mean()
    rsi = ta.rsi(close, length=rsi_window)
    entries = (fast_ma > slow_ma) & (rsi <= rsi_entry)
    exits = (fast_ma < slow_ma) | (rsi >= rsi_exit)
    entries = entries.fillna(False)
    exits = exits.fillna(False)
    return entries, exits


def _to_scalar(value):
    if value is None:
        return None
    if hasattr(value, 'item'):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, 'values'):
        arr = value.values if hasattr(value.values, '__len__') else value
        if getattr(arr, 'size', 0) == 0:
            return None
        return arr.flat[0]
    return value


def compute_metrics(pf: vbt.Portfolio) -> Dict[str, float]:
    trades = pf.trades
    trade_count_raw = _to_scalar(trades.count()) if trades is not None else 0
    if trade_count_raw is None:
        trade_count = 0
    elif isinstance(trade_count_raw, float) and math.isnan(trade_count_raw):
        trade_count = 0
    else:
        trade_count = int(trade_count_raw)

    total_return = _to_scalar(pf.total_return())
    annualized_return = _to_scalar(pf.annualized_return()) if hasattr(pf, 'annualized_return') else None
    sharpe_raw = _to_scalar(pf.sharpe_ratio())
    max_dd = _to_scalar(pf.max_drawdown())
    win_rate_raw = _to_scalar(trades.win_rate()) if trades is not None and trade_count > 0 else None

    if trade_count == 0:
        sharpe = float('nan')
        win_rate = float('nan')
    else:
        if sharpe_raw is None or (isinstance(sharpe_raw, float) and not math.isfinite(sharpe_raw)):
            sharpe = float('nan')
        else:
            sharpe = float(sharpe_raw)
        if win_rate_raw is None or (isinstance(win_rate_raw, float) and not math.isfinite(win_rate_raw)):
            win_rate = float('nan')
        else:
            win_rate = float(win_rate_raw * 100)

    metrics = {
        'total_return_pct': float(total_return * 100) if total_return is not None else float('nan'),
        'annualized_return_pct': float(annualized_return * 100) if annualized_return is not None else float('nan'),
        'sharpe': sharpe,
        'max_drawdown_pct': float(abs(max_dd) * 100) if max_dd is not None else float('nan'),
        'trades': trade_count,
        'win_rate_pct': win_rate,
    }
    return metrics


def run_backtest(close: pd.Series, entries: pd.Series, exits: pd.Series, fees: float, slippage: float,
                 stop_loss: float, take_profit: float, cash: float, freq: str) -> vbt.Portfolio:
    return vbt.Portfolio.from_signals(
        close,
        entries,
        exits,
        init_cash=cash,
        fees=fees,
        slippage=slippage,
        sl_stop=stop_loss,
        tp_stop=take_profit,
        freq=freq,
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.conf)

    exchange = cfg["exchange"]
    symbols: Iterable[str] = cfg["symbols"]
    timeframes: Iterable[str] = cfg["timeframes"]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for symbol in symbols:
        for timeframe in timeframes:
            spec = SeriesSpec(exchange=exchange, symbol=symbol, timeframe=timeframe)
            close = load_close_series(spec)
            freq = ensure_freq(timeframe)
            split_idx = int(len(close) * args.train_ratio)
            close_train = close.iloc[:split_idx]
            close_test = close.iloc[split_idx:]

            rows: List[Dict] = []
            for fast in DEFAULT_PARAMS["fast_windows"]:
                for slow in DEFAULT_PARAMS["slow_windows"]:
                    if fast >= slow:
                        continue
                    for rsi_window in DEFAULT_PARAMS["rsi_windows"]:
                        for rsi_entry in DEFAULT_PARAMS["rsi_entries"]:
                            for rsi_exit in DEFAULT_PARAMS["rsi_exits"]:
                                entries, exits = build_signals(close, fast, slow, rsi_window, rsi_entry, rsi_exit)
                                pf_full = run_backtest(close, entries, exits, args.fees, args.slippage, args.stop_loss, args.take_profit, args.cash, freq)
                                pf_train = run_backtest(close_train, entries.iloc[:split_idx], exits.iloc[:split_idx], args.fees, args.slippage, args.stop_loss, args.take_profit, args.cash, freq)
                                pf_test = run_backtest(close_test, entries.iloc[split_idx:], exits.iloc[split_idx:], args.fees, args.slippage, args.stop_loss, args.take_profit, args.cash, freq)

                                metrics_full = compute_metrics(pf_full)
                                metrics_train = compute_metrics(pf_train)
                                metrics_test = compute_metrics(pf_test)

                                if (
                                    metrics_full['trades'] == 0
                                    or metrics_train['trades'] == 0
                                    or metrics_test['trades'] == 0
                                ):
                                    continue

                                row = {
                                    "symbol": symbol,
                                    "timeframe": timeframe,
                                    "fast": fast,
                                    "slow": slow,
                                    "rsi_window": rsi_window,
                                    "rsi_entry": rsi_entry,
                                    "rsi_exit": rsi_exit,
                                }
                                row.update({f"full_{k}": v for k, v in metrics_full.items()})
                                row.update({f"train_{k}": v for k, v in metrics_train.items()})
                                row.update({f"test_{k}": v for k, v in metrics_test.items()})
                                rows.append(row)

            result_df = pd.DataFrame(rows)
            result_path = RESULTS_DIR / f"{spec.symbol_slug}_{timeframe}_ma_rsi.parquet"
            result_df.to_parquet(result_path, index=False)
            top = result_df.sort_values("test_total_return_pct", ascending=False).head(5)
            print(f"Top results for {symbol} {timeframe} (sorted by test total return):")
            print(top[[
                "fast",
                "slow",
                "rsi_window",
                "rsi_entry",
                "rsi_exit",
                "full_total_return_pct",
                "test_total_return_pct",
                "test_sharpe",
                "test_max_drawdown_pct",
            ]])
            print(f"Saved: {result_path}\n")


if __name__ == "__main__":
    main()
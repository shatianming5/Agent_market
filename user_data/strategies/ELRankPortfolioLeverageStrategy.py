from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.strategy import IStrategy, stoploss_from_open


def _inject_project_paths() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "src" / "agent_market").exists():
            sys.path.insert(0, str(parent / "src"))
            sys.path.insert(0, str(parent))
            return parent
    root = here.parents[2]
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root))
    return root


PROJECT_ROOT = _inject_project_paths()


def _paths():
    from agent_market import paths as _p
    return _p


def _pair_token(pair: str) -> str:
    raw = str(pair).split(":")[0].replace("/", "_")
    if raw.endswith("_USDT"):
        return f"{raw}_USDT"
    return raw


class ELRankPortfolioLeverageStrategy(IStrategy):
    """Signal-file driven rank portfolio strategy for OKX isolated futures."""

    timeframe = (os.environ.get("RP_TIMEFRAME") or "1h").strip() or "1h"
    can_short = True
    minimal_roi = {"0": 0.50, "2880": -1}
    stoploss = -0.30
    use_custom_stoploss = True
    use_exit_signal = True
    process_only_new_candles = True
    startup_candle_count: int = 1

    _signals_by_pair: Dict[str, DataFrame] = {}
    _signal_dir: Optional[Path] = None

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        raw = os.environ.get(name)
        if raw in (None, ""):
            return float(default)
        try:
            return float(raw)
        except Exception:
            return float(default)

    def _resolve_signal_dir(self) -> Path:
        if self._signal_dir is not None:
            return self._signal_dir
        env_dir = os.environ.get("RP_SIGNAL_DIR")
        if env_dir:
            path = _paths().resolve_repo_path(env_dir)
            self._signal_dir = path
            return path
        tag = os.environ.get("RP_TAG", "gpt54_purealpha_v2_full1000_fix1")
        path = _paths().artifacts_root() / "rank_portfolio" / tag / "signals"
        if path.exists():
            self._signal_dir = path
            return path
        root = _paths().artifacts_root() / "rank_portfolio"
        candidates = sorted(root.glob("*/signals"), key=lambda p: p.stat().st_mtime, reverse=True) if root.exists() else []
        self._signal_dir = candidates[0] if candidates else path
        return self._signal_dir

    def _load_pair_signals(self, pair: str) -> DataFrame:
        if pair in self._signals_by_pair:
            return self._signals_by_pair[pair]

        signal_dir = self._resolve_signal_dir()
        token = _pair_token(pair)
        candidates = [
            signal_dir / f"{token}.feather",
            signal_dir / f"{token.removesuffix('_USDT')}.feather",
            signal_dir / "all.feather",
        ]
        sig = pd.DataFrame()
        for path in candidates:
            if not path.exists():
                continue
            loaded = pd.read_feather(path)
            loaded["date"] = pd.to_datetime(loaded["date"], utc=True)
            if path.name == "all.feather" and "pair" in loaded.columns:
                base_pair = str(pair).split(":")[0]
                loaded = loaded.loc[loaded["pair"].astype(str) == base_pair]
            sig = loaded
            break
        if sig.empty:
            sig = pd.DataFrame(columns=[
                "date", "rp_score", "rp_rank", "rp_side", "rp_target_weight",
                "rp_leverage", "rp_stop_pct", "rp_kill_mode", "rp_rebalance",
                "rp_exit_long", "rp_exit_short",
            ])
        keep = [
            "date", "rp_score", "rp_score_z", "rp_rank", "rp_side",
            "rp_target_weight", "rp_leverage", "rp_stop_pct", "rp_kill_mode",
            "rp_rebalance", "rp_exit_long", "rp_exit_short",
        ]
        keep = [col for col in keep if col in sig.columns]
        sig = sig[keep].drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
        self._signals_by_pair[pair] = sig
        return sig

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = str(metadata.get("pair") or "") if isinstance(metadata, dict) else ""
        df = dataframe.copy()
        df["date"] = pd.to_datetime(df["date"], utc=True)
        sig = self._load_pair_signals(pair)
        if not sig.empty:
            df = df.merge(sig, on="date", how="left")
        defaults: Dict[str, Any] = {
            "rp_score": 0.0,
            "rp_score_z": 0.0,
            "rp_rank": 0,
            "rp_side": 0,
            "rp_target_weight": 0.0,
            "rp_leverage": 1.0,
            "rp_stop_pct": 0.10,
            "rp_kill_mode": "normal",
            "rp_rebalance": False,
            "rp_exit_long": False,
            "rp_exit_short": False,
        }
        for col, value in defaults.items():
            if col not in df.columns:
                df[col] = value
            elif col == "rp_kill_mode":
                df[col] = df[col].fillna(str(value)).astype(str)
            elif col in {"rp_rebalance", "rp_exit_long", "rp_exit_short"}:
                default_bool = bool(value)
                df[col] = df[col].map(lambda x: bool(x) if pd.notna(x) else default_bool)
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(float(value))
        return df

    def _current_signal_row(self, pair: str, current_time: Any) -> Optional[pd.Series]:
        try:
            df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        except Exception:
            return None
        if df is None or df.empty:
            return None
        ts = pd.Timestamp(current_time)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        dates = pd.to_datetime(df["date"], utc=True)
        rows = df.loc[dates <= ts]
        if rows.empty:
            return None
        return rows.iloc[-1]

    def leverage(
        self,
        pair: str,
        current_time: Any,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs: Any,
    ) -> float:
        row = self._current_signal_row(pair, current_time)
        lev = float(proposed_leverage or 1.0)
        if row is not None:
            lev = float(row.get("rp_leverage", lev) or lev)
        env_cap = self._env_float("RP_MAX_LEVERAGE", float(max_leverage))
        return max(1.0, min(float(lev), float(max_leverage), env_cap))

    def custom_stake_amount(
        self,
        pair: str,
        current_time: Any,
        current_rate: float,
        proposed_stake: float,
        min_stake: Optional[float],
        max_stake: float,
        leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs: Any,
    ) -> float:
        row = self._current_signal_row(pair, current_time)
        if row is None:
            return proposed_stake
        target_weight = abs(float(row.get("rp_target_weight", 0.0) or 0.0))
        lev = max(1.0, float(leverage or row.get("rp_leverage", 1.0) or 1.0))
        try:
            equity = float(self.wallets.get_total_stake_amount())
        except Exception:
            equity = float(self.config.get("dry_run_wallet", max_stake) if hasattr(self, "config") else max_stake)
        stake = equity * target_weight / lev
        if min_stake is not None:
            stake = max(stake, float(min_stake))
        return max(0.0, min(float(stake), float(max_stake)))

    def custom_stoploss(
        self,
        pair: str,
        trade: Any,
        current_time: Any,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs: Any,
    ) -> float:
        row = self._current_signal_row(pair, current_time)
        stop = float(row.get("rp_stop_pct", 0.10) if row is not None else 0.10)
        stop = max(0.005, min(0.30, stop))
        lev = max(1.0, float(getattr(trade, "leverage", 1.0) or 1.0))
        # Freqtrade custom_stoploss is relative to current_rate and, in
        # futures mode, represents leveraged trade risk. Convert the research
        # engine's entry-relative price stop to a fixed open-relative stop so
        # short trades do not trail down from each candle's low in backtests.
        return stoploss_from_open(
            -min(0.99, stop * lev),
            current_profit,
            is_short=bool(getattr(trade, "is_short", False)),
            leverage=lev,
        )

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        active = (
            (df["volume"] > 0)
            & (df["rp_kill_mode"].astype(str) != "daily_halt")
            & df["rp_rebalance"].astype(bool)
        )
        df.loc[active & (df["rp_side"].astype(float) > 0), ["enter_long", "enter_tag"]] = (1, "rp_long")
        df.loc[active & (df["rp_side"].astype(float) < 0), ["enter_short", "enter_tag"]] = (1, "rp_short")
        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        active = df["volume"] > 0
        side = df["rp_side"].astype(float)
        exit_long = df["rp_exit_long"].astype(bool) if "rp_exit_long" in df.columns else False
        exit_short = df["rp_exit_short"].astype(bool) if "rp_exit_short" in df.columns else False
        df.loc[active & ((side <= 0) | exit_long), "exit_long"] = 1
        df.loc[active & ((side >= 0) | exit_short), "exit_short"] = 1
        return df

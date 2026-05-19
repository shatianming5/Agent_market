"""Local QuantConnect/LEAN validation bridge for rank-portfolio artifacts."""
from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
import time
from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, fields
from datetime import timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from agent_market import paths as repo_paths

from .paths import FUNDING_DIR, OKX_FUTURES_DIR
from .rank_portfolio import FUTURES_VENUES, RISK_PROFILE_AGGRESSIVE, RiskConfig, run_research_backtest
from .timeframes import normalize_timeframe, timeframe_minutes


BRIDGE_VERSION = "lean-bridge-v1"
DEFAULT_THRESHOLDS = {
    "final_equity": 0.05,
    "max_drawdown": 0.10,
    "trades": 0.05,
    "orders": 0.05,
    "turnover": 0.10,
}
REQUIRED_SIGNAL_COLUMNS = ("date", "pair", "rp_target_weight")
REQUIRED_OHLCV_COLUMNS = ("date", "open", "high", "low", "close", "volume")


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"object is not JSON serializable: {type(value)!r}")


def _read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected object JSON artifact: {path}")
    return payload


def _read_json_any(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _resolve_existing_path(raw: Any, *, artifact_path: Path) -> Path:
    if raw in (None, ""):
        raise ValueError(f"missing path in artifact: {artifact_path}")
    path = Path(str(raw)).expanduser()
    if path.is_absolute():
        return path.resolve()
    local = (artifact_path.parent / path).resolve()
    if local.exists():
        return local
    return repo_paths.resolve_repo_path(path)


def normalize_pair(pair: Any) -> str:
    """Normalize OKX/Freqtrade-ish pair spellings to BASE/QUOTE."""
    raw = str(pair or "").strip().split(":", 1)[0]
    if not raw:
        return raw
    if "/" in raw:
        left, right = raw.split("/", 1)
        return f"{left.strip().upper()}/{right.strip().upper()}"
    parts = raw.replace("-", "_").split("_")
    if len(parts) >= 2:
        return f"{parts[0].strip().upper()}/{parts[1].strip().upper()}"
    return raw.upper()


def pair_file_token(pair: Any) -> str:
    base = normalize_pair(pair).replace("/", "_")
    if base.endswith("_USDT"):
        return f"{base}_USDT"
    return base


def lean_symbol(pair: Any) -> str:
    norm = normalize_pair(pair)
    return norm.replace("/", "").replace("-", "").replace("_", "").upper()


def futures_data_root(*, venue: str = "okx", data_root: Optional[str | Path] = None) -> Path:
    venue_s = str(venue or "okx").strip().lower()
    if venue_s not in FUTURES_VENUES:
        raise ValueError(f"LEAN bridge futures venue must be one of {sorted(FUTURES_VENUES)}, got {venue!r}")
    if data_root is not None:
        return Path(data_root)
    root = repo_paths.user_data_root() / "data" / venue_s / "futures"
    if venue_s == "okx" and not root.exists():
        root = OKX_FUTURES_DIR
    return root


def futures_ohlcv_path(
    pair: Any,
    timeframe: str,
    *,
    venue: str = "okx",
    data_root: Optional[str | Path] = None,
) -> Path:
    root = futures_data_root(venue=venue, data_root=data_root)
    return root / f"{pair_file_token(pair)}-{normalize_timeframe(timeframe)}-futures.feather"


def okx_futures_path(pair: Any, timeframe: str, *, data_root: Optional[str | Path] = None) -> Path:
    return futures_ohlcv_path(pair, timeframe, venue="okx", data_root=data_root)


def _artifact_signal_path(payload: Mapping[str, Any]) -> Any:
    signals = payload.get("signals")
    if isinstance(signals, Mapping):
        return signals.get("all") or signals.get("signals") or signals.get("path")
    return signals


def _risk_config_from_payload(payload: Mapping[str, Any], *, timeframe: str) -> RiskConfig:
    profile = str(
        payload.get("risk_profile")
        or (payload.get("risk_config") or {}).get("profile")
        or RISK_PROFILE_AGGRESSIVE
    )
    cfg = RiskConfig.from_profile(profile, timeframe=timeframe)
    risk_raw = payload.get("risk_config")
    if isinstance(risk_raw, Mapping):
        allowed = {f.name for f in fields(RiskConfig)}
        for key, value in risk_raw.items():
            if key not in allowed:
                continue
            if key == "exclude_pairs" and value is not None:
                value = tuple(str(v) for v in value)
            setattr(cfg, key, value)
        cfg.timeframe = normalize_timeframe(timeframe)
        return cfg

    kwargs: Dict[str, Any] = {}
    allowed_kwargs = {
        "gross_cap",
        "net_cap",
        "top_k",
        "single_pair_cap",
        "side_mode",
        "min_abs_score_z",
        "rebalance_hours",
        "rebalance_minutes",
        "risk_per_trade",
        "leverage_cap",
        "edge_mode",
        "edge_lookback_hours",
        "edge_min_periods",
        "edge_deadband",
        "pair_edge_leverage",
        "pair_edge_deadband",
        "pair_edge_strong_ic",
        "pair_edge_very_strong_ic",
        "pair_edge_weak_cap",
        "regime_mode",
        "regime_min_edge_ic",
        "regime_min_pair_edge_ic",
        "regime_min_pair_count",
        "regime_short_max_market_mom_24h",
        "regime_short_max_market_mom_72h",
        "regime_max_market_atr_pct",
        "short_max_mom_24h",
        "short_max_mom_72h",
        "long_min_mom_24h",
        "max_entry_atr_pct",
        "short_max_market_mom_24h",
        "short_max_market_mom_72h",
        "short_max_market_ma_gap",
        "short_exit_mom_24h",
        "short_exit_mom_72h",
        "short_exit_market_mom_24h",
        "short_exit_market_ma_gap",
        "exclude_pairs",
    }
    for key in allowed_kwargs:
        if key in payload and payload.get(key) is not None:
            kwargs[key] = payload.get(key)
    return RiskConfig.from_profile(profile, timeframe=timeframe, **kwargs)


def load_rank_artifact(rank_artifact: str | Path, *, timeframe: Optional[str] = None) -> Dict[str, Any]:
    """Load a rank_export.json or backtest.json artifact and resolve its signals."""
    artifact_path = repo_paths.resolve_repo_path(Path(rank_artifact)).resolve()
    if not artifact_path.exists():
        raise FileNotFoundError(f"rank artifact not found: {artifact_path}")
    payload = _read_json(artifact_path)
    artifact_tf = normalize_timeframe(
        payload.get("timeframe")
        or (payload.get("risk_config") or {}).get("timeframe")
        or timeframe
        or "1h"
    )
    if timeframe is not None:
        cli_tf = normalize_timeframe(timeframe)
        if cli_tf != artifact_tf:
            raise ValueError(f"CLI timeframe={cli_tf} does not match artifact timeframe={artifact_tf}")
    signals_raw = _artifact_signal_path(payload)
    if not signals_raw:
        fallback = artifact_path.parent / "signals" / "all.feather"
        if fallback.exists():
            signals_raw = fallback
    signals_path = _resolve_existing_path(signals_raw, artifact_path=artifact_path)
    if not signals_path.exists():
        raise FileNotFoundError(f"signals file not found: {signals_path}")
    venue = str(payload.get("venue") or "okx").strip().lower()
    if venue not in FUTURES_VENUES:
        raise ValueError(f"LEAN bridge local futures venue must be one of {sorted(FUTURES_VENUES)}, got {venue!r}")
    return {
        "artifact_path": artifact_path,
        "artifact_kind": artifact_path.name,
        "payload": payload,
        "signals_path": signals_path,
        "timeframe": artifact_tf,
        "timeframe_minutes": timeframe_minutes(artifact_tf),
        "venue": venue,
        "data_venue": str(payload.get("data_venue") or venue).strip().lower(),
        "risk_config": _risk_config_from_payload(payload, timeframe=artifact_tf),
        "tag": payload.get("tag") or artifact_path.parent.name,
    }


def load_signals(signals_path: str | Path) -> pd.DataFrame:
    path = Path(signals_path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_feather(path)
    if "pair" not in df.columns and "__pair__" in df.columns:
        df["pair"] = df["__pair__"]
    missing = [col for col in REQUIRED_SIGNAL_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"signals missing required columns {missing}: {path}")
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], utc=True)
    out["pair"] = out["pair"].map(normalize_pair)
    out["rp_target_weight"] = pd.to_numeric(out["rp_target_weight"], errors="coerce").fillna(0.0)
    defaults = {
        "rp_rebalance": True,
        "rp_exit_long": False,
        "rp_exit_short": False,
        "rp_liq_reject": False,
        "rp_kill_mode": "normal",
        "rp_leverage": 1.0,
        "rp_stop_pct": 0.0,
        "rp_liq_distance": np.nan,
        "rp_side": 0,
        "rp_rank": 0,
    }
    for col, default in defaults.items():
        if col not in out.columns:
            out[col] = default
    for col in ("rp_rebalance", "rp_exit_long", "rp_exit_short", "rp_liq_reject"):
        out[col] = out[col].fillna(False).astype(bool)
    out["rp_kill_mode"] = out["rp_kill_mode"].fillna("normal").astype(str)
    dupes = out.duplicated(subset=["date", "pair"], keep=False)
    if bool(dupes.any()):
        sample = out.loc[dupes, ["date", "pair"]].head(5).to_dict("records")
        raise ValueError(f"signals contain duplicate date/pair rows: {sample}")
    return out.sort_values(["date", "pair"]).reset_index(drop=True)


def used_signal_pairs(signals: pd.DataFrame) -> list[str]:
    mask = signals["rp_target_weight"].abs() > 1e-12
    for col in ("rp_exit_long", "rp_exit_short", "rp_liq_reject"):
        if col in signals.columns:
            mask = mask | signals[col].fillna(False).astype(bool)
    if "rp_kill_mode" in signals.columns:
        mask = mask | ~signals["rp_kill_mode"].fillna("normal").astype(str).str.lower().isin({"", "normal", "none"})
    pairs = sorted(signals.loc[mask, "pair"].map(normalize_pair).unique().tolist())
    return pairs or sorted(signals["pair"].map(normalize_pair).unique().tolist())


def normalize_signal_targets(signals: pd.DataFrame) -> pd.DataFrame:
    """Add LEAN-ready target weights with hold/exit/kill semantics applied."""
    out = signals.copy().sort_values(["pair", "date"]).reset_index(drop=True)
    out["lean_previous_target_weight"] = 0.0
    out["lean_target_weight"] = 0.0
    out["lean_target_delta"] = 0.0
    out["lean_force_flat"] = False
    out["lean_action"] = False
    for pair, idx in out.groupby("pair", sort=False).groups.items():
        previous = 0.0
        for row_idx in idx:
            row = out.loc[row_idx]
            previous_before = previous
            kill_mode = str(row.get("rp_kill_mode", "normal") or "normal").strip().lower()
            force_flat = (
                bool(row.get("rp_exit_long", False))
                or bool(row.get("rp_exit_short", False))
                or bool(row.get("rp_liq_reject", False))
                or kill_mode not in {"", "normal", "none"}
            )
            if force_flat:
                target = 0.0
            elif not bool(row.get("rp_rebalance", True)):
                target = previous
            else:
                target = float(row.get("rp_target_weight", 0.0) or 0.0)
            if not math.isfinite(target):
                target = 0.0
            delta = float(target) - float(previous_before)
            is_rebalance = bool(row.get("rp_rebalance", True))
            action = abs(delta) > 1e-12 or (force_flat and abs(previous_before) > 1e-9)
            out.at[row_idx, "lean_target_weight"] = float(target)
            out.at[row_idx, "lean_previous_target_weight"] = float(previous_before)
            out.at[row_idx, "lean_target_delta"] = float(delta)
            out.at[row_idx, "lean_force_flat"] = bool(force_flat)
            out.at[row_idx, "lean_action"] = bool(action)
            previous = float(target)
    return out.sort_values(["date", "pair"]).reset_index(drop=True)


def _read_ohlcv(path: Path, *, pair: str) -> pd.DataFrame:
    df = pd.read_feather(path)
    missing = [col for col in REQUIRED_OHLCV_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"OHLCV file for {pair} missing columns {missing}: {path}")
    out = df.loc[:, list(REQUIRED_OHLCV_COLUMNS)].copy()
    out["date"] = pd.to_datetime(out["date"], utc=True)
    for col in ("open", "high", "low", "close", "volume"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["date", "open", "high", "low", "close"])
    return out.sort_values("date").reset_index(drop=True)


def preflight_coverage(
    signals: pd.DataFrame,
    *,
    timeframe: str,
    venue: str = "okx",
    data_root: Optional[str | Path] = None,
    pairs: Optional[Sequence[str]] = None,
    skip_gap_pairs: bool = True,
) -> Dict[str, Any]:
    """Validate local futures OHLCV coverage for the signal rows to be exported.

    When *skip_gap_pairs=True* (default), pairs whose OHLCV file is missing,
    unreadable, has duplicate timestamps, or has gaps are silently excluded from
    the returned coverage rather than raising.  The caller can inspect
    ``result["excluded_pairs"]`` to see what was dropped.  A ValueError is still
    raised if *all* pairs are excluded.
    """
    tf = normalize_timeframe(timeframe)
    minutes = timeframe_minutes(tf)
    venue_s = str(venue or "okx").strip().lower()
    pair_list = sorted(normalize_pair(p) for p in (pairs or used_signal_pairs(signals)))
    if not pair_list:
        raise ValueError("no signal pairs to export")
    hard_errors: list[str] = []
    excluded_pairs: Dict[str, str] = {}  # pair -> reason
    files: Dict[str, Dict[str, Any]] = {}
    global_start: Optional[pd.Timestamp] = None
    global_end: Optional[pd.Timestamp] = None

    for pair in pair_list:
        pair_signals = signals.loc[signals["pair"].map(normalize_pair) == pair]
        if pair_signals.empty:
            continue
        start = pair_signals["date"].min()
        end = pair_signals["date"].max()
        path = futures_ohlcv_path(pair, tf, venue=venue_s, data_root=data_root)
        if not path.exists():
            msg = f"missing OHLCV for {pair}: {path}"
            if skip_gap_pairs:
                excluded_pairs[pair] = msg
            else:
                hard_errors.append(msg)
            continue
        try:
            ohlcv = _read_ohlcv(path, pair=pair)
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            if skip_gap_pairs:
                excluded_pairs[pair] = msg
            else:
                hard_errors.append(msg)
            continue
        duplicated = ohlcv["date"].duplicated(keep=False)
        if bool(duplicated.any()):
            sample = [d.isoformat() for d in ohlcv.loc[duplicated, "date"].head(5)]
            msg = f"duplicate OHLCV timestamps for {pair}: {sample}"
            if skip_gap_pairs:
                excluded_pairs[pair] = msg
            else:
                hard_errors.append(msg)
            continue
        window = ohlcv.loc[(ohlcv["date"] >= start) & (ohlcv["date"] <= end)].copy()
        actual_dates = pd.DatetimeIndex(window["date"].drop_duplicates().sort_values())
        required_dates = pd.DatetimeIndex(pair_signals["date"].drop_duplicates().sort_values())
        missing_signal_dates = required_dates.difference(actual_dates)
        expected = pd.date_range(start=start, end=end, freq=f"{minutes}min", tz="UTC")
        missing_bars = expected.difference(actual_dates)
        pair_soft_errors: list[str] = []
        if len(missing_signal_dates) > 0:
            sample = [d.isoformat() for d in missing_signal_dates[:5]]
            pair_soft_errors.append(f"missing signal timestamps: {sample}")
        if len(missing_bars) > 0:
            sample = [d.isoformat() for d in missing_bars[:5]]
            pair_soft_errors.append(f"bar gaps: {sample}")
        if pair_soft_errors:
            combined = f"OHLCV coverage issue for {pair}: " + "; ".join(pair_soft_errors)
            if skip_gap_pairs:
                excluded_pairs[pair] = combined
                continue
            else:
                hard_errors.append(combined)
        if pair not in excluded_pairs:
            global_start = start if global_start is None else min(global_start, start)
            global_end = end if global_end is None else max(global_end, end)
            files[pair] = {
                "path": str(path),
                "rows": int(len(window)),
                "start": start.isoformat(),
                "end": end.isoformat(),
            }
    if hard_errors:
        raise ValueError("coverage preflight failed: " + "; ".join(hard_errors[:20]))
    clean_pairs = [p for p in pair_list if p not in excluded_pairs and p in files]
    if not clean_pairs:
        raise ValueError(
            "coverage preflight: all pairs excluded. excluded="
            + "; ".join(f"{p}: {r}" for p, r in list(excluded_pairs.items())[:10])
        )
    if excluded_pairs:
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "[lean_bridge] preflight excluded %d pairs with OHLCV gaps: %s",
            len(excluded_pairs),
            ", ".join(excluded_pairs.keys()),
        )
    return {
        "timeframe": tf,
        "timeframe_minutes": int(minutes),
        "venue": venue_s,
        "pairs": clean_pairs,
        "excluded_pairs": excluded_pairs,
        "start": global_start.isoformat() if global_start is not None else None,
        "end": global_end.isoformat() if global_end is not None else None,
        "files": files,
    }


def _format_time(ts: Any) -> str:
    stamp = pd.Timestamp(ts)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    return stamp.strftime("%Y-%m-%d %H:%M:%S")


def _write_signals_csv(signals: pd.DataFrame, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "time",
        "pair",
        "symbol",
        "rp_target_weight",
        "lean_target_weight",
        "lean_previous_target_weight",
        "lean_target_delta",
        "lean_action",
        "rp_rebalance",
        "lean_force_flat",
        "rp_exit_long",
        "rp_exit_short",
        "rp_liq_reject",
        "rp_kill_mode",
        "rp_leverage",
        "rp_stop_pct",
        "rp_liq_distance",
        "rp_side",
        "rp_rank",
        "close",
    ]
    rows = []
    for _, row in signals.sort_values(["date", "pair"]).iterrows():
        item = {
            "time": _format_time(row["date"]),
            "pair": normalize_pair(row["pair"]),
            "symbol": lean_symbol(row["pair"]),
            "rp_target_weight": float(row.get("rp_target_weight", 0.0) or 0.0),
            "lean_target_weight": float(row.get("lean_target_weight", 0.0) or 0.0),
            "lean_previous_target_weight": float(row.get("lean_previous_target_weight", 0.0) or 0.0),
            "lean_target_delta": float(row.get("lean_target_delta", 0.0) or 0.0),
            "lean_action": bool(row.get("lean_action", False)),
            "rp_rebalance": bool(row.get("rp_rebalance", True)),
            "lean_force_flat": bool(row.get("lean_force_flat", False)),
            "rp_exit_long": bool(row.get("rp_exit_long", False)),
            "rp_exit_short": bool(row.get("rp_exit_short", False)),
            "rp_liq_reject": bool(row.get("rp_liq_reject", False)),
            "rp_kill_mode": str(row.get("rp_kill_mode", "normal") or "normal"),
            "rp_leverage": float(row.get("rp_leverage", 1.0) or 1.0),
            "rp_stop_pct": float(row.get("rp_stop_pct", 0.0) or 0.0),
            "rp_liq_distance": row.get("rp_liq_distance", ""),
            "rp_side": int(row.get("rp_side", 0) or 0),
            "rp_rank": int(row.get("rp_rank", 0) or 0),
            "close": row.get("close", ""),
        }
        rows.append(item)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _write_one_ohlcv(
    pair: str,
    pair_signals: pd.DataFrame,
    timeframe: str,
    venue: str,
    data_root: Optional[str | Path],
    output_dir: Path,
) -> tuple[str, str]:
    start = pair_signals["date"].min()
    end = pair_signals["date"].max()
    end_with_execution_bar = end + pd.Timedelta(minutes=timeframe_minutes(timeframe))
    source = futures_ohlcv_path(pair, timeframe, venue=venue, data_root=data_root)
    ohlcv = _read_ohlcv(source, pair=pair)
    ohlcv = ohlcv.loc[(ohlcv["date"] >= start) & (ohlcv["date"] <= end_with_execution_bar)].copy()
    ohlcv["time"] = ohlcv["date"].map(_format_time)
    path = output_dir / f"{lean_symbol(pair)}.csv"
    ohlcv.loc[:, ["time", "open", "high", "low", "close", "volume"]].to_csv(path, index=False)
    return normalize_pair(pair), str(path)


def _write_ohlcv_csvs(
    *,
    signals: pd.DataFrame,
    timeframe: str,
    venue: str,
    data_root: Optional[str | Path],
    output_dir: Path,
    pairs: Sequence[str],
    max_workers: int = 16,
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    # Pre-group signals by pair once — avoids O(pairs × signals) scanning in workers.
    norm_col = signals["pair"].map(normalize_pair)
    signals_by_pair = {normalize_pair(p): signals.loc[norm_col == normalize_pair(p)] for p in pairs}
    outputs: Dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _write_one_ohlcv,
                pair,
                signals_by_pair[normalize_pair(pair)],
                timeframe,
                venue,
                data_root,
                output_dir,
            ): pair
            for pair in pairs
        }
        for fut in as_completed(futures):
            key, path = fut.result()
            outputs[key] = path
    return outputs


def _write_funding_csv(pairs: Sequence[str], data_dir: Path) -> int:
    """Write data/funding.csv from feather files; returns row count."""
    frames: list[pd.DataFrame] = []
    for pair in pairs:
        base = normalize_pair(pair).replace("/", "_")
        feather_path = FUNDING_DIR / f"{base}-funding.feather"
        if not feather_path.exists():
            continue
        try:
            df = pd.read_feather(feather_path)
        except Exception:
            continue
        if "funding_rate" not in df.columns:
            continue
        date_col = "date" if "date" in df.columns else df.columns[0]
        naive_utc = pd.to_datetime(df[date_col], utc=True).dt.tz_convert(None)
        frames.append(pd.DataFrame({
            "time": naive_utc.dt.strftime("%Y-%m-%d %H:%M:%S"),
            "pair": pair,
            "rate": df["funding_rate"].astype(float).values,
        }))
    csv_path = data_dir / "funding.csv"
    if frames:
        out = pd.concat(frames, ignore_index=True)
        out.to_csv(csv_path, index=False)
        return len(out)
    csv_path.write_text("time,pair,rate\n", encoding="utf-8")
    return 0


def _main_py_template() -> str:
    return '''from AlgorithmImports import *
import csv
import json
import os
from datetime import datetime, timedelta


class BridgeFeeModel(FeeModel):
    def __init__(self, fee_rate):
        self.fee_rate = float(fee_rate or 0.0)

    def GetOrderFee(self, parameters):
        price = float(parameters.Security.Price or 0.0)
        quantity = abs(float(parameters.Order.Quantity or 0.0))
        fee = price * quantity * self.fee_rate
        currency = "USD"
        try:
            currency = parameters.Security.QuoteCurrency.Symbol
        except Exception:
            pass
        return OrderFee(CashAmount(fee, currency))


class BridgeSlippageModel:
    def __init__(self, slippage):
        self.slippage = float(slippage or 0.0)

    def GetSlippageApproximation(self, asset, order):
        return float(asset.Price or 0.0) * self.slippage


class _RiskState:
    """Mirror of rank_portfolio.AccountRiskController for LEAN backtesting."""

    def __init__(self, cfg):
        self.daily_loss_limit   = float(cfg.get("daily_loss_limit",   0.04) or 0.04)
        self.weekly_loss_limit  = float(cfg.get("weekly_loss_limit",  0.08) or 0.08)
        self.drawdown_safe_mode = float(cfg.get("drawdown_safe_mode", 0.20) or 0.20)
        self.consecutive_limit  = int(cfg.get("consecutive_loss_limit", 8)  or 8)
        self.pause_hours        = int(cfg.get("pause_hours", 24) or 24)
        self.peak         = 1.0
        self.day_start    = 1.0
        self.week_start   = 1.0
        self.consec_losses = 0
        self.last_equity  = 1.0
        self.pause_until  = None
        self.mode         = "normal"

    def update(self, ts, equity_frac):
        """Update state; returns (allow_new_entries, allow_reduce)."""
        eq = float(equity_frac)
        if eq > self.peak:
            self.peak = eq
        if ts.hour == 0 and ts.minute == 0:
            if eq < self.last_equity:
                self.consec_losses += 1
            else:
                self.consec_losses = 0
            self.day_start = eq
        if ts.weekday() == 0 and ts.hour == 0:
            self.week_start = eq
        self.last_equity = eq
        if self.pause_until and ts < self.pause_until:
            if self.consec_losses >= self.consecutive_limit:
                self.mode = "loss_pause"
                return False, True
        if self.consec_losses >= self.consecutive_limit and self.pause_until is None:
            self.pause_until = ts + timedelta(hours=self.pause_hours)
            self.mode = "loss_pause"
            return False, True
        dd = (self.peak - eq) / max(self.peak, 1e-9)
        day_loss = (self.day_start - eq) / max(self.day_start, 1e-9)
        week_loss = (self.week_start - eq) / max(self.week_start, 1e-9)
        if day_loss >= self.daily_loss_limit:
            self.mode = "daily_halt"
            return False, True
        if week_loss >= self.weekly_loss_limit:
            self.mode = "weekly_safe"
            return False, True
        if dd >= self.drawdown_safe_mode:
            self.mode = "drawdown_safe"
            return False, True
        self.mode = "normal"
        return True, True


class LocalFuturesOhlcv(PythonData):
    TimeframeMinutes = 60

    def GetSource(self, config, date, isLiveMode):
        root = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(root, "data", "ohlcv", f"{config.Symbol.Value}.csv")
        return SubscriptionDataSource(path, SubscriptionTransportMedium.LocalFile)

    def Reader(self, config, line, date, isLiveMode):
        if not line or line.startswith("time"):
            return None
        parts = line.split(",")
        if len(parts) < 6:
            return None
        bar = LocalFuturesOhlcv()
        bar.Symbol = config.Symbol
        stamp = datetime.strptime(parts[0], "%Y-%m-%d %H:%M:%S")
        bar.Time = stamp
        bar.EndTime = stamp + timedelta(minutes=int(LocalFuturesOhlcv.TimeframeMinutes))
        bar.Value = float(parts[1])  # fill at bar OPEN (no look-ahead)
        bar.high = float(parts[2])
        bar.low = float(parts[3])
        bar.close = float(parts[4])
        bar.volume = float(parts[5])
        return bar


class LeanBridgeAlgorithm(QCAlgorithm):
    def Initialize(self):
        root = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(root, "manifest.json"), "r", encoding="utf-8") as handle:
            self.manifest = json.load(handle)
        self.risk = self.manifest.get("risk_config", {})
        LocalFuturesOhlcv.TimeframeMinutes = int(self.manifest.get("timeframe_minutes", 60))
        self.SetTimeZone(TimeZones.Utc)
        start = datetime.strptime(self.manifest["date_range"]["start"][:10], "%Y-%m-%d")
        end = datetime.strptime(self.manifest["date_range"]["end"][:10], "%Y-%m-%d")
        self.SetStartDate(start.year, start.month, start.day)
        self.SetEndDate(end.year, end.month, end.day)
        self.SetCash(100000)
        self.Settings.MinimumOrderMarginPortfolioPercentage = 0

        resolution = Resolution.Hour
        if int(LocalFuturesOhlcv.TimeframeMinutes) < 60:
            resolution = Resolution.Minute

        self.symbol_by_pair = {}
        first_symbol = None
        fee_rate = float(self.risk.get("fee_rate", 0.0) or 0.0)
        slippage = float(self.risk.get("slippage", 0.0) or 0.0)
        leverage = max(1.0, float(self.risk.get("leverage_cap", 1.0) or 1.0))
        for item in self.manifest.get("pairs", []):
            symbol = self.AddData(LocalFuturesOhlcv, item["symbol"], resolution).Symbol
            if first_symbol is None:
                first_symbol = symbol
            self.symbol_by_pair[item["pair"]] = symbol
            security = self.Securities[symbol]
            security.SetLeverage(leverage)
            security.SetFeeModel(BridgeFeeModel(fee_rate))
            try:
                security.SetSlippageModel(BridgeSlippageModel(slippage))
            except Exception as exc:
                self.Debug(f"slippage model unavailable for {item['symbol']}: {exc}")
        if first_symbol is not None:
            self.SetBenchmark(first_symbol)
        else:
            self.SetBenchmark(lambda _: 0)

        self.signals = {}
        signal_path = os.path.join(root, "data", "signals.csv")
        with open(signal_path, "r", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                stamp = datetime.strptime(row["time"], "%Y-%m-%d %H:%M:%S")
                self.signals.setdefault(stamp, []).append(row)
        self._ohlcv_by_pair = {}
        for item in self.manifest.get("pairs", []):
            pair = item["pair"]
            path = os.path.join(root, "data", "ohlcv", f"{item['symbol']}.csv")
            bars = {}
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as handle:
                    for brow in csv.DictReader(handle):
                        stamp = datetime.strptime(brow["time"], "%Y-%m-%d %H:%M:%S")
                        bars[stamp] = {
                            "open": float(brow["open"]),
                            "high": float(brow["high"]),
                            "low": float(brow["low"]),
                            "close": float(brow["close"]),
                            "volume": float(brow["volume"]),
                        }
            self._ohlcv_by_pair[pair] = bars
        self._current_ohlcv = {}
        self.previous_targets = {}
        self._risk_state = _RiskState(self.risk)
        self._entry_prices = {}   # symbol -> entry price
        self._entry_side   = {}   # symbol -> +1 long / -1 short
        self._stop_pct = float(self.risk.get("max_stop_pct", 0.30) or 0.30)
        self._halted = False      # True after liquidation — stops all trading

        # Funding rates: (pair, naive-UTC-datetime) → rate (8h cadence)
        self._funding = {}
        self._total_funding_paid = 0.0
        funding_path = os.path.join(root, "data", "funding.csv")
        if os.path.exists(funding_path):
            with open(funding_path, "r", encoding="utf-8") as fh:
                for frow in csv.DictReader(fh):
                    stamp = datetime.strptime(frow["time"], "%Y-%m-%d %H:%M:%S")
                    self._funding[(frow["pair"], stamp)] = float(frow["rate"])

    def _portfolio_equity_frac(self):
        total = float(self.Portfolio.TotalPortfolioValue or 0.0)
        start_cash = 100000.0
        return total / start_cash if start_cash > 0 else 1.0

    def _set_symbol_price_at_open(self, pair, symbol, stamp):
        bar = self._ohlcv_by_pair.get(pair, {}).get(stamp)
        if bar is None:
            return None
        point = LocalFuturesOhlcv()
        point.Symbol = symbol
        point.Time = stamp
        point.EndTime = stamp + timedelta(minutes=int(LocalFuturesOhlcv.TimeframeMinutes))
        point.Value = float(bar["open"])
        point.high = float(bar["high"])
        point.low = float(bar["low"])
        point.close = float(bar["close"])
        point.volume = float(bar["volume"])
        self.Securities[symbol].SetMarketPrice(point)
        return bar

    def _check_stop_losses(self, data=None):
        for sym, entry_price in list(self._entry_prices.items()):
            if entry_price <= 0:
                continue
            holding = self.Portfolio[sym]
            if abs(float(holding.Quantity)) < 1e-9:
                self._entry_prices.pop(sym, None)
                self._entry_side.pop(sym, None)
                continue
            side = self._entry_side.get(sym, 1)
            check_price = 0.0
            current_bar = self._current_ohlcv.get(sym)
            if current_bar is not None:
                raw = float(current_bar["low"] if side == 1 else current_bar["high"])
                if raw > 0:
                    check_price = raw
            if check_price <= 0 and data is not None:
                try:
                    if sym in data:
                        bd = data[sym]
                        raw = float(bd.low if side == 1 else bd.high)
                        if raw > 0:
                            check_price = raw
                except Exception:
                    pass
            if check_price <= 0:
                check_price = float(self.Securities[sym].Price or 0.0)
            if check_price <= 0:
                continue
            adverse = side * (entry_price - check_price) / entry_price
            if adverse >= self._stop_pct:
                self.SetHoldings(sym, 0)
                self._entry_prices.pop(sym, None)
                self._entry_side.pop(sym, None)

    def _plot_bridge_metrics(self):
        total = float(self.Portfolio.TotalPortfolioValue or 0.0)
        if total <= 0:
            return
        gross_value = 0.0
        for symbol in self.symbol_by_pair.values():
            gross_value += abs(float(self.Portfolio[symbol].HoldingsValue))
        self.Plot("Bridge Exposure", "Gross", gross_value / total)

    def OnData(self, data):
        if self._halted:
            return

        current_stamp = self.Time.replace(tzinfo=None)
        self._current_ohlcv = {}
        for pair, symbol in self.symbol_by_pair.items():
            bar = self._set_symbol_price_at_open(pair, symbol, current_stamp)
            if bar is not None:
                self._current_ohlcv[symbol] = bar

        eq_frac = self._portfolio_equity_frac()
        # Halt if equity effectively wiped out (< 2% of start)
        if eq_frac < 0.02:
            self.Liquidate()
            self._halted = True
            return

        allow_new, allow_reduce = self._risk_state.update(self.Time, eq_frac)
        self._check_stop_losses(data)

        # 8h funding settlement at 00:00, 08:00, 16:00 UTC
        if self.Time.minute == 0 and self.Time.hour in (0, 8, 16):
            t_key = self.Time.replace(tzinfo=None)
            for pair, symbol in self.symbol_by_pair.items():
                rate = self._funding.get((pair, t_key), 0.0)
                if rate == 0.0:
                    continue
                holding = self.Portfolio[symbol]
                qty = float(holding.Quantity or 0.0)
                if abs(qty) < 1e-9:
                    continue
                price = float(self.Securities[symbol].Price or 0.0)
                if price <= 0:
                    continue
                payment = qty * price * rate  # longs pay (+), shorts receive (-)
                self.Portfolio.CashBook["USD"].AddAmount(-payment)
                self._total_funding_paid += payment

        # Signal stamped at bar T fires at bar T+1 open — shift lookup by 1 bar
        prev_stamp = (self.Time - timedelta(minutes=int(LocalFuturesOhlcv.TimeframeMinutes))).replace(tzinfo=None)
        rows = self.signals.get(prev_stamp)
        if not rows:
            self._plot_bridge_metrics()
            return

        targets = {}
        for row in rows:
            pair = row["pair"]
            target = float(row.get("lean_target_weight") or 0.0)
            targets[pair] = target

        gross = sum(abs(v) for v in targets.values())
        gross_cap = float(self.risk.get("gross_cap", gross or 0.0) or 0.0)
        scale = gross_cap / gross if gross_cap > 0 and gross > gross_cap else 1.0

        for row in rows:
            action = str(row.get("lean_action", "")).strip().lower() in ("1", "true", "yes")
            if not action:
                continue
            pair = row["pair"]
            new_target = targets.get(pair, 0.0) * scale
            symbol = self.symbol_by_pair.get(pair)
            if symbol is None:
                continue
            if symbol not in self._current_ohlcv:
                continue
            prev_target = float(self.previous_targets.get(pair, 0.0))
            is_increase = abs(new_target) > abs(prev_target) + 1e-9
            if is_increase and not allow_new:
                continue
            was_flat = abs(float(self.Portfolio[symbol].Quantity or 0.0)) < 1e-9
            self.SetHoldings(symbol, new_target)
            # Update entry price: fresh open, position increase, or re-entry after stop-out
            if abs(new_target) > 1e-9 and (abs(prev_target) < 1e-9 or is_increase or was_flat):
                price = float(self.Securities[symbol].Price or 0.0)
                if price > 0:
                    self._entry_prices[symbol] = price
                    self._entry_side[symbol] = 1 if new_target > 0 else -1

        self.previous_targets = targets
        self._plot_bridge_metrics()

    def OnEndOfAlgorithm(self):
        self.Debug(f"Total funding paid: {self._total_funding_paid:.6f} USD")
'''


def _config_json() -> Dict[str, Any]:
    return {
        "algorithm-language": "Python",
        "algorithm-type-name": "LeanBridgeAlgorithm",
        "parameters": {
            "signal-file": "data/signals.csv",
            "manifest-file": "manifest.json",
        },
        "description": "Agent_market local LEAN validation bridge project",
        "cloud-id": 0,
        "organization-id": "",
    }


def export_project(
    *,
    rank_artifact: str | Path,
    output: str | Path,
    timeframe: Optional[str] = None,
    data_root: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Export a self-contained local LEAN project from a rank artifact.

    Pairs whose OHLCV data has gaps or is missing are automatically excluded
    (see ``preflight_coverage`` with *skip_gap_pairs=True*).  The returned
    manifest records excluded pairs under ``coverage.excluded_pairs``.
    """
    artifact = load_rank_artifact(rank_artifact, timeframe=timeframe)
    signals = load_signals(artifact["signals_path"])
    used_pairs = used_signal_pairs(signals)
    signals = signals.loc[signals["pair"].isin(used_pairs)].copy()
    signals = normalize_signal_targets(signals)
    coverage = preflight_coverage(
        signals,
        timeframe=artifact["timeframe"],
        venue=artifact["venue"],
        data_root=data_root,
        pairs=used_pairs,
        skip_gap_pairs=True,
    )

    # Drop excluded pairs from signals so LEAN only sees clean data.
    excluded = set(coverage.get("excluded_pairs", {}).keys())
    if excluded:
        used_pairs = [p for p in used_pairs if p not in excluded]
        signals = signals.loc[~signals["pair"].map(normalize_pair).isin(excluded)].copy()

    out_dir = Path(output).expanduser().resolve()
    data_dir = out_dir / "data"
    ohlcv_dir = data_dir / "ohlcv"
    out_dir.mkdir(parents=True, exist_ok=True)
    signal_csv = data_dir / "signals.csv"
    signal_rows = _write_signals_csv(signals, signal_csv)
    ohlcv_csvs = _write_ohlcv_csvs(
        signals=signals,
        timeframe=artifact["timeframe"],
        venue=artifact["venue"],
        data_root=data_root,
        output_dir=ohlcv_dir,
        pairs=used_pairs,
    )
    funding_rows = _write_funding_csv(used_pairs, data_dir)
    date_start = signals["date"].min().isoformat()
    date_end = signals["date"].max().isoformat()
    pair_manifest = [
        {
            "pair": pair,
            "symbol": lean_symbol(pair),
            "futures_token": pair_file_token(pair),
            "okx_token": pair_file_token(pair),
            "ohlcv_csv": str(Path(ohlcv_csvs[pair]).relative_to(out_dir)),
        }
        for pair in used_pairs
    ]
    manifest = {
        "version": BRIDGE_VERSION,
        "created_at": pd.Timestamp.now(tz=timezone.utc).isoformat(),
        "local_only": True,
        "quantconnect_cloud": False,
        "artifact": {
            "path": str(artifact["artifact_path"]),
            "kind": artifact["artifact_kind"],
            "tag": artifact["tag"],
            "signals_path": str(artifact["signals_path"]),
        },
        "venue": artifact["venue"],
        "data_venue": artifact["data_venue"],
        "timeframe": artifact["timeframe"],
        "timeframe_minutes": int(artifact["timeframe_minutes"]),
        "micro_validation": artifact["timeframe"] in {"1m", "5m"},
        "date_range": {"start": date_start, "end": date_end},
        "risk_config": asdict(artifact["risk_config"]),
        "execution_model": {
            "name": "close_to_next_bar_approx",
            "description": "Portfolio-level validation using precomputed target weights and local futures OHLCV; not an order-book simulation.",
            "rebalance_policy": "order_on_lean_action_only",
            "non_rebalance_behavior": "carry target in signals; LEAN holds filled quantity until the next target-change or force-flat action",
        },
        "pairs": pair_manifest,
        "files": {
            "main_py": "main.py",
            "config_json": "config.json",
            "signals_csv": str(signal_csv.relative_to(out_dir)),
            "ohlcv_dir": str(ohlcv_dir.relative_to(out_dir)),
            "funding_csv": "data/funding.csv",
        },
        "funding": {
            "rows": int(funding_rows),
            "csv": "data/funding.csv",
        },
        "coverage": coverage,
        "signals": {
            "rows": int(signal_rows),
            "pairs": int(len(used_pairs)),
            "dates": int(signals["date"].nunique()),
            "action_rows": int(signals["lean_action"].fillna(False).astype(bool).sum()),
        },
    }
    (out_dir / "main.py").write_text(_main_py_template(), encoding="utf-8")
    (out_dir / "config.json").write_text(json.dumps(_config_json(), indent=2), encoding="utf-8")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8")
    return manifest


def find_latest_lean_result(project: str | Path) -> Optional[Path]:
    root = Path(project)
    if root.is_file():
        return root.resolve()
    candidates: list[Path] = []
    for name in ("lean_bridge_result.json", "results.json", "statistics.json"):
        candidates.extend(root.rglob(name))
    candidates.extend(root.rglob("*-summary.json"))
    candidates.extend(
        path
        for path in root.rglob("*.json")
        if not path.name.endswith("-order-events.json")
        and not path.name.startswith("data-monitor-report-")
        and path.name not in {"manifest.json", "config.json", "comparison.json", "lean_backtest_run.json"}
    )
    if not candidates:
        return None
    def _priority(path: Path) -> tuple[int, float]:
        if path.name.endswith("-summary.json"):
            return (2, path.stat().st_mtime)
        return (1, path.stat().st_mtime)
    return max(candidates, key=_priority).resolve()


def run_lean_backtest(
    *,
    lean_project: str | Path,
    lean_bin: str = "lean",
    timeout: Optional[int] = None,
) -> Dict[str, Any]:
    """Run `lean backtest` for a local bridge project and record command output."""
    project = Path(lean_project).expanduser().resolve()
    if not project.exists():
        raise FileNotFoundError(f"LEAN project not found: {project}")
    binary = shutil.which(lean_bin)
    if binary:
        binary = str(Path(binary).expanduser().resolve())
    if not binary:
        binary_path = Path(lean_bin).expanduser()
        if binary_path.exists():
            binary = str(binary_path.resolve())
    if not binary:
        raise FileNotFoundError(f"LEAN binary not found: {lean_bin}")
    started = time.time()
    cmd = [binary, "backtest", str(project)]
    lean_config = repo_paths.artifacts_root() / "lean" / "lean.json"
    if lean_config.exists():
        cmd.extend(["--lean-config", str(lean_config.resolve())])
    proc = subprocess.run(
        cmd,
        cwd=str(project.parent),
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    result_path = find_latest_lean_result(project)
    # Treat as success if a result file was produced, even when the LEAN CLI
    # exits non-zero (e.g. post-processing path-validation warnings).
    effective_ok = result_path is not None
    summary = {
        "command": cmd,
        "returncode": int(proc.returncode),
        "ok": effective_ok,
        "duration_sec": float(time.time() - started),
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
        "result_path": str(result_path) if result_path else None,
    }
    (project / "lean_backtest_run.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not effective_ok:
        raise RuntimeError(
            f"lean backtest failed with returncode={proc.returncode}; "
            f"result_path={result_path}; see {project / 'lean_backtest_run.json'}"
        )
    return summary


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        out = float(value)
        return out if math.isfinite(out) else None
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    is_percent = text.endswith("%")
    if is_percent:
        text = text[:-1].strip()
    for token in ("$", "USDT", "USD"):
        text = text.replace(token, "")
    try:
        out = float(text)
    except ValueError:
        return None
    if is_percent:
        out /= 100.0
    return out if math.isfinite(out) else None


def _lookup_metric(mapping: Mapping[str, Any], names: Iterable[str]) -> Any:
    lower = {str(k).strip().lower(): v for k, v in mapping.items()}
    for name in names:
        key = str(name).strip().lower()
        if key in lower:
            return lower[key]
    for value in mapping.values():
        if isinstance(value, Mapping):
            found = _lookup_metric(value, names)
            if found is not None:
                return found
    return None


def _load_lean_payload(path: str | Path) -> Dict[str, Any]:
    result_path = Path(path).expanduser().resolve()
    if result_path.is_dir():
        latest = find_latest_lean_result(result_path)
        if latest is None:
            raise FileNotFoundError(f"no LEAN result JSON found under {result_path}")
        result_path = latest
    payload = _read_json(result_path)
    source_path = result_path
    if "result_path" in payload and payload.get("result_path"):
        nested = Path(str(payload["result_path"])).expanduser()
        if not nested.is_absolute():
            nested = result_path.parent / nested
        if nested.exists():
            payload = _read_json(nested)
            source_path = nested.resolve()
    payload["_source_path"] = str(source_path)
    return payload


def _lean_result_sibling_paths(source_path: Path) -> Dict[str, Path]:
    name = source_path.name
    stem = source_path.stem
    if name.endswith("-summary.json"):
        prefix = name[: -len("-summary.json")]
    elif name.endswith("-order-events.json"):
        prefix = name[: -len("-order-events.json")]
    elif source_path.suffix == ".json":
        prefix = stem
    else:
        prefix = stem
    return {
        "summary": source_path.with_name(f"{prefix}-summary.json"),
        "full": source_path.with_name(f"{prefix}.json"),
        "order_events": source_path.with_name(f"{prefix}-order-events.json"),
    }


def _load_optional_dict(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        payload = _read_json(path)
    except Exception:
        return None
    payload["_source_path"] = str(path.resolve())
    return payload


def _load_lean_related_payloads(payload: Dict[str, Any]) -> Dict[str, Any]:
    source_path = Path(str(payload.get("_source_path") or "")).expanduser()
    paths = _lean_result_sibling_paths(source_path) if source_path.name else {}
    is_summary_source = source_path.name.endswith("-summary.json")
    summary_payload: Optional[Dict[str, Any]] = None
    full_payload: Optional[Dict[str, Any]] = None
    if "statistics" in payload or "Statistics" in payload:
        summary_payload = payload
    if not is_summary_source and ("charts" in payload or "orders" in payload or "totalPerformance" in payload):
        full_payload = payload
    if not summary_payload and paths:
        summary_payload = _load_optional_dict(paths["summary"])
    if not full_payload and paths:
        full_payload = _load_optional_dict(paths["full"])
    order_events: list[Mapping[str, Any]] = []
    if paths and paths["order_events"].exists():
        raw_events = _read_json_any(paths["order_events"])
        if isinstance(raw_events, list):
            order_events = [event for event in raw_events if isinstance(event, Mapping)]
        elif isinstance(raw_events, Mapping):
            raw_list = raw_events.get("orderEvents") or raw_events.get("OrderEvents") or []
            if isinstance(raw_list, list):
                order_events = [event for event in raw_list if isinstance(event, Mapping)]
    return {
        "summary": summary_payload or payload,
        "full": full_payload,
        "order_events": order_events,
        "paths": paths,
    }


def _series_point_time_value(point: Any) -> Optional[tuple[float, float]]:
    if isinstance(point, Mapping):
        raw_time = point.get("x") or point.get("time") or point.get("Time")
        raw_value = point.get("y") or point.get("value") or point.get("Value")
    elif isinstance(point, (list, tuple)) and len(point) >= 2:
        raw_time = point[0]
        raw_value = point[-1]
    else:
        return None
    timestamp = _as_float(raw_time)
    value = _as_float(raw_value)
    if timestamp is None or value is None:
        return None
    return float(timestamp), float(value)


def _chart_series_values(payload: Optional[Mapping[str, Any]], chart_name: str, series_name: str) -> list[tuple[float, float]]:
    if not isinstance(payload, Mapping):
        return []
    charts = payload.get("charts") or payload.get("Charts") or {}
    if not isinstance(charts, Mapping):
        return []
    chart = charts.get(chart_name)
    if not isinstance(chart, Mapping):
        return []
    series_map = chart.get("series") or chart.get("Series") or {}
    if not isinstance(series_map, Mapping):
        return []
    series = series_map.get(series_name)
    if not isinstance(series, Mapping):
        return []
    raw_values = series.get("values") or series.get("Values") or []
    out: list[tuple[float, float]] = []
    if isinstance(raw_values, list):
        for point in raw_values:
            parsed = _series_point_time_value(point)
            if parsed is not None:
                out.append(parsed)
    return sorted(out, key=lambda item: item[0])


def _equity_at_or_before(equity_series: Sequence[tuple[float, float]], timestamp: float, fallback: float) -> float:
    if not equity_series:
        return float(fallback)
    times = [item[0] for item in equity_series]
    idx = bisect_right(times, float(timestamp)) - 1
    if idx < 0:
        return float(fallback)
    equity = float(equity_series[idx][1])
    if not math.isfinite(equity) or equity <= 0:
        return float(fallback)
    return equity


def _event_timestamp(event: Mapping[str, Any]) -> Optional[float]:
    raw = event.get("time") or event.get("Time") or event.get("lastFillTime") or event.get("LastFillTime")
    if raw is None:
        return None
    numeric = _as_float(raw)
    if numeric is not None:
        return float(numeric)
    try:
        return float(pd.Timestamp(raw).timestamp())
    except Exception:
        return None


def _event_symbol(event: Mapping[str, Any]) -> str:
    symbol = event.get("symbolValue") or event.get("SymbolValue") or event.get("symbol") or event.get("Symbol")
    if isinstance(symbol, Mapping):
        symbol = symbol.get("value") or symbol.get("Value") or symbol.get("permtick") or symbol.get("Permtick")
    return str(symbol or "").split(".", 1)[0]


def _order_lookup(full_payload: Optional[Mapping[str, Any]]) -> Dict[int, Mapping[str, Any]]:
    if not isinstance(full_payload, Mapping):
        return {}
    raw_orders = full_payload.get("orders") or full_payload.get("Orders") or {}
    if not isinstance(raw_orders, Mapping):
        return {}
    out: Dict[int, Mapping[str, Any]] = {}
    for key, value in raw_orders.items():
        if not isinstance(value, Mapping):
            continue
        order_id = _as_float(value.get("id") or value.get("Id") or key)
        if order_id is None:
            continue
        out[int(order_id)] = value
    return out


def _filled_events_from_orders(full_payload: Optional[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    orders = _order_lookup(full_payload)
    events: list[Dict[str, Any]] = []
    for order_id, order in orders.items():
        status = str(order.get("status") or order.get("Status") or "").strip().lower()
        status_num = _as_float(order.get("status") or order.get("Status"))
        if status not in {"filled", "3"} and status_num != 3:
            continue
        quantity = _as_float(order.get("quantity") or order.get("Quantity"))
        price = _as_float(order.get("price") or order.get("Price"))
        timestamp = order.get("lastFillTime") or order.get("time") or order.get("Time")
        if quantity is None or price is None or abs(quantity) <= 1e-12:
            continue
        events.append(
            {
                "orderId": order_id,
                "time": timestamp,
                "status": "filled",
                "fillQuantity": quantity,
                "fillPrice": price,
                "symbol": order.get("symbol") or order.get("Symbol"),
            }
        )
    return events


def _lean_execution_stats(
    *,
    order_events: Sequence[Mapping[str, Any]],
    full_payload: Optional[Mapping[str, Any]],
    start_equity: Optional[float],
) -> Dict[str, Any]:
    events = list(order_events) or _filled_events_from_orders(full_payload)
    equity_fallback = float(start_equity or 100000.0)
    equity_series = _chart_series_values(full_payload, "Strategy Equity", "Equity")
    orders = _order_lookup(full_payload)
    fills: list[Dict[str, Any]] = []
    for event in events:
        status = str(event.get("status") or event.get("Status") or "").strip().lower()
        status_num = _as_float(event.get("status") or event.get("Status"))
        quantity = _as_float(
            event.get("fillQuantity")
            if event.get("fillQuantity") is not None
            else event.get("FillQuantity")
            if event.get("FillQuantity") is not None
            else event.get("quantity")
            if event.get("quantity") is not None
            else event.get("Quantity")
        )
        price = _as_float(
            event.get("fillPrice")
            if event.get("fillPrice") is not None
            else event.get("FillPrice")
            if event.get("FillPrice") is not None
            else event.get("price")
            if event.get("price") is not None
            else event.get("Price")
        )
        timestamp = _event_timestamp(event)
        is_filled = ("filled" in status) or status_num == 3
        if not is_filled or quantity is None or price is None or timestamp is None or abs(quantity) <= 1e-12:
            continue
        fills.append(
            {
                "timestamp": float(timestamp),
                "order_id": int(_as_float(event.get("orderId") or event.get("OrderId") or 0) or 0),
                "symbol": _event_symbol(event),
                "quantity": float(quantity),
                "price": float(price),
                "fee": float(_as_float(event.get("orderFeeAmount") or event.get("OrderFeeAmount")) or 0.0),
            }
        )
    fills.sort(key=lambda item: (item["timestamp"], item["order_id"], item["symbol"]))

    positions: Dict[str, float] = {}
    entries = 0
    exits = 0
    total_notional = 0.0
    turnover_dynamic = 0.0
    fee_dynamic = 0.0
    total_fees = 0.0
    slippage_dynamic = 0.0
    total_slippage = 0.0
    for fill in fills:
        timestamp = float(fill["timestamp"])
        quantity = float(fill["quantity"])
        price = float(fill["price"])
        notional = abs(quantity * price)
        equity = _equity_at_or_before(equity_series, timestamp, equity_fallback)
        total_notional += notional
        turnover_dynamic += notional / max(equity, 1e-12)
        fee = float(fill["fee"])
        total_fees += fee
        fee_dynamic += fee / max(equity, 1e-12)

        order = orders.get(int(fill["order_id"]), {})
        submission = order.get("orderSubmissionData") or order.get("OrderSubmissionData") or {}
        base_price = None
        if isinstance(submission, Mapping):
            base_price = _as_float(submission.get("lastPrice") or submission.get("LastPrice"))
            if base_price is None:
                bid = _as_float(submission.get("bidPrice") or submission.get("BidPrice"))
                ask = _as_float(submission.get("askPrice") or submission.get("AskPrice"))
                if bid is not None and ask is not None:
                    base_price = (bid + ask) / 2.0
        if base_price is not None:
            slip = abs(price - float(base_price)) * abs(quantity)
            total_slippage += slip
            slippage_dynamic += slip / max(equity, 1e-12)

        symbol = str(fill["symbol"])
        before = positions.get(symbol, 0.0)
        after = before + quantity
        before_active = abs(before) > 1e-12
        after_active = abs(after) > 1e-12
        sign_flip = before_active and after_active and (before > 0) != (after > 0)
        if before_active and (not after_active or sign_flip):
            exits += 1
        if (not before_active and after_active) or sign_flip:
            entries += 1
        positions[symbol] = 0.0 if abs(after) <= 1e-12 else after

    return {
        "orders": float(len(fills)),
        "trades": float(entries),
        "entries": float(entries),
        "exits": float(exits),
        "turnover": float(turnover_dynamic),
        "turnover_start_equity": float(total_notional / max(equity_fallback, 1e-12)),
        "fee_cost": float(fee_dynamic),
        "fee_cost_start_equity": float(total_fees / max(equity_fallback, 1e-12)),
        "slippage_cost": float(slippage_dynamic),
        "slippage_cost_start_equity": float(total_slippage / max(equity_fallback, 1e-12)),
        "filled_order_count": float(len(fills)),
        "ending_open_positions": float(sum(1 for qty in positions.values() if abs(qty) > 1e-12)),
        "execution_metric_source": "order_events" if fills else "summary",
    }


def parse_lean_metrics(lean_result: str | Path) -> Dict[str, Any]:
    payload = _load_lean_payload(lean_result)
    related = _load_lean_related_payloads(payload)
    summary_payload = related["summary"]
    full_payload = related["full"]
    stats = summary_payload.get("statistics") or summary_payload.get("Statistics") or summary_payload.get("summary") or summary_payload.get("metrics") or summary_payload
    if not isinstance(stats, Mapping):
        stats = summary_payload
    start_equity = _as_float(_lookup_metric(stats, ("start_equity", "start equity")))
    final_equity = _as_float(_lookup_metric(stats, ("final_equity", "end_equity", "end equity", "final portfolio value", "portfolio value")))
    total_return = _as_float(_lookup_metric(stats, ("total_return", "total return", "net profit", "compounding annual return")))
    if final_equity is not None and start_equity is not None and start_equity > 0:
        final_equity = final_equity / start_equity
    if final_equity is None and total_return is not None:
        final_equity = 1.0 + total_return
    if total_return is None and final_equity is not None:
        total_return = final_equity - 1.0
    max_drawdown = _as_float(_lookup_metric(stats, ("max_drawdown", "max drawdown", "drawdown")))
    trades = _as_float(_lookup_metric(stats, ("trades", "total trades", "entries", "entry count")))
    orders = _as_float(_lookup_metric(stats, ("orders", "filled orders", "total orders", "order count")))
    summary_turnover = _as_float(_lookup_metric(stats, ("turnover", "portfolio turnover", "avg_turnover", "average turnover")))
    turnover = summary_turnover
    avg_gross = _as_float(_lookup_metric(stats, ("avg_gross", "average gross", "gross exposure")))
    max_gross = _as_float(_lookup_metric(stats, ("max_gross", "max gross", "maximum gross exposure")))
    fee_cost = _as_float(_lookup_metric(stats, ("fee_cost", "fee cost", "total fees", "fees")))
    if fee_cost is not None and start_equity is not None and start_equity > 0:
        fee_cost = fee_cost / start_equity
    slippage_cost = _as_float(_lookup_metric(stats, ("slippage_cost", "slippage cost", "estimated slippage")))
    profit_over_dd = _as_float(_lookup_metric(stats, ("profit_over_max_drawdown", "profit/max_drawdown", "profit over max drawdown")))
    if profit_over_dd is None and total_return is not None and max_drawdown is not None:
        profit_over_dd = total_return / max(max_drawdown, 1e-12)
    execution = _lean_execution_stats(
        order_events=related["order_events"],
        full_payload=full_payload,
        start_equity=start_equity,
    )
    if execution["filled_order_count"] > 0:
        trades = execution["trades"]
        orders = execution["orders"]
        turnover = execution["turnover"]
        fee_cost = execution["fee_cost"]
        slippage_cost = execution["slippage_cost"]
    gross_series = _chart_series_values(full_payload, "Bridge Exposure", "Gross")
    if gross_series:
        gross_values = [value for _, value in gross_series if math.isfinite(float(value))]
        if gross_values:
            avg_gross = float(np.mean(gross_values))
            max_gross = float(max(gross_values))
    return {
        "final_equity": final_equity,
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "profit_over_max_drawdown": profit_over_dd,
        "trades": trades,
        "orders": orders,
        "turnover": turnover,
        "avg_gross": avg_gross,
        "max_gross": max_gross,
        "fee_cost": fee_cost,
        "slippage_cost": slippage_cost,
        "start_equity": start_equity,
        "summary_portfolio_turnover": summary_turnover,
        "summary_total_orders": _as_float(_lookup_metric(stats, ("total orders", "order count"))),
        "turnover_start_equity": execution.get("turnover_start_equity"),
        "fee_cost_start_equity": execution.get("fee_cost_start_equity"),
        "slippage_cost_start_equity": execution.get("slippage_cost_start_equity"),
        "entries": execution.get("entries"),
        "exits": execution.get("exits"),
        "filled_order_count": execution.get("filled_order_count"),
        "ending_open_positions": execution.get("ending_open_positions"),
        "execution_metric_source": execution.get("execution_metric_source"),
        "source_path": payload.get("_source_path"),
    }


def _signal_turnover_stats(signals: pd.DataFrame, cfg: RiskConfig) -> Dict[str, float]:
    targets = normalize_signal_targets(signals)
    previous: Dict[str, float] = {}
    turnovers: list[float] = []
    gross_values: list[float] = []
    orders = 0
    entries = 0
    exits = 0
    target_change_dates = 0
    executable_dates = pd.DatetimeIndex(pd.to_datetime(targets["date"], utc=True).drop_duplicates().sort_values())
    terminal_date = executable_dates[-1] if len(executable_dates) else None
    for _, group in targets.groupby("date", sort=True):
        group_date = pd.Timestamp(group["date"].iloc[0])
        if terminal_date is not None and group_date == terminal_date:
            # A signal stamped at T executes at T+1 open. The terminal signal
            # row has no following bar in the exported signal grid, so LEAN and
            # Freqtrade cannot submit that target-change instruction.
            continue
        now = {str(row["pair"]): float(row["lean_target_weight"]) for _, row in group.iterrows()}
        pairs = set(previous) | set(now)
        turnover = sum(abs(now.get(pair, 0.0) - previous.get(pair, 0.0)) for pair in pairs)
        changed_pairs = [
            pair
            for pair in pairs
            if abs(now.get(pair, 0.0) - previous.get(pair, 0.0)) > 1e-12
        ]
        if changed_pairs:
            target_change_dates += 1
        orders += len(changed_pairs)
        for pair in changed_pairs:
            before = previous.get(pair, 0.0)
            after = now.get(pair, 0.0)
            before_active = abs(before) > 1e-12
            after_active = abs(after) > 1e-12
            sign_flip = before_active and after_active and (before > 0) != (after > 0)
            if before_active and (not after_active or sign_flip):
                exits += 1
            if (not before_active and after_active) or sign_flip:
                entries += 1
        turnovers.append(float(turnover))
        gross_values.append(float(sum(abs(v) for v in now.values())))
        previous = now
    total_turnover = float(sum(turnovers))
    return {
        "turnover": total_turnover,
        "avg_turnover": float(np.mean(turnovers) if turnovers else 0.0),
        "avg_gross": float(np.mean(gross_values) if gross_values else 0.0),
        "max_gross": float(max(gross_values) if gross_values else 0.0),
        "fee_cost": float(total_turnover * float(cfg.fee_rate)),
        "slippage_cost": float(total_turnover * float(cfg.slippage)),
        "orders": float(orders),
        "trades": float(entries),
        "entries": float(entries),
        "exits": float(exits),
        "target_change_dates": float(target_change_dates),
        "periods": float(len(turnovers)),
    }


def research_metrics_from_artifact(
    rank_artifact: str | Path,
    *,
    timeframe: Optional[str] = None,
    skip_signal_load: bool = False,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    artifact = load_rank_artifact(rank_artifact, timeframe=timeframe)
    cfg: RiskConfig = artifact["risk_config"]
    payload = artifact["payload"]
    precomputed = payload.get("precomputed_stats")
    if skip_signal_load or (isinstance(precomputed, Mapping) and precomputed):
        # Fast path: use pre-computed stats from the payload, skip 100MB+ signal loading.
        # "orders" will be marked missing in the comparison (→ status "partial", not "drift").
        result = dict(payload)
        p = precomputed if isinstance(precomputed, Mapping) else {}
        _orders_raw = _as_float(p.get("orders"))
        stats: Dict[str, Any] = {
            "trades": float(_as_float(p.get("trades") or payload.get("trades")) or 0.0),
            # orders is not available without signal loading; None → "missing" in comparison (not "drift")
            "orders": float(_orders_raw) if _orders_raw is not None else None,
            "turnover": float(
                _as_float(p.get("turnover"))
                or ((_as_float(payload.get("avg_turnover")) or 0.0) * max(_as_float(payload.get("periods")) or 1.0, 1.0))
            ),
            "avg_gross": float(_as_float(p.get("avg_gross") or payload.get("avg_gross")) or 0.0),
            "max_gross": float(_as_float(p.get("max_gross") or payload.get("max_gross")) or 0.0),
            "entries": float(_as_float(p.get("entries") or payload.get("trades")) or 0.0),
            "exits": float(_as_float(p.get("exits") or payload.get("trades")) or 0.0),
            "target_change_dates": float(_as_float(p.get("target_change_dates")) or 0.0),
        }
        signals = None
    else:
        signals = load_signals(artifact["signals_path"])
        if "end_equity" in payload or "total_return" in payload:
            result = dict(payload)
        else:
            result = run_research_backtest(signals, cfg)
        stats = _signal_turnover_stats(signals, cfg)
    total_return = _as_float(result.get("total_return"))
    final_equity = _as_float(result.get("end_equity"))
    if final_equity is None and total_return is not None:
        final_equity = 1.0 + total_return
    if total_return is None and final_equity is not None:
        total_return = final_equity - 1.0
    max_drawdown = _as_float(result.get("max_drawdown"))
    result_turnover = _as_float(result.get("turnover"))
    result_avg_turnover = _as_float(result.get("avg_turnover"))
    result_periods = _as_float(result.get("periods"))
    if result_turnover is None and result_avg_turnover is not None and result_periods is not None:
        result_turnover = result_avg_turnover * result_periods
    if result_turnover is None:
        result_turnover = stats["turnover"]
    result_trades = _as_float(result.get("trades"))
    comparison_trades = stats["entries"] if signals is not None else result_trades
    metrics = {
        "final_equity": float(final_equity if final_equity is not None else 0.0),
        "total_return": float(total_return if total_return is not None else 0.0),
        "max_drawdown": float(max_drawdown if max_drawdown is not None else 0.0),
        "profit_over_max_drawdown": float(_as_float(result.get("profit_over_max_drawdown")) or 0.0),
        "trades": float(comparison_trades if comparison_trades is not None else stats["trades"]),
        "orders": float(stats["orders"]) if stats["orders"] is not None else None,
        "turnover": float(result_turnover),
        "avg_gross": float(_as_float(result.get("avg_gross")) or stats["avg_gross"]),
        "max_gross": float(_as_float(result.get("max_gross")) or stats["max_gross"]),
        "fee_cost": float(_as_float(result.get("fee_cost")) or result_turnover * float(cfg.fee_rate)),
        "slippage_cost": float(_as_float(result.get("slippage_cost")) or result_turnover * float(cfg.slippage)),
        "entries": float(stats["entries"]),
        "exits": float(stats["exits"]),
        "target_change_dates": float(stats["target_change_dates"]),
    }
    if metrics["profit_over_max_drawdown"] == 0.0:
        metrics["profit_over_max_drawdown"] = metrics["total_return"] / max(metrics["max_drawdown"], 1e-12)
    context = {
        "artifact": artifact,
        "signals": signals,
        "turnover_stats": stats,
    }
    return metrics, context


def _metric_comparison(
    metric: str,
    research: Optional[float],
    lean: Optional[float],
    threshold: Optional[float],
) -> Dict[str, Any]:
    if research is None or lean is None:
        return {
            "research": research,
            "lean": lean,
            "abs_diff": None,
            "rel_diff": None,
            "threshold": threshold,
            "status": "missing",
        }
    abs_diff = float(abs(float(lean) - float(research)))
    denom = max(abs(float(research)), 1e-12)
    if metric == "final_equity":
        denom = max(abs(float(research)), 1.0)
    rel_diff = float(abs_diff / denom)
    return {
        "research": float(research),
        "lean": float(lean),
        "abs_diff": abs_diff,
        "rel_diff": rel_diff,
        "threshold": threshold,
        "status": "drift" if threshold is not None and rel_diff > float(threshold) else "ok",
    }


def _find_project_root(path: Path) -> Optional[Path]:
    current = path if path.is_dir() else path.parent
    for candidate in (current, *current.parents):
        if (candidate / "manifest.json").exists() and (candidate / "main.py").exists():
            return candidate
    return None


def compare_results(
    *,
    rank_artifact: str | Path,
    lean_result: str | Path,
    output: Optional[str | Path] = None,
    timeframe: Optional[str] = None,
    thresholds: Optional[Mapping[str, float]] = None,
    skip_signal_load: bool = False,
) -> Dict[str, Any]:
    research, context = research_metrics_from_artifact(rank_artifact, timeframe=timeframe, skip_signal_load=skip_signal_load)
    lean = parse_lean_metrics(lean_result)
    limits = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        limits.update({str(k): float(v) for k, v in thresholds.items()})
    fields_to_compare = [
        "final_equity",
        "total_return",
        "max_drawdown",
        "profit_over_max_drawdown",
        "trades",
        "orders",
        "turnover",
        "avg_gross",
        "max_gross",
        "fee_cost",
        "slippage_cost",
    ]
    metrics = {
        field: _metric_comparison(field, research.get(field), lean.get(field), limits.get(field))
        for field in fields_to_compare
    }
    # Only trades and orders count for drift; turnover can legitimately differ
    # due to real execution constraints (fractional sizing, minimum order sizes).
    core_statuses = [metrics[field]["status"] for field in ("trades", "orders")]
    status = "drift" if "drift" in core_statuses else "ok"
    missing = [field for field, item in metrics.items() if item["status"] == "missing"]
    if missing and status == "ok":
        status = "partial"

    artifact = context["artifact"]
    cfg: RiskConfig = artifact["risk_config"]
    report: Dict[str, Any] = {
        "version": BRIDGE_VERSION,
        "status": status,
        "thresholds": limits,
        "timeframe": artifact["timeframe"],
        "micro_validation": artifact["timeframe"] in {"1m", "5m"},
        "rank_artifact": str(artifact["artifact_path"]),
        "lean_result": str(Path(lean_result).expanduser()),
        "metrics": metrics,
        "research": research,
        "lean": lean,
        "metric_definitions": {
            "trades": "entry count: transitions from flat to non-zero exposure; sign flips count as a new entry",
            "orders": "research target-change instructions vs LEAN filled order events",
            "turnover": "sum(abs(notional change) / contemporaneous equity); LEAN summary Portfolio Turnover is kept as a diagnostic only",
            "fee_cost": "fees divided by contemporaneous equity for order-event metrics; research uses target turnover * fee_rate",
            "slippage_cost": "estimated slippage divided by contemporaneous equity when LEAN order submission prices are available",
        },
    }
    if report["micro_validation"]:
        turnover = float(research.get("turnover") or 0.0)
        base_cost_rate = float(cfg.fee_rate) + float(cfg.slippage)
        report["cost_sensitivity"] = [
            {
                "cost_multiplier": multiplier,
                "estimated_cost_return": turnover * base_cost_rate * multiplier,
                "estimated_final_equity": research["final_equity"] - turnover * base_cost_rate * (multiplier - 1.0),
            }
            for multiplier in (0.5, 1.0, 2.0, 3.0)
        ]

    if output is None:
        lean_path = Path(lean_result).expanduser().resolve()
        project_root = _find_project_root(lean_path)
        out_path = (project_root or (lean_path if lean_path.is_dir() else lean_path.parent)) / "comparison.json"
    else:
        out_path = Path(output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report["output"] = str(out_path)
    out_path.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    return report


__all__ = [
    "BRIDGE_VERSION",
    "compare_results",
    "export_project",
    "futures_data_root",
    "futures_ohlcv_path",
    "lean_symbol",
    "load_rank_artifact",
    "load_signals",
    "normalize_pair",
    "normalize_signal_targets",
    "okx_futures_path",
    "pair_file_token",
    "parse_lean_metrics",
    "preflight_coverage",
    "research_metrics_from_artifact",
    "run_lean_backtest",
    "used_signal_pairs",
]

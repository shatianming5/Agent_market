from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from agent_market.utils import write_json
from agent_market.config import FreqAISettings
from agent_market.microstructure.ohlcv_features import build_ohlcv_micro_features
from agent_market.microstructure.features.feature_registry import (
    FeatureContext,
    build_default_microstructure_registry,
    compute_registry_features,
)
from agent_market.microstructure.features.ofi_features import (
    align_trade_rollups_to_lob,
    compute_l2_delta_ofi,
    compute_trade_rollups,
    load_kucoin_match_trades,
)


def _resolve(root: Path, path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (root / p).resolve()


@dataclass(frozen=True, slots=True)
class MicroFeatureOutputs:
    features_parquet: Path
    manifest_json: Path


def generate_micro_features_from_freqtrade_config(
    *,
    root: Path,
    freqtrade_config: Path,
    run_id: str,
    out_dir: Path,
    timeframe: Optional[str] = None,
    timerange: Optional[str] = None,
    pairs: Optional[Sequence[str]] = None,
    data_dir: Optional[str | Path] = None,
) -> MicroFeatureOutputs:
    """Generate OHLCV-based micro features.

    Inputs:
    - freqtrade_config: freqtrade JSON config (used to infer exchange/pairs/datadir/timeframe)
    - timerange: YYYYMMDD-YYYYMMDD (optional). When omitted, uses all rows in feather.

    Outputs (in out_dir):
    - features.parquet
    - manifest.json
    """
    root = Path(root).resolve()
    cfg_path = _resolve(root, freqtrade_config)
    settings = FreqAISettings.from_file(cfg_path, timeframe_override=timeframe)

    exchange = settings.exchange
    tf = timeframe or settings.timeframe
    pair_list = [str(p) for p in (pairs or settings.pairs) if str(p).strip()]

    datadir_root = Path(data_dir) if data_dir is not None else settings.data_dir.parent
    if not datadir_root.is_absolute():
        datadir_root = (root / datadir_root).resolve()
    ex_dir = (datadir_root / exchange).resolve()

    out_dir = _resolve(root, out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    features_path = (out_dir / "features.parquet").resolve()
    manifest_path = (out_dir / "manifest.json").resolve()

    df, manifest = build_ohlcv_micro_features(
        ex_dir=ex_dir,
        exchange=exchange,
        pairs=pair_list,
        timeframe=tf,
        timerange=timerange,
    )
    df.to_parquet(features_path, index=False)

    manifest_payload: Dict[str, Any] = {
        "version": 1,
        "kind": "micro_feature",
        "run_id": str(run_id),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "freqtrade_config": str(cfg_path),
            "exchange": exchange,
            "pairs": pair_list,
            "timeframe": tf,
            "timerange": timerange,
            "data_dir": str(datadir_root),
        },
        "data_sources": ["ohlcv_feather"],
        "rows": int(df.shape[0]),
        "columns": [str(c) for c in df.columns],
        "features": manifest,
        "artifacts": {
            "features_parquet": str(features_path),
        },
    }
    write_json(manifest_path, manifest_payload)
    return MicroFeatureOutputs(features_parquet=features_path, manifest_json=manifest_path)


def generate_microstructure_features_from_lob_and_match(
    *,
    root: Path,
    run_id: str,
    out_dir: Path,
    lob_state: Path,
    match_path: Path,
    symbol: Optional[str] = None,
    depth_levels: int = 20,
    windows_sec: Sequence[int] = (10,),
) -> MicroFeatureOutputs:
    """Generate microstructure features from LOB state + trades."""
    try:
        import pandas as pd  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Missing dependency for microstructure features: {exc}") from exc

    root = Path(root).resolve()
    out_dir = _resolve(root, out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    features_path = (out_dir / "features.parquet").resolve()
    manifest_path = (out_dir / "manifest.json").resolve()

    lob_state = _resolve(root, lob_state)
    match_path = _resolve(root, match_path)
    if not lob_state.exists():
        raise FileNotFoundError(f"lob_state parquet not found: {lob_state}")
    if not match_path.exists():
        raise FileNotFoundError(f"match.ndjson.gz not found: {match_path}")

    lob = pd.read_parquet(lob_state)
    required_lob = {"ts", "symbol", "event", "bid1", "ask1", "mid", "spread", "rel_spread", "bid_sz", "ask_sz"}
    missing = required_lob - set(lob.columns)
    if missing:
        raise ValueError(f"lob_state missing columns: {sorted(missing)}")
    lob["ts"] = pd.to_datetime(lob["ts"], utc=True).dt.tz_convert(None)
    lob = lob.sort_values(["ts", "event", "symbol"]).reset_index(drop=True)

    sym = str(symbol).strip() if symbol else None
    if sym:
        lob = lob.loc[lob["symbol"].astype(str) == sym].reset_index(drop=True)

    trades = load_kucoin_match_trades(match_path=match_path, symbol=sym)
    rollups = compute_trade_rollups(trades=trades, windows_sec=windows_sec)
    aligned = align_trade_rollups_to_lob(lob=lob, trade_rollups=rollups)

    registry = build_default_microstructure_registry(depth_levels=int(depth_levels), windows_sec=windows_sec)
    ctx = FeatureContext(lob=lob, trades=trades, aligned_trades=aligned)
    df = compute_registry_features(ctx=ctx, registry=registry)

    # L2 delta-OFI (plan.md §4.2): compute from LOB state changes
    try:
        l2_ofi = compute_l2_delta_ofi(lob, windows_sec=windows_sec)
        for col in l2_ofi.columns:
            if col not in ("ts", "symbol") and col not in df.columns:
                df[col] = l2_ofi[col].values[:len(df)] if len(l2_ofi) >= len(df) else None
    except Exception:
        pass  # graceful degradation if LOB lacks required columns

    df.to_parquet(features_path, index=False)

    manifest_payload: Dict[str, Any] = {
        "version": 1,
        "kind": "micro_feature",
        "run_id": str(run_id),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "lob_state": str(lob_state),
            "match_path": str(match_path),
            "symbol": sym,
            "depth_levels": int(depth_levels),
            "windows_sec": [int(x) for x in windows_sec],
        },
        "data_sources": ["lob_state_parquet", "match_ndjson_gz"],
        "rows": int(df.shape[0]),
        "columns": [str(c) for c in df.columns],
        "features": registry.manifest(),
        "artifacts": {
            "features_parquet": str(features_path),
        },
    }
    write_json(manifest_path, manifest_payload)
    return MicroFeatureOutputs(features_parquet=features_path, manifest_json=manifest_path)


__all__ = [
    "MicroFeatureOutputs",
    "generate_micro_features_from_freqtrade_config",
    "generate_microstructure_features_from_lob_and_match",
]

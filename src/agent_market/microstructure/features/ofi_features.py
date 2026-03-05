from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from ..ms_utils import ensure_pandas, iter_ndjson_gz, topic_symbol


def _parse_trade_ts(msg: Dict[str, Any]) -> Optional[datetime]:
    data = msg.get("data") if isinstance(msg.get("data"), dict) else {}
    ts_ms = data.get("time")
    if ts_ms is not None:
        try:
            return datetime.fromtimestamp(float(ts_ms) / 1000.0, tz=timezone.utc)
        except Exception:
            pass
    recv = msg.get("_received_at")
    if recv:
        try:
            return datetime.fromisoformat(str(recv).replace("Z", "+00:00"))
        except Exception:
            pass
    return None


def load_kucoin_match_trades(*, match_path: Path, symbol: Optional[str] = None) -> Any:
    """Load KuCoin match.ndjson.gz into a dataframe.

    Output columns:
      ts, symbol, side, price, size
    """
    pd = ensure_pandas()
    rows = []
    sym_filter = str(symbol).strip() if symbol else None

    for msg in iter_ndjson_gz(Path(match_path)):
        if msg.get("type") != "message":
            continue
        topic = str(msg.get("topic") or "")
        sym = topic_symbol(topic)
        data = msg.get("data")
        if not sym or not isinstance(data, dict):
            continue
        if sym_filter and sym != sym_filter:
            continue
        ts = _parse_trade_ts(msg)
        if ts is None:
            continue
        side = str(data.get("side") or "").strip().lower()
        try:
            size = float(data.get("size") or 0.0)
            price = float(data.get("price") or 0.0)
        except Exception:
            continue
        if size <= 0 or price <= 0:
            continue
        rows.append({"ts": ts, "symbol": sym, "side": side, "size": size, "price": price})

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(None)
    df = df.sort_values(["symbol", "ts"]).reset_index(drop=True)
    return df


def _trade_sign(side: Any) -> Optional[int]:
    s = str(side or "").strip().lower()
    if s == "buy":
        return 1
    if s == "sell":
        return -1
    return None


def compute_trade_rollups(*, trades: Any, windows_sec: Sequence[int], eps: float = 1e-12) -> Any:
    """Compute rolling trade rollups at trade timestamps (per symbol)."""
    pd = ensure_pandas()
    if trades is None or getattr(trades, "empty", True):
        return pd.DataFrame()

    df = trades.copy()
    df["trade_sign"] = df["side"].apply(_trade_sign).astype("float64")
    df["signed_size"] = df["size"].astype("float64") * df["trade_sign"].astype("float64")
    df["notional"] = df["price"].astype("float64") * df["size"].astype("float64")

    out_frames = []
    for sym, g in df.groupby("symbol", sort=False):
        g = g.sort_values("ts").set_index("ts")
        base = g[["symbol", "trade_sign"]].copy()
        for w in windows_sec:
            w = int(w)
            if w <= 0:
                continue
            win = f"{w}s"

            vol = g["size"].rolling(win).sum()
            signed_vol = g["signed_size"].rolling(win).sum()
            notional = g["notional"].rolling(win).sum()
            cnt = g["size"].rolling(win).count()

            buy_vol = g["size"].where(g["trade_sign"] > 0.0, 0.0).rolling(win).sum()
            sell_vol = g["size"].where(g["trade_sign"] < 0.0, 0.0).rolling(win).sum()

            base[f"vwap_{w}"] = notional / (vol + float(eps))
            base[f"ofi_{w}"] = signed_vol / (vol + float(eps))
            base[f"arrival_intensity_{w}"] = cnt / float(w)
            base[f"buy_vol_{w}"] = buy_vol
            base[f"sell_vol_{w}"] = sell_vol
        base = base.reset_index()
        out_frames.append(base)

    if not out_frames:
        return pd.DataFrame()
    out = pd.concat(out_frames, axis=0, ignore_index=True)
    out = out.sort_values(["symbol", "ts"]).reset_index(drop=True)
    return out


def align_trade_rollups_to_lob(*, lob: Any, trade_rollups: Any) -> Any:
    """Align trade rollups to lob timestamps via backward asof join (per symbol)."""
    pd = ensure_pandas()
    if trade_rollups is None or getattr(trade_rollups, "empty", True):
        return pd.DataFrame(index=lob.index).assign(**{})

    left = lob[["ts", "symbol"]].copy()
    left["ts"] = pd.to_datetime(left["ts"], utc=True).dt.tz_convert(None)

    right = trade_rollups.copy()
    right["ts"] = pd.to_datetime(right["ts"], utc=True).dt.tz_convert(None)
    right = right.sort_values(["symbol", "ts"])

    aligned = pd.merge_asof(
        left.sort_values(["symbol", "ts"]),
        right,
        on="ts",
        by="symbol",
        direction="backward",
        allow_exact_matches=True,
    )
    aligned = aligned.sort_index()
    # Drop duplicated key columns from right.
    for col in ["symbol_y"]:
        if col in aligned.columns:
            aligned = aligned.drop(columns=[col])
    if "symbol_x" in aligned.columns:
        aligned = aligned.rename(columns={"symbol_x": "symbol"})
    return aligned.reset_index(drop=True)


__all__ = [
    "align_trade_rollups_to_lob",
    "compute_trade_rollups",
    "load_kucoin_match_trades",
]

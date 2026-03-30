from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent_market.utils import write_json

logger = logging.getLogger(__name__)
from agent_market.tca.adapters.freqtrade import load_freqtrade_backtest_trades
from agent_market.tca.metrics import summarize_trades
from agent_market.tca.schema import (
    ArtifactRef,
    TCABenchmark,
    TCABenchmarks,
    TCACosts,
    TCACostValue,
    TCAFillQuality,
    TCAImplementationShortfall,
    TCAImplementationShortfallComponents,
    TCAMeta,
    TCAQuantiles,
    TCAReport,
    TCASlippageDistribution,
    TCASource,
    TCAStrategyMeta,
    TCASummary,
    TCATimeRange,
 )


def _ms_to_iso(ms: Any) -> Optional[str]:
    try:
        if ms is None:
            return None
        value = float(ms) / 1000.0
        return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()
    except (ValueError, TypeError, OverflowError, OSError):
        logger.debug("Failed to convert ms timestamp: %r", ms)
        return None


def _pair_slug(pair: str) -> str:
    return str(pair).replace("/", "_")


def _find_ohlcv_feather(
    *,
    root: Path,
    slug: str,
    timeframe: str,
    exchange: Optional[str],
) -> Optional[Path]:
    base_dir = (Path(root) / "user_data" / "data").resolve()
    if exchange:
        candidate = (base_dir / str(exchange) / f"{slug}-{timeframe}.feather").resolve()
        if candidate.exists():
            return candidate

    if base_dir.exists():
        for ex_dir in sorted(p for p in base_dir.iterdir() if p.is_dir()):
            candidate = (ex_dir / f"{slug}-{timeframe}.feather").resolve()
            if candidate.exists():
                return candidate

    return None


def _compute_participation_proxy(
    fills: List[Dict[str, Any]],
    *,
    root: Optional[Path],
    timeframe: Optional[str],
    exchange: Optional[str],
) -> Dict[str, Any]:
    """Best-effort participation proxy based on OHLCV volume.

    Returns a stable object (key present) even when data is missing:
      { method, overall, per_symbol, executed_qty, market_qty }
    """

    out: Dict[str, Any] = {
        "method": "ohlcv_volume_proxy_v1",
        "overall": None,
        "per_symbol": {},
        "executed_qty": None,
        "market_qty": None,
    }
    if root is None or not fills or not timeframe:
        return out

    try:
        import pandas as pd  # noqa: PLC0415
    except ImportError:
        logger.warning("pandas not available; skipping participation proxy")
        return out

    symbols = sorted({str(f.get("symbol") or "").strip() for f in fills if str(f.get("symbol") or "").strip()})
    if not symbols:
        return out

    executed_total = 0.0
    market_total = 0.0
    per_symbol: Dict[str, float] = {}

    for symbol in symbols:
        rows = [f for f in fills if str(f.get("symbol") or "").strip() == symbol]
        if not rows:
            continue

        fill_ts: list[Any] = []
        qty: list[float] = []
        for r in rows:
            try:
                qty_f = float(r.get("qty") or 0.0)
            except (ValueError, TypeError):
                logger.debug("Skipping fill with invalid qty: %r", r.get("qty"))
                continue
            ts = r.get("ts")
            dt = pd.to_datetime(ts, errors="coerce")
            if dt is pd.NaT:
                continue
            fill_ts.append(dt)
            qty.append(qty_f)

        if not fill_ts:
            continue

        fills_df = pd.DataFrame({"fill_ts": fill_ts, "qty": qty})
        tcol = pd.to_datetime(fills_df["fill_ts"], errors="coerce")
        try:
            if getattr(tcol.dt, "tz", None) is not None:
                tcol = tcol.dt.tz_convert(None)
        except Exception:
            logger.debug("tz_convert failed for fill timestamps of %s", symbol)
        fills_df["fill_ts"] = tcol
        fills_df = fills_df.dropna(subset=["fill_ts"]).sort_values("fill_ts").reset_index(drop=True)
        if fills_df.empty:
            continue

        slug = _pair_slug(symbol)
        fp = _find_ohlcv_feather(root=Path(root), slug=slug, timeframe=str(timeframe), exchange=exchange)
        if fp is None:
            continue

        try:
            ohlcv = pd.read_feather(fp)
        except Exception:
            logger.warning("Failed to read OHLCV feather: %s", fp)
            continue
        if "date" not in ohlcv.columns or "volume" not in ohlcv.columns:
            continue
        ohlcv = ohlcv[["date", "volume"]].copy()
        dcol = pd.to_datetime(ohlcv["date"], errors="coerce")
        try:
            if getattr(dcol.dt, "tz", None) is not None:
                dcol = dcol.dt.tz_convert(None)
        except Exception:
            logger.debug("tz_convert failed for OHLCV dates of %s", symbol)
        ohlcv["date"] = dcol
        ohlcv["volume"] = pd.to_numeric(ohlcv["volume"], errors="coerce")
        ohlcv = ohlcv.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        if ohlcv.empty:
            continue

        merged = pd.merge_asof(
            fills_df,
            ohlcv[["date", "volume"]],
            left_on="fill_ts",
            right_on="date",
            direction="backward",
            allow_exact_matches=True,
        )
        merged = merged.dropna(subset=["date"])
        if merged.empty:
            continue

        start_dt = merged["date"].min()
        end_dt = merged["date"].max()
        window = ohlcv.loc[(ohlcv["date"] >= start_dt) & (ohlcv["date"] <= end_dt)]
        try:
            market_qty = float(pd.to_numeric(window["volume"], errors="coerce").fillna(0.0).sum())
        except Exception:
            logger.debug("Failed to compute market volume for %s", symbol)
            market_qty = 0.0
        executed_qty = float(pd.to_numeric(merged["qty"], errors="coerce").abs().fillna(0.0).sum())
        if market_qty <= 0:
            continue

        ratio = executed_qty / market_qty
        per_symbol[symbol] = float(ratio)
        executed_total += executed_qty
        market_total += market_qty

    out["per_symbol"] = per_symbol
    if market_total > 0:
        out["executed_qty"] = float(executed_total)
        out["market_qty"] = float(market_total)
        out["overall"] = float(executed_total / market_total)
    return out


def _extract_orders_fills_from_freqtrade_trades(
    trades: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Extract best-effort orders/fills from freqtrade backtest trades.

    Many freqtrade backtest JSONs include `trade['orders']` with entry/exit fills.
    We map them into the plan.md v1 positions (`orders[]`, `fills[]`).
    """

    orders: List[Dict[str, Any]] = []
    fills: List[Dict[str, Any]] = []

    for ti, trade in enumerate(trades):
        pair = str(trade.get("pair") or "").strip() or "UNKNOWN"
        raw_orders = trade.get("orders") if isinstance(trade.get("orders"), list) else []
        for oi, raw in enumerate(raw_orders):
            if not isinstance(raw, dict):
                continue
            side_raw = str(raw.get("ft_order_side") or "").strip().lower()
            side = "BUY" if side_raw == "buy" else ("SELL" if side_raw == "sell" else "BUY")

            try:
                qty = float(raw.get("amount") or 0.0)
                price = float(raw.get("safe_price") or 0.0)
            except (ValueError, TypeError):
                logger.debug("Skipping order with invalid amount/price in trade %d", ti)
                continue
            if qty <= 0 or price <= 0:
                continue

            ts = _ms_to_iso(raw.get("order_filled_timestamp")) or _ms_to_iso(trade.get("open_timestamp")) or None
            order_id = f"t{ti:06d}_o{oi:02d}"
            orders.append(
                {
                    "order_id": order_id,
                    "symbol": pair,
                    "side": side,
                    "type": "MKT",
                    "qty": qty,
                    "limit_px": None,
                    "submit_ts": ts,
                    "cancel_ts": None,
                }
            )
            fills.append(
                {
                    "order_id": order_id,
                    "fill_id": f"{order_id}_f0",
                    "symbol": pair,
                    "side": side,
                    "qty": qty,
                    "price": price,
                    "ts": ts or datetime.now(timezone.utc).isoformat(),
                    "fee": None,
                    "liquidity": "TAKER",
                }
            )

    return orders, fills


def generate_tca_report(
    backtest_zip: Path,
    *,
    run_id: str,
    out_path: Path,
    root: Optional[Path] = None,
    html_path: Optional[Path] = None,
) -> Dict[str, Any]:
    strategy, trades, meta = load_freqtrade_backtest_trades(backtest_zip)
    summary, distributions, per_pair = summarize_trades(trades)
    orders, fills = _extract_orders_fills_from_freqtrade_trades(trades)

    generated_at = datetime.now(timezone.utc).isoformat()
    symbols = sorted({str(t.get("pair") or "").strip() for t in trades if str(t.get("pair") or "").strip()})
    time_range = TCATimeRange(
        start=(str(meta.get("backtest_start")) if meta.get("backtest_start") is not None else None),
        end=(str(meta.get("backtest_end")) if meta.get("backtest_end") is not None else None),
    )
    meta_block = TCAMeta(
        run_id=str(run_id),
        generated_at=generated_at,
        exchange=(str(meta.get("exchange")) if meta.get("exchange") else None),
        market=(str(meta.get("market")) if meta.get("market") else None),
        symbols=symbols,
        time_range=time_range,
        timeframe=(str(meta.get("timeframe")) if meta.get("timeframe") else None),
        data_sources=["freqtrade_backtest"],
        strategy=TCAStrategyMeta(name=str(strategy), params={}),
    )

    fees_total = summary.get("fees_total")
    try:
        fees_total_f = float(fees_total) if fees_total is not None else None
    except (ValueError, TypeError):
        logger.debug("Non-numeric fees_total: %r", fees_total)
        fees_total_f = None

    total_notional = 0.0
    for f in fills:
        try:
            total_notional += abs(float(f.get("qty") or 0.0) * float(f.get("price") or 0.0))
        except (ValueError, TypeError):
            continue
    fees_bps = None
    if fees_total_f is not None and total_notional > 0:
        fees_bps = float(10000.0 * float(fees_total_f) / float(total_notional))

    is_components = TCAImplementationShortfallComponents(
        spread=TCACostValue(bps=0.0),
        delay=TCACostValue(bps=0.0),
        market_impact=TCACostValue(bps=0.0),
        fees=TCACostValue(bps=fees_bps, quote_ccy=fees_total_f),
    )
    is_block = TCAImplementationShortfall(
        total=TCACostValue(bps=fees_bps, quote_ccy=fees_total_f),
        by_component=is_components,
    )

    fill_quality = TCAFillQuality(
        fill_rate=1.0 if orders else None,
        avg_fill_latency_ms=0.0 if orders else None,
        cancel_rate=0.0 if orders else None,
    )
    slippage_dist = TCASlippageDistribution(
        p50_bps=0.0 if orders else None,
        p90_bps=0.0 if orders else None,
        p99_bps=0.0 if orders else None,
    )
    costs_block = TCACosts(
        implementation_shortfall=is_block,
        slippage_distribution=slippage_dist,
        fill=fill_quality,
    )

    artifacts = [ArtifactRef(name="tca_report", path=str(out_path))]
    if html_path is not None:
        artifacts.append(ArtifactRef(name="tca_html", path=str(html_path)))

    timeframe_used = str(meta.get("timeframe") or meta.get("timeframe_detail") or "1h") if isinstance(meta, dict) else "1h"
    exchange_used = str(meta.get("exchange")) if isinstance(meta, dict) and meta.get("exchange") else None
    participation = _compute_participation_proxy(
        fills,
        root=root,
        timeframe=timeframe_used,
        exchange=exchange_used,
    )

    payload = TCAReport(
        meta=meta_block,
        orders=orders,
        fills=fills,
        benchmarks=TCABenchmarks(
            arrival_mid=TCABenchmark(definition="mid at submit_ts", value_series_ref=None),
            vwap=TCABenchmark(definition="VWAP over exec window", value_series_ref=None),
        ),
        costs=costs_block,
        diagnostics={"participation": participation},
        run_id=str(run_id),
        generated_at=generated_at,
        source=TCASource(backtest_zip=str(backtest_zip), strategy=str(strategy), meta=meta),
        summary=TCASummary(**summary),
        distributions={k: TCAQuantiles(**v) for k, v in distributions.items()},
        per_pair=per_pair,
        artifacts=artifacts,
    )
    try:
        payload = payload.model_dump()  # pydantic v2
    except AttributeError:  # pragma: no cover
        payload = payload.dict()  # pydantic v1

    if root is not None:
        try:
            root_resolved = root.resolve()

            def _rel(p: Path) -> str:
                try:
                    return str(p.resolve().relative_to(root_resolved))
                except ValueError:
                    return str(p.resolve())

            payload["source"]["backtest_zip"] = _rel(Path(payload["source"]["backtest_zip"]))
            for a in payload.get("artifacts") or []:
                a["path"] = _rel(Path(a["path"]))
        except Exception:
            logger.warning("Failed to relativize paths in TCA report", exc_info=True)

    write_json(out_path, payload)

    if html_path is not None:
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text(
            "<html><head><meta charset='utf-8'><title>TCA Report</title></head>"
            "<body><pre>"
            + json.dumps(payload, ensure_ascii=False, indent=2)
            + "</pre></body></html>",
            encoding="utf-8",
        )

    return payload


__all__ = ["generate_tca_report"]

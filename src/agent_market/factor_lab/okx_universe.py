"""OKX futures universe discovery and manifest helpers."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import requests

from .paths import USER_DATA

OKX_API = "https://www.okx.com"
UNIVERSE_DIR = USER_DATA / "data" / "okx" / "universes"


@dataclass(frozen=True)
class OkxInstrument:
    pair: str
    inst_id: str
    base: str
    list_time_ms: int
    list_date: str
    has_spot_usdt: bool
    quote_volume_24h: float
    last: float
    vol_ccy_24h: float
    ct_val: float
    ct_val_ccy: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _date_to_ms(value: str) -> int:
    return int(datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


def _ms_to_date(value: int) -> str:
    return datetime.fromtimestamp(int(value) / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d")


def _request_okx(path: str, params: Mapping[str, str]) -> list[dict[str, Any]]:
    r = requests.get(f"{OKX_API}{path}", params=dict(params), timeout=30)
    r.raise_for_status()
    payload = r.json()
    if payload.get("code") != "0":
        raise RuntimeError(f"OKX {path} code={payload.get('code')} msg={payload.get('msg')}")
    return list(payload.get("data") or [])


def fetch_okx_usdt_swap_instruments() -> list[OkxInstrument]:
    """Fetch live OKX USDT-SWAP instruments with simple liquidity metadata."""
    swaps = _request_okx("/api/v5/public/instruments", {"instType": "SWAP"})
    spots = _request_okx("/api/v5/public/instruments", {"instType": "SPOT"})
    tickers = _request_okx("/api/v5/market/tickers", {"instType": "SWAP"})
    spot_ids = {item.get("instId") for item in spots if item.get("state") == "live"}
    ticker_by_id = {item.get("instId"): item for item in tickers}
    out: list[OkxInstrument] = []
    for item in swaps:
        inst_id = str(item.get("instId") or "")
        if item.get("settleCcy") != "USDT" or item.get("state") != "live" or not inst_id.endswith("-USDT-SWAP"):
            continue
        base = inst_id[: -len("-USDT-SWAP")]
        ticker = ticker_by_id.get(inst_id, {})
        last = float(ticker.get("last") or 0.0)
        vol_ccy_24h = float(ticker.get("volCcy24h") or 0.0)
        list_time_ms = int(item.get("listTime") or 0)
        out.append(
            OkxInstrument(
                pair=f"{base}/USDT",
                inst_id=inst_id,
                base=base,
                list_time_ms=list_time_ms,
                list_date=_ms_to_date(list_time_ms) if list_time_ms else "",
                has_spot_usdt=f"{base}-USDT" in spot_ids,
                quote_volume_24h=float(last * vol_ccy_24h),
                last=last,
                vol_ccy_24h=vol_ccy_24h,
                ct_val=float(item.get("ctVal") or 0.0),
                ct_val_ccy=str(item.get("ctValCcy") or ""),
            )
        )
    return sorted(out, key=lambda row: row.quote_volume_24h, reverse=True)


def select_okx_universe(
    name: str,
    *,
    full_history_start: str = "2025-04-12",
    top_n: int | None = None,
) -> tuple[list[OkxInstrument], dict[str, Any]]:
    """Return selected OKX futures instruments and manifest metadata."""
    instruments = fetch_okx_usdt_swap_instruments()
    start_ms = _date_to_ms(full_history_start)
    key = name.strip().lower().replace("-", "_")
    if key in {"core", "core160", "core_160", "okx_core_160"}:
        selected = [row for row in instruments if row.list_time_ms <= start_ms][: int(top_n or 160)]
        universe_name = "okx_core_160"
        rule = (
            "top live OKX USDT-SWAP instruments by approximate quote_volume_24h "
            f"with list_time <= {full_history_start}; cap=160"
        )
        dynamic = False
    elif key in {"top200", "top_200", "top200_dynamic", "okx_top200_dynamic"}:
        selected = instruments[: int(top_n or 200)]
        universe_name = "okx_top200_dynamic"
        rule = "top 200 live OKX USDT-SWAP instruments by approximate quote_volume_24h; dynamic listing dates"
        dynamic = True
    elif key in {"all", "all303", "all_raw", "okx_all303_raw"}:
        selected = instruments[: int(top_n or len(instruments))]
        universe_name = "okx_all303_raw"
        rule = "all live OKX USDT-SWAP instruments by approximate quote_volume_24h; raw research universe"
        dynamic = True
    else:
        raise ValueError("unknown OKX universe; expected core_160, top200_dynamic, or all_raw")
    metadata = {
        "name": universe_name,
        "requested_name": name,
        "as_of_utc": _utc_now(),
        "full_history_start": full_history_start,
        "selection_rule": rule,
        "dynamic_listings": dynamic,
        "instrument_count_available": len(instruments),
        "pair_count": len(selected),
        "spot_matched_count": int(sum(1 for row in selected if row.has_spot_usdt)),
        "min_list_date": min((row.list_date for row in selected if row.list_date), default=""),
        "max_list_date": max((row.list_date for row in selected if row.list_date), default=""),
    }
    return selected, metadata


def write_okx_universe_manifest(
    name: str,
    *,
    full_history_start: str = "2025-04-12",
    out_dir: Path = UNIVERSE_DIR,
    top_n: int | None = None,
) -> Path:
    selected, metadata = select_okx_universe(name, full_history_start=full_history_start, top_n=top_n)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{metadata['name']}.json"
    payload = {
        **metadata,
        "pairs": [row.pair for row in selected],
        "pair_start_dates": {row.pair: row.list_date for row in selected},
        "instruments": [asdict(row) for row in selected],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    txt = path.with_suffix(".pairs.txt")
    txt.write_text("\n".join(payload["pairs"]) + "\n", encoding="utf-8")
    return path


def build_default_okx_universe_manifests(*, full_history_start: str = "2025-04-12") -> list[Path]:
    return [
        write_okx_universe_manifest("core_160", full_history_start=full_history_start),
        write_okx_universe_manifest("top200_dynamic", full_history_start=full_history_start),
        write_okx_universe_manifest("all_raw", full_history_start=full_history_start),
    ]


def resolve_okx_universe_manifest(value: str | Path) -> Path:
    raw = str(value).strip()
    path = Path(raw)
    if path.exists():
        return path
    key = raw.lower().replace("-", "_")
    aliases = {
        "core": "okx_core_160.json",
        "core160": "okx_core_160.json",
        "core_160": "okx_core_160.json",
        "okx_core_160": "okx_core_160.json",
        "top200": "okx_top200_dynamic.json",
        "top_200": "okx_top200_dynamic.json",
        "top200_dynamic": "okx_top200_dynamic.json",
        "okx_top200_dynamic": "okx_top200_dynamic.json",
        "all": "okx_all303_raw.json",
        "all303": "okx_all303_raw.json",
        "all_raw": "okx_all303_raw.json",
        "okx_all303_raw": "okx_all303_raw.json",
    }
    if key in aliases:
        return UNIVERSE_DIR / aliases[key]
    return UNIVERSE_DIR / raw


def load_okx_universe_manifest(value: str | Path) -> dict[str, Any]:
    path = resolve_okx_universe_manifest(value)
    if not path.exists():
        raise FileNotFoundError(f"OKX universe manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("pairs"), list):
        raise ValueError(f"OKX universe manifest missing pairs: {path}")
    return payload


def manifest_pairs_and_starts(value: str | Path) -> tuple[list[str], dict[str, str]]:
    payload = load_okx_universe_manifest(value)
    pairs = [str(pair) for pair in payload.get("pairs") or []]
    starts_raw = payload.get("pair_start_dates") or {}
    starts = {str(k): str(v) for k, v in starts_raw.items() if v}
    return pairs, starts

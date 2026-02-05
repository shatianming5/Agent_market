#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT))

from agent_market import paths  # noqa: E402


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _resolve_path(path: str) -> Path:
    return paths.resolve_repo_path(path)


def _resolve_timeframe(cfg: Dict[str, Any], cli_timeframe: Optional[str]) -> str:
    if cli_timeframe:
        return cli_timeframe
    freqai = cfg.get("freqai") or {}
    fp = freqai.get("feature_parameters") or {}
    tfs = fp.get("include_timeframes") or []
    if isinstance(tfs, list) and tfs and isinstance(tfs[0], str) and tfs[0]:
        return tfs[0]
    tf = cfg.get("timeframe")
    return str(tf) if tf else "1h"


def _resolve_label_period(cfg: Dict[str, Any], override: Optional[int]) -> int:
    if override is not None:
        return int(override)
    freqai = cfg.get("freqai") or {}
    fp = freqai.get("feature_parameters") or {}
    return int(fp.get("label_period_candles") or 12)


def _resolve_pairs(cfg: Dict[str, Any], override: Optional[List[str]]) -> List[str]:
    if override:
        return [str(p) for p in override if str(p).strip()]
    exchange = cfg.get("exchange") or {}
    pairs = exchange.get("pair_whitelist") or []
    return [str(p) for p in pairs if str(p).strip()]


def build_preset_features() -> List[Dict[str, Any]]:
    """
    Keep this small and stable: the rest of the project (training, expression agent)
    expects the feature file to be predictable and easy to debug.
    """

    return [
        {"name": "rsi_14", "type": "rsi", "period": 14, "description": "RSI(14)"},
        {"name": "mfi_14", "type": "mfi", "period": 14, "description": "MFI(14)"},
        {"name": "adx_14", "type": "adx", "period": 14, "description": "ADX(14)"},
        {"name": "cci_20", "type": "cci", "period": 20, "description": "CCI(20)"},
        {"name": "ema_pct_12", "type": "ema_pct", "period": 12, "description": "EMA(12) / close - 1"},
        {"name": "ema_pct_48", "type": "ema_pct", "period": 48, "description": "EMA(48) / close - 1"},
        {"name": "kama_pct_20", "type": "kama_pct", "period": 20, "description": "KAMA(20) / close - 1"},
        {"name": "macd_diff", "type": "macd_diff", "description": "MACD - signal"},
        {"name": "stoch_k_14", "type": "stoch_k", "period": 14, "description": "Stochastic %K"},
        {"name": "stoch_d_14", "type": "stoch_d", "period": 14, "description": "Stochastic %D"},
        {"name": "psar_ratio", "type": "psar_ratio", "description": "SAR / close - 1"},
        {"name": "donchian_width_20", "type": "donchian_width", "period": 20, "description": "Donchian width / close"},
    ]


def build_preset_combos() -> List[Dict[str, Any]]:
    return [
        {
            "name": "ema_spread",
            "type": "spread",
            "sources": ["ema_pct_12", "ema_pct_48"],
            "formula": "ema_pct_12 - ema_pct_48",
        },
        {
            "name": "stoch_diff",
            "type": "spread",
            "sources": ["stoch_k_14", "stoch_d_14"],
            "formula": "stoch_k_14 - stoch_d_14",
        },
    ]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a small, stable FreqAI feature config JSON.")
    parser.add_argument(
        "--config",
        required=True,
        help="Freqtrade config JSON (contains exchange/pairs/freqai params).",
    )
    parser.add_argument("--output", default="user_data/freqai_features.json", help="Output feature JSON path.")
    parser.add_argument("--timeframe", default=None, help="Override timeframe (e.g. 1h).")
    parser.add_argument("--pairs", nargs="*", default=None, help="Optional override pairs (e.g. BTC/USDT ETH/USDT).")
    parser.add_argument("--label-period", type=int, default=None, help="Override label horizon in candles.")
    args = parser.parse_args(argv)

    cfg_path = _resolve_path(args.config)
    cfg = _load_json(cfg_path)

    exchange = str((cfg.get("exchange") or {}).get("name") or "unknown")
    timeframe = _resolve_timeframe(cfg, args.timeframe)
    label_period = _resolve_label_period(cfg, args.label_period)
    pairs = _resolve_pairs(cfg, args.pairs)

    output_path = _resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "version": 1,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "exchange": exchange,
        "pairs": pairs,
        "timeframe": timeframe,
        "label_period": int(label_period),
        "selection_methods": ["preset"],
        "features": build_preset_features(),
        "feature_combos": build_preset_combos(),
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[feature_agent] wrote: {output_path}")
    print(f"[feature_agent] exchange={exchange} timeframe={timeframe} pairs={len(pairs)} label_period={label_period}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

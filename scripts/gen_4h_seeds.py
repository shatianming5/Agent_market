#!/usr/bin/env python3
"""Build a 4h-safe starter seed library from plain base-column expressions.

The miner's load_seeds() only aggregates from a few JSON paths — including
user_data/factor_lib_archive/freqai_expressions*.json. This script produces
a seed file there with expressions that reference ONLY columns present on
4h feathers (i.e. no mtf4h_* — which only exists on 1h). The 1h miner also
reads this file, which is fine: these are generic base-column seeds.
"""
from __future__ import annotations

import json
import itertools
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "user_data" / "factor_lib_archive" / "freqai_expressions_base_seeds.json"

BASE = [
    "rsi_14", "mfi_14", "adx_14", "cci_20", "ema_pct_12", "ema_pct_48",
    "kama_pct_20", "macd_diff", "stoch_k_14", "stoch_d_14",
    "psar_ratio", "donchian_width_20", "atr_norm_14", "range_pct",
    "volume_ratio_20", "cmf_20", "obv_delta_20", "roc_12", "roc_48",
    "momentum_12", "return_zscore_24", "vwap_pct_20", "sma_pct_20",
    "tema_pct_12", "linearreg_slope_20", "linearreg_slope_48",
    "price_zscore_48", "realized_vol_24", "realized_vol_72",
    "return_skew_48", "volume_zscore_24", "vwap_slope_20",
    "rsi_7", "rsi_28", "adx_7", "adx_28", "plus_di_14", "minus_di_14",
    "dx_14", "stochrsi_k", "stochrsi_d", "linreg_angle_14",
    "linreg_angle_48", "ema_spread", "stoch_diff", "vol_price_div",
    "momentum_spread", "rsi_divergence", "adx_regime_shift",
    "close_to_open", "upper_shadow", "lower_shadow", "body_ratio",
    "close_position", "volume_change", "rsi_mfi_div",
]

# Regimes that work on 4h (no mtf4h)
REGIMES = [
    "adx_14 > 25", "adx_14 < 20",
    "rsi_14 > 60", "rsi_14 < 40",
    "cmf_20 > 0", "cmf_20 < 0",
    "volume_zscore_24 > 1", "realized_vol_24 > realized_vol_72",
    "plus_di_14 > minus_di_14", "macd_diff > 0",
]


def main() -> int:
    seeds: set[str] = set()

    # 1) Plain z-scored single factors (common starting point)
    for c in BASE:
        seeds.add(f"z({c})")
        seeds.add(f"-z({c})")

    # 2) ema-smoothed factors
    for c in ["rsi_14", "mfi_14", "cci_20", "cmf_20", "obv_delta_20",
              "volume_ratio_20", "momentum_12", "roc_12", "stoch_k_14"]:
        seeds.add(f"z(ema({c}, 12))")
        seeds.add(f"z(ema({c}, 24) - ema({c}, 6))")

    # 3) Pairwise z-diff among semantically related factors
    related = [
        ("rsi_14", "mfi_14"),
        ("ema_pct_12", "ema_pct_48"),
        ("linearreg_slope_20", "linearreg_slope_48"),
        ("roc_12", "roc_48"),
        ("realized_vol_24", "realized_vol_72"),
        ("plus_di_14", "minus_di_14"),
        ("stochrsi_k", "stochrsi_d"),
        ("rsi_7", "rsi_28"),
        ("adx_7", "adx_28"),
    ]
    for a, b in related:
        seeds.add(f"z({a}) - z({b})")
        seeds.add(f"tanh(z({a}) - z({b}))")

    # 4) Volume-weighted momentum
    for mom in ["roc_12", "momentum_12", "linearreg_slope_20"]:
        seeds.add(f"z({mom}) * sign(cmf_20)")
        seeds.add(f"z({mom}) * sign(volume_zscore_24)")

    # 5) Regime-switched
    import random
    rng = random.Random(17)
    for _ in range(40):
        a, b = rng.sample(BASE, 2)
        reg = rng.choice(REGIMES)
        seeds.add(f"ifelse({reg}, z({a}), -z({b}))")

    # 6) Nonlinear wrappers on strongest singletons
    for c in ["rsi_14", "rsi_7", "cmf_20", "adx_14", "ema_pct_48"]:
        seeds.add(f"tanh(z({c}))")
        seeds.add(f"sign({c} - 50) * atr_norm_14")

    # 7) Multi-timescale compositions using only base cols
    for short, long in [("rsi_7", "rsi_28"), ("ema_pct_6", "ema_pct_96"),
                         ("adx_7", "adx_28"), ("atr_norm_7", "atr_norm_48")]:
        seeds.add(f"z({short}) - z({long})")

    # 8) Shadow / candle features (useful for mean-reversion)
    seeds.add("z(upper_shadow - lower_shadow)")
    seeds.add("z(body_ratio) * sign(close_to_open)")
    seeds.add("ifelse(rsi_14 > 70, -z(upper_shadow), z(lower_shadow))")

    # 9) Funding-aware seeds (1h only — 4h has no funding, will be skipped)
    for fund in ["funding_rate", "funding_shift_8h", "funding_z_200"]:
        seeds.add(f"-z({fund})")
        seeds.add(f"z({fund}) * sign(cmf_20)")
        seeds.add(f"ifelse(funding_z_200 > 1, -z(rsi_14), z(rsi_14))")

    payload = {
        "version": 1,
        "generated_at": "gen_4h_seeds.py",
        "note": "Base-column-only seeds — safe for 1h, 4h, 15m, etc. No mtf4h deps.",
        "expressions": [{"name": f"bs{i+1:03d}", "expression": e}
                        for i, e in enumerate(sorted(seeds))],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[gen_4h_seeds] wrote {len(seeds)} seeds → {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

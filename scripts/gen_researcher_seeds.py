#!/usr/bin/env python3
"""Generate 5 persona-specific seed libraries for independent researchers.

Each persona has:
  - feature family preference (limits seed column set)
  - composition style (how it builds seed factors)
  - independent random seed

Output: 5 files under user_data/factor_lib_archive/:
  - freqai_expressions_R1_trender.json
  - freqai_expressions_R2_reverter.json
  - freqai_expressions_R3_volwatcher.json
  - freqai_expressions_R4_funding.json
  - freqai_expressions_R5_xsranker.json

Note: load_seeds() reads everything matching "freqai_expressions*.json" under
factor_lib_archive. To keep personas ISOLATED, we put them in subdir and bypass
load_seeds() — miner receives its persona's seed via a separate mechanism
(see --seed-file arg we'll add to mining).
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "user_data" / "researcher_seeds"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# Feature families per persona
TREND_COLS = [
    "ema_pct_6", "ema_pct_12", "ema_pct_48", "ema_pct_96",
    "linearreg_slope_20", "linearreg_slope_48", "linreg_angle_14",
    "roc_12", "roc_48", "kama_pct_20", "tema_pct_12", "dema_pct_20",
    "momentum_12", "t3_pct_10", "ema_spread", "sma_pct_20",
    "macd_diff", "psar_ratio",
]
REVERT_COLS = [
    "rsi_7", "rsi_14", "rsi_28", "mfi_7", "mfi_14", "mfi_28",
    "stochrsi_k", "stochrsi_d", "stochf_k", "stoch_k_14", "stoch_d_14",
    "cci_10", "cci_20", "return_zscore_24", "price_zscore_48",
    "vwap_pct_20", "stoch_diff", "rsi_divergence",
]
VOL_COLS = [
    "atr_norm_7", "atr_norm_14", "atr_norm_48", "realized_vol_24",
    "realized_vol_72", "donchian_width_20", "donchian_width_48",
    "range_pct", "adx_7", "adx_14", "adx_28", "dx_14",
    "return_skew_48", "return_skew_96", "plus_di_14", "minus_di_14",
    "vol_term_structure", "adx_regime_shift",
]
FUNDING_COLS = [
    "funding_rate", "funding_abs_ma24", "funding_cumsum_72",
    "funding_shift_8h", "funding_z_200",
]
XS_COLS = [
    "xs_ret_24h", "xs_ret_72h", "xs_vol_24h", "xs_volume_24h",
    "xs_momentum_168h", "xs_ret_vs_btc", "xs_mtf_rsi_rank",
    "mtf4h_rsi_14", "mtf4h_adx_14", "mtf4h_cmf_20", "mtf4h_ema_pct_12",
    "mtf4h_ema_pct_48", "mtf4h_plus_di_14", "mtf4h_minus_di_14",
    "mtf4h_return_zscore_24", "mtf4h_realized_vol_24",
]

# Regimes per persona (reflects research bias)
R1_REG = ["adx_14 > 25", "di_spread > 0", "macd_diff > 0", "ema_pct_48 > 0"]
R2_REG = ["rsi_14 > 70", "rsi_14 < 30", "cci_20 > 100", "cci_20 < -100"]
R3_REG = ["atr_norm_14 > 0.02", "realized_vol_24 > realized_vol_72",
          "adx_14 < 20", "donchian_width_48 > 0.05"]
R4_REG = ["funding_z_200 > 1", "funding_z_200 < -1",
          "funding_cumsum_72 > 0", "abs(funding_rate) > 0.0002"]
R5_REG = ["xs_ret_vs_btc > 0.5", "xs_ret_vs_btc < -0.5",
          "mtf4h_rsi_14 > 60", "mtf4h_rsi_14 < 40"]


def _make_seeds(cols: List[str], regimes: List[str], seed: int, count: int = 80) -> List[str]:
    rng = random.Random(seed)
    out: set = set()
    # 1. raw z-scored singletons (both signs)
    for c in cols:
        out.add(f"z({c})")
        out.add(f"-z({c})")
    # 2. EMA-smoothed
    for c in rng.sample(cols, min(len(cols), 12)):
        out.add(f"z(ema({c}, 12))")
        out.add(f"z(ema({c}, 24) - ema({c}, 6))")
    # 3. pairwise z-diff
    for _ in range(20):
        a, b = rng.sample(cols, 2)
        out.add(f"z({a}) - z({b})")
        out.add(f"tanh(z({a}) - z({b}))")
    # 4. regime-switched
    for _ in range(30):
        a, b = rng.sample(cols, 2)
        r = rng.choice(regimes)
        out.add(f"ifelse({r}, z({a}), -z({b}))")
        out.add(f"ifelse({r}, -z({a}), z({b}))")
    # 5. nonlinear wrappers on strong singletons
    for c in rng.sample(cols, min(len(cols), 6)):
        out.add(f"tanh(z({c}))")
        out.add(f"sign({c} - roll_mean({c}, 24))")
    # 6. rolling combos
    for _ in range(10):
        a = rng.choice(cols)
        out.add(f"z(rolling_max({a}, 24) - rolling_min({a}, 24))")
        out.add(f"sign({a}) * roll_std({a}, 24)")
    # keep only unique, short
    return [s for s in out if 4 < len(s) < 400][:count]


def main() -> int:
    personas: Dict[str, Dict] = {
        "R1_trender":     {"cols": TREND_COLS,   "reg": R1_REG, "seed": 17},
        "R2_reverter":    {"cols": REVERT_COLS,  "reg": R2_REG, "seed": 31},
        "R3_volwatcher":  {"cols": VOL_COLS,     "reg": R3_REG, "seed": 47},
        "R4_funding":     {"cols": FUNDING_COLS + REVERT_COLS[:6],
                           "reg": R4_REG, "seed": 59},
        "R5_xsranker":    {"cols": XS_COLS,      "reg": R5_REG, "seed": 71},
    }
    for name, cfg in personas.items():
        seeds = _make_seeds(cfg["cols"], cfg["reg"], cfg["seed"])
        out = OUT_DIR / f"seeds_{name}.json"
        out.write_text(json.dumps({
            "version": 1,
            "persona": name,
            "generated_at": "gen_researcher_seeds.py",
            "feature_cols": cfg["cols"],
            "regimes": cfg["reg"],
            "expressions": [{"name": f"{name}_s{i+1:03d}", "expression": s}
                            for i, s in enumerate(seeds)],
        }, indent=2), encoding="utf-8")
        print(f"  [{name}] {len(seeds)} seeds → {out.name}")
    print(f"[done] all seeds in {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

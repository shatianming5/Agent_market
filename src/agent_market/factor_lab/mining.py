"""Unified factor miner: IC + model-driven (LGB gain) + LLM + composition.

Combines the best of v2/v3/v4/v5:
  • Iterative loop with checkpointing (v5)
  • Fast IC + sign-consistency gate (v3)
  • Python compositional operators (v2/v3/v4)
  • LLM candidate generation + review (v5)
  • Model-driven LightGBM gain re-rank every K loops (v4)
  • Novelty penalty (v3)
"""
from __future__ import annotations

import ast
import json
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .paths import (BINANCE_FUTURES_DIR, BYBIT_FUTURES_DIR, KUCOIN_DIR, OKX_FUTURES_DIR, FEATURE_FILE, EXPRESSIONS_SCORED,
                    DEFAULT_TRAIN, DEFAULT_OOS,
                    DEFAULT_TRAIN3, DEFAULT_VAL3, DEFAULT_REAL_TEST3,
                    DEFAULT_VAL_WINDOWS, DEFAULT_LABEL_PERIOD,
                    DEFAULT_CLASS_THRESHOLD, LAB_STATE,
                    TIMEFRAME_LABEL_BARS, feather_for_pair, resolve_pairs)
from . import fitness as F
from .cache import CACHE_VERSION, DEFAULT_CACHE_DIR, file_fingerprint, get_cache, panel_fingerprint, stable_hash
from .purification import DEFAULT_EXPOSURE_GROUPS, PurifyConfig, apply_purification, parse_exposure_groups
from .timeframes import (
    LANE_DEFAULTS,
    bps_to_rate,
    lane_manifest,
    normalize_lane,
    parse_label_horizons,
    primary_label_horizon,
)

# import existing engine
_SRC = str(Path(__file__).resolve().parents[2])
if _SRC not in sys.path: sys.path.insert(0, _SRC)
from agent_market.freqai.expression_engine import safe_eval_expression
from agent_market.freqai.features import apply_configured_features


# ============================================================
# Factor Hub integration (optional — failures are non-fatal)
# ============================================================

def _open_hub_client(tag: str):
    """Return a Factor Hub Client, or None if initialization fails."""
    try:
        from agent_market.factor_hub import Client
        c = Client()
        c.init_db()
        return c
    except Exception as exc:
        print(f"[mining] factor-hub unavailable ({exc!s:.120}), continuing without logging")
        return None


def _hub_register(hub, cand: "CandidateRecord", cfg: "MiningConfig",
                  tag: str, loop_idx: int) -> None:
    if hub is None: return
    try:
        annotate_diversity(cand)
        sub_origin = cand.origin or "unknown"
        category = "mined"
        if "llm" in sub_origin:   category = "llm_mined"
        if "seed" in sub_origin:  category = "seed"
        if "py" in sub_origin:    category = "composition"
        # Prefix origin so a consumer can filter by `origin LIKE '%mining%'`
        # without losing the sub-origin detail.
        origin = f"mining:{tag}:{sub_origin}"
        # composite/portfolio eval modes never touch REAL_TEST3 — tag as clean.
        # legacy mode uses DEFAULT_OOS as IC gate → tag as oos_snooped.
        snoop_level = "clean" if _is_composite_eval_mode(cfg) else "oos_snooped"
        fid = hub.propose(
            expression=cand.expression,
            name=f"{tag}_loop{loop_idx:04d}_{len(cand.expression)%10000}",
            category=category, origin=origin,
            status="candidate", source_lib=f"mining:{tag}",
            metadata={"train_ic": cand.train_ic, "oos_ic": cand.oos_ic,
                      "sign_agree": cand.sign_agree, "combined": cand.combined,
                      "loop_found": cand.loop_found, "mining_tag": tag,
                      "sub_origin": sub_origin,
                      "eval_mode": cfg.eval_mode,
                      "purify_mode": cand.purify_mode,
                      "raw_ic": cand.raw_ic,
                      "clean_ic": cand.clean_ic,
                      "neutralized_ic": cand.neutralized_ic,
                      "residual_ic_ratio": cand.residual_ic_ratio,
                      "exposure_r2": cand.exposure_r2,
                      "max_exposure_corr": cand.max_exposure_corr,
                      "snoop_level": snoop_level,
                      "timeframe": cfg.timeframe,
                      "primary_family": cand.primary_family,
                      "family_tags": list(cand.family_tags),
                      "canonical_signature": cand.canonical_signature,
                      "max_corr_to_kept": cand.max_corr_to_kept},
        )
        hub.add_evaluation(fid, eval_type="ic", metric_name="oos_ic",
                           metric_value=cand.oos_ic,
                           period_start=cfg.oos[0], period_end=cfg.oos[1],
                           sign_agree=cand.sign_agree,
                           notes=f"mining:{tag} loop {loop_idx}",
                           metadata={"train_ic": cand.train_ic,
                                     "combined": cand.combined,
                                     "origin": sub_origin})
    except Exception as exc:  # noqa: BLE001
        print(f"[mining] factor-hub register failed: {exc!s:.120}")


def _hub_event(hub, event_type: str, **payload) -> None:
    if hub is None: return
    try:
        hub.log(event_type, payload=payload or None)
    except Exception:
        pass


# ============================================================
# Configuration
# ============================================================
MAX_EXPR_LEN = 450
DEFAULT_FAMILIES = (
    "trend",
    "volatility",
    "micro",
    "funding",
    "cross_sectional",
    "mtf",
    "regime",
)
CORE_PORTFOLIO_FAMILIES = ("trend", "volatility", "funding", "micro", "cross_sectional")


@dataclass
class MiningConfig:
    rounds: int = 100
    top_k: int = 40
    llm_per_loop: int = 6
    py_per_loop: int = 10
    ic_gate: float = 0.025
    sign_gate: int = 7
    checkpoint_every: int = 10
    lgb_rerank_every: int = 0  # 0 = disabled
    novelty_gate: float = 0.85  # legacy alias; reject if corr > threshold to survivors
    hard_corr_gate: float = 0.85
    soft_corr_penalty_start: float = 0.55
    max_same_family_in_top40: int = 8
    max_same_signature: int = 2
    label_period: int = DEFAULT_LABEL_PERIOD
    # Legacy two-section windows (kept for backward compat when eval_mode="legacy")
    train: Tuple[str, str] = DEFAULT_TRAIN
    oos: Tuple[str, str] = DEFAULT_OOS
    # Three-section windows for composite fitness path
    train3: Tuple[str, str] = DEFAULT_TRAIN3
    val3: Tuple[str, str] = DEFAULT_VAL3
    real_test3: Tuple[str, str] = DEFAULT_REAL_TEST3
    val_windows: Tuple = DEFAULT_VAL_WINDOWS
    timeframe: str = "1h"
    evaluation_lane: str = "auto"
    data_venue: str = "kucoin"
    label_horizons: Tuple[int, ...] = ()
    embargo_bars: int = 0
    micro_data_quality: str = "unknown"
    class_threshold: float = DEFAULT_CLASS_THRESHOLD
    use_llm: bool = False
    llm_required: bool = False
    seed_file: Optional[str] = None   # override default load_seeds() — multi-researcher isolation
    llm_retries: int = 3
    llm_timeout: float = 120.0
    llm_max_tokens: int = 0
    llm_reasoning_effort: str = ""
    # Composite-fitness knobs
    eval_mode: str = "legacy"          # "legacy" | "composite"
    xs_weight: float = 0.0             # 0 = pure TS; 1.0 = pure XS
    turnover_weight: float = 1.0       # 0 = disable cost penalty
    stability_mode: str = "min_abs"    # "min_abs" | "mean" | "median"
    fee_rate: float = 0.0008           # taker fee for composite fitness
    slippage: float = 0.0003
    # Label target mode
    label_mode: str = "forward_return"  # forward_return | pair_spread_btc | pair_beta_resid_btc
    pair_reference: str = "BTC/USDT"
    data_dir: Optional[str] = None
    pairs: str = "auto"
    # Optional cross-sectional purification. Defaults keep legacy behavior.
    purify_mode: str = "off"  # off | clean | neutralized | blend
    purify_winsor: str = "mad"  # mad | quantile | iqr | none
    purify_standardize: str = "zscore"  # zscore | rank | rank_gaussianize | none
    purify_neutralize: str = "ridge"  # none | ols | ridge
    purify_exposures: str = ",".join(DEFAULT_EXPOSURE_GROUPS)
    purify_ridge_alpha: float = 1e-3
    alpha_objective: str = "blend"  # blend | pure_residual
    prompt_profile: str = "default"  # default | residual_alpha_v2
    llm_filter_low_coverage: bool = True
    llm_min_feature_coverage: float = 0.60
    llm_min_feature_rows: int = 300
    pure_residual_ic_gate: float = 0.008
    pure_residual_sign_gate: int = 6
    pure_residual_ratio_gate: float = 0.15
    pure_residual_exposure_r2_gate: float = 0.90
    pure_residual_max_exposure_corr_gate: float = 0.50
    cache_dir: str = str(DEFAULT_CACHE_DIR)
    no_cache: bool = False
    # Optional post-loop LEAN gate. Disabled by default because each trigger runs
    # a full rank portfolio backtest, LEAN export, LEAN backtest, and comparison.
    lean_gate_every: int = 0
    lean_gate_fail_fast: bool = False
    lean_gate_force: bool = True
    lean_gate_n: int = 30
    lean_gate_venue: str = "auto"
    lean_gate_data_venue: str = "auto"
    lean_gate_start: str = "2025-12-01"
    lean_gate_end: str = "2026-04-12"
    lean_gate_bin: str = "lean"
    lean_gate_timeout: int = 0
    lean_gate_data_root: str = ""
    lean_gate_required_status: str = "ok"
    lean_gate_min_final_equity: float = 1.0
    lean_gate_max_drawdown_pct: float = 25.0
    lean_gate_min_trades: int = 80
    lean_gate_rank_top_k: int = 2
    lean_gate_gross_cap: float = 2.0
    lean_gate_net_cap: float = 2.0
    lean_gate_single_pair_cap: float = 2.0
    lean_gate_side_mode: str = "short"
    lean_gate_score_threshold: float = 1.5
    lean_gate_rebalance_hours: int = 8
    lean_gate_rebalance_minutes: int = 0
    lean_gate_risk_per_trade: float = 0.08
    lean_gate_leverage_cap: float = 5.0
    lean_gate_recompute_corr: bool = False


@dataclass
class CandidateRecord:
    expression: str
    origin: str
    train_ic: float = 0.0
    oos_ic: float = 0.0
    sign_agree: int = 0
    combined: float = 0.0
    lgb_gain: float = 0.0
    loop_found: int = 0
    # Composite-fitness fields (filled when eval_mode="composite")
    xs_ic: float = 0.0
    turnover: float = 0.0
    cost_mult: float = 1.0
    stability_ic: float = 0.0       # min-over-window (or whichever agg)
    fitness: float = 0.0            # composite score — supersedes `combined`
    # Diversity metadata
    primary_family: str = "trend"
    family_tags: Tuple[str, ...] = ()
    canonical_signature: str = ""
    max_corr_to_kept: float = 0.0
    cluster_id: int = -1
    # Purification diagnostics (filled only when purify_mode != off).
    raw_ic: float = 0.0
    clean_ic: float = 0.0
    neutralized_ic: float = 0.0
    residual_ic_ratio: float = 0.0
    exposure_r2: float = 0.0
    max_exposure_corr: float = 0.0
    exposure_count: int = 0
    purify_mode: str = "off"
    eval_cache_key: str = ""


# ============================================================
# Data loading
# ============================================================

FUTURES_VENUE_DIRS = {
    "okx": OKX_FUTURES_DIR,
    "bybit": BYBIT_FUTURES_DIR,
    "binance": BINANCE_FUTURES_DIR,
}


def _pair_file_token(pair: str) -> str:
    base = str(pair).split(":", 1)[0].replace("/", "_").upper()
    if base.endswith("_USDT"):
        return f"{base}_USDT"
    return base


def _futures_feather(pair: str, *, timeframe: str, data_dir: Optional[str | Path] = None) -> Path:
    root = Path(data_dir) if data_dir is not None else OKX_FUTURES_DIR
    return root / f"{_pair_file_token(pair)}-{timeframe}-futures.feather"


def _pair_from_futures_token(token: str) -> str:
    parts = str(token).upper().split("_")
    if len(parts) >= 3 and parts[-2:] == ["USDT", "USDT"]:
        return f"{'_'.join(parts[:-2])}/USDT"
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    return str(token).upper()


def _discover_futures_pairs(data_dir: Optional[str | Path], timeframe: str) -> list[str]:
    root = Path(data_dir) if data_dir is not None else OKX_FUTURES_DIR
    if not root.exists():
        return []
    suffix = f"-{timeframe}-futures.feather"
    pairs: list[str] = []
    for path in sorted(root.glob(f"*{suffix}")):
        token = path.name[: -len(suffix)]
        pairs.append(_pair_from_futures_token(token))
    return pairs


def _resolve_mining_data(
    *,
    data_venue: str,
    data_dir: Optional[str | Path],
    timeframe: str,
    pairs: Optional[Sequence[str] | str],
) -> tuple[Path, list[str], list[Path]]:
    venue = str(data_venue or "kucoin").strip().lower()
    if venue in FUTURES_VENUE_DIRS:
        data_root = Path(data_dir) if data_dir is not None else FUTURES_VENUE_DIRS[venue]
        if isinstance(pairs, str) and pairs.strip().lower() == "auto":
            pair_list = _discover_futures_pairs(data_root, timeframe) or resolve_pairs("default")
        else:
            pair_list = resolve_pairs(pairs, data_dir=data_root, timeframe=timeframe)
        return data_root, pair_list, [_futures_feather(pair, timeframe=timeframe, data_dir=data_root) for pair in pair_list]

    data_root = Path(data_dir) if data_dir is not None else KUCOIN_DIR
    pair_list = resolve_pairs(pairs, data_dir=data_root, timeframe=timeframe)
    return data_root, pair_list, [feather_for_pair(pair, timeframe=timeframe, data_dir=data_root) for pair in pair_list]


def build_big(
    timeframe: str = "1h",
    label_bars: Optional[int] = None,
    *,
    label_mode: str = "forward_return",
    pair_reference: str = "BTC/USDT",
    data_dir: Optional[str | Path] = None,
    data_venue: str = "auto",
    pairs: Optional[Sequence[str] | str] = None,
    cache_dir: Optional[str | Path] = None,
    no_cache: bool = False,
) -> Tuple[pd.DataFrame, List[str]]:
    """Load all pairs, apply base+mtf4h+xs+funding+micro, create labels.

    timeframe: "1h" (default) / "4h" / "1m" — must have feather data at that freq.
    label_bars: forward-return horizon in bars. Defaults to 12 for 1h, 3 for 4h, etc.
    """
    if label_bars is None:
        label_bars = TIMEFRAME_LABEL_BARS.get(timeframe, DEFAULT_LABEL_PERIOD)
    venue = str(data_venue or "kucoin").strip().lower()
    data_root, pair_list, feather_paths = _resolve_mining_data(
        data_venue=venue,
        data_dir=data_dir,
        timeframe=timeframe,
        pairs=pairs,
    )
    existing_paths = [p for p in feather_paths if p.exists()]
    cache_enabled = bool(cache_dir) and not bool(no_cache)
    cache = get_cache(cache_dir, no_cache=not cache_enabled)
    panel_key_payload = {
        "kind": "panel",
        "cache_version": CACHE_VERSION,
        "timeframe": timeframe,
        "label_bars": int(label_bars),
        "label_mode": str(label_mode or "forward_return"),
        "pair_reference": pair_reference,
        "data_venue": venue,
        "data_root": str(data_root),
        "pairs": pair_list,
        "feature_file": file_fingerprint(FEATURE_FILE),
        "data_files": [file_fingerprint(path) for path in existing_paths],
    }
    panel_key = stable_hash(panel_key_payload)
    if cache_enabled:
        loaded = cache.load_panel(panel_key)
        if loaded is not None:
            cached_big, meta = loaded
            base_cols = list(meta.get("base_cols") or [])
            cached_big.attrs["factor_lab_panel_key"] = panel_key
            cached_big.attrs["factor_lab_panel_payload"] = panel_key_payload
            cached_big.attrs["factor_lab_cache_dir"] = str(cache.root)
            return cached_big, base_cols

    feat_cfg = json.loads(FEATURE_FILE.read_text(encoding="utf-8-sig"))
    frames = []
    for pair, f in zip(pair_list, feather_paths):
        if not f.exists(): continue
        df = pd.read_feather(f)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df = df.reset_index(drop=True)
        df = apply_configured_features(df, feat_cfg)
        df["__pair__"] = pair
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no {venue} {timeframe} feather data found under {data_root}")
    big = pd.concat(frames, ignore_index=True)
    big["date"] = pd.to_datetime(big["date"], utc=True)
    big = big.sort_values(["__pair__", "date"]).reset_index(drop=True)
    big["__fwd_raw__"] = (big.groupby("__pair__")["close"].shift(-int(label_bars)) / big["close"]) - 1.0

    mode = str(label_mode or "forward_return").strip().lower()
    if mode == "forward_return":
        big["__fwd_ret__"] = big["__fwd_raw__"]
    else:
        ref = big.loc[big["__pair__"] == pair_reference, ["date", "__fwd_raw__", "close"]].rename(
            columns={"__fwd_raw__": "__fwd_ref__", "close": "__ref_close__"}
        )
        if ref.empty:
            raise ValueError(f"pair reference `{pair_reference}` not found for label_mode={mode}")
        big = big.merge(ref, on="date", how="left")
        if mode == "pair_spread_btc":
            big["__fwd_ret__"] = big["__fwd_raw__"] - big["__fwd_ref__"]
        elif mode == "pair_beta_resid_btc":
            big["__ret_1__"] = big.groupby("__pair__")["close"].pct_change(1)
            ref_r = big.loc[big["__pair__"] == pair_reference, ["date", "__ret_1__"]].rename(
                columns={"__ret_1__": "__ret_ref_1__"}
            )
            big = big.merge(ref_r, on="date", how="left")
            beta_window = max(48, int(label_bars) * 6)
            min_periods = max(24, beta_window // 3)
            beta = np.full(len(big), np.nan, dtype=float)
            for _, idx in big.groupby("__pair__").groups.items():
                sub = big.loc[idx]
                cov = sub["__ret_1__"].rolling(beta_window, min_periods=min_periods).cov(sub["__ret_ref_1__"])
                var = sub["__ret_ref_1__"].rolling(beta_window, min_periods=min_periods).var(ddof=0)
                beta[idx] = (cov / (var + 1e-9)).to_numpy()
            big["__beta_ref__"] = beta
            big["__fwd_ret__"] = big["__fwd_raw__"] - big["__beta_ref__"] * big["__fwd_ref__"]
        else:
            raise ValueError(f"unknown label_mode: {label_mode}")

        # Reference pair against itself has no pair-trade label.
        big.loc[big["__pair__"] == pair_reference, "__fwd_ret__"] = np.nan

    exclude = {
        "date","open","high","low","close","volume","__pair__","__fwd_ret__",
        "__fwd_raw__","__fwd_ref__","__ref_close__","__ret_1__","__ret_ref_1__","__beta_ref__",
    }
    base_cols = [c for c in big.columns if c not in exclude]
    big.attrs["factor_lab_panel_key"] = panel_key
    big.attrs["factor_lab_panel_payload"] = panel_key_payload
    if cache_enabled:
        big.attrs["factor_lab_cache_dir"] = str(cache.root)
        cache.save_panel(
            panel_key,
            big,
            {
                "base_cols": base_cols,
                "rows": int(len(big)),
                "pairs": pair_list,
                "timeframe": timeframe,
                "label_bars": int(label_bars),
                "label_mode": str(label_mode or "forward_return"),
                "pair_reference": pair_reference,
                "data_venue": venue,
            },
        )
    return big, base_cols


_PAIR_INDEXERS_ATTR = "factor_lab_pair_indexers"


def _pair_indexers_in_big(big: pd.DataFrame) -> List[Tuple[str, slice | np.ndarray]]:
    cached = big.attrs.get(_PAIR_INDEXERS_ATTR)
    if isinstance(cached, dict) and int(cached.get("length", -1)) == len(big):
        indexers = cached.get("indexers")
        if isinstance(indexers, list):
            return indexers

    pair_values = big["__pair__"].to_numpy()
    if len(pair_values) == 0:
        big.attrs[_PAIR_INDEXERS_ATTR] = {"length": 0, "indexers": []}
        return []

    boundaries = np.flatnonzero(pair_values[1:] != pair_values[:-1]) + 1
    starts = np.r_[0, boundaries]
    stops = np.r_[boundaries, len(pair_values)]

    seen: set[str] = set()
    indexers: List[Tuple[str, slice | np.ndarray]] = []
    contiguous = True
    for start, stop in zip(starts, stops):
        pair = str(pair_values[int(start)])
        if pair in seen:
            contiguous = False
            break
        seen.add(pair)
        indexers.append((pair, slice(int(start), int(stop))))

    if not contiguous:
        indexers = [
            (str(pair), np.asarray(idx, dtype=np.int64))
            for pair, idx in big.groupby("__pair__", sort=False).indices.items()
        ]

    big.attrs[_PAIR_INDEXERS_ATTR] = {"length": len(big), "indexers": indexers}
    return indexers


def _iter_pair_frames(big: pd.DataFrame):
    for pair, indexer in _pair_indexers_in_big(big):
        yield pair, indexer, big.iloc[indexer]


def _pairs_in_big(big: pd.DataFrame) -> List[str]:
    return [pair for pair, _ in _pair_indexers_in_big(big)]


def _is_composite_eval_mode(cfg: MiningConfig) -> bool:
    return str(cfg.eval_mode or "legacy").lower() in {"composite", "portfolio"}


def _alpha_objective(cfg: MiningConfig) -> str:
    mode = str(getattr(cfg, "alpha_objective", "blend") or "blend").strip().lower()
    return mode if mode in {"blend", "pure_residual"} else "blend"


def _pure_residual_enabled(cfg: MiningConfig) -> bool:
    return _alpha_objective(cfg) == "pure_residual"


def _pure_residual_gates(cfg: MiningConfig) -> Dict[str, float | int]:
    return {
        "min_abs_neutralized_ic": float(cfg.pure_residual_ic_gate),
        "min_sign_agree": int(cfg.pure_residual_sign_gate),
        "min_residual_ic_ratio": float(cfg.pure_residual_ratio_gate),
        "max_exposure_r2": float(cfg.pure_residual_exposure_r2_gate),
        "max_exposure_corr": float(cfg.pure_residual_max_exposure_corr_gate),
    }


def _effective_purify_mode(cfg: MiningConfig) -> str:
    if _pure_residual_enabled(cfg):
        return "neutralized"
    return str(cfg.purify_mode or "off").lower()


def _purify_config(cfg: MiningConfig) -> PurifyConfig:
    return PurifyConfig(
        mode=_effective_purify_mode(cfg),
        winsor=str(cfg.purify_winsor or "mad").lower(),
        standardize=str(cfg.purify_standardize or "zscore").lower(),
        neutralize=str(cfg.purify_neutralize or "ridge").lower(),
        exposures=parse_exposure_groups(cfg.purify_exposures),
        ridge_alpha=float(cfg.purify_ridge_alpha),
        cache_dir=str(cfg.cache_dir) if getattr(cfg, "cache_dir", None) else None,
        no_cache=bool(getattr(cfg, "no_cache", False)),
    )


def _purification_enabled(cfg: MiningConfig) -> bool:
    return _effective_purify_mode(cfg) != "off"


def _cache_enabled(cfg: MiningConfig) -> bool:
    return bool(getattr(cfg, "cache_dir", None)) and not bool(getattr(cfg, "no_cache", False))


def _cache_for_cfg(cfg: MiningConfig):
    return get_cache(getattr(cfg, "cache_dir", None), no_cache=not _cache_enabled(cfg))


def _purify_config_for_panel(big: pd.DataFrame, cfg: MiningConfig) -> PurifyConfig:
    base = _purify_config(cfg)
    return PurifyConfig(
        mode=base.mode,
        winsor=base.winsor,
        standardize=base.standardize,
        neutralize=base.neutralize,
        exposures=base.exposures,
        ridge_alpha=base.ridge_alpha,
        cache_dir=base.cache_dir,
        no_cache=base.no_cache,
        panel_fingerprint=panel_fingerprint(big),
    )


def _purify_payload(pcfg: PurifyConfig) -> Dict[str, Any]:
    return {
        "mode": pcfg.mode,
        "winsor": pcfg.winsor,
        "standardize": pcfg.standardize,
        "neutralize": pcfg.neutralize,
        "exposures": list(pcfg.exposures),
        "ridge_alpha": float(pcfg.ridge_alpha),
    }


def _eval_payload(big: pd.DataFrame, expr: str, cfg: MiningConfig) -> Dict[str, Any]:
    return {
        "kind": "factor_eval",
        "cache_version": CACHE_VERSION,
        "panel": panel_fingerprint(big),
        "expression": stable_hash({"expr": str(expr)}),
        "evaluation_lane": str(cfg.evaluation_lane),
        "label_horizons": list(_effective_label_horizons(cfg)),
        "embargo_bars": int(cfg.embargo_bars or 0),
        "eval_mode": "composite" if _is_composite_eval_mode(cfg) else "legacy",
        "train": list(cfg.train),
        "oos": list(cfg.oos),
        "train3": list(cfg.train3),
        "val3": list(cfg.val3),
        "val_windows": [list(w) for w in cfg.val_windows],
        "ic_gate": float(cfg.ic_gate),
        "sign_gate": int(cfg.sign_gate),
        "xs_weight": float(cfg.xs_weight),
        "turnover_weight": float(cfg.turnover_weight),
        "stability_mode": str(cfg.stability_mode),
        "fee_rate": float(cfg.fee_rate),
        "slippage": float(cfg.slippage),
        "purify": _purify_payload(_purify_config_for_panel(big, cfg)),
        "alpha_objective": _alpha_objective(cfg),
        "pure_residual_gates": _pure_residual_gates(cfg) if _pure_residual_enabled(cfg) else None,
    }


def _effective_lane(cfg: MiningConfig):
    return normalize_lane(getattr(cfg, "evaluation_lane", "") or "auto", timeframe=cfg.timeframe)


def _effective_label_horizons(cfg: MiningConfig) -> tuple[int, ...]:
    lane = _effective_lane(cfg)
    return parse_label_horizons(getattr(cfg, "label_horizons", ()), default=lane.label_horizons)


def _effective_embargo_bars(cfg: MiningConfig) -> int:
    lane = _effective_lane(cfg)
    raw = int(getattr(cfg, "embargo_bars", 0) or 0)
    return raw if raw > 0 else int(lane.embargo_bars)


def mining_lane_manifest(cfg: MiningConfig) -> dict[str, Any]:
    horizons = _effective_label_horizons(cfg)
    fee_bps = float(cfg.fee_rate) * 10_000.0
    slippage_bps = float(cfg.slippage) * 10_000.0
    quality = str(getattr(cfg, "micro_data_quality", "") or "unknown")
    if _effective_lane(cfg).lane in {"1m_micro", "5m_micro"} and quality == "unknown":
        quality = "ohlcv_only"
    return lane_manifest(
        lane=_effective_lane(cfg).lane,
        timeframe=cfg.timeframe,
        data_venue=str(getattr(cfg, "data_venue", "kucoin") or "kucoin"),
        label_horizons=horizons,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        embargo_bars=_effective_embargo_bars(cfg),
        micro_data_quality=quality,
    )


def factor_versions(big: pd.DataFrame, expr: str, cfg: MiningConfig) -> Dict[str, Any]:
    """Return raw/clean/neutralized/selected factor series, using persistent cache."""
    pcfg = _purify_config_for_panel(big, cfg)
    payload = {
        "kind": "factor_version",
        "cache_version": CACHE_VERSION,
        "panel": panel_fingerprint(big),
        "expression": stable_hash({"expr": str(expr)}),
        "purify": _purify_payload(pcfg),
    }
    key = stable_hash(payload)
    cache = _cache_for_cfg(cfg)
    if _cache_enabled(cfg):
        loaded = cache.load_npz_bundle("factor_version", key)
        if loaded is not None:
            arrays, meta = loaded
            try:
                if len(arrays.get("raw", [])) == len(big):
                    raw = pd.Series(arrays["raw"], index=big.index, dtype="float64")
                    clean = pd.Series(arrays.get("clean", arrays["raw"]), index=big.index, dtype="float64")
                    neutralized = pd.Series(arrays.get("neutralized", arrays["raw"]), index=big.index, dtype="float64")
                    selected_name = str(meta.get("selected_name") or "selected")
                    if selected_name in arrays:
                        selected = pd.Series(arrays[selected_name], index=big.index, dtype="float64")
                    else:
                        selected = pd.Series(arrays.get("selected", arrays["raw"]), index=big.index, dtype="float64")
                    return {
                        "raw": raw,
                        "clean": clean,
                        "neutralized": neutralized,
                        "selected": selected,
                        "diagnostics": dict(meta.get("diagnostics") or {}),
                        "cache_key": key,
                        "cache_hit": True,
                    }
            except Exception:
                pass

    raw = _eval_factor_by_pair(big, expr)
    if pcfg.mode == "off":
        versions = {
            "raw": raw,
            "clean": raw,
            "neutralized": raw,
            "selected": raw,
            "diagnostics": {},
            "cache_key": key,
            "cache_hit": False,
        }
        selected_name = "raw"
    else:
        pur = apply_purification(big, raw, pcfg)
        versions = {
            "raw": pur.raw,
            "clean": pur.clean,
            "neutralized": pur.neutralized,
            "selected": pur.selected,
            "diagnostics": pur.diagnostics,
            "cache_key": key,
            "cache_hit": False,
        }
        selected_name = "clean" if pcfg.mode == "clean" else ("neutralized" if pcfg.mode in {"neutralized", "blend"} else "raw")

    if _cache_enabled(cfg):
        arrays = {
            "raw": np.asarray(versions["raw"], dtype=np.float64),
            "clean": np.asarray(versions["clean"], dtype=np.float64),
            "neutralized": np.asarray(versions["neutralized"], dtype=np.float64),
            "selected": np.asarray(versions["selected"], dtype=np.float64),
        }
        cache.save_npz_bundle(
            "factor_version",
            key,
            arrays,
            {
                "payload": payload,
                "selected_name": selected_name,
                "diagnostics": versions.get("diagnostics", {}),
            },
        )
    return versions


def _effective_purify_sign_gate(big: pd.DataFrame, cfg: MiningConfig) -> int:
    return min(int(cfg.sign_gate), max(1, int(big["__pair__"].nunique())))


def _eval_factor_by_pair(big: pd.DataFrame, expr: str) -> pd.Series:
    """Evaluate an expression per pair to preserve rolling-window causality."""
    out = pd.Series(np.nan, index=big.index, dtype="float64")
    for _, indexer, sub in _iter_pair_frames(big):
        if sub.empty:
            continue
        values = safe_eval_expression(expr, sub)
        out.iloc[indexer] = np.asarray(values, dtype=np.float64)
    return out.replace([np.inf, -np.inf], np.nan)


def _legacy_metrics_for_series(
    big: pd.DataFrame,
    factor: pd.Series,
    cfg: MiningConfig,
    return_oos_series: bool = False,
) -> Dict:
    ics_tr, ics_oo = [], []
    oos_chunks: List[np.ndarray] = []
    series = pd.Series(factor, index=big.index, dtype="float64").replace([np.inf, -np.inf], np.nan)
    for _, _, sub in _iter_pair_frames(big):
        s_pair = series.loc[sub.index]
        m_tr = ((sub["date"] >= pd.Timestamp(cfg.train[0], tz="UTC"))
                & (sub["date"] < pd.Timestamp(cfg.train[1], tz="UTC")))
        m_oo = ((sub["date"] >= pd.Timestamp(cfg.oos[0], tz="UTC"))
                & (sub["date"] < pd.Timestamp(cfg.oos[1], tz="UTC")))
        ic_t = spearman(s_pair[m_tr.values], sub.loc[m_tr, "__fwd_ret__"])
        ic_o = spearman(s_pair[m_oo.values], sub.loc[m_oo, "__fwd_ret__"])
        if not np.isnan(ic_t): ics_tr.append(ic_t)
        if not np.isnan(ic_o): ics_oo.append(ic_o)
        if return_oos_series:
            oos_chunks.append(np.asarray(s_pair[m_oo.values], dtype=np.float64))
    if not ics_tr or not ics_oo:
        return {"status": "insufficient"}
    tr = float(np.mean(ics_tr)); oo = float(np.mean(ics_oo))
    sig = sum(1 for ic in ics_oo if ic * oo > 0)
    result = {
        "status": "ok",
        "train_ic": tr,
        "oos_ic": oo,
        "sign_agree": sig,
        "n_pairs": len(ics_oo),
        "combined": abs(oo) * sig / 10.0,
        "fitness": abs(oo) * sig / 10.0,
        "passes": abs(oo) >= cfg.ic_gate and sig >= cfg.sign_gate,
        "eval_mode": "legacy",
    }
    if return_oos_series and oos_chunks:
        result["oos_series"] = np.concatenate(oos_chunks)
    return result


def _merge_purification_metrics(
    selected: Dict,
    raw_metrics: Dict,
    clean_metrics: Dict,
    neutralized_metrics: Dict,
    diagnostics: Dict[str, Any],
    mode: str,
) -> Dict:
    out = dict(selected)
    raw_ic = float(raw_metrics.get("oos_ic", 0.0) or 0.0)
    clean_ic = float(clean_metrics.get("oos_ic", 0.0) or 0.0)
    neutralized_ic = float(neutralized_metrics.get("oos_ic", 0.0) or 0.0)
    out.update({
        "raw_ic": raw_ic,
        "clean_ic": clean_ic,
        "neutralized_ic": neutralized_ic,
        "residual_ic_ratio": float(abs(neutralized_ic) / (abs(clean_ic) + 1e-9)),
        "exposure_r2": float(diagnostics.get("exposure_r2", 0.0) or 0.0),
        "max_exposure_corr": float(diagnostics.get("max_exposure_corr", 0.0) or 0.0),
        "exposure_count": int(diagnostics.get("exposure_count", 0) or 0),
        "purify_mode": mode,
    })
    return out


def _blend_selected_metrics(
    raw_metrics: Dict,
    clean_metrics: Dict,
    neutralized_metrics: Dict,
    diagnostics: Dict[str, Any],
) -> Dict:
    selected = dict(neutralized_metrics if neutralized_metrics.get("status") == "ok" else clean_metrics)
    raw_ic = float(raw_metrics.get("oos_ic", 0.0) or 0.0)
    clean_ic = float(clean_metrics.get("oos_ic", 0.0) or 0.0)
    neutralized_ic = float(neutralized_metrics.get("oos_ic", 0.0) or 0.0)
    ratio = abs(neutralized_ic) / (abs(clean_ic) + 1e-9)
    sign_source = neutralized_ic if abs(neutralized_ic) > 1e-12 else clean_ic
    if abs(sign_source) <= 1e-12:
        sign_source = raw_ic
    sign = float(np.sign(sign_source) or 1.0)
    blend_abs = 0.10 * abs(raw_ic) + 0.45 * abs(clean_ic) + 0.45 * abs(neutralized_ic)
    collapse_penalty = max(0.0, min(1.0, ratio / 0.50))
    sign_agree = int(selected.get("sign_agree", 0) or 0)
    selected["oos_ic"] = float(sign * blend_abs)
    selected["combined"] = float(blend_abs * sign_agree / 10.0 * collapse_penalty)
    selected["fitness"] = selected["combined"]
    selected["raw_ic"] = raw_ic
    selected["clean_ic"] = clean_ic
    selected["neutralized_ic"] = neutralized_ic
    selected["residual_ic_ratio"] = float(ratio)
    selected["exposure_r2"] = float(diagnostics.get("exposure_r2", 0.0) or 0.0)
    selected["max_exposure_corr"] = float(diagnostics.get("max_exposure_corr", 0.0) or 0.0)
    selected["exposure_count"] = int(diagnostics.get("exposure_count", 0) or 0)
    selected["purify_mode"] = "blend"
    return selected


def _pure_residual_rejection_reasons(metrics: Dict[str, Any], cfg: MiningConfig) -> List[str]:
    if metrics.get("status") != "ok":
        return ["invalid_expr"]

    reasons: List[str] = []
    neutralized_ic = abs(float(metrics.get("neutralized_ic", metrics.get("oos_ic", 0.0)) or 0.0))
    sign_agree = int(metrics.get("sign_agree", 0) or 0)
    residual_ratio = float(metrics.get("residual_ic_ratio", 0.0) or 0.0)
    exposure_r2 = float(metrics.get("exposure_r2", 0.0) or 0.0)
    max_exposure_corr = float(metrics.get("max_exposure_corr", 0.0) or 0.0)

    if neutralized_ic < float(cfg.pure_residual_ic_gate):
        reasons.append("low_neutralized_ic")
    if sign_agree < int(cfg.pure_residual_sign_gate):
        reasons.append("sign_instability")
    if residual_ratio < float(cfg.pure_residual_ratio_gate):
        reasons.append("low_residual_ratio")
    if exposure_r2 > float(cfg.pure_residual_exposure_r2_gate):
        reasons.append("high_exposure_r2")
    if max_exposure_corr > float(cfg.pure_residual_max_exposure_corr_gate):
        reasons.append("high_exposure_r2")
    return reasons


def _apply_pure_residual_objective(metrics: Dict[str, Any], cfg: MiningConfig) -> Dict[str, Any]:
    out = dict(metrics)
    neutralized_ic = float(out.get("neutralized_ic", out.get("oos_ic", 0.0)) or 0.0)
    sign_agree = int(out.get("sign_agree", 0) or 0)
    n_pairs = max(1, int(out.get("n_pairs", 10) or 10))
    cost_mult = float(out.get("cost_mult", 1.0) or 1.0)
    residual_ratio = max(0.0, float(out.get("residual_ic_ratio", 0.0) or 0.0))
    exposure_r2 = max(0.0, min(1.0, float(out.get("exposure_r2", 0.0) or 0.0)))
    max_exposure_corr = max(0.0, min(1.0, float(out.get("max_exposure_corr", 0.0) or 0.0)))

    sign_agree_ratio = max(0.0, min(1.0, sign_agree / float(n_pairs)))
    residual_ratio_bonus = min(2.0, residual_ratio / max(float(cfg.pure_residual_ratio_gate), 1e-9))
    exposure_penalty = max(0.0, 1.0 - exposure_r2) * max(0.0, 1.0 - max_exposure_corr)
    score = abs(neutralized_ic) * sign_agree_ratio * cost_mult * residual_ratio_bonus * exposure_penalty

    out.update({
        "alpha_objective": "pure_residual",
        "objective_score": float(score),
        "oos_ic": neutralized_ic,
        "combined": float(score),
        "fitness": float(score),
        "pure_residual_gates": _pure_residual_gates(cfg),
    })
    reasons = _pure_residual_rejection_reasons(out, cfg)
    out["reject_reasons"] = reasons
    out["passes"] = out.get("status") == "ok" and not reasons and score > 0.0
    return out


def _candidate_pure_residual_rejection_reasons(cand: CandidateRecord, cfg: MiningConfig) -> List[str]:
    return _pure_residual_rejection_reasons({
        "status": "ok",
        "neutralized_ic": cand.neutralized_ic if cand.neutralized_ic else cand.oos_ic,
        "sign_agree": cand.sign_agree,
        "residual_ic_ratio": cand.residual_ic_ratio,
        "exposure_r2": cand.exposure_r2,
        "max_exposure_corr": cand.max_exposure_corr,
    }, cfg)


# ============================================================
# IC evaluation
# ============================================================

def spearman(x, y) -> float:
    mask = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 300: return float("nan")
    xa, ya = x[mask], y[mask]
    if xa.std() == 0 or ya.std() == 0: return float("nan")
    return float(xa.rank(method="average").corr(ya.rank(method="average")))


def _eval_legacy_purified(big: pd.DataFrame, expr: str, cfg: MiningConfig,
                          return_oos_series: bool) -> Dict:
    versions = factor_versions(big, expr, cfg)
    raw_m = _legacy_metrics_for_series(big, versions["raw"], cfg, return_oos_series=False)
    clean_m = _legacy_metrics_for_series(big, versions["clean"], cfg, return_oos_series=False)
    neutral_m = _legacy_metrics_for_series(big, versions["neutralized"], cfg, return_oos_series=return_oos_series)

    metrics = {"raw": raw_m, "clean": clean_m, "neutralized": neutral_m}
    mode = _effective_purify_mode(cfg)
    if _pure_residual_enabled(cfg):
        out = _merge_purification_metrics(neutral_m, raw_m, clean_m, neutral_m, versions["diagnostics"], "neutralized")
        out["eval_mode"] = "legacy"
        out = _apply_pure_residual_objective(out, cfg)
    elif mode == "blend":
        if clean_m.get("status") != "ok" and neutral_m.get("status") != "ok":
            return {"status": "insufficient"}
        out = _blend_selected_metrics(raw_m, clean_m, neutral_m, versions["diagnostics"])
    else:
        selected_name = "neutralized" if mode == "neutralized" else "clean"
        selected = metrics[selected_name]
        if selected.get("status") != "ok":
            return selected
        out = _merge_purification_metrics(selected, raw_m, clean_m, neutral_m, versions["diagnostics"], mode)

    if not _pure_residual_enabled(cfg):
        out["eval_mode"] = "legacy"
        out["passes"] = (
            out.get("status") == "ok"
            and abs(float(out.get("oos_ic", 0.0) or 0.0)) >= cfg.ic_gate
            and int(out.get("sign_agree", 0) or 0) >= _effective_purify_sign_gate(big, cfg)
            and float(out.get("fitness", out.get("combined", 0.0)) or 0.0) > 0.0
        )
    if return_oos_series and "oos_series" not in out:
        selected_series = versions["selected"] if mode != "clean" else versions["clean"]
        oo = _legacy_metrics_for_series(big, selected_series, cfg, return_oos_series=True)
        if "oos_series" in oo:
            out["oos_series"] = oo["oos_series"]
    out["factor_cache_key"] = str(versions.get("cache_key", ""))
    out["factor_cache_hit"] = bool(versions.get("cache_hit", False))
    return out


def _eval_legacy(big: pd.DataFrame, expr: str, cfg: MiningConfig,
                 return_oos_series: bool) -> Dict:
    """Backwards-compatible path: Spearman IC on train + OOS two sections."""
    if _purification_enabled(cfg):
        return _eval_legacy_purified(big, expr, cfg, return_oos_series)
    ics_tr, ics_oo = [], []
    oos_chunks: List[np.ndarray] = []
    for _, _, sub in _iter_pair_frames(big):
        series = safe_eval_expression(expr, sub)
        m_tr = ((sub["date"] >= pd.Timestamp(cfg.train[0], tz="UTC"))
                & (sub["date"] < pd.Timestamp(cfg.train[1], tz="UTC")))
        m_oo = ((sub["date"] >= pd.Timestamp(cfg.oos[0], tz="UTC"))
                & (sub["date"] < pd.Timestamp(cfg.oos[1], tz="UTC")))
        ic_t = spearman(series[m_tr.values], sub.loc[m_tr, "__fwd_ret__"])
        ic_o = spearman(series[m_oo.values], sub.loc[m_oo, "__fwd_ret__"])
        if not np.isnan(ic_t): ics_tr.append(ic_t)
        if not np.isnan(ic_o): ics_oo.append(ic_o)
        if return_oos_series:
            oos_chunks.append(np.asarray(series[m_oo.values], dtype=np.float64))
    if not ics_tr or not ics_oo:
        return {"status": "insufficient"}
    tr = float(np.mean(ics_tr)); oo = float(np.mean(ics_oo))
    sig = sum(1 for ic in ics_oo if ic * oo > 0)
    passes = abs(oo) >= cfg.ic_gate and sig >= cfg.sign_gate
    result = {"status": "ok", "train_ic": tr, "oos_ic": oo, "sign_agree": sig,
              "n_pairs": len(ics_oo),
              "combined": abs(oo) * sig / 10.0, "fitness": abs(oo) * sig / 10.0,
              "passes": passes, "eval_mode": "legacy"}
    if return_oos_series and oos_chunks:
        result["oos_series"] = np.concatenate(oos_chunks)
    return result


def _eval_composite_series(big: pd.DataFrame, factor: pd.Series, cfg: MiningConfig,
                           return_oos_series: bool) -> Dict:
    factor_wide_parts: Dict[str, pd.Series] = {}
    fwd_wide_parts: Dict[str, pd.Series] = {}
    ts_ics_per_pair_window: List[List[float]] = []
    train_ics: List[float] = []
    per_pair_val_ic: List[float] = []
    oos_chunks: List[np.ndarray] = []
    turnovers: List[float] = []

    series_all = pd.Series(factor, index=big.index, dtype="float64").replace([np.inf, -np.inf], np.nan)
    ts_train = pd.Timestamp(cfg.train3[0], tz="UTC")
    te_train = pd.Timestamp(cfg.train3[1], tz="UTC")
    ts_val   = pd.Timestamp(cfg.val3[0],   tz="UTC")
    te_val   = pd.Timestamp(cfg.val3[1],   tz="UTC")

    for pair, _, sub in _iter_pair_frames(big):
        if len(sub) < 500:
            continue
        s_arr = np.asarray(series_all.loc[sub.index], dtype=np.float64)
        fwd = np.asarray(sub["__fwd_ret__"].values, dtype=np.float64)
        dates = sub["date"].values
        m_tr = (sub["date"] >= ts_train) & (sub["date"] < te_train)
        m_val_full = (sub["date"] >= ts_val) & (sub["date"] < te_val)
        ic_train = F.timeseries_ic(s_arr[m_tr.values], fwd[m_tr.values])
        ic_val = F.timeseries_ic(s_arr[m_val_full.values], fwd[m_val_full.values])
        train_ics.append(ic_train)
        per_pair_val_ic.append(ic_val)
        per_win = F.ics_by_windows(s_arr, fwd, sub["date"], cfg.val_windows)
        ts_ics_per_pair_window.append(per_win)
        turnovers.append(F.turnover(s_arr[m_tr.values]))
        ds_val = pd.Series(dates[m_val_full.values])
        if m_val_full.sum() > 0:
            factor_wide_parts[pair] = pd.Series(s_arr[m_val_full.values],
                                                index=ds_val.values)
            fwd_wide_parts[pair] = pd.Series(fwd[m_val_full.values],
                                             index=ds_val.values)
        if return_oos_series:
            oos_chunks.append(s_arr[m_val_full.values])

    if not train_ics or not per_pair_val_ic:
        return {"status": "insufficient"}

    per_pair_mean_val_ic = [float(np.mean(w)) for w in ts_ics_per_pair_window]
    val_ic_mean = float(np.mean(per_pair_mean_val_ic))
    sign_agree = sum(1 for ic in per_pair_mean_val_ic if ic * val_ic_mean > 0)

    arr = np.asarray(ts_ics_per_pair_window, dtype=np.float64)
    per_window_cross_pair = [float(np.nanmean(arr[:, w])) for w in range(arr.shape[1])]

    if factor_wide_parts:
        factor_wide = pd.DataFrame(factor_wide_parts).sort_index()
        fwd_wide = pd.DataFrame(fwd_wide_parts).sort_index()
        xs_ic = F.cross_sectional_ic(factor_wide, fwd_wide)
    else:
        xs_ic = 0.0

    turnover_mean = float(np.mean(turnovers)) if turnovers else 0.0
    fcfg = F.FitnessConfig(
        fee=cfg.fee_rate, slippage=cfg.slippage,
        turnover_weight=cfg.turnover_weight,
        stability_mode=cfg.stability_mode,
        xs_weight=cfg.xs_weight,
    )
    fit = F.composite_fitness(
        ts_ics=per_window_cross_pair, xs_ic=xs_ic,
        turnover_val=turnover_mean, sign_agree=sign_agree,
        n_pairs=len(per_pair_mean_val_ic),
        train_ic=float(np.mean(train_ics)),
        cfg=fcfg,
    )
    passes = (abs(fit["combined_ic"]) >= cfg.ic_gate
              and sign_agree >= cfg.sign_gate
              and fit["fitness"] > 0)
    result = {
        "status": "ok", "eval_mode": "composite",
        "train_ic": float(np.mean(train_ics)),
        "oos_ic": float(fit["combined_ic"]),
        "ts_agg": float(fit["ts_agg"]),
        "xs_ic": float(xs_ic),
        "turnover": turnover_mean,
        "cost_mult": fit["cost_mult"],
        "sign_agree": sign_agree,
        "n_pairs": len(per_pair_mean_val_ic),
        "combined": fit["fitness"],
        "fitness": fit["fitness"],
        "per_window_ic": per_window_cross_pair,
        "passes": passes,
    }
    if return_oos_series and oos_chunks:
        result["oos_series"] = np.concatenate(oos_chunks)
    return result


def _eval_composite_purified(big: pd.DataFrame, expr: str, cfg: MiningConfig,
                             return_oos_series: bool) -> Dict:
    versions = factor_versions(big, expr, cfg)
    raw_m = _eval_composite_series(big, versions["raw"], cfg, return_oos_series=False)
    clean_m = _eval_composite_series(big, versions["clean"], cfg, return_oos_series=False)
    neutral_m = _eval_composite_series(big, versions["neutralized"], cfg, return_oos_series=return_oos_series)

    mode = _effective_purify_mode(cfg)
    if _pure_residual_enabled(cfg):
        out = _merge_purification_metrics(neutral_m, raw_m, clean_m, neutral_m, versions["diagnostics"], "neutralized")
        out["eval_mode"] = "composite"
        out = _apply_pure_residual_objective(out, cfg)
    elif mode == "blend":
        if clean_m.get("status") != "ok" and neutral_m.get("status") != "ok":
            return {"status": "insufficient"}
        out = _blend_selected_metrics(raw_m, clean_m, neutral_m, versions["diagnostics"])
    else:
        selected = neutral_m if mode == "neutralized" else clean_m
        if selected.get("status") != "ok":
            return selected
        out = _merge_purification_metrics(selected, raw_m, clean_m, neutral_m, versions["diagnostics"], mode)

    if not _pure_residual_enabled(cfg):
        out["eval_mode"] = "composite"
        out["passes"] = (
            out.get("status") == "ok"
            and abs(float(out.get("oos_ic", 0.0) or 0.0)) >= cfg.ic_gate
            and int(out.get("sign_agree", 0) or 0) >= _effective_purify_sign_gate(big, cfg)
            and float(out.get("fitness", out.get("combined", 0.0)) or 0.0) > 0.0
        )
    if return_oos_series and "oos_series" not in out:
        selected_series = versions["selected"] if mode != "clean" else versions["clean"]
        selected_m = _eval_composite_series(big, selected_series, cfg, return_oos_series=True)
        if "oos_series" in selected_m:
            out["oos_series"] = selected_m["oos_series"]
    out["factor_cache_key"] = str(versions.get("cache_key", ""))
    out["factor_cache_hit"] = bool(versions.get("cache_hit", False))
    return out


def _eval_composite(big: pd.DataFrame, expr: str, cfg: MiningConfig,
                    return_oos_series: bool) -> Dict:
    """Composite-fitness path: TS IC in multi-period + XS IC + turnover penalty.

    train3 is used to measure decay/stability; fitness decisions are made on
    VAL3's rolling sub-windows. REAL_TEST3 is never inspected here.
    """
    if _purification_enabled(cfg):
        return _eval_composite_purified(big, expr, cfg, return_oos_series)
    # Per-pair TS IC on each VAL sub-window + full TRAIN3
    factor_wide_parts: Dict[str, pd.Series] = {}
    fwd_wide_parts: Dict[str, pd.Series] = {}
    ts_ics_per_pair_window: List[List[float]] = []  # [pair][window]
    train_ics: List[float] = []
    per_pair_val_ic: List[float] = []  # for sign_agree vs val mean
    oos_chunks: List[np.ndarray] = []
    turnovers: List[float] = []

    ts_train = pd.Timestamp(cfg.train3[0], tz="UTC")
    te_train = pd.Timestamp(cfg.train3[1], tz="UTC")
    ts_val   = pd.Timestamp(cfg.val3[0],   tz="UTC")
    te_val   = pd.Timestamp(cfg.val3[1],   tz="UTC")

    for pair, _, sub in _iter_pair_frames(big):
        if len(sub) < 500:
            continue
        series = safe_eval_expression(expr, sub)
        s_arr = np.asarray(series, dtype=np.float64)
        fwd = np.asarray(sub["__fwd_ret__"].values, dtype=np.float64)
        dates = sub["date"].values
        m_tr = (sub["date"] >= ts_train) & (sub["date"] < te_train)
        m_val_full = (sub["date"] >= ts_val) & (sub["date"] < te_val)
        ic_train = F.timeseries_ic(s_arr[m_tr.values], fwd[m_tr.values])
        ic_val = F.timeseries_ic(s_arr[m_val_full.values], fwd[m_val_full.values])
        train_ics.append(ic_train)
        per_pair_val_ic.append(ic_val)
        # Multi-period IC inside VAL
        per_win = F.ics_by_windows(s_arr, fwd, sub["date"], cfg.val_windows)
        ts_ics_per_pair_window.append(per_win)
        # Turnover on TRAIN (where the signal should be stable)
        turnovers.append(F.turnover(s_arr[m_tr.values]))
        # Keep wide-format factor for XS IC (on VAL only)
        ds_val = pd.Series(dates[m_val_full.values])
        if m_val_full.sum() > 0:
            factor_wide_parts[pair] = pd.Series(s_arr[m_val_full.values],
                                                index=ds_val.values)
            fwd_wide_parts[pair] = pd.Series(fwd[m_val_full.values],
                                             index=ds_val.values)
        if return_oos_series:
            oos_chunks.append(s_arr[m_val_full.values])

    if not train_ics or not per_pair_val_ic:
        return {"status": "insufficient"}

    # Mean per-pair IC across all VAL sub-windows (one number per pair)
    per_pair_mean_val_ic = [float(np.mean(w)) for w in ts_ics_per_pair_window]
    val_ic_mean = float(np.mean(per_pair_mean_val_ic))
    sign_agree = sum(1 for ic in per_pair_mean_val_ic if ic * val_ic_mean > 0)

    # Collapse multi-period list: we already have [pair][window]. For stability
    # we average across pairs per window first, then aggregate across windows.
    arr = np.asarray(ts_ics_per_pair_window, dtype=np.float64)
    per_window_cross_pair = [float(np.nanmean(arr[:, w])) for w in range(arr.shape[1])]

    # XS IC across pairs on VAL
    if factor_wide_parts:
        factor_wide = pd.DataFrame(factor_wide_parts).sort_index()
        fwd_wide = pd.DataFrame(fwd_wide_parts).sort_index()
        xs_ic = F.cross_sectional_ic(factor_wide, fwd_wide)
    else:
        xs_ic = 0.0

    turnover_mean = float(np.mean(turnovers)) if turnovers else 0.0
    fcfg = F.FitnessConfig(
        fee=cfg.fee_rate, slippage=cfg.slippage,
        turnover_weight=cfg.turnover_weight,
        stability_mode=cfg.stability_mode,
        xs_weight=cfg.xs_weight,
    )
    fit = F.composite_fitness(
        ts_ics=per_window_cross_pair, xs_ic=xs_ic,
        turnover_val=turnover_mean, sign_agree=sign_agree,
        n_pairs=len(per_pair_mean_val_ic),
        train_ic=float(np.mean(train_ics)),
        cfg=fcfg,
    )
    passes = (abs(fit["combined_ic"]) >= cfg.ic_gate
              and sign_agree >= cfg.sign_gate
              and fit["fitness"] > 0)
    result = {
        "status": "ok", "eval_mode": "composite",
        "train_ic": float(np.mean(train_ics)),
        "oos_ic": float(fit["combined_ic"]),
        "ts_agg": float(fit["ts_agg"]),
        "xs_ic": float(xs_ic),
        "turnover": turnover_mean,
        "cost_mult": fit["cost_mult"],
        "sign_agree": sign_agree,
        "n_pairs": len(per_pair_mean_val_ic),
        "combined": fit["fitness"],
        "fitness": fit["fitness"],
        "per_window_ic": per_window_cross_pair,
        "passes": passes,
    }
    if return_oos_series and oos_chunks:
        result["oos_series"] = np.concatenate(oos_chunks)
    return result


def eval_ic(big: pd.DataFrame, expr: str, cfg: MiningConfig,
            return_oos_series: bool = False) -> Dict:
    cache = _cache_for_cfg(cfg)
    cache_key = stable_hash(_eval_payload(big, expr, cfg))
    if _cache_enabled(cfg):
        cached = cache.load_eval(cache_key, need_oos_series=return_oos_series)
        if cached is not None:
            return cached
    try:
        if _is_composite_eval_mode(cfg):
            out = _eval_composite(big, expr, cfg, return_oos_series)
        else:
            out = _eval_legacy(big, expr, cfg, return_oos_series)
    except Exception as e:
        out = {"status": "error", "error": str(e)[:200]}
    out["cache_key"] = cache_key
    out["cache_hit"] = False
    if _cache_enabled(cfg):
        cache.save_eval(cache_key, out)
    return out


# ============================================================
# Novelty gate — reject candidates whose |Spearman rank corr| with any
# existing survivor exceeds a threshold. Stops 30 monotone-variant
# wrappers of the same signal from all surviving.
# ============================================================

def _series_to_ranks(series: np.ndarray) -> Optional[np.ndarray]:
    """Convert an arbitrary series into dense ranks, dropping NaN/inf.
    Returns None if too few valid observations to compare reliably."""
    mask = np.isfinite(series)
    if mask.sum() < 200:
        return None
    vals = series[mask]
    # Fast dense rank via argsort — ties get average ranks via bottleneck-like pattern.
    order = np.argsort(vals, kind="mergesort")
    ranks = np.empty(len(vals), dtype=np.float64)
    ranks[order] = np.arange(1, len(vals) + 1, dtype=np.float64)
    # Full array aligned to original length (NaN where missing).
    out = np.full(series.shape, np.nan, dtype=np.float64)
    out[mask] = ranks
    return out


def _rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson corr on pre-ranked arrays == Spearman rank corr.
    Both inputs must be aligned (same length) with NaN where invalid."""
    if a.shape != b.shape:
        n = min(len(a), len(b))
        if n <= 0:
            return 0.0
        a = a[:n]
        b = b[:n]
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 200:
        return 0.0
    av, bv = a[mask], b[mask]
    if av.std() == 0 or bv.std() == 0:
        return 0.0
    return float(np.corrcoef(av, bv)[0, 1])


def _novelty_ok(cand_ranks: Optional[np.ndarray],
                survivor_ranks: Dict[str, np.ndarray],
                gate: float) -> Tuple[bool, float, str]:
    """Return (is_novel, max_abs_corr, collision_expr)."""
    if cand_ranks is None:
        return True, 0.0, ""
    best = 0.0
    worst_expr = ""
    for expr, r in survivor_ranks.items():
        c = abs(_rank_corr(cand_ranks, r))
        if c > best:
            best = c
            worst_expr = expr
        if c >= gate:
            return False, c, expr
    return True, best, worst_expr


# ============================================================
# Diversity metadata + selection
# ============================================================

_CANONICAL_WRAPPERS = {
    "z", "zscore", "ts_z", "robust_z", "ema", "roll_mean", "rolling_mean",
    "decay_linear", "sign", "tanh", "clip", "winsorize", "abs",
}
_CALL_SYNONYMS = {"rolling_mean": "roll_mean"}


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id.lower()
    if isinstance(node, ast.Attribute):
        return node.attr.lower()
    return ""


def _canonical_fallback(expr: str) -> str:
    text = re.sub(r"\s+", "", expr.lower())
    text = re.sub(r"\b(ema|z|zscore|ts_z|robust_z|roll_mean|rolling_mean|sign|tanh)\(", "wrap(", text)
    text = re.sub(r"\d+(?:\.\d+)?", "n", text)
    return text[:240]


def _canonical_ast(node: ast.AST) -> str:
    if isinstance(node, ast.Expression):
        return _canonical_ast(node.body)
    if isinstance(node, ast.Name):
        return node.id.lower()
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return "num"
        return repr(node.value).lower()
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        # `-x` is a sign-flipped variant, not an independent source.
        return _canonical_ast(node.operand)
    if isinstance(node, ast.BinOp):
        left = _canonical_ast(node.left)
        right = _canonical_ast(node.right)
        if isinstance(node.op, (ast.Add, ast.Sub)):
            return "lin(" + ",".join(sorted((left, right))) + ")"
        if isinstance(node.op, ast.Mult):
            return "mul(" + ",".join(sorted((left, right))) + ")"
        if isinstance(node.op, ast.Div):
            return f"div({left},{right})"
        return f"bin({type(node.op).__name__.lower()},{left},{right})"
    if isinstance(node, ast.BoolOp):
        op = "and" if isinstance(node.op, ast.And) else "or"
        vals = sorted(_canonical_ast(v) for v in node.values)
        return f"{op}(" + ",".join(vals) + ")"
    if isinstance(node, ast.Compare):
        left = _canonical_ast(node.left)
        parts = [left]
        for op, comp in zip(node.ops, node.comparators):
            parts.append(type(op).__name__.lower())
            parts.append(_canonical_ast(comp))
        return "cmp(" + ",".join(parts) + ")"
    if isinstance(node, ast.Call):
        name = _CALL_SYNONYMS.get(_call_name(node.func), _call_name(node.func))
        if name in _CANONICAL_WRAPPERS and node.args:
            return _canonical_ast(node.args[0])
        args = [_canonical_ast(arg) for arg in node.args]
        if name in {"max", "min"}:
            args = sorted(args)
        return f"{name}(" + ",".join(args) + ")"
    if isinstance(node, ast.IfExp):
        return f"if({ _canonical_ast(node.test)},{_canonical_ast(node.body)},{_canonical_ast(node.orelse)})"
    return _canonical_fallback(ast.dump(node, include_attributes=False))


def canonical_signature(expr: str) -> str:
    """Return a source-level signature where monotone wrappers/sign flips collapse."""
    raw = str(expr or "").strip()
    if not raw:
        return ""
    try:
        tree = ast.parse(raw, mode="eval")
    except SyntaxError:
        return _canonical_fallback(raw)
    return _canonical_ast(tree)[:240]


def infer_family_tags(expr: str) -> Tuple[str, ...]:
    """Infer coarse signal families from expression text.

    Families are intentionally broad and conservative; the OOS rank correlation
    remains the primary independence test.
    """
    text = str(expr or "").lower()
    patterns = {
        "funding": r"\bfunding|open_?interest|\boi\b|premium|perp|basis",
        "micro": r"\bmicro|lob|order_?book|bid|ask|spread|ofi|imbalance|depth|queue|taker|trade_|vwap",
        "cross_sectional": r"rank_xs|\bxs_|cross_?section|relative_strength|btc_rel|btc_.*rel|rel_?btc|pair_rank|\bpair_",
        "volatility": r"volatility|\bvol\b|realized_vol|atr|roll_std|rolling_std|std|range|drawdown|crash|bb_|band",
        "mtf": r"\bmtf|4h|1d|higher_?tf|multi_?tf",
        "regime": r"\bifelse\b|\bwhere\b|adx|regime|trend_filter|risk_?on|risk_?off",
        "trend": r"ema|sma|tema|macd|momentum|pct_change|diff|return|close|di_spread|trend|rsi|mfi",
    }
    tags = [family for family in DEFAULT_FAMILIES if re.search(patterns[family], text)]
    if not tags:
        tags = ["trend"]
    return tuple(tags)


def primary_family(expr: str) -> str:
    tags = infer_family_tags(expr)
    priority = ("funding", "micro", "cross_sectional", "volatility", "mtf", "regime", "trend")
    for family in priority:
        if family in tags:
            return family
    return tags[0] if tags else "trend"


def annotate_diversity(cand: CandidateRecord) -> CandidateRecord:
    tags = infer_family_tags(cand.expression)
    cand.family_tags = tuple(tags)
    cand.primary_family = primary_family(cand.expression)
    cand.canonical_signature = canonical_signature(cand.expression)
    return cand


def _base_score(cand: CandidateRecord, score_mode: str = "combined") -> float:
    mode = (score_mode or "combined").lower()
    if mode in {"fitness", "portfolio"} and cand.fitness:
        return float(cand.fitness)
    return float(cand.combined)


def _portfolio_key(cand: CandidateRecord, score_mode: str = "combined") -> Tuple[float, float, float, float]:
    base = _base_score(cand, score_mode)
    return (
        base,
        abs(float(cand.stability_ic or cand.oos_ic or 0.0)),
        float(cand.sign_agree or 0),
        abs(float(cand.oos_ic or 0.0)),
    )


def _family_limit_for_top_n(max_same_family_in_top40: int, top_n: int) -> int:
    if max_same_family_in_top40 <= 0:
        return top_n
    return max(1, int(np.ceil(max_same_family_in_top40 * max(top_n, 1) / 40.0)))


def _max_corr_to_selected(
    cand: CandidateRecord,
    selected: Sequence[CandidateRecord],
    rank_cache: Dict[str, np.ndarray],
) -> Tuple[float, str]:
    ranks = rank_cache.get(cand.expression)
    if ranks is None or not selected:
        return 0.0, ""
    best = 0.0
    best_expr = ""
    for kept in selected:
        kept_ranks = rank_cache.get(kept.expression)
        if kept_ranks is None:
            continue
        corr = abs(_rank_corr(ranks, kept_ranks))
        if corr > best:
            best = corr
            best_expr = kept.expression
    return best, best_expr


def select_diverse_candidates(
    candidates: Sequence[CandidateRecord],
    rank_cache: Dict[str, np.ndarray],
    *,
    top_n: int,
    hard_corr_gate: float = 0.85,
    soft_corr_penalty_start: float = 0.55,
    max_same_family: int = 8,
    max_same_signature: int = 2,
    score_mode: str = "combined",
) -> Tuple[List[CandidateRecord], List[Dict[str, Any]]]:
    """Greedy low-correlation selector used by both mining and export."""
    remaining = [annotate_diversity(c) for c in candidates if c.expression]
    original_order = {id(c): i for i, c in enumerate(remaining)}
    selected: List[CandidateRecord] = []
    rejected: List[Dict[str, Any]] = []
    family_counts: Dict[str, int] = {}
    signature_counts: Dict[str, int] = {}

    hard_gate = float(hard_corr_gate)
    soft_start = float(soft_corr_penalty_start)

    while remaining and len(selected) < top_n:
        best_idx: Optional[int] = None
        best_key: Optional[Tuple[float, float, float, float, float]] = None
        best_corr = 0.0
        best_collision = ""

        for idx, cand in enumerate(remaining):
            family = cand.primary_family or primary_family(cand.expression)
            signature = cand.canonical_signature or canonical_signature(cand.expression)
            if max_same_family > 0 and family_counts.get(family, 0) >= max_same_family:
                continue
            if max_same_signature > 0 and signature_counts.get(signature, 0) >= max_same_signature:
                continue

            max_corr, collision = _max_corr_to_selected(cand, selected, rank_cache)
            if hard_gate < 1.0 and max_corr >= hard_gate:
                continue

            base, stability, sign_agree, abs_ic = _portfolio_key(cand, score_mode)
            penalty = 0.0
            if soft_start < 1.0 and max_corr > soft_start:
                penalty = (max_corr - soft_start) * max(abs(base), 0.01)
            adjusted = base - penalty
            key = (adjusted, stability, sign_agree, abs_ic, -float(original_order[id(cand)]))
            if best_key is None or key > best_key:
                best_idx = idx
                best_key = key
                best_corr = max_corr
                best_collision = collision

        if best_idx is None:
            break

        chosen = remaining.pop(best_idx)
        chosen.max_corr_to_kept = float(best_corr)
        family_counts[chosen.primary_family] = family_counts.get(chosen.primary_family, 0) + 1
        signature_counts[chosen.canonical_signature] = signature_counts.get(chosen.canonical_signature, 0) + 1
        selected.append(chosen)

    selected_exprs = {c.expression for c in selected}
    for cand in remaining:
        family = cand.primary_family or primary_family(cand.expression)
        signature = cand.canonical_signature or canonical_signature(cand.expression)
        max_corr, collision = _max_corr_to_selected(cand, selected, rank_cache)
        if max_same_family > 0 and family_counts.get(family, 0) >= max_same_family:
            reason = "family_quota"
        elif max_same_signature > 0 and signature_counts.get(signature, 0) >= max_same_signature:
            reason = "signature_quota"
        elif hard_gate < 1.0 and max_corr >= hard_gate:
            reason = "corr_gate"
        else:
            reason = "not_selected"
        rejected.append({
            "expression": cand.expression,
            "primary_family": family,
            "canonical_signature": signature,
            "reason": reason,
            "max_corr_to_kept": float(max_corr),
            "collision_expression": collision,
        })
    # Include any duplicate object that might have been removed upstream.
    for cand in candidates:
        if cand.expression not in selected_exprs and all(r["expression"] != cand.expression for r in rejected):
            rejected.append({"expression": cand.expression, "reason": "not_selected"})
    return selected, rejected


def _select_for_mining(
    candidates: Sequence[CandidateRecord],
    rank_cache: Dict[str, np.ndarray],
    cfg: MiningConfig,
) -> Tuple[List[CandidateRecord], List[Dict[str, Any]]]:
    family_limit = _family_limit_for_top_n(cfg.max_same_family_in_top40, cfg.top_k)
    hard_gate = min(float(cfg.hard_corr_gate), float(cfg.novelty_gate))
    return select_diverse_candidates(
        candidates,
        rank_cache,
        top_n=cfg.top_k,
        hard_corr_gate=hard_gate,
        soft_corr_penalty_start=cfg.soft_corr_penalty_start,
        max_same_family=family_limit,
        max_same_signature=cfg.max_same_signature,
        score_mode="portfolio",
    )


def _needs_diversity_rank_cache(cfg: MiningConfig) -> bool:
    return (
        min(float(cfg.hard_corr_gate), float(cfg.novelty_gate)) < 1.0
        or float(cfg.soft_corr_penalty_start) < 1.0
    )


def _rank_cache_for_selected(
    selected: Sequence[CandidateRecord],
    rank_cache: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    selected_exprs = {c.expression for c in selected}
    return {expr: ranks for expr, ranks in rank_cache.items() if expr in selected_exprs}


_DSL_FUNCTION_NAMES = {
    "z", "ts_z", "zscore", "shift", "diff", "roll_mean", "roll_std",
    "rolling_mean", "rolling_std", "rolling_sum", "pct_change", "sign",
    "clip", "ema", "rolling_max", "rolling_min", "decay_linear",
    "winsorize", "robust_z", "log1p", "log", "exp", "sqrt", "tanh",
    "abs", "ifelse", "rank_xs", "zscore_xs", "corr_xs", "neutralize",
    "fill_prob", "impact_proxy", "queue_pos_proxy", "max", "min", "sum",
    "mean", "median",
}


def _expression_feature_names(expr: str) -> set[str]:
    try:
        tree = ast.parse(str(expr or "").strip(), mode="eval")
    except SyntaxError:
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id not in _DSL_FUNCTION_NAMES:
            if node.id not in {"True", "False", "None"}:
                names.add(node.id)
    return names


def _feature_quality_stats(big: pd.DataFrame, columns: Sequence[str]) -> Dict[str, Dict[str, float | int]]:
    n_rows = max(1, int(len(big)))
    stats: Dict[str, Dict[str, float | int]] = {}
    for col in columns:
        if col not in big.columns:
            continue
        # Some parquet-backed arrays can produce misleading reductions when
        # viewed without copying on this stack; force a plain NumPy buffer once.
        values = pd.to_numeric(big[col], errors="coerce").to_numpy(dtype=np.float64, copy=True)
        finite = np.isfinite(values)
        finite_count = int(finite.sum())
        if finite_count > 1:
            std = float(np.std(values[finite]))
        else:
            std = 0.0
        stats[col] = {
            "coverage": float(finite_count / n_rows),
            "finite_count": finite_count,
            "std": std,
        }
    return stats


def _feature_quality_ok(stat: Optional[Dict[str, float | int]], cfg: MiningConfig) -> bool:
    if not bool(getattr(cfg, "llm_filter_low_coverage", True)):
        return True
    if not stat:
        return True
    return (
        float(stat.get("coverage", 0.0) or 0.0) >= float(cfg.llm_min_feature_coverage)
        and int(stat.get("finite_count", 0) or 0) >= int(cfg.llm_min_feature_rows)
        and float(stat.get("std", 0.0) or 0.0) > 1e-12
    )


def _llm_usable_feature_cols(
    base_cols: Sequence[str],
    feature_stats: Dict[str, Dict[str, float | int]],
    cfg: MiningConfig,
) -> List[str]:
    if not bool(getattr(cfg, "llm_filter_low_coverage", True)):
        return list(base_cols)
    return [col for col in base_cols if _feature_quality_ok(feature_stats.get(col), cfg)]


def _feature_rejection_detail(
    expr: str,
    *,
    allowed_columns: set[str],
    feature_stats: Dict[str, Dict[str, float | int]],
    cfg: MiningConfig,
) -> Tuple[str, str]:
    refs = _expression_feature_names(expr)
    missing = sorted(ref for ref in refs if ref not in allowed_columns)
    if missing:
        return "invalid_expr", "unknown features: " + ", ".join(missing[:8])
    if not bool(getattr(cfg, "llm_filter_low_coverage", True)):
        return "", ""
    bad = sorted(ref for ref in refs if ref in feature_stats and not _feature_quality_ok(feature_stats.get(ref), cfg))
    if not bad:
        return "", ""
    parts = []
    for ref in bad[:8]:
        stat = feature_stats.get(ref, {})
        parts.append(
            f"{ref}(coverage={float(stat.get('coverage', 0.0) or 0.0):.4f},"
            f"rows={int(stat.get('finite_count', 0) or 0)})"
        )
    return "low_feature_coverage", "; ".join(parts)


def _format_feature_filter_note(
    base_cols: Sequence[str],
    usable_cols: Sequence[str],
    feature_stats: Dict[str, Dict[str, float | int]],
    cfg: MiningConfig,
) -> str:
    if not bool(getattr(cfg, "llm_filter_low_coverage", True)):
        return ""
    usable = set(usable_cols)
    dropped = [col for col in base_cols if col not in usable]
    if not dropped:
        return (
            "\n\n## Feature availability guard\n"
            "Only use columns listed above; all listed columns passed the data coverage and non-constant checks."
        )
    dropped_sorted = sorted(
        dropped,
        key=lambda c: (
            float(feature_stats.get(c, {}).get("coverage", 0.0) or 0.0),
            int(feature_stats.get(c, {}).get("finite_count", 0) or 0),
            c,
        ),
    )
    examples = []
    for col in dropped_sorted[:18]:
        stat = feature_stats.get(col, {})
        examples.append(
            f"{col}(coverage={float(stat.get('coverage', 0.0) or 0.0):.4f},"
            f"rows={int(stat.get('finite_count', 0) or 0)})"
        )
    return (
        "\n\n## Feature availability guard\n"
        "Use ONLY columns listed in the Available features section. Columns omitted from that list failed "
        f"coverage >= {float(cfg.llm_min_feature_coverage):.2f}, "
        f"rows >= {int(cfg.llm_min_feature_rows)}, and non-constant checks. "
        "Do not use omitted sparse L2/orderbook columns even if their names are familiar.\n"
        "Omitted examples: " + "; ".join(examples)
    )


def _prompt_elites_by_family(survivors: Sequence[CandidateRecord], limit: int = 12) -> List[str]:
    """Pick prompt examples by family instead of a single top-score leaderboard."""
    by_family: Dict[str, List[CandidateRecord]] = {family: [] for family in DEFAULT_FAMILIES}
    for cand in sorted((annotate_diversity(c) for c in survivors), key=lambda c: _portfolio_key(c, "portfolio"), reverse=True):
        by_family.setdefault(cand.primary_family, []).append(cand)

    out: List[str] = []
    for family in DEFAULT_FAMILIES:
        if by_family.get(family):
            out.append(by_family[family][0].expression)
            if len(out) >= limit:
                return out
    for cand in sorted(survivors, key=lambda c: _portfolio_key(c, "portfolio"), reverse=True):
        if cand.expression not in out:
            out.append(cand.expression)
            if len(out) >= limit:
                break
    return out


# ============================================================
# Python compositional offspring
# ============================================================

def gen_python_offspring(top_exprs: List[str], round_idx: int, n: int = 10,
                          available_cols: Optional[set] = None) -> List[str]:
    if not top_exprs: return []
    out = set()
    rng = random.Random(42 + round_idx)
    wrappers = ["z({x})", "ema({x}, 12)", "ema({x}, 24)", "sign({x})", "tanh({x})",
                "rolling_max({x}, 24)", "rolling_min({x}, 24)", "roll_std({x}, 24)", "-({x})"]
    base_regimes = ["adx_14 > 25", "adx_14 < 20", "di_spread > 0", "di_spread < 0",
                    "rsi_14 > 70", "rsi_14 < 30"]
    mtf_regimes = ["mtf4h_rsi_14 > 60", "mtf4h_rsi_14 < 40"]
    funding_regimes = [
        "funding_z_200 > 1", "funding_z_200 < -1",
        "funding_cumsum_72 > 0", "funding_cumsum_72 < 0",
        "abs(funding_rate) > 0.0002",
    ]
    vol_regimes = [
        "realized_vol_24 > realized_vol_72",
        "volume_zscore_24 > 1", "volume_zscore_24 < -1",
        "atr_norm_14 > 0.02",
    ]
    # Only include a regime family if its columns exist in the feature matrix.
    # This lets the same generator work on 1h (with mtf4h+funding), 4h (neither),
    # and future timeframes without manual per-tf branches.
    regimes = list(base_regimes)
    if available_cols is None or "mtf4h_rsi_14" in available_cols:
        regimes += mtf_regimes
    if available_cols is None or "funding_z_200" in available_cols:
        regimes += funding_regimes
    if available_cols is None or "volume_zscore_24" in available_cols:
        regimes += vol_regimes
    deep_tmpls = ["z(ema({x}, 12) - ema({x}, 48))",
                  "tanh(rolling_max({x}, 24) - rolling_min({x}, 24))",
                  "sign({x}) * roll_std({x}, 24)"]

    attempts = 0
    while len(out) < n * 3 and attempts < n * 10:
        attempts += 1
        mode = rng.choice(["arith","wrap","regime","deep"])
        if mode == "arith" and len(top_exprs) >= 2:
            a, b = rng.sample(top_exprs, 2); op = rng.choice(["+","-","*","/"])
            out.add(f"({a}) / ((abs({b})) + 1e-6)" if op == "/" else f"({a}) {op} ({b})")
        elif mode == "wrap":
            out.add(rng.choice(wrappers).format(x=rng.choice(top_exprs)))
        elif mode == "regime" and len(top_exprs) >= 2:
            a, b = rng.sample(top_exprs, 2)
            out.add(f"ifelse({rng.choice(regimes)}, ({a}), ({b}))")
        elif mode == "deep":
            try: out.add(rng.choice(deep_tmpls).format(x=rng.choice(top_exprs)))
            except: pass
    return [e for e in out if 4 < len(e) < MAX_EXPR_LEN][:n]


# ============================================================
# Seed pool
# ============================================================

def load_seeds() -> List[str]:
    from .paths import USER_DATA, EXPRESSIONS_FILE

    seeds: set = set()

    def _pull_json(path: Path) -> None:
        if not path.exists(): return
        try:
            d = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            return
        # Shape A: scored miner output
        for c in d.get("candidates", []) or []:
            e = (c.get("expression") or "").strip()
            if e: seeds.add(e)
        # Shape B: canonical freqai_expressions.json
        for e in d.get("expressions", []) or []:
            expr = (e.get("expression") or "").strip()
            if expr: seeds.add(expr)

    _pull_json(EXPRESSIONS_SCORED)
    _pull_json(EXPRESSIONS_FILE)
    _pull_json(USER_DATA / "freqai_expressions.g13.bak.json")
    _pull_json(USER_DATA / "freqai_expressions.production_backup.json")
    # archived libraries contribute too (dedup via set)
    archive = USER_DATA / "factor_lib_archive"
    if archive.exists():
        for p in sorted(archive.glob("freqai_expressions*.json")):
            _pull_json(p)
    return list(seeds)


# ============================================================
# State management (resume-safe)
# ============================================================

def state_dir(tag: str) -> Path:
    d = LAB_STATE / "mining" / tag
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_state(
    tag: str,
    loop: int,
    survivors: List[CandidateRecord],
    evaluated: set,
    cfg: Optional[MiningConfig] = None,
    rejection_summary: Optional[Dict[str, Any]] = None,
):
    sd = state_dir(tag)
    state = {
        "loop": loop, "survivors": [asdict(c) for c in survivors],
        "all_evaluated_count": len(evaluated),
        "all_evaluated": list(evaluated),
        "timestamp": time.time(),
    }
    if cfg is not None:
        state["config"] = asdict(cfg)
        state.update(mining_lane_manifest(cfg))
        state["intraday"] = mining_lane_manifest(cfg)
        state["alpha_objective"] = _alpha_objective(cfg)
        state["cache_stats"] = _cache_for_cfg(cfg).snapshot()
        if _pure_residual_enabled(cfg):
            state["pure_residual_gates"] = _pure_residual_gates(cfg)
    if rejection_summary is not None:
        state["rejection_summary"] = rejection_summary
    (sd / "latest.json").write_text(json.dumps(state), encoding="utf-8")
    (sd / f"state_{loop:04d}.json").write_text(json.dumps(state), encoding="utf-8")


def load_state_config(tag: str) -> Dict[str, Any]:
    sd = LAB_STATE / "mining" / tag / "latest.json"
    if not sd.exists():
        return {}
    try:
        d = json.loads(sd.read_text(encoding="utf-8"))
    except Exception:
        return {}
    cfg = d.get("config")
    return cfg if isinstance(cfg, dict) else {}


def load_state(tag: str) -> Optional[Tuple[int, List[CandidateRecord], set]]:
    sd = state_dir(tag) / "latest.json"
    if not sd.exists(): return None
    d = json.loads(sd.read_text())
    allowed = {f.name for f in fields(CandidateRecord)}
    sur = [CandidateRecord(**{k: v for k, v in r.items() if k in allowed}) for r in d["survivors"]]
    return int(d["loop"]), sur, set(d.get("all_evaluated", []))


FUTURES_LEAN_VENUES = {"okx", "bybit", "binance"}


def _lean_gate_enabled(cfg: MiningConfig) -> bool:
    try:
        return int(getattr(cfg, "lean_gate_every", 0) or 0) > 0
    except (TypeError, ValueError):
        return False


def _should_run_loop_lean_gate(cfg: MiningConfig, loop: int) -> bool:
    if not _lean_gate_enabled(cfg):
        return False
    every = max(1, int(getattr(cfg, "lean_gate_every", 0) or 0))
    return loop % every == 0 or loop == int(cfg.rounds)


def _effective_lean_gate_venue(cfg: MiningConfig) -> str:
    raw = str(getattr(cfg, "lean_gate_venue", "auto") or "auto").strip().lower()
    if raw != "auto":
        return raw
    data_venue = str(getattr(cfg, "data_venue", "") or "").strip().lower()
    return data_venue if data_venue in FUTURES_LEAN_VENUES else "okx"


def _effective_lean_gate_data_venue(cfg: MiningConfig) -> str:
    raw = str(getattr(cfg, "lean_gate_data_venue", "auto") or "auto").strip().lower()
    if raw != "auto":
        return raw
    data_venue = str(getattr(cfg, "data_venue", "") or "").strip().lower()
    return data_venue if data_venue else "auto"


def _lean_gate_rank_kwargs(cfg: MiningConfig) -> Dict[str, Any]:
    return {
        "top_k": int(getattr(cfg, "lean_gate_rank_top_k", 2) or 2),
        "gross_cap": float(getattr(cfg, "lean_gate_gross_cap", 2.0) or 2.0),
        "net_cap": float(getattr(cfg, "lean_gate_net_cap", 2.0) or 2.0),
        "single_pair_cap": float(getattr(cfg, "lean_gate_single_pair_cap", 2.0) or 2.0),
        "side_mode": str(getattr(cfg, "lean_gate_side_mode", "short") or "short"),
        "min_abs_score_z": float(getattr(cfg, "lean_gate_score_threshold", 1.5) or 0.0),
        "rebalance_hours": int(getattr(cfg, "lean_gate_rebalance_hours", 8) or 8),
        "rebalance_minutes": (
            int(getattr(cfg, "lean_gate_rebalance_minutes", 0))
            if int(getattr(cfg, "lean_gate_rebalance_minutes", 0) or 0) > 0
            else None
        ),
        "risk_per_trade": float(getattr(cfg, "lean_gate_risk_per_trade", 0.08) or 0.08),
        "leverage_cap": float(getattr(cfg, "lean_gate_leverage_cap", 5.0) or 5.0),
        "recompute_corr": bool(getattr(cfg, "lean_gate_recompute_corr", False)),
    }


def _cfg_float(cfg: MiningConfig, name: str, default: float) -> float:
    raw = getattr(cfg, name, default)
    if raw is None or raw == "":
        return float(default)
    return float(raw)


def _cfg_int(cfg: MiningConfig, name: str, default: int) -> int:
    raw = getattr(cfg, name, default)
    if raw is None or raw == "":
        return int(default)
    return int(raw)


def _lean_gate_summary(result: Mapping[str, Any], loop: int) -> Dict[str, Any]:
    artifacts = result.get("artifacts") if isinstance(result.get("artifacts"), dict) else {}
    return {
        "loop": int(loop),
        "status": result.get("status"),
        "reason": result.get("reason"),
        "comparison_status": result.get("comparison_status"),
        "violations": result.get("violations") if isinstance(result.get("violations"), list) else [],
        "duration_sec": result.get("duration_sec"),
        "summary": artifacts.get("summary"),
        "comparison_json": artifacts.get("comparison_json"),
        "lean_project": artifacts.get("lean_project"),
        "lean_result": artifacts.get("lean_result"),
    }


def _record_loop_lean_gate(tag: str, loop: int, result: Mapping[str, Any]) -> None:
    sd = state_dir(tag)
    summary = _lean_gate_summary(result, loop)
    history_path = sd / "lean_gate_history.jsonl"
    with history_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(summary, sort_keys=True, default=str) + "\n")
    for state_path in (sd / "latest.json", sd / f"state_{loop:04d}.json"):
        if not state_path.exists():
            continue
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state["lean_gate_latest"] = summary
            state["lean_gate_history"] = str(history_path)
            state_path.write_text(json.dumps(state), encoding="utf-8")
        except Exception:
            pass


def _run_loop_lean_gate(tag: str, loop: int, cfg: MiningConfig) -> Dict[str, Any]:
    from . import mine_lean_gate

    sd = state_dir(tag)
    candidate_state = sd / f"state_{loop:04d}.json"
    if not candidate_state.exists():
        raise FileNotFoundError(f"mining state snapshot missing for LEAN gate: {candidate_state}")
    run_id = f"loop_{loop:04d}"
    result = mine_lean_gate.run_mine_lean_gate(
        tag=tag,
        n=int(getattr(cfg, "lean_gate_n", 30) or 30),
        candidate_state=candidate_state,
        run_id=run_id,
        output=sd / "lean_gate" / run_id,
        rank_tag=f"{tag}_lean_{run_id}",
        venue=_effective_lean_gate_venue(cfg),
        timeframe=cfg.timeframe,
        data_venue=_effective_lean_gate_data_venue(cfg),
        start=str(getattr(cfg, "lean_gate_start", "2025-12-01") or "2025-12-01"),
        end=str(getattr(cfg, "lean_gate_end", "2026-04-12") or "2026-04-12"),
        lean_bin=str(getattr(cfg, "lean_gate_bin", "lean") or "lean"),
        lean_timeout=(
            int(getattr(cfg, "lean_gate_timeout", 0))
            if int(getattr(cfg, "lean_gate_timeout", 0) or 0) > 0
            else None
        ),
        lean_data_root=(str(getattr(cfg, "lean_gate_data_root", "") or "") or None),
        lean_required_status=str(getattr(cfg, "lean_gate_required_status", "ok") or "ok"),
        min_final_equity=_cfg_float(cfg, "lean_gate_min_final_equity", 1.0),
        max_drawdown_pct=_cfg_float(cfg, "lean_gate_max_drawdown_pct", 25.0),
        min_trades=_cfg_int(cfg, "lean_gate_min_trades", 80),
        force=bool(getattr(cfg, "lean_gate_force", True)),
        rank_kwargs=_lean_gate_rank_kwargs(cfg),
    )
    _record_loop_lean_gate(tag, loop, result)
    return dict(result)


REJECTION_REASON_ORDER = (
    "low_neutralized_ic",
    "low_residual_ratio",
    "high_exposure_r2",
    "sign_instability",
    "low_feature_coverage",
    "duplicate_corr",
    "invalid_expr",
)


def _new_rejection_summary() -> Dict[str, Any]:
    return {
        "total": {reason: 0 for reason in REJECTION_REASON_ORDER},
        "recent": [],
        "last_loop": {},
    }


def _normalize_rejection_reason(reason: str) -> str:
    if reason in {"corr_gate", "signature_quota", "duplicate", "already_evaluated"}:
        return "duplicate_corr"
    if reason in {"family_quota", "not_selected", "ranked_below_cut"}:
        return "duplicate_corr"
    if reason in {"error", "insufficient", "status_error", "status_insufficient"}:
        return "invalid_expr"
    if reason in REJECTION_REASON_ORDER:
        return reason
    return "invalid_expr"


def _record_rejection(
    summary: Dict[str, Any],
    loop_counts: Dict[str, int],
    reason: str,
    *,
    expression: str = "",
    detail: str = "",
) -> None:
    normalized = _normalize_rejection_reason(reason)
    summary.setdefault("total", {})
    summary["total"][normalized] = int(summary["total"].get(normalized, 0)) + 1
    loop_counts[normalized] = int(loop_counts.get(normalized, 0)) + 1
    recent = summary.setdefault("recent", [])
    if expression or detail:
        recent.append({
            "reason": normalized,
            "expression": str(expression or "")[:160],
            "detail": str(detail or "")[:180],
        })
        del recent[:-20]


def _record_eval_rejection(
    summary: Dict[str, Any],
    loop_counts: Dict[str, int],
    metrics: Dict[str, Any],
    cfg: MiningConfig,
    *,
    expression: str,
) -> None:
    status = str(metrics.get("status") or "unknown")
    if status != "ok":
        _record_rejection(
            summary,
            loop_counts,
            f"status_{status}",
            expression=expression,
            detail=str(metrics.get("error", status)),
        )
        return
    reasons = list(metrics.get("reject_reasons") or [])
    if not reasons and _pure_residual_enabled(cfg):
        reasons = _pure_residual_rejection_reasons(metrics, cfg)
    if not reasons:
        if abs(float(metrics.get("oos_ic", 0.0) or 0.0)) < float(cfg.ic_gate):
            reasons.append("low_neutralized_ic" if _pure_residual_enabled(cfg) else "invalid_expr")
        if int(metrics.get("sign_agree", 0) or 0) < int(cfg.sign_gate):
            reasons.append("sign_instability")
    for reason in reasons or ["invalid_expr"]:
        _record_rejection(summary, loop_counts, reason, expression=expression)


def _summarize_rejections_for_prompt(summary: Dict[str, Any]) -> Dict[str, Any]:
    total = summary.get("total", {}) if isinstance(summary, dict) else {}
    recent = summary.get("recent", []) if isinstance(summary, dict) else []
    return {
        "counts": {reason: int(total.get(reason, 0)) for reason in REJECTION_REASON_ORDER},
        "recent": list(recent[-8:]),
    }


# ============================================================
# LLM helper
# ============================================================

def _llm_generate(
    cfg: MiningConfig,
    elites: Sequence[CandidateRecord | str],
    feature_glossary: str,
    round_idx: int,
    rejection_summary: Optional[Dict[str, Any]] = None,
) -> List[Dict]:
    try:
        from agent_market.freqai.llm import LLMConfig, request_completion, _format_allowed_functions
        from agent_market.freqai.llm_miner_v2 import (
            build_system_prompt, build_generation_prompt_v2,
            parse_llm_response, KNOWN_FAILURES, FactorExample,
        )
    except Exception as exc:
        print(f"[llm] import/setup failed: {exc!s:.200}")
        return []

    llm = LLMConfig()
    llm.timeout = cfg.llm_timeout
    max_token_default = 20000 if str(cfg.llm_reasoning_effort or "").strip().lower() == "max" else 4096
    llm.max_tokens = int(cfg.llm_max_tokens or os.environ.get("LLM_MAX_TOKENS", str(max_token_default)))
    llm.temperature = 0.5
    llm.retries = cfg.llm_retries
    llm.reasoning_effort = cfg.llm_reasoning_effort or llm.reasoning_effort

    prompt_good: List[FactorExample] = []
    for i, elite in enumerate(elites[:10]):
        if isinstance(elite, CandidateRecord):
            prompt_good.append(FactorExample(
                name=f"c{i}",
                expression=elite.expression,
                category=elite.primary_family or "evolved",
                abs_ic=abs(float(elite.neutralized_ic or elite.oos_ic or 0.0)),
                oos_ic=float(elite.oos_ic or 0.0),
                raw_ic=float(elite.raw_ic or 0.0),
                clean_ic=float(elite.clean_ic or 0.0),
                neutralized_ic=float(elite.neutralized_ic or 0.0),
                residual_ic_ratio=float(elite.residual_ic_ratio or 0.0),
                exposure_r2=float(elite.exposure_r2 or 0.0),
                max_exposure_corr=float(elite.max_exposure_corr or 0.0),
            ))
        else:
            prompt_good.append(FactorExample(name=f"c{i}", expression=str(elite), category="evolved",
                                             abs_ic=0.0, oos_ic=0.0))

    prompt_profile = str(cfg.prompt_profile or "default")
    mechanism_quota: Optional[Dict[str, int]] = None
    if prompt_profile == "residual_alpha_v2" and len(elites) < 10:
        mechanism_quota = {
            "microstructure surprise": 2,
            "funding dislocation after beta/vol control": 1,
            "liquidity shock reversal residual": 1,
            "cross-asset disagreement residual": 1,
            "regime-conditioned residual momentum": 1,
        }

    sys_p = build_system_prompt(prompt_profile=prompt_profile)
    gen_p = build_generation_prompt_v2(
        feature_glossary=feature_glossary,
        functions_doc=_format_allowed_functions()[:800],
        success_examples=prompt_good,
        avoid_examples=[],
        failure_patterns=KNOWN_FAILURES[:3],
        round_idx=round_idx,
        request_count=cfg.llm_per_loop,
        label_period=cfg.label_period,
        category_quota=mechanism_quota,
        prompt_profile=prompt_profile,
        rejection_summary=_summarize_rejections_for_prompt(rejection_summary or {}),
    )
    try:
        raw = request_completion(gen_p, llm, system_prompt=sys_p)
        if isinstance(raw, tuple): raw = raw[0]
        return parse_llm_response(raw)
    except Exception as exc:
        print(f"[llm] request failed (round={round_idx}): {exc!s:.240}")
        return []


# ============================================================
# Main mining loop
# ============================================================

def mine(cfg: MiningConfig, tag: str = "default", resume: bool = True) -> List[CandidateRecord]:
    random.seed(42); np.random.seed(42)
    lane = _effective_lane(cfg)
    cfg.evaluation_lane = lane.lane
    cfg.timeframe = lane.timeframe
    cfg.label_horizons = _effective_label_horizons(cfg)
    cfg.embargo_bars = _effective_embargo_bars(cfg)
    cfg.label_period = primary_label_horizon(cfg.label_horizons, default=lane.label_horizons)
    print(f"[mining] loading {cfg.timeframe} feature matrix "
          f"(lane={cfg.evaluation_lane}, venue={cfg.data_venue}, horizons={list(cfg.label_horizons)}, "
          f"eval_mode={cfg.eval_mode}, turnover_w={cfg.turnover_weight}, "
          f"xs_w={cfg.xs_weight}, label_mode={cfg.label_mode}, pairs={cfg.pairs}, "
          f"purify={_effective_purify_mode(cfg)}/{cfg.purify_neutralize}, "
          f"alpha_objective={_alpha_objective(cfg)}, prompt_profile={cfg.prompt_profile})...", flush=True)
    big, base_cols = build_big(
        timeframe=cfg.timeframe,
        label_bars=int(cfg.label_period),
        label_mode=cfg.label_mode,
        pair_reference=cfg.pair_reference,
        data_dir=cfg.data_dir,
        data_venue=cfg.data_venue,
        pairs=cfg.pairs,
        cache_dir=cfg.cache_dir,
        no_cache=cfg.no_cache,
    )
    print(f"  rows={len(big):,}  pairs={big['__pair__'].nunique()}  base_features={len(base_cols)}")
    feature_stats = _feature_quality_stats(big, base_cols)
    llm_base_cols = _llm_usable_feature_cols(base_cols, feature_stats, cfg)
    allowed_expr_columns = set(base_cols) | {"open", "high", "low", "close", "volume"}
    if cfg.use_llm and bool(getattr(cfg, "llm_filter_low_coverage", True)):
        dropped_n = len(base_cols) - len(llm_base_cols)
        print(
            f"[llm] feature filter: usable={len(llm_base_cols)}/{len(base_cols)} "
            f"dropped={dropped_n} min_coverage={float(cfg.llm_min_feature_coverage):.2f} "
            f"min_rows={int(cfg.llm_min_feature_rows)}",
            flush=True,
        )

    hub = _open_hub_client(tag)
    _hub_event(hub, "mining.started", tag=tag, rounds=cfg.rounds,
               ic_gate=cfg.ic_gate, sign_gate=cfg.sign_gate,
               novelty_gate=cfg.novelty_gate,
               hard_corr_gate=cfg.hard_corr_gate,
               soft_corr_penalty_start=cfg.soft_corr_penalty_start,
               max_same_family_in_top40=cfg.max_same_family_in_top40,
               max_same_signature=cfg.max_same_signature,
               label_period=cfg.label_period,
               label_horizons=list(cfg.label_horizons),
               evaluation_lane=cfg.evaluation_lane,
               data_venue=cfg.data_venue,
               embargo_bars=cfg.embargo_bars,
               label_mode=cfg.label_mode,
               pair_reference=cfg.pair_reference,
               data_dir=cfg.data_dir,
               pairs=cfg.pairs,
               purify_mode=cfg.purify_mode,
               purify_neutralize=cfg.purify_neutralize,
               purify_exposures=cfg.purify_exposures,
               alpha_objective=_alpha_objective(cfg),
               prompt_profile=cfg.prompt_profile,
               pure_residual_gates=_pure_residual_gates(cfg) if _pure_residual_enabled(cfg) else None,
               train=list(cfg.train), oos=list(cfg.oos))

    survivors: List[CandidateRecord] = []
    evaluated: set = set()
    survivor_ranks: Dict[str, np.ndarray] = {}
    novelty_rejected = 0
    rejection_summary = _new_rejection_summary()
    start = 1
    if resume:
        state = load_state(tag)
        if state:
            start, survivors, evaluated = state
            start += 1
            survivors = [annotate_diversity(c) for c in survivors]
            print(f"[mining] resumed from loop {start-1}, {len(survivors)} survivors, {len(evaluated)} evaluated")
            # Rebuild rank cache for current survivors so novelty gate keeps working
            if _needs_diversity_rank_cache(cfg):
                hard_gate = min(float(cfg.hard_corr_gate), float(cfg.novelty_gate))
                print(f"[mining] rebuilding rank cache for diversity gate (gate={hard_gate})...")
                for c in survivors:
                    r = eval_ic(big, c.expression, cfg, return_oos_series=True)
                    if r.get("status") == "ok" and "oos_series" in r:
                        ranks = _series_to_ranks(r["oos_series"])
                        if ranks is not None:
                            survivor_ranks[c.expression] = ranks
            _hub_event(hub, "mining.started", tag=tag, resumed=True, loop=start-1,
                       survivors=len(survivors), evaluated=len(evaluated))

    # Seed round (only if no checkpoint)
    if not survivors:
        if cfg.seed_file:
            sp = Path(cfg.seed_file)
            if not sp.exists():
                raise FileNotFoundError(f"seed_file {sp} not found")
            sd = json.loads(sp.read_text(encoding="utf-8-sig"))
            seeds = [e["expression"] for e in sd.get("expressions", []) if e.get("expression")]
            print(f"[mining] seed pool (from {sp.name}): {len(seeds)} expressions  novelty_gate={cfg.novelty_gate}")
        else:
            seeds = load_seeds()
            print(f"[mining] seed pool: {len(seeds)} expressions  novelty_gate={cfg.novelty_gate}")
        # Seed pass: score first, then pick a diversity-aware survivor set.
        seed_candidates: List[CandidateRecord] = []
        seed_rank_cache: Dict[str, np.ndarray] = {}
        seed_rejections: Dict[str, int] = {reason: 0 for reason in REJECTION_REASON_ORDER}
        for i, expr in enumerate(seeds):
            if i == 0 or i % 5 == 0:
                print(f"  seed eval [{i}/{len(seeds)}] scored={len(seed_candidates)}", flush=True)
            if expr in evaluated: continue
            evaluated.add(expr)
            m = eval_ic(big, expr, cfg, return_oos_series=_needs_diversity_rank_cache(cfg))
            if m.get("status") == "ok" and m.get("passes"):
                cand = CandidateRecord(
                    expression=expr, origin="seed",
                    train_ic=m["train_ic"], oos_ic=m["oos_ic"],
                    sign_agree=m["sign_agree"], combined=m["combined"],
                    xs_ic=float(m.get("xs_ic", 0.0)),
                    turnover=float(m.get("turnover", 0.0)),
                    cost_mult=float(m.get("cost_mult", 1.0)),
                    stability_ic=float(m.get("ts_agg", m["oos_ic"])),
                    fitness=float(m.get("fitness", m["combined"])),
                    raw_ic=float(m.get("raw_ic", 0.0)),
                    clean_ic=float(m.get("clean_ic", 0.0)),
                    neutralized_ic=float(m.get("neutralized_ic", 0.0)),
                    residual_ic_ratio=float(m.get("residual_ic_ratio", 0.0)),
                    exposure_r2=float(m.get("exposure_r2", 0.0)),
                    max_exposure_corr=float(m.get("max_exposure_corr", 0.0)),
                    exposure_count=int(m.get("exposure_count", 0)),
                    purify_mode=str(m.get("purify_mode", cfg.purify_mode)),
                    eval_cache_key=str(m.get("cache_key", "")),
                )
                seed_candidates.append(annotate_diversity(cand))
                if "oos_series" in m:
                    ranks = _series_to_ranks(m["oos_series"])
                    if ranks is not None:
                        seed_rank_cache[expr] = ranks
            else:
                _record_eval_rejection(rejection_summary, seed_rejections, m, cfg, expression=expr)
        print(f"  seed scoring complete: {len(seed_candidates)} pass IC gate; "
              f"now applying diversity gates...")

        survivors, seed_rejected = _select_for_mining(seed_candidates, seed_rank_cache, cfg)
        survivor_ranks = _rank_cache_for_selected(survivors, seed_rank_cache)
        novelty_rejected += len(seed_rejected)
        for row in seed_rejected:
            if row.get("reason") in {"corr_gate", "family_quota", "signature_quota"}:
                _record_rejection(
                    rejection_summary,
                    seed_rejections,
                    str(row.get("reason") or "duplicate_corr"),
                    expression=str(row.get("expression") or ""),
                    detail=str(row.get("collision_expression") or ""),
                )
        rejection_summary["last_loop"] = dict(seed_rejections)
        for cand in survivors:
            _hub_register(hub, cand, cfg, tag, loop_idx=0)
        save_state(tag, 0, survivors, evaluated, cfg, rejection_summary=rejection_summary)
        print(f"[mining] seed round: kept {len(survivors)} top-K={cfg.top_k}  "
              f"diversity_rejected={novelty_rejected}")
        _hub_event(hub, "mining.loop_completed", tag=tag, loop=0, phase="seed",
                   survivors=len(survivors), evaluated=len(evaluated),
                   novelty_rejected=novelty_rejected)

    # Feature glossary for LLM
    from agent_market.freqai.llm import build_feature_glossary
    feat_cfg = json.loads(FEATURE_FILE.read_text(encoding="utf-8-sig"))
    glossary = build_feature_glossary(feat_cfg, llm_base_cols, combos=[], max_items=max(200, len(llm_base_cols)))
    glossary += _format_feature_filter_note(base_cols, llm_base_cols, feature_stats, cfg)

    t0 = time.time()
    llm_calls = 0
    for loop in range(start, cfg.rounds + 1):
        new_cands = []
        loop_rejections: Dict[str, int] = {reason: 0 for reason in REJECTION_REASON_ORDER}
        elite_records = sorted(
            (annotate_diversity(c) for c in survivors),
            key=lambda c: _portfolio_key(c, "portfolio"),
            reverse=True,
        )[:max(12, min(cfg.top_k, 24))]
        elite_exprs = _prompt_elites_by_family(survivors, limit=max(12, min(cfg.top_k, 24)))
        if cfg.use_llm:
            llm_rows = _llm_generate(cfg, elite_records[:12], glossary, loop, rejection_summary)
            llm_queued = 0
            for c in llm_rows:
                e = (c.get("expression") or "").strip()
                if not e or len(e) >= MAX_EXPR_LEN:
                    _record_rejection(rejection_summary, loop_rejections, "invalid_expr", expression=e)
                else:
                    reason, detail = _feature_rejection_detail(
                        e,
                        allowed_columns=allowed_expr_columns,
                        feature_stats=feature_stats,
                        cfg=cfg,
                    )
                    if reason:
                        _record_rejection(
                            rejection_summary,
                            loop_rejections,
                            reason,
                            expression=e,
                            detail=detail,
                        )
                    elif e in evaluated:
                        _record_rejection(rejection_summary, loop_rejections, "already_evaluated", expression=e)
                    else:
                        new_cands.append(("llm", e))
                        llm_queued += 1
            if bool(getattr(cfg, "llm_required", False)) and llm_queued <= 0:
                raise RuntimeError(
                    f"LLM required but produced no usable candidates at loop {loop}; "
                    f"raw_candidates={len(llm_rows)}"
                )
            llm_calls += 1
        for e in gen_python_offspring(elite_exprs[:cfg.top_k], loop, cfg.py_per_loop, available_cols=set(base_cols)):
            reason, detail = _feature_rejection_detail(
                e,
                allowed_columns=allowed_expr_columns,
                feature_stats=feature_stats,
                cfg=cfg,
            )
            if reason:
                _record_rejection(
                    rejection_summary,
                    loop_rejections,
                    reason,
                    expression=e,
                    detail=detail,
                )
            elif e not in evaluated:
                new_cands.append(("py", e))
            else:
                _record_rejection(rejection_summary, loop_rejections, "already_evaluated", expression=e)

        passed = 0
        rejected_novelty = 0
        for origin, expr in new_cands:
            evaluated.add(expr)
            m = eval_ic(big, expr, cfg, return_oos_series=_needs_diversity_rank_cache(cfg))
            if m.get("status") != "ok" or not m.get("passes"):
                _record_eval_rejection(rejection_summary, loop_rejections, m, cfg, expression=expr)
                continue
            if any(c.expression == expr for c in survivors):
                _record_rejection(rejection_summary, loop_rejections, "duplicate", expression=expr)
                continue
            cand = CandidateRecord(
                expression=expr, origin=f"{origin}_loop{loop}",
                train_ic=m["train_ic"], oos_ic=m["oos_ic"],
                sign_agree=m["sign_agree"], combined=m["combined"],
                loop_found=loop,
                xs_ic=float(m.get("xs_ic", 0.0)),
                turnover=float(m.get("turnover", 0.0)),
                cost_mult=float(m.get("cost_mult", 1.0)),
                stability_ic=float(m.get("ts_agg", m["oos_ic"])),
                fitness=float(m.get("fitness", m["combined"])),
                raw_ic=float(m.get("raw_ic", 0.0)),
                clean_ic=float(m.get("clean_ic", 0.0)),
                neutralized_ic=float(m.get("neutralized_ic", 0.0)),
                residual_ic_ratio=float(m.get("residual_ic_ratio", 0.0)),
                exposure_r2=float(m.get("exposure_r2", 0.0)),
                max_exposure_corr=float(m.get("max_exposure_corr", 0.0)),
                exposure_count=int(m.get("exposure_count", 0)),
                purify_mode=str(m.get("purify_mode", cfg.purify_mode)),
                eval_cache_key=str(m.get("cache_key", "")),
            )
            survivors.append(annotate_diversity(cand))
            if "oos_series" in m:
                cand_ranks = _series_to_ranks(m.get("oos_series"))
                if cand_ranks is not None:
                    survivor_ranks[expr] = cand_ranks
            _hub_register(hub, cand, cfg, tag, loop_idx=loop)
            passed += 1

        survivors, diversity_rejected = _select_for_mining(survivors, survivor_ranks, cfg)
        survivor_ranks = _rank_cache_for_selected(survivors, survivor_ranks)
        rejected_novelty = sum(
            1 for r in diversity_rejected
            if r.get("reason") in {"corr_gate", "family_quota", "signature_quota"}
        )
        for row in diversity_rejected:
            if row.get("reason") in {"corr_gate", "family_quota", "signature_quota"}:
                _record_rejection(
                    rejection_summary,
                    loop_rejections,
                    str(row.get("reason") or "duplicate_corr"),
                    expression=str(row.get("expression") or ""),
                    detail=str(row.get("collision_expression") or ""),
                )
        novelty_rejected += rejected_novelty
        rejection_summary["last_loop"] = dict(loop_rejections)

        elapsed = (time.time() - t0) / 60
        avg_loop = elapsed / max(loop - start + 1, 1)
        eta = avg_loop * (cfg.rounds - loop)
        best = survivors[0].oos_ic if survivors else 0
        print(f"[loop {loop:>3}/{cfg.rounds}] new={len(new_cands):>2} pass={passed} "
              f"div_rej={rejected_novelty} sur={len(survivors)} seen={len(evaluated):,} "
              f"best={best:+.3f} elapsed={elapsed:.1f}m eta={eta:.1f}m llm={llm_calls}",
              flush=True)
        if _pure_residual_enabled(cfg) and loop % 50 == 0:
            breakdown = ", ".join(
                f"{reason}={int(loop_rejections.get(reason, 0))}"
                for reason in REJECTION_REASON_ORDER
                if int(loop_rejections.get(reason, 0)) > 0
            ) or "none"
            total = ", ".join(
                f"{reason}={int(rejection_summary['total'].get(reason, 0))}"
                for reason in REJECTION_REASON_ORDER
                if int(rejection_summary["total"].get(reason, 0)) > 0
            ) or "none"
            print(f"[pure-alpha] loop_rejects: {breakdown} | total_rejects: {total}", flush=True)
        if loop % 50 == 0:
            cs = _cache_for_cfg(cfg).snapshot()
            print(
                "[cache] "
                f"panel h/m={cs.get('panel_hits', 0)}/{cs.get('panel_misses', 0)} "
                f"exposure h/m={cs.get('exposure_hits', 0)}/{cs.get('exposure_misses', 0)} "
                f"factor h/m={cs.get('factor_version_hits', 0)}/{cs.get('factor_version_misses', 0)} "
                f"eval h/m={cs.get('eval_hits', 0)}/{cs.get('eval_misses', 0)}",
                flush=True,
            )

        _hub_event(hub, "mining.loop_completed", tag=tag, loop=loop,
                   new=len(new_cands), passed=passed,
                   novelty_rejected_loop=rejected_novelty,
                   survivors=len(survivors),
                   best_oos_ic=best, evaluated=len(evaluated),
                   elapsed_min=round(elapsed, 2))

        should_checkpoint = loop % cfg.checkpoint_every == 0
        should_lean_gate = _should_run_loop_lean_gate(cfg, loop)
        if should_checkpoint or should_lean_gate:
            save_state(tag, loop, survivors, evaluated, cfg, rejection_summary=rejection_summary)
        if should_lean_gate:
            print(f"[lean-gate loop {loop:>3}/{cfg.rounds}] running LEAN gate...", flush=True)
            try:
                gate_result = _run_loop_lean_gate(tag, loop, cfg)
            except Exception as exc:
                gate_result = {
                    "status": "failed",
                    "reason": f"LEAN gate hook failed: {exc}",
                    "violations": [f"LEAN gate hook failed: {exc}"],
                    "artifacts": {},
                }
                _record_loop_lean_gate(tag, loop, gate_result)
            status = str(gate_result.get("status") or "unknown")
            comparison_status = str(gate_result.get("comparison_status") or "missing")
            reason = str(gate_result.get("reason") or "")
            print(
                f"[lean-gate loop {loop:>3}/{cfg.rounds}] "
                f"status={status} comparison={comparison_status} reason={reason[:220]}",
                flush=True,
            )
            _hub_event(
                hub,
                "mining.lean_gate_completed",
                tag=tag,
                loop=loop,
                status=status,
                comparison_status=comparison_status,
                reason=reason,
                artifacts=gate_result.get("artifacts"),
            )
            if bool(getattr(cfg, "lean_gate_fail_fast", False)) and status != "passed":
                raise RuntimeError(f"LEAN gate failed at loop {loop}: {reason or status}")

    final_saved_in_loop = (
        (int(cfg.rounds) % int(cfg.checkpoint_every) == 0)
        or _should_run_loop_lean_gate(cfg, int(cfg.rounds))
    )
    if not final_saved_in_loop:
        save_state(tag, cfg.rounds, survivors, evaluated, cfg, rejection_summary=rejection_summary)
    _hub_event(hub, "mining.finished", tag=tag, rounds=cfg.rounds,
               survivors=len(survivors), evaluated=len(evaluated),
               novelty_rejected_total=novelty_rejected,
               novelty_gate=cfg.novelty_gate,
               best_oos_ic=(survivors[0].oos_ic if survivors else 0.0))
    return survivors


def _candidate_export_row(cand: CandidateRecord, idx: int) -> Dict[str, Any]:
    annotate_diversity(cand)
    return {
        "name": f"f{idx+1:03d}",
        "expression": cand.expression,
        "origin": cand.origin,
        "train_ic": cand.train_ic,
        "oos_ic": cand.oos_ic,
        "sign_agree": cand.sign_agree,
        "combined": cand.combined,
        "fitness": cand.fitness,
        "stability_ic": cand.stability_ic,
        "loop_found": cand.loop_found,
        "primary_family": cand.primary_family,
        "family_tags": list(cand.family_tags),
        "canonical_signature": cand.canonical_signature,
        "max_corr_to_kept": cand.max_corr_to_kept,
        "cluster_id": cand.cluster_id,
        "raw_ic": cand.raw_ic,
        "clean_ic": cand.clean_ic,
        "neutralized_ic": cand.neutralized_ic,
        "residual_ic_ratio": cand.residual_ic_ratio,
        "exposure_r2": cand.exposure_r2,
        "max_exposure_corr": cand.max_exposure_corr,
        "exposure_count": cand.exposure_count,
        "purify_mode": cand.purify_mode,
        "eval_cache_key": cand.eval_cache_key,
    }


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def cluster_by_abs_corr(
    candidates: Sequence[CandidateRecord],
    rank_cache: Dict[str, np.ndarray],
    *,
    corr_gate: float,
) -> Tuple[Dict[int, List[CandidateRecord]], List[Dict[str, Any]]]:
    """Cluster candidates by connected components of abs Spearman rank corr."""
    ranked = [annotate_diversity(c) for c in candidates if c.expression in rank_cache]
    uf = _UnionFind(len(ranked))
    pair_rows: List[Dict[str, Any]] = []
    for i in range(len(ranked)):
        for j in range(i + 1, len(ranked)):
            corr = abs(_rank_corr(rank_cache[ranked[i].expression], rank_cache[ranked[j].expression]))
            if corr >= corr_gate:
                uf.union(i, j)
            pair_rows.append({
                "a": ranked[i].expression,
                "b": ranked[j].expression,
                "abs_corr": float(corr),
            })

    root_to_cluster: Dict[int, int] = {}
    clusters: Dict[int, List[CandidateRecord]] = {}
    for i, cand in enumerate(ranked):
        root = uf.find(i)
        if root not in root_to_cluster:
            root_to_cluster[root] = len(root_to_cluster)
        cluster_id = root_to_cluster[root]
        cand.cluster_id = cluster_id
        clusters.setdefault(cluster_id, []).append(cand)
    return clusters, pair_rows


def _effective_factor_count(selected: Sequence[CandidateRecord], rank_cache: Dict[str, np.ndarray]) -> float:
    if len(selected) <= 1:
        return float(len(selected))
    mat = []
    for i, a in enumerate(selected):
        row = []
        for j, b in enumerate(selected):
            if i == j:
                row.append(1.0)
            else:
                row.append(abs(_rank_corr(rank_cache[a.expression], rank_cache[b.expression])))
        mat.append(row)
    corr = np.asarray(mat, dtype=np.float64)
    try:
        eig = np.linalg.eigvalsh(corr)
    except np.linalg.LinAlgError:
        return float(len(selected))
    denom = float(np.sum(eig ** 2))
    if denom <= 0:
        return float(len(selected))
    return float((np.sum(eig) ** 2) / denom)


def _family_distribution(candidates: Sequence[CandidateRecord]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for cand in candidates:
        family = cand.primary_family or primary_family(cand.expression)
        out[family] = out.get(family, 0) + 1
    return out


def _select_diverse_export(
    candidates: Sequence[CandidateRecord],
    rank_cache: Dict[str, np.ndarray],
    *,
    n: int,
    corr_gate: float,
    score_mode: str,
    family_max: int,
) -> Tuple[List[CandidateRecord], Dict[str, Any]]:
    clusters, pair_rows = cluster_by_abs_corr(candidates, rank_cache, corr_gate=corr_gate)
    medoids: List[CandidateRecord] = []
    for members in clusters.values():
        medoids.append(max(members, key=lambda c: _portfolio_key(c, score_mode)))

    selected: List[CandidateRecord] = []
    selected_exprs: set[str] = set()
    family_counts: Dict[str, int] = {}
    selected_clusters: Dict[int, int] = {}
    exemptions: List[Dict[str, Any]] = []

    def can_add(cand: CandidateRecord, *, allow_same_cluster: bool = False) -> Tuple[bool, str, float, str]:
        family = cand.primary_family or primary_family(cand.expression)
        if family_max > 0 and family_counts.get(family, 0) >= family_max:
            return False, "family_quota", 0.0, ""
        if not allow_same_cluster and cand.cluster_id in selected_clusters:
            return False, "cluster_medoid_only", 0.0, ""
        max_corr, collision = _max_corr_to_selected(cand, selected, rank_cache)
        if corr_gate < 1.0 and max_corr >= corr_gate:
            return False, "corr_gate", max_corr, collision
        return True, "", max_corr, collision

    def add(cand: CandidateRecord, *, allow_same_cluster: bool = False) -> bool:
        ok, reason, max_corr, collision = can_add(cand, allow_same_cluster=allow_same_cluster)
        if not ok:
            return False
        cand.max_corr_to_kept = float(max_corr)
        selected.append(cand)
        selected_exprs.add(cand.expression)
        family = cand.primary_family or primary_family(cand.expression)
        family_counts[family] = family_counts.get(family, 0) + 1
        selected_clusters[cand.cluster_id] = selected_clusters.get(cand.cluster_id, 0) + 1
        if allow_same_cluster and selected_clusters[cand.cluster_id] > 1:
            exemptions.append({
                "cluster_id": cand.cluster_id,
                "expression": cand.expression,
                "reason": "same_cluster_different_family_pairwise_below_gate",
                "max_corr_to_kept": float(max_corr),
                "collision_expression": collision,
            })
        return True

    sorted_medoids = sorted(medoids, key=lambda c: _portfolio_key(c, score_mode), reverse=True)
    for family in CORE_PORTFOLIO_FAMILIES:
        family_options = [c for c in sorted_medoids if c.primary_family == family and c.expression not in selected_exprs]
        for cand in family_options:
            if add(cand):
                break
        if len(selected) >= n:
            break

    for cand in sorted_medoids:
        if len(selected) >= n:
            break
        if cand.expression not in selected_exprs:
            add(cand)

    if len(selected) < n:
        extras = sorted(
            [c for c in candidates if c.expression in rank_cache and c.expression not in selected_exprs],
            key=lambda c: _portfolio_key(c, score_mode),
            reverse=True,
        )
        selected_cluster_families: Dict[int, set[str]] = {}
        for cand in selected:
            selected_cluster_families.setdefault(cand.cluster_id, set()).add(cand.primary_family)
        for cand in extras:
            if len(selected) >= n:
                break
            same_cluster_families = selected_cluster_families.get(cand.cluster_id, set())
            allow_same_cluster = bool(same_cluster_families and cand.primary_family not in same_cluster_families)
            if add(cand, allow_same_cluster=allow_same_cluster):
                selected_cluster_families.setdefault(cand.cluster_id, set()).add(cand.primary_family)

    strict_selected_n = len(selected)
    if len(selected) < n:
        relaxed_pool = sorted(
            [c for c in candidates if c.expression in rank_cache and c.expression not in selected_exprs],
            key=lambda c: _portfolio_key(c, score_mode),
            reverse=True,
        )
        for cand in relaxed_pool:
            if len(selected) >= n:
                break
            family = cand.primary_family or primary_family(cand.expression)
            max_corr, collision = _max_corr_to_selected(cand, selected, rank_cache)
            violates_family_max = family_max > 0 and family_counts.get(family, 0) >= family_max
            same_cluster = cand.cluster_id in selected_clusters
            cand.max_corr_to_kept = float(max_corr)
            selected.append(cand)
            selected_exprs.add(cand.expression)
            family_counts[family] = family_counts.get(family, 0) + 1
            selected_clusters[cand.cluster_id] = selected_clusters.get(cand.cluster_id, 0) + 1
            exemptions.append({
                "cluster_id": cand.cluster_id,
                "expression": cand.expression,
                "reason": "relaxed_fill_to_requested_n",
                "max_corr_to_kept": float(max_corr),
                "collision_expression": collision,
                "violates_corr_gate": bool(corr_gate < 1.0 and max_corr >= corr_gate),
                "violates_family_max": bool(violates_family_max),
                "same_cluster": bool(same_cluster),
            })

    violations: List[Dict[str, Any]] = []
    max_pair = {"abs_corr": 0.0, "a": "", "b": ""}
    for i, a in enumerate(selected):
        for b in selected[i + 1:]:
            corr = abs(_rank_corr(rank_cache[a.expression], rank_cache[b.expression]))
            if corr > max_pair["abs_corr"]:
                max_pair = {"abs_corr": float(corr), "a": a.expression, "b": b.expression}
            if corr >= corr_gate:
                violations.append({"a": a.expression, "b": b.expression, "abs_corr": float(corr)})

    selected_cluster_ids = {c.cluster_id for c in selected}
    rejected: List[Dict[str, Any]] = []
    for cand in sorted(candidates, key=lambda c: _portfolio_key(c, score_mode), reverse=True):
        if cand.expression in selected_exprs:
            continue
        if cand.expression not in rank_cache:
            reason = "eval_failed"
            max_corr, collision = 0.0, ""
        else:
            ok, reason, max_corr, collision = can_add(cand, allow_same_cluster=False)
            if ok and cand.cluster_id not in selected_cluster_ids:
                reason = "ranked_below_cut"
        rejected.append({
            "expression": cand.expression,
            "primary_family": cand.primary_family,
            "cluster_id": cand.cluster_id,
            "reason": reason,
            "max_corr_to_kept": float(max_corr),
            "collision_expression": collision,
        })

    report = {
        "requested_n": n,
        "selected_n": len(selected),
        "strict_selected_n": strict_selected_n,
        "relaxed_fill_n": max(0, len(selected) - strict_selected_n),
        "corr_gate": corr_gate,
        "score_mode": score_mode,
        "family_max": family_max,
        "family_distribution": _family_distribution(selected),
        "cluster_count": len(clusters),
        "selected_cluster_count": len(selected_cluster_ids),
        "largest_cluster_size": max((len(v) for v in clusters.values()), default=0),
        "max_pairwise_abs_corr": max_pair,
        "effective_factor_count": _effective_factor_count(selected, rank_cache),
        "correlation_violations": violations,
        "same_cluster_exemptions": exemptions,
        "selected": [
            {
                "name": f"f{i+1:03d}",
                "expression": cand.expression,
                "primary_family": cand.primary_family,
                "family_tags": list(cand.family_tags),
                "cluster_id": cand.cluster_id,
                "combined": cand.combined,
                "fitness": cand.fitness,
                "stability_ic": cand.stability_ic,
                "sign_agree": cand.sign_agree,
                "max_corr_to_kept": cand.max_corr_to_kept,
                "raw_ic": cand.raw_ic,
                "clean_ic": cand.clean_ic,
                "neutralized_ic": cand.neutralized_ic,
                "residual_ic_ratio": cand.residual_ic_ratio,
                "exposure_r2": cand.exposure_r2,
                "max_exposure_corr": cand.max_exposure_corr,
                "purify_mode": cand.purify_mode,
            }
            for i, cand in enumerate(selected)
        ],
        "rejected_similar": rejected[:200],
        "top_correlated_pairs": sorted(pair_rows, key=lambda row: row["abs_corr"], reverse=True)[:50],
    }
    return selected, report


def export_top(
    tag: str,
    n: int = 30,
    out_name: str = None,
    *,
    diverse: bool = False,
    corr_gate: float = 0.65,
    score_mode: str = "combined",
    family_max: int = 6,
    timeframe: str = "1h",
    evaluation_lane: str = "auto",
    data_venue: str = "auto",
    label_horizons: Sequence[int] | str | None = None,
    eval_mode: str = "legacy",
    label_mode: str = "forward_return",
    pair_reference: str = "BTC/USDT",
    data_dir: Optional[str] = None,
    pairs: str = "auto",
    purify_mode: str = "off",
    purify_winsor: str = "mad",
    purify_standardize: str = "zscore",
    purify_neutralize: str = "ridge",
    purify_exposures: str = ",".join(DEFAULT_EXPOSURE_GROUPS),
    cache_dir: Optional[str | Path] = DEFAULT_CACHE_DIR,
    no_cache: bool = False,
) -> Path:
    """Export survivors from checkpoint to user_data/freqai_expressions_<tag>.json."""
    default_cache_dir = str(DEFAULT_CACHE_DIR)
    cache_dir_str = str(cache_dir) if cache_dir is not None else ""
    state = load_state(tag)
    if not state:
        raise FileNotFoundError(f"no state for tag={tag}")
    _, survivors, _ = state
    survivors = [annotate_diversity(c) for c in survivors]
    run_cfg = load_state_config(tag)
    if run_cfg:
        if timeframe == "1h":
            timeframe = str(run_cfg.get("timeframe") or timeframe)
        if eval_mode == "legacy":
            eval_mode = str(run_cfg.get("eval_mode") or eval_mode)
        if label_mode == "forward_return":
            label_mode = str(run_cfg.get("label_mode") or label_mode)
        if pair_reference == "BTC/USDT":
            pair_reference = str(run_cfg.get("pair_reference") or pair_reference)
        if data_dir is None and run_cfg.get("data_dir") is not None:
            data_dir = str(run_cfg.get("data_dir"))
        if pairs == "auto":
            pairs = str(run_cfg.get("pairs") or pairs)
        if purify_mode == "off":
            purify_mode = str(run_cfg.get("purify_mode") or purify_mode)
        if purify_winsor == "mad":
            purify_winsor = str(run_cfg.get("purify_winsor") or purify_winsor)
        if purify_standardize == "zscore":
            purify_standardize = str(run_cfg.get("purify_standardize") or purify_standardize)
        if purify_neutralize == "ridge":
            purify_neutralize = str(run_cfg.get("purify_neutralize") or purify_neutralize)
        if purify_exposures == ",".join(DEFAULT_EXPOSURE_GROUPS):
            purify_exposures = str(run_cfg.get("purify_exposures") or purify_exposures)
        if cache_dir_str == default_cache_dir and run_cfg.get("cache_dir"):
            cache_dir = str(run_cfg.get("cache_dir"))
            cache_dir_str = str(cache_dir)
        if not no_cache:
            no_cache = bool(run_cfg.get("no_cache", no_cache))
    cfg_kwargs = {f.name: run_cfg[f.name] for f in fields(MiningConfig) if f.name in run_cfg}
    export_cfg = MiningConfig(**cfg_kwargs) if cfg_kwargs else MiningConfig()
    lane_raw = str(evaluation_lane or getattr(export_cfg, "evaluation_lane", "auto") or "auto").strip().lower()
    if lane_raw not in {"", "auto", "1h"} and str(timeframe or "1h").strip().lower() == "1h":
        lane = normalize_lane(lane_raw)
        timeframe = lane.timeframe
    else:
        lane = normalize_lane(lane_raw, timeframe=timeframe)
    export_cfg.timeframe = timeframe
    export_cfg.evaluation_lane = lane.lane
    export_cfg.label_horizons = parse_label_horizons(label_horizons, default=lane.label_horizons) if label_horizons not in (None, "") else parse_label_horizons(getattr(export_cfg, "label_horizons", ()), default=lane.label_horizons)
    export_cfg.label_period = primary_label_horizon(export_cfg.label_horizons, default=lane.label_horizons)
    export_cfg.embargo_bars = int(getattr(export_cfg, "embargo_bars", 0) or lane.embargo_bars)
    if str(data_venue or "auto").lower() == "auto":
        export_cfg.data_venue = str(getattr(export_cfg, "data_venue", "kucoin") or "kucoin")
    else:
        export_cfg.data_venue = str(data_venue)
    export_cfg.eval_mode = eval_mode
    export_cfg.label_mode = label_mode
    export_cfg.pair_reference = pair_reference
    export_cfg.data_dir = data_dir
    export_cfg.pairs = pairs
    export_cfg.purify_mode = purify_mode
    export_cfg.purify_winsor = purify_winsor
    export_cfg.purify_standardize = purify_standardize
    export_cfg.purify_neutralize = purify_neutralize
    export_cfg.purify_exposures = purify_exposures
    export_cfg.cache_dir = cache_dir_str
    export_cfg.no_cache = bool(no_cache)
    if _pure_residual_enabled(export_cfg):
        survivors = [
            cand for cand in survivors
            if not _candidate_pure_residual_rejection_reasons(cand, export_cfg)
        ]
    from .paths import USER_DATA

    report: Optional[Dict[str, Any]] = None
    if diverse:
        cfg = export_cfg
        print(f"[export] recomputing OOS rank series for {len(survivors)} survivors...")
        big, _ = build_big(
            timeframe=timeframe,
            label_bars=int(export_cfg.label_period),
            label_mode=label_mode,
            pair_reference=pair_reference,
            data_dir=data_dir,
            data_venue=export_cfg.data_venue,
            pairs=pairs,
            cache_dir=cache_dir,
            no_cache=no_cache,
        )
        rank_cache: Dict[str, np.ndarray] = {}
        eval_errors: List[Dict[str, str]] = []
        for cand in survivors:
            metrics = eval_ic(big, cand.expression, cfg, return_oos_series=True)
            if metrics.get("status") == "ok" and "oos_series" in metrics:
                ranks = _series_to_ranks(metrics["oos_series"])
                if ranks is not None:
                    rank_cache[cand.expression] = ranks
                    continue
            eval_errors.append({"expression": cand.expression, "status": str(metrics.get("status")), "error": str(metrics.get("error", ""))})
        top, report = _select_diverse_export(
            survivors,
            rank_cache,
            n=n,
            corr_gate=corr_gate,
            score_mode=score_mode,
            family_max=family_max,
        )
        report.update({
            "version": f"lab-{tag}-diverse",
            "tag": tag,
            "ranked_survivors": len(rank_cache),
            "eval_errors": eval_errors[:100],
            "timestamp": time.time(),
        })
        out_name = out_name or f"freqai_expressions_{tag}_diverse.json"
        report_path = USER_DATA / f"factor_diversity_report_{tag}.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    else:
        top = sorted(survivors, key=lambda c: _portfolio_key(c, score_mode), reverse=True)[:n]
        out_name = out_name or f"freqai_expressions_{tag}.json"

    out = USER_DATA / out_name
    manifest = mining_lane_manifest(export_cfg)
    out.write_text(json.dumps({
        "version": f"lab-{tag}{'-diverse' if diverse else ''}",
        "diverse": bool(diverse),
        "corr_gate": corr_gate if diverse else None,
        "score_mode": score_mode,
        **manifest,
        "intraday": manifest,
        "expressions": [
            _candidate_export_row(s, i)
            for i, s in enumerate(top)
        ],
    }, indent=2), encoding="utf-8")
    return out

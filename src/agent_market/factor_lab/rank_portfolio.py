"""Cross-sectional rank portfolio engine with dynamic leverage controls."""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field, fields
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from agent_market import paths as repo_paths
from agent_market.freqai.expression_engine import safe_eval_expression
from agent_market.freqai.features import apply_configured_features

from . import mining
from .cache import DEFAULT_CACHE_DIR
from .paths import (
    DEFAULT_FUNDING_DAILY,
    DEFAULT_PAIRS,
    FEATURE_FILE,
    FUNDING_DIR,
    KUCOIN_DIR,
    OKX_FUTURES_DIR,
    USER_DATA,
    feather_for_pair,
)
from .timeframes import bars_for_hours, bars_for_minutes, manifest_matches_profile, normalize_timeframe


DEFAULT_TAG = "gpt54_purealpha_v2_full1000_fix1"
RISK_PROFILE_AGGRESSIVE = "aggressive"
ALLOWED_LEVERAGES = (10, 8, 5, 3, 2, 1)
FUTURES_VENUES = {"okx", "bybit", "binance"}


@dataclass
class RiskConfig:
    profile: str = RISK_PROFILE_AGGRESSIVE
    gross_cap: float = 10.0
    net_cap: float = 2.5
    single_pair_cap: float = 2.0
    risk_per_trade: float = 0.008
    daily_loss_limit: float = 0.06
    weekly_loss_limit: float = 0.08
    drawdown_safe_mode: float = 0.20
    consecutive_loss_limit: int = 8
    pause_hours: int = 24
    maintenance_margin: float = 0.005
    fee_buffer: float = 0.003
    fee_rate: float = 0.0004
    slippage: float = 0.0003
    min_stop_pct: float = 0.01
    max_stop_pct: float = 0.06
    atr_stop_mult: float = 2.0
    top_k: int = 3
    min_pairs_for_top_k: int = 8
    low_pair_top_k: int = 2
    side_mode: str = "both"
    min_abs_score_z: float = 0.0
    rebalance_hours: int = 1
    timeframe: str = "1h"
    rebalance_minutes: int = 60
    leverage_cap: float = 10.0
    edge_mode: str = "off"
    edge_lookback_hours: int = 336
    edge_min_periods: int = 168
    edge_deadband: float = 0.005
    pair_edge_leverage: bool = True
    pair_edge_deadband: float = 0.01
    pair_edge_strong_ic: float = 0.05
    pair_edge_very_strong_ic: float = 0.10
    pair_edge_weak_cap: float = 2.0
    regime_mode: str = "off"
    regime_min_edge_ic: float = 0.0
    regime_min_pair_edge_ic: float = 0.0
    regime_min_pair_count: int = 0
    regime_short_max_market_mom_24h: Optional[float] = None
    regime_short_max_market_mom_72h: Optional[float] = None
    regime_max_market_atr_pct: Optional[float] = None
    short_max_mom_24h: Optional[float] = None
    short_max_mom_72h: Optional[float] = None
    long_min_mom_24h: Optional[float] = None
    max_entry_atr_pct: Optional[float] = None
    short_max_market_mom_24h: Optional[float] = None
    short_max_market_mom_72h: Optional[float] = None
    short_max_market_ma_gap: Optional[float] = None
    short_exit_mom_24h: Optional[float] = None
    short_exit_mom_72h: Optional[float] = None
    short_exit_market_mom_24h: Optional[float] = None
    short_exit_market_ma_gap: Optional[float] = None
    exclude_pairs: Tuple[str, ...] = ()

    @classmethod
    def from_profile(
        cls,
        profile: str = RISK_PROFILE_AGGRESSIVE,
        *,
        gross_cap: Optional[float] = None,
        net_cap: Optional[float] = None,
        top_k: Optional[int] = None,
        min_pairs_for_top_k: Optional[int] = None,
        low_pair_top_k: Optional[int] = None,
        single_pair_cap: Optional[float] = None,
        side_mode: Optional[str] = None,
        min_abs_score_z: Optional[float] = None,
        rebalance_hours: Optional[int] = None,
        rebalance_minutes: Optional[int] = None,
        timeframe: str = "1h",
        risk_per_trade: Optional[float] = None,
        leverage_cap: Optional[float] = None,
        edge_mode: Optional[str] = None,
        edge_lookback_hours: Optional[int] = None,
        edge_min_periods: Optional[int] = None,
        edge_deadband: Optional[float] = None,
        pair_edge_leverage: Optional[bool] = None,
        pair_edge_deadband: Optional[float] = None,
        pair_edge_strong_ic: Optional[float] = None,
        pair_edge_very_strong_ic: Optional[float] = None,
        pair_edge_weak_cap: Optional[float] = None,
        regime_mode: Optional[str] = None,
        regime_min_edge_ic: Optional[float] = None,
        regime_min_pair_edge_ic: Optional[float] = None,
        regime_min_pair_count: Optional[int] = None,
        regime_short_max_market_mom_24h: Optional[float] = None,
        regime_short_max_market_mom_72h: Optional[float] = None,
        regime_max_market_atr_pct: Optional[float] = None,
        short_max_mom_24h: Optional[float] = None,
        short_max_mom_72h: Optional[float] = None,
        long_min_mom_24h: Optional[float] = None,
        max_entry_atr_pct: Optional[float] = None,
        short_max_market_mom_24h: Optional[float] = None,
        short_max_market_mom_72h: Optional[float] = None,
        short_max_market_ma_gap: Optional[float] = None,
        short_exit_mom_24h: Optional[float] = None,
        short_exit_mom_72h: Optional[float] = None,
        short_exit_market_mom_24h: Optional[float] = None,
        short_exit_market_ma_gap: Optional[float] = None,
        exclude_pairs: Optional[Sequence[str] | str] = None,
    ) -> "RiskConfig":
        profile_name = str(profile or RISK_PROFILE_AGGRESSIVE).lower()
        cfg = cls(profile=profile_name)
        cfg.timeframe = normalize_timeframe(timeframe)
        if profile_name == RISK_PROFILE_AGGRESSIVE:
            # Evidence-backed default after OKX holdout diagnostics: keep the
            # high-conviction short rank sleeve, rebalance daily, and avoid the
            # churn-heavy long sleeve that costs overwhelm.
            cfg.gross_cap = 1.0
            cfg.net_cap = 1.0
            cfg.single_pair_cap = 1.0
            cfg.risk_per_trade = 0.04
            cfg.top_k = 2
            cfg.low_pair_top_k = 1
            cfg.side_mode = "short"
            cfg.min_abs_score_z = 1.5
            cfg.rebalance_hours = 8
            cfg.leverage_cap = 5.0
            cfg.edge_mode = "rolling_ic"
            cfg.edge_lookback_hours = 336
            cfg.edge_min_periods = 168
            cfg.edge_deadband = 0.005
        env_overrides = {
            "gross_cap": ("RP_GROSS_CAP", float),
            "daily_loss_limit": ("RP_DAILY_LOSS_LIMIT", float),
            "risk_per_trade": ("RP_RISK_PER_TRADE", float),
            "min_abs_score_z": ("RP_SCORE_THRESHOLD", float),
            "rebalance_hours": ("RP_REBALANCE_HOURS", int),
            "leverage_cap": ("RP_MAX_LEVERAGE", float),
            "min_pairs_for_top_k": ("RP_MIN_PAIRS_FOR_TOP_K", int),
            "low_pair_top_k": ("RP_LOW_PAIR_TOP_K", int),
            "edge_lookback_hours": ("RP_EDGE_LOOKBACK_HOURS", int),
            "edge_min_periods": ("RP_EDGE_MIN_PERIODS", int),
            "edge_deadband": ("RP_EDGE_DEADBAND", float),
            "pair_edge_deadband": ("RP_PAIR_EDGE_DEADBAND", float),
            "pair_edge_strong_ic": ("RP_PAIR_EDGE_STRONG_IC", float),
            "pair_edge_very_strong_ic": ("RP_PAIR_EDGE_VERY_STRONG_IC", float),
            "pair_edge_weak_cap": ("RP_PAIR_EDGE_WEAK_CAP", float),
            "regime_min_edge_ic": ("RP_REGIME_MIN_EDGE_IC", float),
            "regime_min_pair_edge_ic": ("RP_REGIME_MIN_PAIR_EDGE_IC", float),
            "regime_min_pair_count": ("RP_REGIME_MIN_PAIR_COUNT", int),
            "regime_short_max_market_mom_24h": ("RP_REGIME_SHORT_MAX_MARKET_MOM_24H", float),
            "regime_short_max_market_mom_72h": ("RP_REGIME_SHORT_MAX_MARKET_MOM_72H", float),
            "regime_max_market_atr_pct": ("RP_REGIME_MAX_MARKET_ATR_PCT", float),
            "short_max_mom_24h": ("RP_SHORT_MAX_MOM_24H", float),
            "short_max_mom_72h": ("RP_SHORT_MAX_MOM_72H", float),
            "long_min_mom_24h": ("RP_LONG_MIN_MOM_24H", float),
            "max_entry_atr_pct": ("RP_MAX_ENTRY_ATR_PCT", float),
            "short_max_market_mom_24h": ("RP_SHORT_MAX_MARKET_MOM_24H", float),
            "short_max_market_mom_72h": ("RP_SHORT_MAX_MARKET_MOM_72H", float),
            "short_max_market_ma_gap": ("RP_SHORT_MAX_MARKET_MA_GAP", float),
            "short_exit_mom_24h": ("RP_SHORT_EXIT_MOM_24H", float),
            "short_exit_mom_72h": ("RP_SHORT_EXIT_MOM_72H", float),
            "short_exit_market_mom_24h": ("RP_SHORT_EXIT_MARKET_MOM_24H", float),
            "short_exit_market_ma_gap": ("RP_SHORT_EXIT_MARKET_MA_GAP", float),
        }
        for attr, (env_name, caster) in env_overrides.items():
            raw = os.environ.get(env_name)
            if raw not in (None, ""):
                try:
                    setattr(cfg, attr, caster(raw))
                except Exception:
                    pass
        env_side = os.environ.get("RP_SIDE_MODE")
        if env_side:
            cfg.side_mode = str(env_side).strip().lower()
        env_edge = os.environ.get("RP_EDGE_MODE")
        if env_edge:
            cfg.edge_mode = str(env_edge).strip().lower()
        env_pair_edge = os.environ.get("RP_PAIR_EDGE_LEVERAGE")
        if env_pair_edge not in (None, ""):
            cfg.pair_edge_leverage = str(env_pair_edge).strip().lower() not in {"0", "false", "no", "off"}
        env_regime = os.environ.get("RP_REGIME_MODE")
        if env_regime:
            cfg.regime_mode = str(env_regime).strip().lower()
        if gross_cap is not None:
            cfg.gross_cap = float(gross_cap)
            if net_cap is None and profile_name == RISK_PROFILE_AGGRESSIVE:
                cfg.net_cap = float(gross_cap)
            if single_pair_cap is None and profile_name == RISK_PROFILE_AGGRESSIVE:
                cfg.single_pair_cap = min(float(gross_cap), 2.0)
        if net_cap is not None:
            cfg.net_cap = float(net_cap)
        if top_k is not None:
            cfg.top_k = int(top_k)
        if min_pairs_for_top_k is not None:
            cfg.min_pairs_for_top_k = int(min_pairs_for_top_k)
        if low_pair_top_k is not None:
            cfg.low_pair_top_k = int(low_pair_top_k)
        if single_pair_cap is not None:
            cfg.single_pair_cap = float(single_pair_cap)
        if side_mode is not None:
            cfg.side_mode = str(side_mode).strip().lower()
        if min_abs_score_z is not None:
            cfg.min_abs_score_z = float(min_abs_score_z)
        if rebalance_minutes is not None:
            cfg.rebalance_minutes = int(rebalance_minutes)
            cfg.rebalance_hours = bars_for_minutes(int(rebalance_minutes), cfg.timeframe)
        elif rebalance_hours is not None:
            cfg.rebalance_minutes = int(rebalance_hours) * 60
            cfg.rebalance_hours = bars_for_hours(int(rebalance_hours), cfg.timeframe)
        else:
            cfg.rebalance_minutes = int(cfg.rebalance_hours) * 60
            cfg.rebalance_hours = bars_for_hours(int(cfg.rebalance_hours), cfg.timeframe)
        if risk_per_trade is not None:
            cfg.risk_per_trade = float(risk_per_trade)
        if leverage_cap is not None:
            cfg.leverage_cap = float(leverage_cap)
        if edge_mode is not None:
            cfg.edge_mode = str(edge_mode).strip().lower()
        if edge_lookback_hours is not None:
            cfg.edge_lookback_hours = int(edge_lookback_hours)
        if edge_min_periods is not None:
            cfg.edge_min_periods = int(edge_min_periods)
        if edge_deadband is not None:
            cfg.edge_deadband = float(edge_deadband)
        if pair_edge_leverage is not None:
            cfg.pair_edge_leverage = bool(pair_edge_leverage)
        if pair_edge_deadband is not None:
            cfg.pair_edge_deadband = float(pair_edge_deadband)
        if pair_edge_strong_ic is not None:
            cfg.pair_edge_strong_ic = float(pair_edge_strong_ic)
        if pair_edge_very_strong_ic is not None:
            cfg.pair_edge_very_strong_ic = float(pair_edge_very_strong_ic)
        if pair_edge_weak_cap is not None:
            cfg.pair_edge_weak_cap = float(pair_edge_weak_cap)
        if regime_mode is not None:
            cfg.regime_mode = str(regime_mode).strip().lower()
        if regime_min_edge_ic is not None:
            cfg.regime_min_edge_ic = float(regime_min_edge_ic)
        if regime_min_pair_edge_ic is not None:
            cfg.regime_min_pair_edge_ic = float(regime_min_pair_edge_ic)
        if regime_min_pair_count is not None:
            cfg.regime_min_pair_count = int(regime_min_pair_count)
        if regime_short_max_market_mom_24h is not None:
            cfg.regime_short_max_market_mom_24h = float(regime_short_max_market_mom_24h)
        if regime_short_max_market_mom_72h is not None:
            cfg.regime_short_max_market_mom_72h = float(regime_short_max_market_mom_72h)
        if regime_max_market_atr_pct is not None:
            cfg.regime_max_market_atr_pct = float(regime_max_market_atr_pct)
        if short_max_mom_24h is not None:
            cfg.short_max_mom_24h = float(short_max_mom_24h)
        if short_max_mom_72h is not None:
            cfg.short_max_mom_72h = float(short_max_mom_72h)
        if long_min_mom_24h is not None:
            cfg.long_min_mom_24h = float(long_min_mom_24h)
        if max_entry_atr_pct is not None:
            cfg.max_entry_atr_pct = float(max_entry_atr_pct)
        if short_max_market_mom_24h is not None:
            cfg.short_max_market_mom_24h = float(short_max_market_mom_24h)
        if short_max_market_mom_72h is not None:
            cfg.short_max_market_mom_72h = float(short_max_market_mom_72h)
        if short_max_market_ma_gap is not None:
            cfg.short_max_market_ma_gap = float(short_max_market_ma_gap)
        if short_exit_mom_24h is not None:
            cfg.short_exit_mom_24h = float(short_exit_mom_24h)
        if short_exit_mom_72h is not None:
            cfg.short_exit_mom_72h = float(short_exit_mom_72h)
        if short_exit_market_mom_24h is not None:
            cfg.short_exit_market_mom_24h = float(short_exit_market_mom_24h)
        if short_exit_market_ma_gap is not None:
            cfg.short_exit_market_ma_gap = float(short_exit_market_ma_gap)
        env_exclude = os.environ.get("RP_EXCLUDE_PAIRS")
        if exclude_pairs is None and env_exclude:
            exclude_pairs = env_exclude
        if exclude_pairs is not None:
            cfg.exclude_pairs = _parse_pair_list(exclude_pairs)
        return cfg


@dataclass
class SelectionConfig:
    n: int = 50
    strict_abs_ic: float = 0.012
    fallback_abs_ic: float = 0.008
    min_residual_ratio: float = 0.8
    min_sign_agree: int = 6
    strict_corr_gate: float = 0.65
    fallback_corr_gate: float = 0.75
    min_before_fallback: int = 30
    family_cap: int = 10
    score_mode: str = "portfolio"


@dataclass
class SelectedFactor:
    name: str
    expression: str
    direction: float
    weight: float
    ensemble_score: float
    neutralized_ic: float
    oos_ic: float
    residual_ic_ratio: float
    sign_agree: int
    primary_family: str
    max_corr_to_kept: float = 0.0
    origin: str = ""
    selection_mode: str = "strict"


@dataclass
class AccountRiskStatus:
    mode: str = "normal"
    allow_new_entries: bool = True
    leverage_cap: float = 10.0
    gross_cap: Optional[float] = None


@dataclass
class AccountRiskController:
    cfg: RiskConfig
    day: Optional[pd.Timestamp] = None
    week: Optional[Tuple[int, int]] = None
    day_start_equity: float = 1.0
    week_start_equity: float = 1.0
    high_water: float = 1.0
    consecutive_losses: int = 0
    pause_until: Optional[pd.Timestamp] = None
    safe_mode_reason: str = ""
    last_status: AccountRiskStatus = field(default_factory=AccountRiskStatus)

    def update(
        self,
        timestamp: Any,
        equity: float,
        realized_pnl: Optional[float] = None,
    ) -> AccountRiskStatus:
        ts = pd.Timestamp(timestamp)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        equity = float(max(equity, 1e-12))
        day = ts.normalize()
        iso = ts.isocalendar()
        week = (int(iso.year), int(iso.week))

        if self.day is None or day != self.day:
            self.day = day
            self.day_start_equity = equity
        if self.week is None or week != self.week:
            self.week = week
            self.week_start_equity = equity

        self.high_water = max(float(self.high_water), equity)
        if realized_pnl is not None:
            pnl = float(realized_pnl)
            if pnl < 0:
                self.consecutive_losses += 1
            elif pnl > 0:
                self.consecutive_losses = 0
            if self.consecutive_losses >= int(self.cfg.consecutive_loss_limit):
                self.pause_until = ts + timedelta(hours=int(self.cfg.pause_hours))
                self.consecutive_losses = 0

        day_loss = (self.day_start_equity - equity) / max(self.day_start_equity, 1e-12)
        week_loss = (self.week_start_equity - equity) / max(self.week_start_equity, 1e-12)
        drawdown = (self.high_water - equity) / max(self.high_water, 1e-12)

        if self.pause_until is not None and ts < self.pause_until:
            status = AccountRiskStatus(mode="loss_pause", allow_new_entries=False, leverage_cap=0.0, gross_cap=0.0)
        elif day_loss >= float(self.cfg.daily_loss_limit):
            status = AccountRiskStatus(mode="daily_halt", allow_new_entries=False, leverage_cap=0.0, gross_cap=0.0)
        elif week_loss >= float(self.cfg.weekly_loss_limit):
            self.safe_mode_reason = "weekly_safe"
            status = AccountRiskStatus(mode="weekly_safe", allow_new_entries=True, leverage_cap=1.0, gross_cap=1.0)
        elif drawdown >= float(self.cfg.drawdown_safe_mode):
            self.safe_mode_reason = "drawdown_safe"
            status = AccountRiskStatus(mode="drawdown_safe", allow_new_entries=True, leverage_cap=1.0, gross_cap=1.0)
        else:
            status = AccountRiskStatus(mode="normal", allow_new_entries=True, leverage_cap=float(max(ALLOWED_LEVERAGES)), gross_cap=None)

        self.last_status = status
        return status


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def _candidate_from_row(row: Dict[str, Any]) -> mining.CandidateRecord:
    allowed = {f.name for f in fields(mining.CandidateRecord)}
    data = {k: v for k, v in row.items() if k in allowed}
    if "expression" not in data:
        data["expression"] = str(row.get("formula") or "")
    data.setdefault("origin", str(row.get("origin") or "rank_fallback"))
    return mining.CandidateRecord(**data)


def load_candidates(
    tag: str = DEFAULT_TAG,
    *,
    candidate_state: Optional[str | Path] = None,
) -> Tuple[List[mining.CandidateRecord], str]:
    """Load mining checkpoint survivors first, then fallback to exported JSON."""
    if candidate_state:
        path = Path(candidate_state).expanduser()
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("survivors") or payload.get("expressions") or []
        if not rows:
            raise ValueError(f"candidate state has no survivors/expressions: {path}")
        return [mining.annotate_diversity(_candidate_from_row(row)) for row in rows], str(path)

    state = mining.load_state(tag)
    if state:
        _, survivors, _ = state
        return [mining.annotate_diversity(c) for c in survivors], "checkpoint"

    path = repo_paths.user_data_root() / f"freqai_expressions_{tag}.json"
    if not path.exists():
        path = USER_DATA / f"freqai_expressions_{tag}.json"
    if not path.exists():
        raise FileNotFoundError(f"no mining checkpoint or expression file found for tag={tag}")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    rows = payload.get("expressions") or []
    return [mining.annotate_diversity(_candidate_from_row(row)) for row in rows], str(path)


def _ic_for_direction(cand: mining.CandidateRecord) -> float:
    neutralized = _safe_float(getattr(cand, "neutralized_ic", 0.0))
    if abs(neutralized) > 1e-12:
        return neutralized
    return _safe_float(getattr(cand, "oos_ic", 0.0))


def _residual_ratio(cand: mining.CandidateRecord) -> float:
    ratio = _safe_float(getattr(cand, "residual_ic_ratio", 0.0), default=0.0)
    return ratio if ratio > 0 else 1.0


def _factor_score(cand: mining.CandidateRecord) -> float:
    ic = abs(_ic_for_direction(cand))
    sign_ratio = max(0.0, min(1.0, float(getattr(cand, "sign_agree", 0) or 0) / 10.0))
    residual_bonus = max(0.1, min(2.0, _residual_ratio(cand))) / 2.0
    fitness = _safe_float(getattr(cand, "fitness", 0.0), default=0.0)
    fitness_bonus = 1.0 + min(1.0, abs(fitness) / max(ic, 1e-9)) * 0.10 if ic > 0 else 1.0
    return float(ic * sign_ratio * residual_bonus * fitness_bonus)


def _passes_factor_gates(cand: mining.CandidateRecord, *, abs_ic: float, residual_ratio: float, sign_agree: int) -> bool:
    return (
        abs(_ic_for_direction(cand)) >= float(abs_ic)
        and _residual_ratio(cand) >= float(residual_ratio)
        and int(getattr(cand, "sign_agree", 0) or 0) >= int(sign_agree)
    )


def _rank_corr_from_cache(a_expr: str, b_expr: str, rank_cache: Dict[str, np.ndarray]) -> float:
    a = rank_cache.get(a_expr)
    b = rank_cache.get(b_expr)
    if a is None or b is None:
        return 0.0
    return abs(mining._rank_corr(a, b))  # noqa: SLF001


def _greedy_select(
    candidates: Sequence[mining.CandidateRecord],
    *,
    n: int,
    corr_gate: float,
    family_cap: int,
    rank_cache: Optional[Dict[str, np.ndarray]],
) -> Tuple[List[mining.CandidateRecord], Dict[str, Any]]:
    rank_cache = rank_cache or {}
    selected: List[mining.CandidateRecord] = []
    rejected: List[Dict[str, Any]] = []
    family_counts: Dict[str, int] = {}
    sorted_candidates = sorted(
        (mining.annotate_diversity(c) for c in candidates),
        key=lambda c: (_factor_score(c), abs(_ic_for_direction(c)), int(c.sign_agree or 0)),
        reverse=True,
    )

    for cand in sorted_candidates:
        if len(selected) >= int(n):
            break
        family = cand.primary_family or mining.primary_family(cand.expression)
        if family_cap > 0 and family_counts.get(family, 0) >= int(family_cap):
            rejected.append({"expression": cand.expression, "reason": "family_cap", "primary_family": family})
            continue
        max_corr = 0.0
        collision = ""
        for kept in selected:
            corr = _rank_corr_from_cache(cand.expression, kept.expression, rank_cache)
            if corr > max_corr:
                max_corr = corr
                collision = kept.expression
        if max_corr >= float(corr_gate):
            rejected.append({
                "expression": cand.expression,
                "reason": "corr_gate",
                "max_corr_to_kept": float(max_corr),
                "collision_expression": collision,
            })
            continue
        cand.max_corr_to_kept = float(max_corr)
        selected.append(cand)
        family_counts[family] = family_counts.get(family, 0) + 1

    return selected, {
        "family_distribution": dict(family_counts),
        "rejected": rejected[:200],
        "rank_cache_size": len(rank_cache),
    }


def select_factor_records(
    candidates: Sequence[mining.CandidateRecord],
    *,
    config: Optional[SelectionConfig] = None,
    rank_cache: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[List[SelectedFactor], Dict[str, Any]]:
    """Select, orient, and weight factors for the rank ensemble."""
    cfg = config or SelectionConfig()
    strict_pool = [
        c for c in candidates
        if _passes_factor_gates(
            c,
            abs_ic=cfg.strict_abs_ic,
            residual_ratio=cfg.min_residual_ratio,
            sign_agree=cfg.min_sign_agree,
        )
    ]
    strict_selected, strict_report = _greedy_select(
        strict_pool,
        n=cfg.n,
        corr_gate=cfg.strict_corr_gate,
        family_cap=cfg.family_cap,
        rank_cache=rank_cache,
    )
    use_fallback = len(strict_selected) < min(int(cfg.min_before_fallback), int(cfg.n))

    selection_mode = "strict"
    selected = strict_selected
    report = {
        "mode": selection_mode,
        "strict_pool": len(strict_pool),
        "strict_selected": len(strict_selected),
        "strict": strict_report,
    }
    if use_fallback:
        fallback_pool = [
            c for c in candidates
            if _passes_factor_gates(
                c,
                abs_ic=cfg.fallback_abs_ic,
                residual_ratio=cfg.min_residual_ratio,
                sign_agree=cfg.min_sign_agree,
            )
        ]
        fallback_selected, fallback_report = _greedy_select(
            fallback_pool,
            n=cfg.n,
            corr_gate=cfg.fallback_corr_gate,
            family_cap=cfg.family_cap,
            rank_cache=rank_cache,
        )
        selected = fallback_selected
        selection_mode = "fallback_relaxed"
        report.update({
            "mode": selection_mode,
            "fallback_pool": len(fallback_pool),
            "fallback_selected": len(fallback_selected),
            "fallback": fallback_report,
        })

    raw_scores = np.asarray([_factor_score(c) for c in selected], dtype=np.float64)
    if raw_scores.size == 0:
        weights = np.asarray([], dtype=np.float64)
    elif float(np.nansum(raw_scores)) > 0:
        weights = raw_scores / float(np.nansum(raw_scores))
    else:
        weights = np.full(raw_scores.shape, 1.0 / float(len(raw_scores)))

    selected_factors: List[SelectedFactor] = []
    for i, (cand, weight) in enumerate(zip(selected, weights)):
        ic = _ic_for_direction(cand)
        direction = float(np.sign(ic) or 1.0)
        selected_factors.append(
            SelectedFactor(
                name=f"rp_f{i + 1:03d}",
                expression=cand.expression,
                direction=direction,
                weight=float(weight),
                ensemble_score=float(_factor_score(cand)),
                neutralized_ic=_safe_float(getattr(cand, "neutralized_ic", 0.0)),
                oos_ic=_safe_float(getattr(cand, "oos_ic", 0.0)),
                residual_ic_ratio=_residual_ratio(cand),
                sign_agree=int(getattr(cand, "sign_agree", 0) or 0),
                primary_family=cand.primary_family or mining.primary_family(cand.expression),
                max_corr_to_kept=float(getattr(cand, "max_corr_to_kept", 0.0) or 0.0),
                origin=str(getattr(cand, "origin", "") or ""),
                selection_mode=selection_mode,
            )
        )

    report.update({
        "requested_n": int(cfg.n),
        "selected_n": len(selected_factors),
        "corr_gate": cfg.fallback_corr_gate if use_fallback else cfg.strict_corr_gate,
        "family_cap": int(cfg.family_cap),
        "selected": [asdict(f) for f in selected_factors],
    })
    return selected_factors, report


def _mining_config_from_tag(tag: str) -> mining.MiningConfig:
    run_cfg = mining.load_state_config(tag)
    cfg_kwargs = {f.name: run_cfg[f.name] for f in fields(mining.MiningConfig) if f.name in run_cfg}
    cfg = mining.MiningConfig(**cfg_kwargs) if cfg_kwargs else mining.MiningConfig()
    if not getattr(cfg, "cache_dir", None):
        cfg.cache_dir = str(DEFAULT_CACHE_DIR)
    return cfg


def _mining_config_from_candidate_state(candidate_state: Optional[str | Path], *, tag: str) -> mining.MiningConfig:
    if not candidate_state:
        return _mining_config_from_tag(tag)
    path = repo_paths.resolve_repo_path(candidate_state)
    if not path.exists():
        return _mining_config_from_tag(tag)
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return _mining_config_from_tag(tag)
    run_cfg = payload.get("config") if isinstance(payload, Mapping) else {}
    cfg_kwargs = {f.name: run_cfg[f.name] for f in fields(mining.MiningConfig) if isinstance(run_cfg, Mapping) and f.name in run_cfg}
    cfg = mining.MiningConfig(**cfg_kwargs) if cfg_kwargs else _mining_config_from_tag(tag)
    if not getattr(cfg, "cache_dir", None):
        cfg.cache_dir = str(DEFAULT_CACHE_DIR)
    return cfg


def _candidate_state_manifest(candidate_state: Optional[str | Path]) -> dict[str, Any]:
    if not candidate_state:
        return {}
    path = repo_paths.resolve_repo_path(candidate_state)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}
    if not isinstance(payload, Mapping):
        return {}
    manifest = {
        "timeframe": payload.get("timeframe"),
        "evaluation_lane": payload.get("evaluation_lane") or payload.get("lane"),
        "data_venue": payload.get("data_venue"),
        "label_horizons": payload.get("label_horizons"),
        "pairs": payload.get("pairs"),
    }
    if isinstance(payload.get("intraday"), Mapping):
        intraday = payload["intraday"]
        for key in ("timeframe", "evaluation_lane", "data_venue", "label_horizons", "pairs"):
            manifest[key] = manifest.get(key) or intraday.get(key)
    if isinstance(payload.get("config"), Mapping):
        cfg = payload["config"]
        for key in ("timeframe", "evaluation_lane", "data_venue", "label_horizons", "pairs"):
            manifest[key] = manifest.get(key) or cfg.get(key)
    return {k: v for k, v in manifest.items() if v not in (None, "", [])}


def _resolve_rank_pair_universe(
    *,
    tag: str,
    candidate_state: Optional[str | Path],
    timeframe: str,
    feature_venue: str,
    pairs: Optional[Sequence[str] | str] = None,
) -> tuple[list[str], dict[str, Any]]:
    mine_cfg = _mining_config_from_candidate_state(candidate_state, tag=tag)
    requested: Optional[Sequence[str] | str] = pairs
    source = "argument"
    if requested in (None, ""):
        requested = getattr(mine_cfg, "pairs", None) or "default"
        source = "mining_config"
    if isinstance(requested, str) and requested.strip().lower() == "all":
        requested = "auto"
    cfg_venue = str(getattr(mine_cfg, "data_venue", "") or "").strip().lower()
    data_dir = getattr(mine_cfg, "data_dir", None) if cfg_venue == str(feature_venue).strip().lower() else None
    data_root, pair_list, _ = mining._resolve_mining_data(  # noqa: SLF001
        data_venue=feature_venue,
        data_dir=data_dir,
        timeframe=timeframe,
        pairs=requested,
    )
    if not pair_list:
        pair_list = list(DEFAULT_PAIRS)
    return pair_list, {
        "source": source,
        "requested": requested,
        "count": len(pair_list),
        "pairs": pair_list,
        "timeframe": timeframe,
        "data_venue": feature_venue,
        "data_root": str(data_root),
    }


def build_rank_cache_for_selection(
    tag: str,
    candidates: Sequence[mining.CandidateRecord],
    *,
    config: Optional[SelectionConfig] = None,
    candidate_state: Optional[str | Path] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Recompute OOS rank series for correlation-gated factor selection."""
    cfg = config or SelectionConfig()
    loose_pool = [
        c for c in candidates
        if _passes_factor_gates(
            c,
            abs_ic=cfg.fallback_abs_ic,
            residual_ratio=cfg.min_residual_ratio,
            sign_agree=cfg.min_sign_agree,
        )
    ]
    mine_cfg = _mining_config_from_candidate_state(candidate_state, tag=tag)
    rank_cache: Dict[str, np.ndarray] = {}
    errors: List[Dict[str, str]] = []
    try:
        big, _ = mining.build_big(
            timeframe=mine_cfg.timeframe,
            label_bars=int(getattr(mine_cfg, "label_period", mining.DEFAULT_LABEL_PERIOD) or mining.DEFAULT_LABEL_PERIOD),
            label_mode=mine_cfg.label_mode,
            pair_reference=mine_cfg.pair_reference,
            data_dir=mine_cfg.data_dir,
            data_venue=getattr(mine_cfg, "data_venue", "kucoin"),
            pairs=mine_cfg.pairs,
            cache_dir=mine_cfg.cache_dir,
            no_cache=mine_cfg.no_cache,
        )
        for cand in loose_pool:
            metrics = mining.eval_ic(big, cand.expression, mine_cfg, return_oos_series=True)
            if metrics.get("status") == "ok" and "oos_series" in metrics:
                ranks = mining._series_to_ranks(np.asarray(metrics["oos_series"], dtype=np.float64))  # noqa: SLF001
                if ranks is not None:
                    rank_cache[cand.expression] = ranks
                    continue
            errors.append({"expression": cand.expression, "status": str(metrics.get("status")), "error": str(metrics.get("error", ""))})
    except Exception as exc:  # noqa: BLE001
        errors.append({"expression": "*", "status": "rank_cache_failed", "error": str(exc)[:240]})
    return rank_cache, {"loose_pool": len(loose_pool), "ranked": len(rank_cache), "errors": errors[:100]}


def _normalize_pair(pair: str) -> str:
    raw = str(pair).strip()
    raw = raw.split(":")[0]
    raw = raw.replace("_", "/") if "/" not in raw and "_" in raw else raw
    parts = raw.split("_")
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    return raw


def _parse_pair_list(value: Optional[Sequence[str] | str]) -> Tuple[str, ...]:
    if value in (None, ""):
        return ()
    if isinstance(value, str):
        raw_items = value.replace(";", ",").split(",")
    else:
        raw_items = list(value)
    pairs = []
    for item in raw_items:
        text = str(item).strip()
        if not text:
            continue
        pairs.append(_normalize_pair(text))
    return tuple(dict.fromkeys(pairs))


def _excluded_pair_set(cfg: RiskConfig) -> set[str]:
    return set(_parse_pair_list(getattr(cfg, "exclude_pairs", ())))


def _pair_file_token(pair: str) -> str:
    base = _normalize_pair(pair).replace("/", "_")
    if base.endswith("_USDT"):
        return f"{base}_USDT"
    return base


def _market_data_root(*, data_venue: str = "kucoin", data_dir: Optional[str | Path] = None) -> Path:
    if data_dir is not None:
        return Path(data_dir)
    venue = str(data_venue or "kucoin").strip().lower()
    if venue in FUTURES_VENUES:
        root = repo_paths.user_data_root() / "data" / venue / "futures"
        if venue == "okx" and not root.exists():
            return OKX_FUTURES_DIR
        return root
    return KUCOIN_DIR


def _market_data_path(pair: str, *, timeframe: str, data_venue: str = "kucoin", data_dir: Optional[str | Path] = None) -> Path:
    tf = normalize_timeframe(timeframe)
    venue = str(data_venue or "kucoin").strip().lower()
    root = _market_data_root(data_venue=venue, data_dir=data_dir)
    if venue in FUTURES_VENUES:
        return root / f"{_pair_file_token(pair)}-{tf}-futures.feather"
    return feather_for_pair(pair, timeframe=tf, data_dir=root)


def load_feature_panel(
    *,
    pairs: Optional[Sequence[str]] = None,
    timeframe: str = "1h",
    data_venue: str = "kucoin",
    data_dir: Optional[str | Path] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    pair_list = list(pairs or DEFAULT_PAIRS)
    tf = normalize_timeframe(timeframe)
    root = _market_data_root(data_venue=data_venue, data_dir=data_dir)
    feat_cfg = json.loads(FEATURE_FILE.read_text(encoding="utf-8-sig")) if FEATURE_FILE.exists() else {"features": []}
    frames: List[pd.DataFrame] = []
    missing: List[str] = []
    for pair in pair_list:
        path = _market_data_path(pair, timeframe=tf, data_venue=data_venue, data_dir=root)
        if not path.exists():
            missing.append(str(path))
            continue
        df = pd.read_feather(path)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df = apply_configured_features(df, feat_cfg)
        df["__pair__"] = _normalize_pair(pair)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no {data_venue} {tf} feature feathers found under {root}; missing={missing[:10]}")
    panel = pd.concat(frames, ignore_index=True).sort_values(["__pair__", "date"]).reset_index(drop=True)
    if start:
        panel = panel.loc[panel["date"] >= pd.Timestamp(start, tz="UTC")]
    if end:
        panel = panel.loc[panel["date"] < pd.Timestamp(end, tz="UTC")]
    return panel.reset_index(drop=True)


def load_venue_ohlcv(
    *,
    venue: str = "okx",
    timeframe: str = "1h",
    pairs: Optional[Sequence[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    venue_s = str(venue or "okx").lower()
    if venue_s not in FUTURES_VENUES:
        raise ValueError(f"rank portfolio futures venue must be one of {sorted(FUTURES_VENUES)}, got {venue!r}")
    tf = normalize_timeframe(timeframe)
    pair_list = list(pairs or DEFAULT_PAIRS)
    root = repo_paths.user_data_root() / "data" / venue_s / "futures"
    if venue_s == "okx" and not root.exists():
        root = OKX_FUTURES_DIR
    frames: List[pd.DataFrame] = []
    missing: List[str] = []
    for pair in pair_list:
        token = _pair_file_token(pair)
        path = root / f"{token}-{tf}-futures.feather"
        if not path.exists():
            missing.append(str(path))
            continue
        df = pd.read_feather(path)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df["__pair__"] = _normalize_pair(pair)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no {venue_s} {tf} futures feathers found under {root}; missing={missing[:10]}")
    out = pd.concat(frames, ignore_index=True).sort_values(["__pair__", "date"]).reset_index(drop=True)
    if start:
        out = out.loc[out["date"] >= pd.Timestamp(start, tz="UTC")]
    if end:
        out = out.loc[out["date"] < pd.Timestamp(end, tz="UTC")]
    return out.reset_index(drop=True)


def _eval_expression_by_pair(panel: pd.DataFrame, expression: str) -> pd.Series:
    out = pd.Series(np.nan, index=panel.index, dtype="float64")
    for _, idx in panel.groupby("__pair__", sort=False).groups.items():
        sub = panel.loc[idx]
        values = safe_eval_expression(expression, sub)
        out.loc[idx] = np.asarray(values, dtype=np.float64)
    return out.replace([np.inf, -np.inf], np.nan)


def _xs_z(values: pd.Series, dates: pd.Series) -> pd.Series:
    s = pd.Series(values, index=values.index, dtype="float64").replace([np.inf, -np.inf], np.nan)
    date_key = pd.Series(pd.to_datetime(pd.Series(dates).to_numpy(), utc=True), index=s.index)
    mean = s.groupby(date_key, sort=False).transform("mean")
    sq_mean = s.pow(2).groupby(date_key, sort=False).transform("mean")
    std = np.sqrt((sq_mean - mean.pow(2)).clip(lower=0.0))
    z = (s - mean) / (std + 1e-9)
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def compute_ensemble_scores(
    feature_panel: pd.DataFrame,
    selected_factors: Sequence[SelectedFactor | Dict[str, Any]],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    panel = feature_panel.copy()
    if "__pair__" not in panel.columns:
        raise ValueError("feature_panel must include __pair__")
    panel["date"] = pd.to_datetime(panel["date"], utc=True)
    score = pd.Series(0.0, index=panel.index, dtype="float64")
    errors: List[Dict[str, str]] = []
    used = 0
    factors = [
        f if isinstance(f, SelectedFactor) else SelectedFactor(**{k: f[k] for k in SelectedFactor.__dataclass_fields__ if k in f})
        for f in selected_factors
    ]
    for factor in factors:
        try:
            raw = _eval_expression_by_pair(panel, factor.expression)
            directed = raw * float(factor.direction)
            score = score + _xs_z(directed, panel["date"]) * float(factor.weight)
            used += 1
        except Exception as exc:  # noqa: BLE001
            errors.append({"name": factor.name, "expression": factor.expression, "error": str(exc)[:240]})
    out = panel[["date", "__pair__"]].copy()
    out["rp_score"] = score.replace([np.inf, -np.inf], np.nan)
    out["rp_score_z"] = _xs_z(out["rp_score"], out["date"])
    return out, {"factor_count": len(factors), "used_factor_count": used, "errors": errors[:100]}


def add_risk_columns(venue_panel: pd.DataFrame, *, timeframe: str = "1h") -> pd.DataFrame:
    tf = normalize_timeframe(timeframe)
    df = venue_panel.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    pieces: List[pd.DataFrame] = []
    bars_24h = bars_for_hours(24, tf)
    bars_72h = bars_for_hours(72, tf)
    bars_96h = bars_for_hours(96, tf)
    bars_30d = bars_for_hours(24 * 30, tf)
    min_vol_periods = min(bars_30d, bars_for_hours(24, tf))
    for _, sub in df.sort_values(["__pair__", "date"]).groupby("__pair__", sort=False):
        sub = sub.copy()
        close = sub["close"].astype("float64")
        high = sub["high"].astype("float64")
        low = sub["low"].astype("float64")
        prev = close.shift(1)
        tr = np.maximum(high - low, np.maximum((high - prev).abs(), (low - prev).abs()))
        atr = pd.Series(tr, index=sub.index).ewm(span=14, adjust=False).mean()
        sub["rp_atr_pct"] = (atr / (close.abs() + 1e-12)).replace([np.inf, -np.inf], np.nan)
        sub["rp_mom_24h"] = (close / close.shift(bars_24h) - 1.0).replace([np.inf, -np.inf], np.nan)
        sub["rp_mom_72h"] = (close / close.shift(bars_72h) - 1.0).replace([np.inf, -np.inf], np.nan)
        ema = close.ewm(span=bars_96h, adjust=False).mean()
        sub["rp_ma_gap_96h"] = (close / (ema + 1e-12) - 1.0).replace([np.inf, -np.inf], np.nan)
        med = sub["volume"].astype("float64").rolling(bars_30d, min_periods=min_vol_periods).median()
        sub["rp_volume_ratio"] = (sub["volume"].astype("float64") / (med + 1e-12)).replace([np.inf, -np.inf], np.nan)
        pieces.append(sub)
    out = pd.concat(pieces, ignore_index=True).sort_values(["date", "__pair__"]).reset_index(drop=True)
    market = (
        out.groupby("date", sort=True)[["rp_mom_24h", "rp_mom_72h", "rp_ma_gap_96h", "rp_atr_pct"]]
        .median(numeric_only=True)
        .rename(
            columns={
                "rp_mom_24h": "rp_market_mom_24h",
                "rp_mom_72h": "rp_market_mom_72h",
                "rp_ma_gap_96h": "rp_market_ma_gap_96h",
                "rp_atr_pct": "rp_market_atr_pct",
            }
        )
        .reset_index()
    )
    return out.merge(market, on="date", how="left").sort_values(["date", "__pair__"]).reset_index(drop=True)


def stop_pct_from_atr(atr_pct: Any, cfg: RiskConfig) -> float:
    atr = _safe_float(atr_pct, default=0.02)
    if atr <= 0:
        atr = 0.02
    stop = float(cfg.atr_stop_mult) * atr
    return float(max(float(cfg.min_stop_pct), min(float(cfg.max_stop_pct), stop)))


def liquidation_distance(leverage: float, *, maintenance_margin: float = 0.005, fee_buffer: float = 0.003) -> float:
    lev = max(1.0, float(leverage))
    return float((1.0 / lev) - float(maintenance_margin) - float(fee_buffer))


def reduce_leverage_for_liq(
    requested_leverage: float,
    stop_pct: float,
    *,
    maintenance_margin: float = 0.005,
    fee_buffer: float = 0.003,
    ten_x_requested: bool = False,
) -> Tuple[float, float, bool]:
    requested = max(1.0, float(requested_leverage))
    levels = [lev for lev in ALLOWED_LEVERAGES if lev <= requested + 1e-9]
    if not levels:
        levels = [1]
    for lev in levels:
        dist = liquidation_distance(lev, maintenance_margin=maintenance_margin, fee_buffer=fee_buffer)
        required = (5.0 if ten_x_requested and lev >= 10 else 3.0) * float(stop_pct)
        if dist >= required:
            return float(lev), float(dist), False
    lev = 1.0
    dist = liquidation_distance(lev, maintenance_margin=maintenance_margin, fee_buffer=fee_buffer)
    return lev, dist, bool(dist < 3.0 * float(stop_pct))


def choose_dynamic_leverage(
    *,
    side_rank: int,
    score_z: float,
    atr_pct: float,
    volume_ratio: float,
    stop_pct: float,
    pair_edge_ic: float = 0.0,
    edge_sign: float = 1.0,
    cfg: RiskConfig,
) -> Tuple[float, float, bool]:
    abs_z = abs(_safe_float(score_z, default=0.0))
    atr = max(0.0, _safe_float(atr_pct, default=0.02))
    vol_ratio = _safe_float(volume_ratio, default=1.0)
    if abs_z < 1.0:
        requested = 2.0
    elif abs_z < 1.25:
        requested = 3.0
    elif abs_z < 1.75:
        requested = 5.0
    else:
        requested = 8.0

    ten_x_ok = side_rank == 1 and abs_z >= 1.75 and atr <= 0.025 and vol_ratio >= 1.0
    if ten_x_ok:
        requested = 10.0
    if atr > 0.04:
        requested = min(requested, 3.0)
    elif atr > 0.025:
        requested = min(requested, 5.0)
    if vol_ratio < 0.6:
        requested = min(requested, 3.0)

    if bool(getattr(cfg, "pair_edge_leverage", True)) and _edge_mode(cfg) == "rolling_ic":
        edge = _safe_float(pair_edge_ic, default=0.0)
        global_sign = _safe_float(edge_sign, default=0.0)
        edge_abs = abs(edge)
        aligned = global_sign != 0.0 and (edge * global_sign) > float(cfg.pair_edge_deadband)
        if not aligned:
            requested = min(requested, float(cfg.pair_edge_weak_cap))
        elif edge_abs < float(cfg.pair_edge_strong_ic):
            requested = min(requested, 3.0)
        elif edge_abs < float(cfg.pair_edge_very_strong_ic):
            requested = min(requested, 5.0)

    requested = min(requested, float(cfg.leverage_cap))

    return reduce_leverage_for_liq(
        requested,
        stop_pct,
        maintenance_margin=cfg.maintenance_margin,
        fee_buffer=cfg.fee_buffer,
        ten_x_requested=ten_x_ok,
    )


def _effective_top_k(valid_count: int, cfg: RiskConfig) -> int:
    if valid_count < 4:
        return 0
    k = int(cfg.top_k) if valid_count >= int(cfg.min_pairs_for_top_k) else int(cfg.low_pair_top_k)
    return max(1, min(k, valid_count // 2))


def _side_mode(cfg: RiskConfig) -> str:
    mode = str(getattr(cfg, "side_mode", "both") or "both").strip().lower()
    return mode if mode in {"both", "long", "short"} else "both"


def _edge_mode(cfg: RiskConfig) -> str:
    mode = str(getattr(cfg, "edge_mode", "off") or "off").strip().lower()
    return mode if mode in {"off", "rolling_ic"} else "off"


def _regime_mode(cfg: RiskConfig) -> str:
    mode = str(getattr(cfg, "regime_mode", "off") or "off").strip().lower()
    return mode if mode in {"off", "hq"} else "off"


def _cross_sectional_rank_ic(group: pd.DataFrame) -> float:
    valid = (
        group["rp_score"].notna()
        & group["rp_fwd_ret"].notna()
        & np.isfinite(group["rp_score"].astype(float))
        & np.isfinite(group["rp_fwd_ret"].astype(float))
    )
    if int(valid.sum()) < 4:
        return np.nan
    scores = group.loc[valid, "rp_score"].astype(float).rank(method="average")
    returns = group.loc[valid, "rp_fwd_ret"].astype(float).rank(method="average")
    corr = scores.corr(returns)
    return float(corr) if np.isfinite(corr) else np.nan


def add_causal_edge_columns(merged: pd.DataFrame, cfg: RiskConfig) -> pd.DataFrame:
    """Attach a causal rolling IC direction estimate to each timestamp.

    `rp_edge_ic` at timestamp t is computed from IC observations ending at
    t-1. Each IC observation uses score_t versus close-to-close return t->t+1,
    which is only known after the next candle closes. This keeps the rank
    direction used for candle t from reading candle t->t+1.
    """
    out = merged.copy()
    out["rp_edge_ic"] = 0.0
    out["rp_edge_sign"] = 1.0
    out["rp_pair_edge_ic"] = 0.0
    out["rp_pair_edge_sign"] = 0.0
    out["rp_sort_score"] = out["rp_score"].astype("float64")
    out["rp_sort_score_z"] = out["rp_score_z"].astype("float64")
    if _edge_mode(cfg) != "rolling_ic":
        return out

    out["rp_fwd_ret"] = (
        out.sort_values(["__pair__", "date"])
        .groupby("__pair__", sort=False)["close"]
        .shift(-1)
        .reindex(out.index)
        / out["close"]
        - 1.0
    )
    by_pair = out.sort_values(["__pair__", "date"]).groupby("__pair__", sort=False)
    ic_by_date = out.groupby("date", sort=True)[["rp_score", "rp_fwd_ret"]].apply(_cross_sectional_rank_ic)
    tf = normalize_timeframe(getattr(cfg, "timeframe", "1h"))
    lookback = max(1, bars_for_hours(float(getattr(cfg, "edge_lookback_hours", 336) or 336), tf))
    min_period_hours = float(getattr(cfg, "edge_min_periods", getattr(cfg, "edge_lookback_hours", 336)) or getattr(cfg, "edge_lookback_hours", 336))
    min_periods = max(1, min(lookback, bars_for_hours(min_period_hours, tf)))
    edge = ic_by_date.rolling(lookback, min_periods=min_periods).mean().shift(1)
    deadband = abs(float(getattr(cfg, "edge_deadband", 0.0) or 0.0))
    signs = pd.Series(0.0, index=edge.index, dtype="float64")
    signs.loc[edge > deadband] = 1.0
    signs.loc[edge < -deadband] = -1.0
    out["rp_edge_ic"] = out["date"].map(edge).astype("float64").fillna(0.0)
    out["rp_edge_sign"] = out["date"].map(signs).astype("float64").fillna(0.0)

    def _pair_rolling_ic(sub: pd.DataFrame) -> pd.Series:
        score = sub["rp_score"].astype("float64")
        fwd = sub["rp_fwd_ret"].astype("float64")
        return score.rolling(lookback, min_periods=min_periods).corr(fwd).shift(1)

    pair_edge = by_pair[["rp_score", "rp_fwd_ret"]].apply(_pair_rolling_ic).reset_index(level=0, drop=True)
    pair_edge = pair_edge.reindex(out.index).replace([np.inf, -np.inf], np.nan)
    pair_sign = pd.Series(0.0, index=out.index, dtype="float64")
    pair_sign.loc[pair_edge > float(cfg.pair_edge_deadband)] = 1.0
    pair_sign.loc[pair_edge < -float(cfg.pair_edge_deadband)] = -1.0
    out["rp_pair_edge_ic"] = pair_edge.astype("float64").fillna(0.0)
    out["rp_pair_edge_sign"] = pair_sign.fillna(0.0)
    out["rp_sort_score"] = out["rp_score"].astype("float64") * out["rp_edge_sign"]
    out["rp_sort_score_z"] = out["rp_score_z"].astype("float64") * out["rp_edge_sign"]
    out = out.drop(columns=["rp_fwd_ret"])
    return out


def _enforce_exposure_caps(group: pd.DataFrame, cfg: RiskConfig) -> pd.DataFrame:
    out = group.copy()
    weights = out["rp_target_weight"].astype("float64")
    gross = float(weights.abs().sum())
    if gross > float(cfg.gross_cap) and gross > 0:
        weights = weights * (float(cfg.gross_cap) / gross)
    pos_sum = float(weights[weights > 0].sum())
    neg_sum = float((-weights[weights < 0]).sum())
    net = pos_sum - neg_sum
    if net > float(cfg.net_cap) and pos_sum > 0:
        target_pos = float(cfg.net_cap) + neg_sum
        weights.loc[weights > 0] = weights.loc[weights > 0] * max(0.0, target_pos / pos_sum)
    elif net < -float(cfg.net_cap) and neg_sum > 0:
        target_neg = float(cfg.net_cap) + pos_sum
        weights.loc[weights < 0] = weights.loc[weights < 0] * max(0.0, target_neg / neg_sum)
    out["rp_target_weight"] = weights
    return out


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return out if np.isfinite(out) else None


def _regime_allows_entries(group: pd.DataFrame, cfg: RiskConfig, side_mode: str) -> bool:
    if _regime_mode(cfg) != "hq":
        return True
    if group.empty:
        return False

    min_edge = float(max(0.0, getattr(cfg, "regime_min_edge_ic", 0.0) or 0.0))
    edge_ic = _optional_float(group["rp_edge_ic"].iloc[0] if "rp_edge_ic" in group.columns else None)
    if min_edge > 0.0 and (edge_ic is None or abs(edge_ic) < min_edge):
        return False

    max_mkt_atr = _optional_float(getattr(cfg, "regime_max_market_atr_pct", None))
    mkt_atr = _optional_float(group["rp_market_atr_pct"].iloc[0] if "rp_market_atr_pct" in group.columns else None)
    if max_mkt_atr is not None and mkt_atr is not None and mkt_atr > max_mkt_atr:
        return False

    if side_mode in {"both", "short"}:
        regime_m24 = _optional_float(getattr(cfg, "regime_short_max_market_mom_24h", None))
        regime_m72 = _optional_float(getattr(cfg, "regime_short_max_market_mom_72h", None))
        m24 = _optional_float(group["rp_market_mom_24h"].iloc[0] if "rp_market_mom_24h" in group.columns else None)
        m72 = _optional_float(group["rp_market_mom_72h"].iloc[0] if "rp_market_mom_72h" in group.columns else None)
        if regime_m24 is not None and m24 is not None and m24 > regime_m24:
            return False
        if regime_m72 is not None and m72 is not None and m72 > regime_m72:
            return False

    min_pair_edge = float(max(0.0, getattr(cfg, "regime_min_pair_edge_ic", 0.0) or 0.0))
    min_pair_count = int(max(0, getattr(cfg, "regime_min_pair_count", 0) or 0))
    if min_pair_edge > 0.0 or min_pair_count > 0:
        if "rp_pair_edge_ic" not in group.columns:
            return False
        edge_sign = _safe_float(group["rp_edge_sign"].iloc[0] if "rp_edge_sign" in group.columns else 0.0, default=0.0)
        pair_edge = group["rp_pair_edge_ic"].astype("float64")
        if abs(edge_sign) <= 1e-12:
            aligned_count = 0
        else:
            aligned = (pair_edge * edge_sign) > float(getattr(cfg, "pair_edge_deadband", 0.0) or 0.0)
            if min_pair_edge > 0.0:
                aligned = aligned & (pair_edge.abs() >= min_pair_edge)
            aligned_count = int(aligned.sum())
        if aligned_count < min_pair_count:
            return False

    return True


def _passes_entry_filters(row: pd.Series, side: int, cfg: RiskConfig) -> bool:
    pair = str(row.get("__pair__") or row.get("pair") or "")
    if pair and pair in _excluded_pair_set(cfg):
        return False

    max_atr = _optional_float(getattr(cfg, "max_entry_atr_pct", None))
    atr = _optional_float(row.get("rp_atr_pct"))
    if max_atr is not None and atr is not None and atr > max_atr:
        return False

    mom24 = _optional_float(row.get("rp_mom_24h"))
    if side < 0:
        short_max_24 = _optional_float(getattr(cfg, "short_max_mom_24h", None))
        if short_max_24 is not None and mom24 is not None and mom24 > short_max_24:
            return False
        mom72 = _optional_float(row.get("rp_mom_72h"))
        short_max_72 = _optional_float(getattr(cfg, "short_max_mom_72h", None))
        if short_max_72 is not None and mom72 is not None and mom72 > short_max_72:
            return False
        market24 = _optional_float(row.get("rp_market_mom_24h"))
        market_max_24 = _optional_float(getattr(cfg, "short_max_market_mom_24h", None))
        if market_max_24 is not None and market24 is not None and market24 > market_max_24:
            return False
        market72 = _optional_float(row.get("rp_market_mom_72h"))
        market_max_72 = _optional_float(getattr(cfg, "short_max_market_mom_72h", None))
        if market_max_72 is not None and market72 is not None and market72 > market_max_72:
            return False
        market_gap = _optional_float(row.get("rp_market_ma_gap_96h"))
        market_max_gap = _optional_float(getattr(cfg, "short_max_market_ma_gap", None))
        if market_max_gap is not None and market_gap is not None and market_gap > market_max_gap:
            return False
    elif side > 0:
        long_min_24 = _optional_float(getattr(cfg, "long_min_mom_24h", None))
        if long_min_24 is not None and mom24 is not None and mom24 < long_min_24:
            return False
    return True


def _should_exit_held(row: pd.Series, side: int, cfg: RiskConfig) -> bool:
    if side < 0:
        mom24 = _optional_float(row.get("rp_mom_24h"))
        exit_24 = _optional_float(getattr(cfg, "short_exit_mom_24h", None))
        if exit_24 is not None and mom24 is not None and mom24 > exit_24:
            return True
        mom72 = _optional_float(row.get("rp_mom_72h"))
        exit_72 = _optional_float(getattr(cfg, "short_exit_mom_72h", None))
        if exit_72 is not None and mom72 is not None and mom72 > exit_72:
            return True
        market24 = _optional_float(row.get("rp_market_mom_24h"))
        market_exit_24 = _optional_float(getattr(cfg, "short_exit_market_mom_24h", None))
        if market_exit_24 is not None and market24 is not None and market24 > market_exit_24:
            return True
        market_gap = _optional_float(row.get("rp_market_ma_gap_96h"))
        market_exit_gap = _optional_float(getattr(cfg, "short_exit_market_ma_gap", None))
        if market_exit_gap is not None and market_gap is not None and market_gap > market_exit_gap:
            return True
    return False


def _passes_score_threshold(row: pd.Series, side: int, cfg: RiskConfig) -> bool:
    threshold = float(getattr(cfg, "min_abs_score_z", 0.0) or 0.0)
    if threshold <= 0:
        return True
    z = _safe_float(row.get("rp_score_z", 0.0), default=0.0)
    return abs(z) >= threshold


def _coerce_utc_timestamp(value: Any) -> Optional[pd.Timestamp]:
    if value in (None, ""):
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _warmup_start_for_rank_signals(start: Optional[str], cfg: RiskConfig) -> Tuple[Optional[str], Dict[str, Any]]:
    start_ts = _coerce_utc_timestamp(start)
    if start_ts is None or _edge_mode(cfg) != "rolling_ic":
        return start, {"enabled": False, "requested_start": str(start) if start else None}

    warmup_hours = int(max(
        0,
        int(getattr(cfg, "edge_lookback_hours", 0) or 0),
        int(getattr(cfg, "edge_min_periods", 0) or 0),
    ))
    if warmup_hours <= 0:
        return start, {"enabled": False, "requested_start": start_ts.isoformat()}

    load_start_ts = start_ts - pd.Timedelta(hours=warmup_hours)
    load_start = load_start_ts.strftime("%Y-%m-%d %H:%M:%S")
    return load_start, {
        "enabled": True,
        "requested_start": start_ts.isoformat(),
        "load_start": load_start_ts.isoformat(),
        "warmup_hours": warmup_hours,
    }


def build_rank_signals(
    score_frame: pd.DataFrame,
    venue_panel: pd.DataFrame,
    cfg: RiskConfig,
    *,
    trading_start: Optional[Any] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    venue = add_risk_columns(venue_panel, timeframe=getattr(cfg, "timeframe", "1h"))
    scores = score_frame.copy()
    scores["date"] = pd.to_datetime(scores["date"], utc=True)
    merged = venue.merge(scores, on=["date", "__pair__"], how="left")
    merged["rp_score"] = merged["rp_score"].replace([np.inf, -np.inf], np.nan)
    merged["rp_score_z"] = merged["rp_score_z"].replace([np.inf, -np.inf], np.nan)
    merged = add_causal_edge_columns(merged, cfg)
    merged["rp_rank"] = 0
    merged["rp_side"] = 0
    merged["rp_target_weight"] = 0.0
    merged["rp_leverage"] = 1.0
    merged["rp_stop_pct"] = merged["rp_atr_pct"].map(lambda x: stop_pct_from_atr(x, cfg))
    merged["rp_liq_distance"] = liquidation_distance(1.0, maintenance_margin=cfg.maintenance_margin, fee_buffer=cfg.fee_buffer)
    merged["rp_liq_reject"] = False
    merged["rp_kill_mode"] = "normal"
    merged["rp_rebalance"] = False
    merged["rp_exit_long"] = False
    merged["rp_exit_short"] = False

    rows: List[pd.DataFrame] = []
    liquidation_rejects = 0
    regime_blocks = 0
    held: Dict[str, Dict[str, float]] = {}
    side_mode = _side_mode(cfg)
    rebalance_bars = max(1, int(getattr(cfg, "rebalance_hours", 1) or 1))
    trading_start_ts = _coerce_utc_timestamp(trading_start)
    trade_date_i = 0
    for _, group in merged.groupby("date", sort=True):
        group_date = pd.Timestamp(group["date"].iloc[0])
        if trading_start_ts is not None and group_date < trading_start_ts:
            continue
        date_i = trade_date_i
        trade_date_i += 1
        g = group.copy()
        valid = (
            g["rp_score"].notna()
            & np.isfinite(g["rp_score"])
            & g["rp_sort_score"].notna()
            & np.isfinite(g["rp_sort_score"])
            & (g["rp_edge_sign"].astype(float) != 0.0)
            & g["close"].notna()
        )
        valid_count = int(valid.sum())
        if valid_count:
            ranks = g.loc[valid, "rp_sort_score"].rank(method="first", ascending=False).astype(int)
            g.loc[valid, "rp_rank"] = ranks
        k = _effective_top_k(valid_count, cfg)
        is_rebalance = (date_i % rebalance_bars) == 0
        if not is_rebalance:
            for idx in g.index:
                pair = str(g.at[idx, "__pair__"])
                prev = held.get(pair)
                if not prev:
                    continue
                held_side = int(prev["side"])
                if _should_exit_held(g.loc[idx], held_side, cfg):
                    if held_side > 0:
                        g.at[idx, "rp_exit_long"] = True
                    elif held_side < 0:
                        g.at[idx, "rp_exit_short"] = True
                    held.pop(pair, None)
                    continue
                g.at[idx, "rp_side"] = int(prev["side"])
                g.at[idx, "rp_target_weight"] = float(prev["target_weight"])
                g.at[idx, "rp_leverage"] = float(prev["leverage"])
                g.at[idx, "rp_stop_pct"] = stop_pct_from_atr(g.at[idx, "rp_atr_pct"], cfg)
                g.at[idx, "rp_liq_distance"] = float(prev["liq_distance"])
            rows.append(g)
            continue
        g["rp_rebalance"] = True
        held = {}
        if k <= 0:
            rows.append(g)
            continue
        if not _regime_allows_entries(g.loc[valid], cfg, side_mode):
            regime_blocks += 1
            g["rp_kill_mode"] = "regime_hq"
            rows.append(g)
            continue
        valid_idx = g.loc[valid].sort_values("rp_sort_score", ascending=False).index.tolist()
        long_idx = valid_idx[:k] if side_mode in {"both", "long"} else []
        short_idx = valid_idx[-k:] if side_mode in {"both", "short"} else []
        sleeve_count = (1 if long_idx else 0) + (1 if short_idx else 0)
        gross_slot = float(cfg.gross_cap) / float(max(1, sleeve_count * k))

        for side, indices in ((1, long_idx), (-1, short_idx)):
            for side_rank, idx in enumerate(indices, start=1):
                if not _passes_score_threshold(g.loc[idx], side, cfg):
                    continue
                if not _passes_entry_filters(g.loc[idx], side, cfg):
                    continue
                stop_pct = stop_pct_from_atr(g.at[idx, "rp_atr_pct"], cfg)
                risk_cap = float(cfg.risk_per_trade) / max(stop_pct, 1e-9)
                target_abs = min(gross_slot, float(cfg.single_pair_cap), risk_cap)
                lev, liq_dist, rejected = choose_dynamic_leverage(
                    side_rank=side_rank,
                    score_z=float(g.at[idx, "rp_score_z"] or 0.0),
                    atr_pct=float(g.at[idx, "rp_atr_pct"] if pd.notna(g.at[idx, "rp_atr_pct"]) else 0.02),
                    volume_ratio=float(g.at[idx, "rp_volume_ratio"] if pd.notna(g.at[idx, "rp_volume_ratio"]) else 1.0),
                    stop_pct=stop_pct,
                    pair_edge_ic=float(g.at[idx, "rp_pair_edge_ic"] if pd.notna(g.at[idx, "rp_pair_edge_ic"]) else 0.0),
                    edge_sign=float(g.at[idx, "rp_edge_sign"] if pd.notna(g.at[idx, "rp_edge_sign"]) else 0.0),
                    cfg=cfg,
                )
                if rejected:
                    liquidation_rejects += 1
                    g.at[idx, "rp_liq_reject"] = True
                    g.at[idx, "rp_liq_distance"] = liq_dist
                    continue
                g.at[idx, "rp_side"] = side
                g.at[idx, "rp_target_weight"] = float(side) * float(target_abs)
                g.at[idx, "rp_leverage"] = float(lev)
                g.at[idx, "rp_stop_pct"] = float(stop_pct)
                g.at[idx, "rp_liq_distance"] = float(liq_dist)

        g = _enforce_exposure_caps(g, cfg)
        for _, row in g.loc[g["rp_target_weight"].abs() > 0].iterrows():
            held[str(row["__pair__"])] = {
                "side": float(row["rp_side"]),
                "target_weight": float(row["rp_target_weight"]),
                "leverage": float(row["rp_leverage"]),
                "liq_distance": float(row["rp_liq_distance"]),
            }
        rows.append(g)

    signals = pd.concat(rows, ignore_index=True).sort_values(["date", "__pair__"]).reset_index(drop=True)
    signals["pair"] = signals["__pair__"]
    cols_first = [
        "date", "pair", "rp_score", "rp_score_z", "rp_rank", "rp_side",
        "rp_target_weight", "rp_leverage", "rp_stop_pct", "rp_kill_mode",
        "rp_rebalance", "rp_edge_ic", "rp_edge_sign", "rp_sort_score", "rp_sort_score_z",
        "rp_pair_edge_ic", "rp_pair_edge_sign",
        "rp_atr_pct", "rp_mom_24h", "rp_mom_72h", "rp_volume_ratio",
        "rp_ma_gap_96h", "rp_market_mom_24h", "rp_market_mom_72h", "rp_market_ma_gap_96h",
        "rp_market_atr_pct", "rp_liq_distance", "rp_liq_reject", "rp_exit_long", "rp_exit_short",
        "open", "high", "low", "close", "volume",
    ]
    keep = [c for c in cols_first if c in signals.columns]
    other = [c for c in signals.columns if c not in keep and not c.startswith("__")]
    signals = signals[keep + other]
    diagnostics = {
        "timeframe": getattr(cfg, "timeframe", "1h"),
        "rebalance_bars": int(rebalance_bars),
        "rebalance_minutes": int(getattr(cfg, "rebalance_minutes", rebalance_bars * 60) or rebalance_bars * 60),
        "rows": int(len(signals)),
        "dates": int(signals["date"].nunique()),
        "pairs": int(signals["pair"].nunique()),
        "liquidation_rejects": int(liquidation_rejects),
        "regime_blocks": int(regime_blocks),
        "avg_gross": float(signals.groupby("date")["rp_target_weight"].apply(lambda s: s.abs().sum()).mean() or 0.0),
        "max_gross": float(signals.groupby("date")["rp_target_weight"].apply(lambda s: s.abs().sum()).max() or 0.0),
        "max_abs_net": float(signals.groupby("date")["rp_target_weight"].sum().abs().max() or 0.0),
    }
    return signals, diagnostics


def _artifact_dir(tag: str) -> Path:
    out = repo_paths.artifacts_root() / "rank_portfolio" / tag
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_signal_files(signals: pd.DataFrame, signal_dir: Path) -> Dict[str, Any]:
    signal_dir.mkdir(parents=True, exist_ok=True)
    all_path = signal_dir / "all.feather"
    signals.reset_index(drop=True).to_feather(all_path)
    per_pair: Dict[str, str] = {}
    for pair, sub in signals.groupby("pair", sort=True):
        token = _pair_file_token(pair)
        path = signal_dir / f"{token}.feather"
        sub.reset_index(drop=True).to_feather(path)
        per_pair[str(pair)] = str(path)
    return {"all": str(all_path), "per_pair": per_pair}


def write_selected_factors(selected: Sequence[SelectedFactor], path: Path, report: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "rank-portfolio-v1",
        "timestamp": time.time(),
        "selected_n": len(selected),
        "selection_report": report,
        "factors": [asdict(f) for f in selected],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def rank_export(
    *,
    tag: str = DEFAULT_TAG,
    n: int = 50,
    risk_profile: str = RISK_PROFILE_AGGRESSIVE,
    venue: str = "okx",
    timeframe: str = "1h",
    data_venue: str = "auto",
    pairs: Optional[Sequence[str] | str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    top_k: Optional[int] = None,
    min_pairs_for_top_k: Optional[int] = None,
    low_pair_top_k: Optional[int] = None,
    gross_cap: Optional[float] = None,
    net_cap: Optional[float] = None,
    single_pair_cap: Optional[float] = None,
    side_mode: Optional[str] = None,
    min_abs_score_z: Optional[float] = None,
    rebalance_hours: Optional[int] = None,
    rebalance_minutes: Optional[int] = None,
    risk_per_trade: Optional[float] = None,
    leverage_cap: Optional[float] = None,
    edge_mode: Optional[str] = None,
    edge_lookback_hours: Optional[int] = None,
    edge_min_periods: Optional[int] = None,
    edge_deadband: Optional[float] = None,
    pair_edge_leverage: Optional[bool] = None,
    pair_edge_deadband: Optional[float] = None,
    pair_edge_strong_ic: Optional[float] = None,
    pair_edge_very_strong_ic: Optional[float] = None,
    pair_edge_weak_cap: Optional[float] = None,
    regime_mode: Optional[str] = None,
    regime_min_edge_ic: Optional[float] = None,
    regime_min_pair_edge_ic: Optional[float] = None,
    regime_min_pair_count: Optional[int] = None,
    regime_short_max_market_mom_24h: Optional[float] = None,
    regime_short_max_market_mom_72h: Optional[float] = None,
    regime_max_market_atr_pct: Optional[float] = None,
    short_max_mom_24h: Optional[float] = None,
    short_max_mom_72h: Optional[float] = None,
    long_min_mom_24h: Optional[float] = None,
    max_entry_atr_pct: Optional[float] = None,
    short_max_market_mom_24h: Optional[float] = None,
    short_max_market_mom_72h: Optional[float] = None,
    short_max_market_ma_gap: Optional[float] = None,
    short_exit_mom_24h: Optional[float] = None,
    short_exit_mom_72h: Optional[float] = None,
    short_exit_market_mom_24h: Optional[float] = None,
    short_exit_market_ma_gap: Optional[float] = None,
    exclude_pairs: Optional[Sequence[str] | str] = None,
    candidate_state: Optional[str | Path] = None,
    recompute_corr: bool = True,
) -> Dict[str, Any]:
    tf = normalize_timeframe(timeframe)
    feature_venue = str(data_venue or "auto").strip().lower()
    if feature_venue == "auto":
        venue_s = str(venue or "okx").strip().lower()
        feature_venue = "kucoin" if tf == "1h" and venue_s == "okx" else venue_s
    candidates, source = load_candidates(tag, candidate_state=candidate_state)
    state_manifest = _candidate_state_manifest(candidate_state)
    if str(data_venue or "auto").strip().lower() == "auto" and state_manifest.get("data_venue"):
        feature_venue = str(state_manifest.get("data_venue") or feature_venue).strip().lower()
    ok, reason = manifest_matches_profile(state_manifest, {"timeframe": tf})
    if not ok:
        raise ValueError(reason)
    selection_cfg = SelectionConfig(n=int(n))
    rank_cache: Dict[str, np.ndarray] = {}
    rank_cache_report: Dict[str, Any] = {"ranked": 0, "skipped": True}
    if recompute_corr:
        rank_cache, rank_cache_report = build_rank_cache_for_selection(
            tag,
            candidates,
            config=selection_cfg,
            candidate_state=candidate_state,
        )
    selected, selection_report = select_factor_records(candidates, config=selection_cfg, rank_cache=rank_cache)
    selection_report["candidate_source"] = source
    selection_report["rank_cache"] = rank_cache_report
    selection_report["timeframe"] = tf
    selection_report["data_venue"] = feature_venue
    selection_report["candidate_state_manifest"] = state_manifest
    pairs, pair_report = _resolve_rank_pair_universe(
        tag=tag,
        candidate_state=candidate_state,
        timeframe=tf,
        feature_venue=feature_venue,
        pairs=pairs,
    )
    selection_report["pair_universe"] = pair_report

    risk_cfg = RiskConfig.from_profile(
        risk_profile,
        gross_cap=gross_cap,
        net_cap=net_cap,
        top_k=top_k,
        min_pairs_for_top_k=min_pairs_for_top_k,
        low_pair_top_k=low_pair_top_k,
        single_pair_cap=single_pair_cap,
        side_mode=side_mode,
        min_abs_score_z=min_abs_score_z,
        rebalance_hours=rebalance_hours,
        rebalance_minutes=rebalance_minutes,
        timeframe=tf,
        risk_per_trade=risk_per_trade,
        leverage_cap=leverage_cap,
        edge_mode=edge_mode,
        edge_lookback_hours=edge_lookback_hours,
        edge_min_periods=edge_min_periods,
        edge_deadband=edge_deadband,
        pair_edge_leverage=pair_edge_leverage,
        pair_edge_deadband=pair_edge_deadband,
        pair_edge_strong_ic=pair_edge_strong_ic,
        pair_edge_very_strong_ic=pair_edge_very_strong_ic,
        pair_edge_weak_cap=pair_edge_weak_cap,
        regime_mode=regime_mode,
        regime_min_edge_ic=regime_min_edge_ic,
        regime_min_pair_edge_ic=regime_min_pair_edge_ic,
        regime_min_pair_count=regime_min_pair_count,
        regime_short_max_market_mom_24h=regime_short_max_market_mom_24h,
        regime_short_max_market_mom_72h=regime_short_max_market_mom_72h,
        regime_max_market_atr_pct=regime_max_market_atr_pct,
        short_max_mom_24h=short_max_mom_24h,
        short_max_mom_72h=short_max_mom_72h,
        long_min_mom_24h=long_min_mom_24h,
        max_entry_atr_pct=max_entry_atr_pct,
        short_max_market_mom_24h=short_max_market_mom_24h,
        short_max_market_mom_72h=short_max_market_mom_72h,
        short_max_market_ma_gap=short_max_market_ma_gap,
        short_exit_mom_24h=short_exit_mom_24h,
        short_exit_mom_72h=short_exit_mom_72h,
        short_exit_market_mom_24h=short_exit_market_mom_24h,
        short_exit_market_ma_gap=short_exit_market_ma_gap,
        exclude_pairs=exclude_pairs,
    )
    load_start, warmup_report = _warmup_start_for_rank_signals(start, risk_cfg)
    feature_panel = load_feature_panel(pairs=pairs, timeframe=tf, data_venue=feature_venue, start=load_start, end=end)
    venue_panel = load_venue_ohlcv(venue=venue, timeframe=tf, pairs=pairs, start=load_start, end=end)
    scores, score_report = compute_ensemble_scores(feature_panel, selected)
    if int(score_report.get("used_factor_count", 0) or 0) <= 0:
        raise ValueError(f"rank ensemble could not evaluate any selected factors: {score_report.get('errors', [])[:5]}")
    signals, signal_report = build_rank_signals(scores, venue_panel, risk_cfg, trading_start=start)

    out_dir = _artifact_dir(tag)
    selected_path = out_dir / "selected_factors.json"
    signal_dir = out_dir / "signals"
    write_selected_factors(selected, selected_path, selection_report)
    signal_paths = write_signal_files(signals, signal_dir)
    summary = {
        "tag": tag,
        "risk_profile": risk_profile,
        "venue": venue,
        "timeframe": tf,
        "data_venue": feature_venue,
        "pair_count": len(pairs),
        "pair_universe": pair_report,
        "candidate_state_manifest": state_manifest,
        "risk_config": asdict(risk_cfg),
        "selected_factors": str(selected_path),
        "signals": signal_paths,
        "selection": selection_report,
        "scores": score_report,
        "signal_report": signal_report,
        "signal_warmup": warmup_report,
    }
    (out_dir / "rank_export.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    high = equity.cummax()
    dd = (high - equity) / high.replace(0.0, np.nan)
    return float(dd.max() or 0.0)


def _load_funding_panel(pairs: Iterable[str]) -> Dict[Tuple[str, pd.Timestamp], float]:
    """Load per-8h funding rates for known pairs.

    Returns dict keyed by (normalised_pair, utc_timestamp).
    Pairs whose feather is absent are silently skipped; callers use DEFAULT_FUNDING_8H as fallback.
    """
    panel: Dict[Tuple[str, pd.Timestamp], float] = {}
    for raw_pair in pairs:
        sym = raw_pair.split("/")[0].upper()
        pf = FUNDING_DIR / f"{sym}_USDT-funding.feather"
        if not pf.exists():
            continue
        try:
            df = pd.read_feather(pf)
            df["date"] = pd.to_datetime(df["date"], utc=True)
            norm = raw_pair.upper().replace(" ", "")
            for ts, rate in zip(df["date"], df["funding_rate"]):
                panel[(norm, ts)] = float(rate)
        except Exception:
            pass
    return panel


DEFAULT_FUNDING_8H: float = DEFAULT_FUNDING_DAILY / 3  # per-8h fallback when feather absent


def run_research_backtest(signals: pd.DataFrame, cfg: RiskConfig) -> Dict[str, Any]:
    df = signals.copy().sort_values(["pair", "date"]).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    pieces: List[pd.DataFrame] = []
    for _, sub in df.groupby("pair", sort=False):
        sub = sub.copy()
        sub["next_open"] = sub["open"].shift(-1)
        sub["next_close"] = sub["close"].shift(-1)
        sub["next_high"] = sub["high"].shift(-1)
        sub["next_low"] = sub["low"].shift(-1)
        # Full-bar return: entry at open[T+1], exit at open[T+2] (next-open-to-next-next-open).
        # This captures overnight gaps while preserving per-bar early-exit semantics.
        sub["next_next_open"] = sub["open"].shift(-2)
        sub["ret_next"] = (sub["next_next_open"] / sub["next_open"].clip(lower=1e-12)) - 1.0
        pieces.append(sub)
    df = pd.concat(pieces, ignore_index=True)

    # Load funding rates (8h cadence). Missing pairs fall back to DEFAULT_FUNDING_8H.
    unique_pairs = list(df["pair"].unique())
    funding_panel = _load_funding_panel(unique_pairs)
    total_funding_paid: float = 0.0

    controller = AccountRiskController(cfg)
    prev_weights: Dict[str, float] = {}
    equity = 1.0
    rows: List[Dict[str, Any]] = []
    simulated_liquidations = 0
    liquidation_terminated_at: Any = None
    trades = 0
    for date, group in df.groupby("date", sort=True):
        if liquidation_terminated_at is not None:
            break
        status = controller.update(date, equity)
        g = group.copy()
        if not status.allow_new_entries:
            g["rp_target_weight"] = 0.0
            g["rp_leverage"] = 1.0
            g["rp_kill_mode"] = status.mode
        elif status.gross_cap is not None:
            gross = float(g["rp_target_weight"].abs().sum())
            if gross > float(status.gross_cap) and gross > 0:
                g["rp_target_weight"] = g["rp_target_weight"] * (float(status.gross_cap) / gross)
            g["rp_leverage"] = np.minimum(g["rp_leverage"].astype(float), float(status.leverage_cap))
            g["rp_kill_mode"] = status.mode

        pnl = 0.0
        weights_now: Dict[str, float] = {}
        for _, row in g.iterrows():
            pair = str(row["pair"])
            weight = float(row.get("rp_target_weight", 0.0) or 0.0)
            weights_now[pair] = weight
            if abs(weight) <= 0 or not np.isfinite(row.get("ret_next", np.nan)):
                continue
            side = 1.0 if weight > 0 else -1.0
            stop = float(row.get("rp_stop_pct", 0.02) or 0.02)
            # Use next_open as fill/reference price (avoids look-ahead vs current close)
            entry = float(row.get("next_open") or row["close"])
            if entry <= 0:
                entry = float(row["close"])
            if side > 0:
                adverse = (entry - float(row.get("next_low", entry))) / max(entry, 1e-12)
                side_ret = float(row["ret_next"])
            else:
                adverse = (float(row.get("next_high", entry)) - entry) / max(entry, 1e-12)
                side_ret = -float(row["ret_next"])
            if adverse >= float(row.get("rp_liq_distance", 999.0) or 999.0):
                simulated_liquidations += 1
                side_ret = -float(row.get("rp_liq_distance", stop))
                liquidation_terminated_at = date
            elif adverse >= stop:
                side_ret = -stop
            pnl += abs(weight) * side_ret

        # Funding cost at 8h settlement periods (00:00 / 08:00 / 16:00 UTC)
        if hasattr(date, "hour") and date.hour in (0, 8, 16):
            for pair, w in prev_weights.items():
                if abs(w) < 1e-9:
                    continue
                norm = pair.upper().replace(" ", "")
                fr = funding_panel.get((norm, date), DEFAULT_FUNDING_8H)
                # Positive funding: longs pay, shorts receive (w encodes sign)
                funding_adj = -w * fr
                pnl += funding_adj
                total_funding_paid -= funding_adj  # track total paid by portfolio

        all_pairs = set(prev_weights) | set(weights_now)
        turnover = sum(abs(weights_now.get(pair, 0.0) - prev_weights.get(pair, 0.0)) for pair in all_pairs)
        entry_count = sum(1 for pair in all_pairs if abs(prev_weights.get(pair, 0.0)) <= 1e-12 and abs(weights_now.get(pair, 0.0)) > 1e-12)
        trades += int(entry_count)
        cost = turnover * (float(cfg.fee_rate) + float(cfg.slippage))
        net_ret = pnl - cost
        equity *= max(0.0, 1.0 + net_ret)
        # `consecutive_losses` is trade-level risk. Hourly portfolio marks are
        # not closed trades, so do not feed them into that kill switch here.
        controller.update(date, equity)
        prev_weights = weights_now
        rows.append({
            "date": date,
            "equity": equity,
            "return": net_ret,
            "gross": sum(abs(v) for v in weights_now.values()),
            "net": sum(weights_now.values()),
            "turnover": turnover,
            "risk_mode": status.mode,
        })

    curve = pd.DataFrame(rows)
    equity_series = curve["equity"] if not curve.empty else pd.Series(dtype="float64")
    leverage_dist = (
        signals.loc[signals["rp_target_weight"].abs() > 0, "rp_leverage"]
        .round(2)
        .value_counts()
        .sort_index()
        .to_dict()
    )
    risk_mode_counts = curve["risk_mode"].value_counts().to_dict() if not curve.empty else {}
    total_return = float(equity - 1.0)
    max_dd = _max_drawdown(equity_series)
    return {
        "start_equity": 1.0,
        "end_equity": float(equity),
        "total_return": total_return,
        "total_return_pct": total_return * 100.0,
        "max_drawdown": max_dd,
        "max_drawdown_pct": max_dd * 100.0,
        "profit_over_max_drawdown": float(total_return / max(max_dd, 1e-12)),
        "trades": int(trades),
        "simulated_liquidations": int(simulated_liquidations),
        "liquidation_terminated_at": str(liquidation_terminated_at) if liquidation_terminated_at is not None else None,
        "liquidation_rejects": int(signals.get("rp_liq_reject", pd.Series(dtype=bool)).sum()),
        "leverage_distribution": {str(k): int(v) for k, v in leverage_dist.items()},
        "risk_mode_counts": {str(k): int(v) for k, v in risk_mode_counts.items()},
        "avg_gross": float(curve["gross"].mean() if not curve.empty else 0.0),
        "max_gross": float(curve["gross"].max() if not curve.empty else 0.0),
        "avg_turnover": float(curve["turnover"].mean() if not curve.empty else 0.0),
        "total_funding_cost": float(total_funding_paid),
        "periods": int(len(curve)),
        "curve": (
            [
                {"date": str(row["date"]), "equity": float(row["equity"])}
                for _, row in curve[["date", "equity"]].iterrows()
            ]
            if not curve.empty
            else []
        ),
    }


def rank_backtest(
    *,
    tag: str = DEFAULT_TAG,
    venue: str = "okx",
    timeframe: str = "1h",
    data_venue: str = "auto",
    pairs: Optional[Sequence[str] | str] = None,
    top_k: int = 2,
    min_pairs_for_top_k: Optional[int] = None,
    low_pair_top_k: Optional[int] = None,
    gross_cap: float = 2.0,
    net_cap: Optional[float] = None,
    single_pair_cap: Optional[float] = None,
    risk_profile: str = RISK_PROFILE_AGGRESSIVE,
    n: int = 50,
    start: str = "2025-12-01",
    end: str = "2026-04-12",
    side_mode: Optional[str] = None,
    min_abs_score_z: Optional[float] = None,
    rebalance_hours: Optional[int] = None,
    rebalance_minutes: Optional[int] = None,
    risk_per_trade: Optional[float] = None,
    leverage_cap: Optional[float] = None,
    edge_mode: Optional[str] = None,
    edge_lookback_hours: Optional[int] = None,
    edge_min_periods: Optional[int] = None,
    edge_deadband: Optional[float] = None,
    pair_edge_leverage: Optional[bool] = None,
    pair_edge_deadband: Optional[float] = None,
    pair_edge_strong_ic: Optional[float] = None,
    pair_edge_very_strong_ic: Optional[float] = None,
    pair_edge_weak_cap: Optional[float] = None,
    regime_mode: Optional[str] = None,
    regime_min_edge_ic: Optional[float] = None,
    regime_min_pair_edge_ic: Optional[float] = None,
    regime_min_pair_count: Optional[int] = None,
    regime_short_max_market_mom_24h: Optional[float] = None,
    regime_short_max_market_mom_72h: Optional[float] = None,
    regime_max_market_atr_pct: Optional[float] = None,
    short_max_mom_24h: Optional[float] = None,
    short_max_mom_72h: Optional[float] = None,
    long_min_mom_24h: Optional[float] = None,
    max_entry_atr_pct: Optional[float] = None,
    short_max_market_mom_24h: Optional[float] = None,
    short_max_market_mom_72h: Optional[float] = None,
    short_max_market_ma_gap: Optional[float] = None,
    short_exit_mom_24h: Optional[float] = None,
    short_exit_mom_72h: Optional[float] = None,
    short_exit_market_mom_24h: Optional[float] = None,
    short_exit_market_ma_gap: Optional[float] = None,
    exclude_pairs: Optional[Sequence[str] | str] = None,
    candidate_state: Optional[str | Path] = None,
    recompute_corr: bool = True,
) -> Dict[str, Any]:
    tf = normalize_timeframe(timeframe)
    export_summary = rank_export(
        tag=tag,
        n=n,
        risk_profile=risk_profile,
        venue=venue,
        timeframe=tf,
        data_venue=data_venue,
        pairs=pairs,
        start=start,
        end=end,
        top_k=top_k,
        min_pairs_for_top_k=min_pairs_for_top_k,
        low_pair_top_k=low_pair_top_k,
        gross_cap=gross_cap,
        net_cap=net_cap,
        single_pair_cap=single_pair_cap,
        side_mode=side_mode,
        min_abs_score_z=min_abs_score_z,
        rebalance_hours=rebalance_hours,
        rebalance_minutes=rebalance_minutes,
        risk_per_trade=risk_per_trade,
        leverage_cap=leverage_cap,
        edge_mode=edge_mode,
        edge_lookback_hours=edge_lookback_hours,
        edge_min_periods=edge_min_periods,
        edge_deadband=edge_deadband,
        pair_edge_leverage=pair_edge_leverage,
        pair_edge_deadband=pair_edge_deadband,
        pair_edge_strong_ic=pair_edge_strong_ic,
        pair_edge_very_strong_ic=pair_edge_very_strong_ic,
        pair_edge_weak_cap=pair_edge_weak_cap,
        regime_mode=regime_mode,
        regime_min_edge_ic=regime_min_edge_ic,
        regime_min_pair_edge_ic=regime_min_pair_edge_ic,
        regime_min_pair_count=regime_min_pair_count,
        regime_short_max_market_mom_24h=regime_short_max_market_mom_24h,
        regime_short_max_market_mom_72h=regime_short_max_market_mom_72h,
        regime_max_market_atr_pct=regime_max_market_atr_pct,
        short_max_mom_24h=short_max_mom_24h,
        short_max_mom_72h=short_max_mom_72h,
        long_min_mom_24h=long_min_mom_24h,
        max_entry_atr_pct=max_entry_atr_pct,
        short_max_market_mom_24h=short_max_market_mom_24h,
        short_max_market_mom_72h=short_max_market_mom_72h,
        short_max_market_ma_gap=short_max_market_ma_gap,
        short_exit_mom_24h=short_exit_mom_24h,
        short_exit_mom_72h=short_exit_mom_72h,
        short_exit_market_mom_24h=short_exit_market_mom_24h,
        short_exit_market_ma_gap=short_exit_market_ma_gap,
        exclude_pairs=exclude_pairs,
        candidate_state=candidate_state,
        recompute_corr=recompute_corr,
    )
    signals_path = Path(export_summary["signals"]["all"])
    signals = pd.read_feather(signals_path)
    risk_cfg = RiskConfig.from_profile(
        risk_profile,
        gross_cap=gross_cap,
        net_cap=net_cap,
        top_k=top_k,
        single_pair_cap=single_pair_cap,
        side_mode=side_mode,
        min_abs_score_z=min_abs_score_z,
        rebalance_hours=rebalance_hours,
        rebalance_minutes=rebalance_minutes,
        timeframe=tf,
        risk_per_trade=risk_per_trade,
        leverage_cap=leverage_cap,
        edge_mode=edge_mode,
        edge_lookback_hours=edge_lookback_hours,
        edge_min_periods=edge_min_periods,
        edge_deadband=edge_deadband,
        pair_edge_leverage=pair_edge_leverage,
        pair_edge_deadband=pair_edge_deadband,
        pair_edge_strong_ic=pair_edge_strong_ic,
        pair_edge_very_strong_ic=pair_edge_very_strong_ic,
        pair_edge_weak_cap=pair_edge_weak_cap,
        regime_mode=regime_mode,
        regime_min_edge_ic=regime_min_edge_ic,
        regime_min_pair_edge_ic=regime_min_pair_edge_ic,
        regime_min_pair_count=regime_min_pair_count,
        regime_short_max_market_mom_24h=regime_short_max_market_mom_24h,
        regime_short_max_market_mom_72h=regime_short_max_market_mom_72h,
        regime_max_market_atr_pct=regime_max_market_atr_pct,
        short_max_mom_24h=short_max_mom_24h,
        short_max_mom_72h=short_max_mom_72h,
        long_min_mom_24h=long_min_mom_24h,
        max_entry_atr_pct=max_entry_atr_pct,
        short_max_market_mom_24h=short_max_market_mom_24h,
        short_max_market_mom_72h=short_max_market_mom_72h,
        short_max_market_ma_gap=short_max_market_ma_gap,
        short_exit_mom_24h=short_exit_mom_24h,
        short_exit_mom_72h=short_exit_mom_72h,
        short_exit_market_mom_24h=short_exit_market_mom_24h,
        short_exit_market_ma_gap=short_exit_market_ma_gap,
        exclude_pairs=exclude_pairs,
    )
    result = run_research_backtest(signals, risk_cfg)
    result.update({
        "tag": tag,
        "venue": venue,
        "timeframe": tf,
        "data_venue": export_summary.get("data_venue"),
        "pair_count": export_summary.get("pair_count"),
        "pair_universe": export_summary.get("pair_universe"),
        "top_k": int(top_k),
        "gross_cap": float(gross_cap),
        "net_cap": float(risk_cfg.net_cap),
        "single_pair_cap": float(risk_cfg.single_pair_cap),
        "side_mode": risk_cfg.side_mode,
        "min_abs_score_z": float(risk_cfg.min_abs_score_z),
        "rebalance_hours": int(risk_cfg.rebalance_hours),
        "rebalance_minutes": int(risk_cfg.rebalance_minutes),
        "risk_per_trade": float(risk_cfg.risk_per_trade),
        "leverage_cap": float(risk_cfg.leverage_cap),
        "edge_mode": risk_cfg.edge_mode,
        "edge_lookback_hours": int(risk_cfg.edge_lookback_hours),
        "edge_min_periods": int(risk_cfg.edge_min_periods),
        "edge_deadband": float(risk_cfg.edge_deadband),
        "pair_edge_leverage": bool(risk_cfg.pair_edge_leverage),
        "pair_edge_deadband": float(risk_cfg.pair_edge_deadband),
        "pair_edge_strong_ic": float(risk_cfg.pair_edge_strong_ic),
        "pair_edge_very_strong_ic": float(risk_cfg.pair_edge_very_strong_ic),
        "pair_edge_weak_cap": float(risk_cfg.pair_edge_weak_cap),
        "regime_mode": risk_cfg.regime_mode,
        "regime_min_edge_ic": float(risk_cfg.regime_min_edge_ic),
        "regime_min_pair_edge_ic": float(risk_cfg.regime_min_pair_edge_ic),
        "regime_min_pair_count": int(risk_cfg.regime_min_pair_count),
        "regime_short_max_market_mom_24h": risk_cfg.regime_short_max_market_mom_24h,
        "regime_short_max_market_mom_72h": risk_cfg.regime_short_max_market_mom_72h,
        "regime_max_market_atr_pct": risk_cfg.regime_max_market_atr_pct,
        "short_max_mom_24h": risk_cfg.short_max_mom_24h,
        "short_max_mom_72h": risk_cfg.short_max_mom_72h,
        "long_min_mom_24h": risk_cfg.long_min_mom_24h,
        "max_entry_atr_pct": risk_cfg.max_entry_atr_pct,
        "short_max_market_mom_24h": risk_cfg.short_max_market_mom_24h,
        "short_max_market_mom_72h": risk_cfg.short_max_market_mom_72h,
        "short_max_market_ma_gap": risk_cfg.short_max_market_ma_gap,
        "short_exit_mom_24h": risk_cfg.short_exit_mom_24h,
        "short_exit_mom_72h": risk_cfg.short_exit_mom_72h,
        "short_exit_market_mom_24h": risk_cfg.short_exit_market_mom_24h,
        "short_exit_market_ma_gap": risk_cfg.short_exit_market_ma_gap,
        "exclude_pairs": list(risk_cfg.exclude_pairs),
        "start": start,
        "end": end,
        "signals": str(signals_path),
        "selected_factors": export_summary["selected_factors"],
        "candidate_source": export_summary.get("selection", {}).get("candidate_source"),
    })
    out = _artifact_dir(tag) / "backtest.json"
    out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result


def rank_sweep(
    *,
    tag: str = DEFAULT_TAG,
    venue: str = "okx",
    timeframe: str = "1h",
    data_venue: str = "auto",
    pairs: Optional[Sequence[str] | str] = None,
    risk_profile: str = RISK_PROFILE_AGGRESSIVE,
    n: int = 50,
    start: str = "2025-12-01",
    end: str = "2026-04-12",
    gross_caps: Sequence[float] = (1.0, 2.0, 3.0),
    top_ks: Sequence[int] = (1, 2, 3),
    side_modes: Sequence[str] = ("short",),
    score_thresholds: Sequence[float] = (1.5, 2.0),
    rebalance_hours_values: Sequence[int] = (8, 12, 24),
    rebalance_minutes_values: Optional[Sequence[int]] = None,
    risk_per_trade: Optional[float] = 0.08,
    leverage_cap: Optional[float] = None,
    single_pair_cap: Optional[float] = None,
    net_cap: Optional[float] = None,
    edge_mode: Optional[str] = None,
    edge_lookback_hours: Optional[int] = None,
    edge_min_periods: Optional[int] = None,
    edge_deadband: Optional[float] = None,
    pair_edge_leverage: Optional[bool] = None,
    pair_edge_deadband: Optional[float] = None,
    pair_edge_strong_ic: Optional[float] = None,
    pair_edge_very_strong_ic: Optional[float] = None,
    pair_edge_weak_cap: Optional[float] = None,
    regime_mode: Optional[str] = None,
    regime_min_edge_ic: Optional[float] = None,
    regime_min_pair_edge_ic: Optional[float] = None,
    regime_min_pair_count: Optional[int] = None,
    regime_short_max_market_mom_24h: Optional[float] = None,
    regime_short_max_market_mom_72h: Optional[float] = None,
    regime_max_market_atr_pct: Optional[float] = None,
    short_max_mom_24h: Optional[float] = None,
    short_max_mom_72h: Optional[float] = None,
    long_min_mom_24h: Optional[float] = None,
    max_entry_atr_pct: Optional[float] = None,
    short_max_market_mom_24h: Optional[float] = None,
    short_max_market_mom_72h: Optional[float] = None,
    short_max_market_ma_gap: Optional[float] = None,
    short_exit_mom_24h: Optional[float] = None,
    short_exit_mom_72h: Optional[float] = None,
    short_exit_market_mom_24h: Optional[float] = None,
    short_exit_market_ma_gap: Optional[float] = None,
    exclude_pairs: Optional[Sequence[str] | str] = None,
    candidate_state: Optional[str | Path] = None,
    recompute_corr: bool = True,
) -> Dict[str, Any]:
    tf = normalize_timeframe(timeframe)
    feature_venue = str(data_venue or "auto").strip().lower()
    if feature_venue == "auto":
        venue_s = str(venue or "okx").strip().lower()
        feature_venue = "kucoin" if tf == "1h" and venue_s == "okx" else venue_s
    candidates, source = load_candidates(tag, candidate_state=candidate_state)
    state_manifest = _candidate_state_manifest(candidate_state)
    if str(data_venue or "auto").strip().lower() == "auto" and state_manifest.get("data_venue"):
        feature_venue = str(state_manifest.get("data_venue") or feature_venue).strip().lower()
    ok, reason = manifest_matches_profile(state_manifest, {"timeframe": tf})
    if not ok:
        raise ValueError(reason)
    selection_cfg = SelectionConfig(n=int(n))
    rank_cache: Dict[str, np.ndarray] = {}
    rank_cache_report: Dict[str, Any] = {"ranked": 0, "skipped": True}
    if recompute_corr:
        rank_cache, rank_cache_report = build_rank_cache_for_selection(
            tag,
            candidates,
            config=selection_cfg,
            candidate_state=candidate_state,
        )
    selected, selection_report = select_factor_records(candidates, config=selection_cfg, rank_cache=rank_cache)
    selection_report["candidate_source"] = source
    selection_report["rank_cache"] = rank_cache_report
    selection_report["timeframe"] = tf
    selection_report["data_venue"] = feature_venue
    selection_report["candidate_state_manifest"] = state_manifest
    pairs, pair_report = _resolve_rank_pair_universe(
        tag=tag,
        candidate_state=candidate_state,
        timeframe=tf,
        feature_venue=feature_venue,
        pairs=pairs,
    )
    selection_report["pair_universe"] = pair_report

    warmup_cfg = RiskConfig.from_profile(
        risk_profile,
        timeframe=tf,
        edge_mode=edge_mode,
        edge_lookback_hours=edge_lookback_hours,
        edge_min_periods=edge_min_periods,
    )
    load_start, warmup_report = _warmup_start_for_rank_signals(start, warmup_cfg)
    feature_panel = load_feature_panel(pairs=pairs, timeframe=tf, data_venue=feature_venue, start=load_start, end=end)
    venue_panel = load_venue_ohlcv(venue=venue, timeframe=tf, pairs=pairs, start=load_start, end=end)
    scores, score_report = compute_ensemble_scores(feature_panel, selected)
    if int(score_report.get("used_factor_count", 0) or 0) <= 0:
        raise ValueError(f"rank ensemble could not evaluate any selected factors: {score_report.get('errors', [])[:5]}")

    out_dir = _artifact_dir(tag)
    selected_path = out_dir / "selected_factors.json"
    write_selected_factors(selected, selected_path, selection_report)

    rows: List[Dict[str, Any]] = []
    for gross_cap in gross_caps:
        for top_k in top_ks:
            for side_mode in side_modes:
                for threshold in score_thresholds:
                    minutes_iter = list(rebalance_minutes_values or [])
                    if not minutes_iter:
                        minutes_iter = [int(v) * 60 for v in rebalance_hours_values]
                    for rebalance_minutes in minutes_iter:
                        risk_cfg = RiskConfig.from_profile(
                            risk_profile,
                            gross_cap=float(gross_cap),
                            net_cap=net_cap,
                            top_k=int(top_k),
                            single_pair_cap=single_pair_cap,
                            side_mode=side_mode,
                            min_abs_score_z=float(threshold),
                            rebalance_minutes=int(rebalance_minutes),
                            timeframe=tf,
                            risk_per_trade=risk_per_trade,
                            leverage_cap=leverage_cap,
                            edge_mode=edge_mode,
                            edge_lookback_hours=edge_lookback_hours,
                            edge_min_periods=edge_min_periods,
                            edge_deadband=edge_deadband,
                            pair_edge_leverage=pair_edge_leverage,
                            pair_edge_deadband=pair_edge_deadband,
                            pair_edge_strong_ic=pair_edge_strong_ic,
                            pair_edge_very_strong_ic=pair_edge_very_strong_ic,
                            pair_edge_weak_cap=pair_edge_weak_cap,
                            regime_mode=regime_mode,
                            regime_min_edge_ic=regime_min_edge_ic,
                            regime_min_pair_edge_ic=regime_min_pair_edge_ic,
                            regime_min_pair_count=regime_min_pair_count,
                            regime_short_max_market_mom_24h=regime_short_max_market_mom_24h,
                            regime_short_max_market_mom_72h=regime_short_max_market_mom_72h,
                            regime_max_market_atr_pct=regime_max_market_atr_pct,
                            short_max_mom_24h=short_max_mom_24h,
                            short_max_mom_72h=short_max_mom_72h,
                            long_min_mom_24h=long_min_mom_24h,
                            max_entry_atr_pct=max_entry_atr_pct,
                            short_max_market_mom_24h=short_max_market_mom_24h,
                            short_max_market_mom_72h=short_max_market_mom_72h,
                            short_max_market_ma_gap=short_max_market_ma_gap,
                            short_exit_mom_24h=short_exit_mom_24h,
                            short_exit_mom_72h=short_exit_mom_72h,
                            short_exit_market_mom_24h=short_exit_market_mom_24h,
                            short_exit_market_ma_gap=short_exit_market_ma_gap,
                            exclude_pairs=exclude_pairs,
                        )
                        signals, signal_report = build_rank_signals(scores, venue_panel, risk_cfg, trading_start=start)
                        result = run_research_backtest(signals, risk_cfg)
                        result.update({
                            "tag": tag,
                            "venue": venue,
                            "timeframe": tf,
                            "data_venue": feature_venue,
                            "top_k": int(top_k),
                            "gross_cap": float(gross_cap),
                            "net_cap": float(risk_cfg.net_cap),
                            "single_pair_cap": float(risk_cfg.single_pair_cap),
                            "side_mode": risk_cfg.side_mode,
                            "min_abs_score_z": float(risk_cfg.min_abs_score_z),
                            "rebalance_hours": int(risk_cfg.rebalance_hours),
                            "rebalance_minutes": int(risk_cfg.rebalance_minutes),
                            "risk_per_trade": float(risk_cfg.risk_per_trade),
                            "leverage_cap": float(risk_cfg.leverage_cap),
                            "edge_mode": risk_cfg.edge_mode,
                            "edge_lookback_hours": int(risk_cfg.edge_lookback_hours),
                            "edge_min_periods": int(risk_cfg.edge_min_periods),
                            "edge_deadband": float(risk_cfg.edge_deadband),
                            "pair_edge_leverage": bool(risk_cfg.pair_edge_leverage),
                            "pair_edge_deadband": float(risk_cfg.pair_edge_deadband),
                            "pair_edge_strong_ic": float(risk_cfg.pair_edge_strong_ic),
                            "pair_edge_very_strong_ic": float(risk_cfg.pair_edge_very_strong_ic),
                            "pair_edge_weak_cap": float(risk_cfg.pair_edge_weak_cap),
                            "regime_mode": risk_cfg.regime_mode,
                            "regime_min_edge_ic": float(risk_cfg.regime_min_edge_ic),
                            "regime_min_pair_edge_ic": float(risk_cfg.regime_min_pair_edge_ic),
                            "regime_min_pair_count": int(risk_cfg.regime_min_pair_count),
                            "regime_short_max_market_mom_24h": risk_cfg.regime_short_max_market_mom_24h,
                            "regime_short_max_market_mom_72h": risk_cfg.regime_short_max_market_mom_72h,
                            "regime_max_market_atr_pct": risk_cfg.regime_max_market_atr_pct,
                            "short_max_mom_24h": risk_cfg.short_max_mom_24h,
                            "short_max_mom_72h": risk_cfg.short_max_mom_72h,
                            "long_min_mom_24h": risk_cfg.long_min_mom_24h,
                            "max_entry_atr_pct": risk_cfg.max_entry_atr_pct,
                            "short_max_market_mom_24h": risk_cfg.short_max_market_mom_24h,
                            "short_max_market_mom_72h": risk_cfg.short_max_market_mom_72h,
                            "short_max_market_ma_gap": risk_cfg.short_max_market_ma_gap,
                            "short_exit_mom_24h": risk_cfg.short_exit_mom_24h,
                            "short_exit_mom_72h": risk_cfg.short_exit_mom_72h,
                            "short_exit_market_mom_24h": risk_cfg.short_exit_market_mom_24h,
                            "short_exit_market_ma_gap": risk_cfg.short_exit_market_ma_gap,
                            "exclude_pairs": list(risk_cfg.exclude_pairs),
                            "start": start,
                            "end": end,
                            "selected_factors": str(selected_path),
                            "candidate_source": source,
                            "signal_report": signal_report,
                            "signal_warmup": warmup_report,
                        })
                        rows.append(result)
    summary = {
        "tag": tag,
        "venue": venue,
        "timeframe": tf,
        "data_venue": feature_venue,
        "pair_count": len(pairs),
        "pair_universe": pair_report,
        "candidate_state_manifest": state_manifest,
        "start": start,
        "end": end,
        "selected_factors": str(selected_path),
        "selection": selection_report,
        "scores": score_report,
        "signal_warmup": warmup_report,
        "results": rows,
        "best_by_profit_over_dd": max(rows, key=lambda r: r.get("profit_over_max_drawdown", -1e9)) if rows else None,
    }
    out = out_dir / "sweep.json"
    out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary

"""Agent-driven factor strategy loop orchestration.

The loop keeps the candidate-writing agent confined to an iteration workspace.
Expensive and stateful actions such as signal export, research backtest,
scoring, and promotion stay in this Python controller so candidate artifacts
are auditable and resumable.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from agent_market import paths as repo_paths
from agent_market.backtest_results import build_backtest_summary
from agent_market.factor_lab import lean_bridge, rank_portfolio
from agent_market.factor_lab.lean_analysis import compute_lean_analysis
from agent_market.factor_lab.timeframes import manifest_matches_profile, normalize_lane, normalize_timeframe


PHASE_PREPARE = "PREPARE"
PHASE_CODE_GEN = "CODE_GEN"
PHASE_SIGNAL_EXPORT = "SIGNAL_EXPORT"
PHASE_BACKTEST = "BACKTEST"
PHASE_LEAN_ANALYSIS = "LEAN_ANALYSIS"
PHASE_EVALUATION = "EVALUATION"
PHASE_ANALYSIS = "ANALYSIS"
PHASE_COMPLETE = "COMPLETE"

PHASES = (
    PHASE_PREPARE,
    PHASE_CODE_GEN,
    PHASE_SIGNAL_EXPORT,
    PHASE_BACKTEST,
    PHASE_LEAN_ANALYSIS,
    PHASE_EVALUATION,
    PHASE_ANALYSIS,
    PHASE_COMPLETE,
)

CANDIDATE_RANK_PROFILE = "rank_profile"
CANDIDATE_FREQTRADE_STRATEGY = "freqtrade_strategy"
CANDIDATE_TYPES = {CANDIDATE_RANK_PROFILE, CANDIDATE_FREQTRADE_STRATEGY}

AGENT_HERMES = "hermes"
AGENT_OPENAI = "openai"
AGENT_OPENCODE = "opencode"
AGENT_TYPES = {AGENT_HERMES, AGENT_OPENAI, AGENT_OPENCODE}
HERMES_REASONING_EFFORTS = {"", "none", "minimal", "low", "medium", "high", "xhigh"}

EVAL_RESEARCH = "research"
EVAL_TWO_STAGE = "two_stage"
EVAL_FREQTRADE = "freqtrade"
EVAL_MODES = {EVAL_RESEARCH, EVAL_TWO_STAGE, EVAL_FREQTRADE}

SCORE_RESEARCH = "research"
SCORE_FREQTRADE = "freqtrade"
SCORE_COMPOSITE = "composite"
SCORE_MODES = {SCORE_RESEARCH, SCORE_FREQTRADE, SCORE_COMPOSITE}

PROMOTE_IMMEDIATE = "immediate"
PROMOTE_FINAL = "final"
PROMOTE_NONE = "none"
PROMOTE_POLICIES = {PROMOTE_IMMEDIATE, PROMOTE_FINAL, PROMOTE_NONE}

VALIDATION_SINGLE = "single"
VALIDATION_TRIPLE_HOLDOUT = "triple_holdout"
VALIDATION_WALKFORWARD = "walkforward"
VALIDATION_PROTOCOLS = {VALIDATION_SINGLE, VALIDATION_TRIPLE_HOLDOUT, VALIDATION_WALKFORWARD}

VERIFY_PARETO = "pareto"
VERIFY_BEST = "best"
VERIFY_ALL = "all"
VERIFY_NONE = "none"
VERIFY_POLICIES = {VERIFY_PARETO, VERIFY_BEST, VERIFY_ALL, VERIFY_NONE}

LEAN_GATE_OFF = "off"
LEAN_GATE_FINAL = "final"
LEAN_GATE_PARETO = "pareto"
LEAN_GATE_ALL = "all"
LEAN_GATE_MODES = {LEAN_GATE_OFF, LEAN_GATE_FINAL, LEAN_GATE_PARETO, LEAN_GATE_ALL}

VERIFICATION_PASSED = "passed"
VERIFICATION_FAILED = "failed"
VERIFICATION_INCONCLUSIVE = "inconclusive"
VERIFICATION_PENDING = "pending"
VERIFICATION_STATUSES = {
    VERIFICATION_PASSED,
    VERIFICATION_FAILED,
    VERIFICATION_INCONCLUSIVE,
    VERIFICATION_PENDING,
}

LOOP_RUNNING = "RUNNING"
LOOP_COMPLETED = "COMPLETED"
LOOP_STOPPED_STAGNATED = "STOPPED_STAGNATED"
STAGNATION_EXPLORE_AFTER = 15
STAGNATION_STOP_AFTER = 30

DEFAULT_START = "2025-12-01"
DEFAULT_END = "2026-04-12"
DEFAULT_SEARCH_TIMERANGE = "20251201-20260228"
DEFAULT_VALIDATION_TIMERANGE = "20260301-20260331"
DEFAULT_BLIND_TIMERANGE = "20260401-20260412"
FAILED_ITERATION_SCORE = -1_000_000.0
FIXED_FREQTRADE_STRATEGY = "ELRankPortfolioLeverageStrategy"
FIXED_FREQTRADE_CONFIG = "user_data/config_okx_futures_rank_backtest.json"

_VENUE_EXCHANGE: dict[str, str] = {
    "okx": "okx",
    "binance": "binance",
    "bybit": "bybit",
    "kucoin": "kucoin",
}
_VENUE_DATADIR: dict[str, str] = {
    "okx": "user_data/data/okx",
    "binance": "user_data/data/binance",
    "bybit": "user_data/data/bybit",
    "kucoin": "user_data/data/kucoin",
}
PARETO_MAX_TOTAL = 12
PARETO_AXES = (
    "best_validation_composite",
    "best_validation_freqtrade_profit",
    "best_validation_freqtrade_profit_over_drawdown",
    "lowest_validation_drawdown_positive_profit",
    "best_research_robustness",
    "best_regime_stability",
)
STRUCTURAL_RANK_KEYS = {
    "candidate_state",
    "timeframe",
    "data_venue",
    "evaluation_lane",
    "rebalance_minutes",
    "side_mode",
    "rebalance_hours",
    "edge_mode",
    "regime_mode",
    "top_k",
    "gross_cap",
    "net_cap",
    "single_pair_cap",
    "exclude_pairs",
}

RANK_PROFILE_KEYS = {
    "n",
    "top_k",
    "gross_cap",
    "net_cap",
    "single_pair_cap",
    "side_mode",
    "min_abs_score_z",
    "score_threshold",
    "rebalance_hours",
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
    "candidate_state",
    "recompute_corr",
    "timeframe",
    "data_venue",
    "evaluation_lane",
    "rebalance_minutes",
    "short_max_market_mom_24h",
    "short_max_market_mom_72h",
    "short_max_market_ma_gap",
    "short_exit_mom_24h",
    "short_exit_mom_72h",
    "short_exit_market_mom_24h",
    "short_exit_market_ma_gap",
    "exclude_pairs",
}

NUMERIC_LIMITS = {
    "n": (1, 200),
    "top_k": (1, 10),
    "gross_cap": (0.0, 10.0),
    "net_cap": (0.0, 5.0),
    "single_pair_cap": (0.0, 2.0),
    "min_abs_score_z": (0.0, 5.0),
    "score_threshold": (0.0, 5.0),
    "rebalance_hours": (1, 168),
    "rebalance_minutes": (1, 7 * 24 * 60),
    "risk_per_trade": (0.0, 0.25),
    "leverage_cap": (1.0, 10.0),
    "edge_lookback_hours": (24, 24 * 90),
    "edge_min_periods": (1, 24 * 90),
    "edge_deadband": (0.0, 0.25),
    "pair_edge_deadband": (0.0, 0.25),
    "pair_edge_strong_ic": (0.0, 1.0),
    "pair_edge_very_strong_ic": (0.0, 1.0),
    "pair_edge_weak_cap": (1.0, 10.0),
    "regime_min_edge_ic": (0.0, 1.0),
    "regime_min_pair_edge_ic": (0.0, 1.0),
    "regime_min_pair_count": (0, 50),
    "regime_short_max_market_mom_24h": (-1.0, 1.0),
    "regime_short_max_market_mom_72h": (-1.0, 1.0),
    "regime_max_market_atr_pct": (0.0, 1.0),
    "short_max_mom_24h": (-1.0, 1.0),
    "short_max_mom_72h": (-1.0, 1.0),
    "long_min_mom_24h": (-1.0, 1.0),
    "max_entry_atr_pct": (0.0, 1.0),
    "short_max_market_mom_24h": (-1.0, 1.0),
    "short_max_market_mom_72h": (-1.0, 1.0),
    "short_max_market_ma_gap": (-1.0, 1.0),
    "short_exit_mom_24h": (-1.0, 1.0),
    "short_exit_mom_72h": (-1.0, 1.0),
    "short_exit_market_mom_24h": (-1.0, 1.0),
    "short_exit_market_ma_gap": (-1.0, 1.0),
}

ENUM_LIMITS = {
    "side_mode": {"both", "long", "short"},
    "edge_mode": {"off", "rolling_ic"},
    "regime_mode": {"off", "hq"},
    "timeframe": {"1m", "5m", "15m", "1h"},
    "data_venue": {"auto", "kucoin", "okx", "bybit", "binance"},
}

BANNED_STRATEGY_IMPORTS = {
    "asyncio",
    "httpx",
    "requests",
    "socket",
    "subprocess",
    "urllib",
}


@dataclass
class StrategyLoopConfig:
    tag: str
    venue: str = "okx"
    agent: str = AGENT_HERMES
    model: str = ""
    risk_profile: str = "aggressive"
    timeframe: str = "1h"
    data_venue: str = "auto"
    evaluation_lane: str = "1h"
    max_iterations: int = 30
    timerange: str = f"{DEFAULT_START.replace('-', '')}-{DEFAULT_END.replace('-', '')}"
    n: int = 50
    run_id: str = ""
    resume: bool = False
    start: str = DEFAULT_START
    end: str = DEFAULT_END
    min_trades: int = 80
    max_drawdown_pct: float = 25.0
    min_profit_over_dd: float = 1.2
    target_profit_pct: float = 25.0
    promote: bool = True
    max_turns: int = 30
    stale_timeout: float = 180.0
    max_retries: int = 2
    candidate_type: str = CANDIDATE_RANK_PROFILE
    opencode_mode: str = "cli"
    hermes_provider: str = ""
    hermes_toolsets: str = "terminal,file"
    hermes_reasoning_effort: str = ""
    hermes_yolo: bool = False
    candidate_state: str = ""
    recompute_corr: Optional[bool] = None
    baseline_profile_path: str = ""
    eval_mode: str = EVAL_TWO_STAGE
    score_mode: str = SCORE_RESEARCH
    promote_policy: str = PROMOTE_IMMEDIATE
    validation_protocol: str = VALIDATION_SINGLE
    search_timerange: str = DEFAULT_SEARCH_TIMERANGE
    validation_timerange: str = DEFAULT_VALIDATION_TIMERANGE
    blind_timerange: str = DEFAULT_BLIND_TIMERANGE
    verify_policy: str = VERIFY_NONE
    pareto_size_per_axis: int = 3
    lean_gate_mode: str = LEAN_GATE_OFF
    lean_bin: str = "lean"
    lean_timeout: Optional[int] = None
    lean_required_status: str = "ok"
    lean_data_root: str = ""
    score_lean_weight: float = 0.7

    @classmethod
    def from_args(
        cls,
        *,
        tag: str = rank_portfolio.DEFAULT_TAG,
        venue: str = "okx",
        agent: str = AGENT_HERMES,
        model: str = "",
        risk_profile: str = "aggressive",
        timeframe: str = "1h",
        data_venue: str = "auto",
        evaluation_lane: str = "auto",
        max_iterations: int = 30,
        timerange: Optional[str] = None,
        run_id: Optional[str] = None,
        resume: bool = False,
        n: int = 50,
        max_turns: int = 30,
        stale_timeout: float = 180.0,
        max_retries: int = 2,
        promote: bool = True,
        candidate_type: str = CANDIDATE_RANK_PROFILE,
        opencode_mode: str = "cli",
        hermes_provider: Optional[str] = None,
        hermes_toolsets: Optional[str] = None,
        hermes_reasoning_effort: Optional[str] = None,
        hermes_yolo: bool = False,
        candidate_state: Optional[str] = None,
        recompute_corr: Optional[bool] = None,
        baseline_profile: Optional[str] = None,
        eval_mode: str = EVAL_TWO_STAGE,
        score_mode: str = SCORE_RESEARCH,
        promote_policy: str = PROMOTE_IMMEDIATE,
        validation_protocol: str = VALIDATION_SINGLE,
        search_timerange: Optional[str] = None,
        validation_timerange: Optional[str] = None,
        blind_timerange: Optional[str] = None,
        verify_policy: Optional[str] = None,
        pareto_size_per_axis: int = 3,
        lean_gate_mode: str = LEAN_GATE_OFF,
        lean_bin: str = "lean",
        lean_timeout: Optional[int] = None,
        lean_required_status: str = "ok",
        lean_data_root: Optional[str] = None,
        score_lean_weight: float = 0.7,
    ) -> "StrategyLoopConfig":
        protocol = str(validation_protocol or VALIDATION_SINGLE).strip().lower()
        if protocol not in VALIDATION_PROTOCOLS:
            raise ValueError(f"validation_protocol must be one of {sorted(VALIDATION_PROTOCOLS)}, got {validation_protocol!r}")
        search_range = str(search_timerange or DEFAULT_SEARCH_TIMERANGE)
        validation_range = str(validation_timerange or DEFAULT_VALIDATION_TIMERANGE)
        blind_range = str(blind_timerange or DEFAULT_BLIND_TIMERANGE)
        if protocol == VALIDATION_SINGLE:
            start, end = parse_timerange(timerange or f"{DEFAULT_START.replace('-', '')}-{DEFAULT_END.replace('-', '')}")
            effective_timerange = timerange or f"{start.replace('-', '')}-{end.replace('-', '')}"
        else:
            search_start, _ = parse_timerange(search_range)
            _, blind_end = parse_timerange(blind_range)
            start, end = search_start, blind_end
            effective_timerange = f"{search_start.replace('-', '')}-{blind_end.replace('-', '')}"
            parse_timerange(validation_range)
        ctype = str(candidate_type or CANDIDATE_RANK_PROFILE).strip().lower()
        if ctype not in {"auto", CANDIDATE_RANK_PROFILE, CANDIDATE_FREQTRADE_STRATEGY}:
            raise ValueError(f"candidate_type must be auto, {CANDIDATE_RANK_PROFILE}, or {CANDIDATE_FREQTRADE_STRATEGY}")
        agent_s = str(agent or AGENT_HERMES).strip().lower()
        if agent_s not in AGENT_TYPES:
            raise ValueError(f"agent must be one of {sorted(AGENT_TYPES)}, got {agent!r}")
        mode = str(opencode_mode or "cli").strip().lower()
        if mode not in {"server", "cli", "auto"}:
            raise ValueError("opencode_mode must be server, cli, or auto")
        effort = str(hermes_reasoning_effort or "").strip().lower()
        if effort not in HERMES_REASONING_EFFORTS:
            raise ValueError(f"hermes_reasoning_effort must be one of {sorted(HERMES_REASONING_EFFORTS)}, got {hermes_reasoning_effort!r}")
        emode = str(eval_mode or EVAL_TWO_STAGE).strip().lower()
        if emode not in EVAL_MODES:
            raise ValueError(f"eval_mode must be one of {sorted(EVAL_MODES)}, got {eval_mode!r}")
        smode = str(score_mode or SCORE_RESEARCH).strip().lower()
        if smode not in SCORE_MODES:
            raise ValueError(f"score_mode must be one of {sorted(SCORE_MODES)}, got {score_mode!r}")
        policy = str(promote_policy or PROMOTE_IMMEDIATE).strip().lower()
        if policy not in PROMOTE_POLICIES:
            raise ValueError(f"promote_policy must be one of {sorted(PROMOTE_POLICIES)}, got {promote_policy!r}")
        verify = str(verify_policy if verify_policy is not None else VERIFY_NONE).strip().lower()
        if verify not in VERIFY_POLICIES:
            raise ValueError(f"verify_policy must be one of {sorted(VERIFY_POLICIES)}, got {verify_policy!r}")
        venue_s = str(venue or "okx").strip().lower()
        futures_venues = set(getattr(rank_portfolio, "FUTURES_VENUES", {"okx"}))
        if venue_s not in futures_venues:
            raise ValueError(f"venue must be one of {sorted(futures_venues)}, got {venue!r}")
        lean_mode = str(lean_gate_mode or LEAN_GATE_OFF).strip().lower()
        if lean_mode not in LEAN_GATE_MODES:
            raise ValueError(f"lean_gate_mode must be one of {sorted(LEAN_GATE_MODES)}, got {lean_gate_mode!r}")
        lean_status = str(lean_required_status or "ok").strip().lower()
        if not lean_status:
            lean_status = "ok"
        lane_raw = str(evaluation_lane or "auto").strip().lower()
        if lane_raw not in {"", "auto", "1h"} and str(timeframe or "1h").strip().lower() == "1h":
            lane = normalize_lane(lane_raw)
            tf = lane.timeframe
        else:
            tf = normalize_timeframe(timeframe)
            lane = normalize_lane(evaluation_lane, timeframe=tf)
        data_venue_s = str(data_venue or "auto").strip().lower()
        if data_venue_s not in {"auto", "kucoin", "okx", "bybit", "binance"}:
            raise ValueError("data_venue must be auto, kucoin, okx, bybit, or binance")
        # Freqtrade now supports all venues via per-run venue override config.
        promote_enabled = bool(promote) and policy != PROMOTE_NONE
        return cls(
            tag=tag,
            venue=venue_s,
            agent=agent_s,
            model=model,
            risk_profile=risk_profile,
            timeframe=tf,
            data_venue=data_venue_s,
            evaluation_lane=lane.lane,
            max_iterations=int(max_iterations),
            timerange=effective_timerange,
            run_id=str(run_id or ""),
            resume=bool(resume),
            n=int(n),
            start=start,
            end=end,
            max_turns=int(max_turns),
            stale_timeout=float(stale_timeout),
            max_retries=int(max_retries),
            promote=promote_enabled,
            candidate_type=ctype,
            opencode_mode=mode,
            hermes_provider=str(hermes_provider or ""),
            hermes_toolsets=str(hermes_toolsets or "terminal,file"),
            hermes_reasoning_effort=effort,
            hermes_yolo=bool(hermes_yolo),
            candidate_state=str(candidate_state or ""),
            recompute_corr=recompute_corr if recompute_corr is None else bool(recompute_corr),
            baseline_profile_path=str(baseline_profile or ""),
            eval_mode=emode,
            score_mode=smode,
            promote_policy=policy if promote_enabled else PROMOTE_NONE,
            validation_protocol=protocol,
            search_timerange=search_range,
            validation_timerange=validation_range,
            blind_timerange=blind_range,
            verify_policy=verify,
            pareto_size_per_axis=max(1, int(pareto_size_per_axis)),
            lean_gate_mode=lean_mode,
            lean_bin=str(lean_bin or "lean"),
            lean_timeout=None if lean_timeout is None else int(lean_timeout),
            lean_required_status=lean_status,
            lean_data_root=str(lean_data_root or ""),
            score_lean_weight=float(score_lean_weight) if score_lean_weight is not None else 0.7,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StrategyLoopConfig":
        data = dict(payload)
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class StrategyLoopState:
    run_id: str
    iteration: int = 1
    phase: str = PHASE_PREPARE
    best_candidate: Optional[dict[str, Any]] = None
    best_score: float = float("-inf")
    score_history: list[dict[str, Any]] = field(default_factory=list)
    candidate_paths: list[str] = field(default_factory=list)
    token_cost: dict[str, Any] = field(default_factory=dict)
    status: str = LOOP_RUNNING
    stopped_reason: str = ""
    best_composite_score: float = float("-inf")
    no_composite_improvement_count: int = 0
    valid_candidate_count: int = 0
    exploration_mode: str = "local"
    pareto_pool: dict[str, Any] = field(default_factory=dict)
    final_blind_status: Optional[dict[str, Any]] = None
    final_promotion: Optional[dict[str, Any]] = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StrategyLoopState":
        return cls(
            run_id=str(payload.get("run_id") or ""),
            iteration=int(payload.get("iteration") or 1),
            phase=str(payload.get("phase") or PHASE_PREPARE),
            best_candidate=payload.get("best_candidate"),
            best_score=float(payload.get("best_score", float("-inf"))),
            score_history=list(payload.get("score_history") or []),
            candidate_paths=list(payload.get("candidate_paths") or []),
            token_cost=dict(payload.get("token_cost") or {}),
            status=str(payload.get("status") or LOOP_RUNNING),
            stopped_reason=str(payload.get("stopped_reason") or ""),
            best_composite_score=float(payload.get("best_composite_score", float("-inf"))),
            no_composite_improvement_count=int(payload.get("no_composite_improvement_count") or 0),
            valid_candidate_count=int(payload.get("valid_candidate_count") or 0),
            exploration_mode=str(payload.get("exploration_mode") or "local"),
            pareto_pool=dict(payload.get("pareto_pool") or {}),
            final_blind_status=payload.get("final_blind_status") if isinstance(payload.get("final_blind_status"), dict) else None,
            final_promotion=payload.get("final_promotion") if isinstance(payload.get("final_promotion"), dict) else None,
        )


def _pairs_from_signal_dir(signal_dir: Path, *, exchange_name: str = "binance") -> list[str]:
    """Return futures pair_whitelist for freqtrade from signal directory.

    Reads all.feather to get unique pairs, converts 'BTC/USDT' -> 'BTC/USDT:USDT'.
    Falls back to scanning data files if feather unreadable.
    """
    try:
        import pandas as pd  # lazy import – only needed when freqtrade runs
        # Try all.feather first (consolidated signals)
        all_file = signal_dir / "all.feather"
        if all_file.exists():
            df = pd.read_feather(all_file)
            if "pair" in df.columns:
                raw_pairs: list[str] = sorted(df["pair"].dropna().unique().tolist())
                result = []
                for p in raw_pairs:
                    p = str(p)
                    if ":" not in p and "/" in p:
                        p = p + ":USDT"
                    result.append(p)
                if result:
                    return result
        # Fallback: derive pairs from individual BASE_USDT_USDT.feather files
        result = []
        import re as _re
        for f in sorted(signal_dir.glob("*_USDT_USDT.feather")):
            stem = f.stem  # e.g. "BTC_USDT_USDT"
            base = _re.sub(r"_USDT_USDT$", "", stem)
            if base:
                result.append(f"{base}/USDT:USDT")
        if result:
            return result
    except Exception:
        pass
    # Fallback: return empty → freqtrade uses base config whitelist
    return []


def parse_timerange(timerange: str | None) -> tuple[str, str]:
    raw = str(timerange or "").strip()
    if not raw:
        return DEFAULT_START, DEFAULT_END
    match = re.fullmatch(r"(\d{8})-(\d{8})", raw)
    if not match:
        raise ValueError(f"timerange must be YYYYMMDD-YYYYMMDD, got {timerange!r}")

    def fmt(s: str) -> str:
        return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"

    return fmt(match.group(1)), fmt(match.group(2))


def _date_from_iso(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _timerange_days(timerange: str) -> int:
    start, end = parse_timerange(timerange)
    return max(1, (_date_from_iso(end) - _date_from_iso(start)).days)


def _format_timerange(start: str, end: str) -> str:
    return f"{start.replace('-', '')}-{end.replace('-', '')}"


def _full_gate_days(config: Optional[StrategyLoopConfig] = None) -> int:
    if config is not None and config.validation_protocol != VALIDATION_SINGLE:
        start, _ = parse_timerange(config.search_timerange)
        _, end = parse_timerange(config.blind_timerange)
        return max(1, (_date_from_iso(end) - _date_from_iso(start)).days)
    return max(1, (_date_from_iso(DEFAULT_END) - _date_from_iso(DEFAULT_START)).days)


def scaled_gate_values(config: StrategyLoopConfig, timerange: str) -> dict[str, Any]:
    """Scale count/return gates for a sub-window while keeping risk gates fixed."""
    if config.validation_protocol == VALIDATION_SINGLE:
        return {
            "min_trades": int(config.min_trades),
            "max_drawdown_pct": float(config.max_drawdown_pct),
            "min_profit_over_dd": float(config.min_profit_over_dd),
            "target_profit_pct": float(config.target_profit_pct),
            "window_days": _timerange_days(timerange),
            "full_days": _full_gate_days(config),
        }
    window_days = _timerange_days(timerange)
    full_days = _full_gate_days(config)
    return {
        "min_trades": max(5, int(math.ceil(float(config.min_trades) * window_days / max(full_days, 1)))),
        "max_drawdown_pct": float(config.max_drawdown_pct),
        "min_profit_over_dd": float(config.min_profit_over_dd),
        "target_profit_pct": float(config.target_profit_pct) * window_days / max(full_days, 1),
        "window_days": window_days,
        "full_days": full_days,
    }


def validation_protocol_summary(config: StrategyLoopConfig) -> dict[str, Any]:
    if config.validation_protocol == VALIDATION_SINGLE:
        return {
            "protocol": VALIDATION_SINGLE,
            "windows": {
                "single": {
                    "timerange": config.timerange,
                    "start": config.start,
                    "end": config.end,
                    "gates": scaled_gate_values(config, config.timerange),
                }
            },
            "promotion_basis": "single_window",
        }
    windows: dict[str, Any] = {}
    for stage, timerange in (
        ("search", config.search_timerange),
        ("validation", config.validation_timerange),
        ("blind", config.blind_timerange),
    ):
        start, end = parse_timerange(timerange)
        windows[stage] = {
            "timerange": timerange,
            "start": start,
            "end": end,
            "gates": scaled_gate_values(config, timerange),
        }
    return {
        "protocol": config.validation_protocol,
        "windows": windows,
        "promotion_basis": "blind_only_after_validation_and_verification",
        "verify_policy": config.verify_policy,
    }


def make_run_id(tag: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", tag).strip("._") or "strategy_loop"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return f"{safe}_{stamp}_{uuid.uuid4().hex[:8]}"


def loop_root(run_id: str) -> Path:
    return repo_paths.artifacts_root() / "factor_strategy_loop" / str(run_id)


def shadow_root(profile_id: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(profile_id)).strip("._") or "profile"
    return repo_paths.artifacts_root() / "strategy_shadow" / safe


def append_shadow_event(profile_id: str, event: Mapping[str, Any], *, timestamp: Optional[float] = None) -> Path:
    ts = datetime.utcfromtimestamp(timestamp or time.time())
    root = shadow_root(profile_id)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{ts:%Y%m%d_%H}.jsonl"
    payload = {
        "ts": ts.isoformat(timespec="seconds") + "Z",
        "input_data_timestamp": event.get("input_data_timestamp"),
        "rank_signal": event.get("rank_signal"),
        "target_position": event.get("target_position"),
        "freqtrade_dataframe_status": event.get("freqtrade_dataframe_status"),
        "expected_entry_exit": event.get("expected_entry_exit"),
        "paper_event": event.get("paper_event"),
        "slippage": event.get("slippage"),
        "missing_data": event.get("missing_data"),
        "exchange_restrictions": event.get("exchange_restrictions"),
        "raw": dict(event),
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, default=str, sort_keys=True) + "\n")
    return path


def write_shadow_reconciliation(profile_id: str, payload: Mapping[str, Any], *, day: Optional[str] = None) -> Path:
    root = shadow_root(profile_id)
    root.mkdir(parents=True, exist_ok=True)
    name = f"shadow_reconciliation_{day or time.strftime('%Y%m%d')}.json"
    signal_match = float(payload.get("signal_match_rate", 0.0) or 0.0)
    blockers = list(payload.get("blockers") or [])
    if signal_match < 1.0:
        blockers.append("signal_match_rate < 100%")
    for key in ("signal_mismatch_count", "dataframe_drift_count", "missing_candle_count"):
        if int(payload.get(key) or 0) > 0:
            blockers.append(f"{key}={payload.get(key)}")
    report = {
        "version": "strategy-shadow-reconciliation-v1",
        "profile_id": profile_id,
        "day": day or time.strftime("%Y-%m-%d"),
        "live_blocked": bool(blockers),
        "blockers": blockers,
        "payload": dict(payload),
    }
    path = root / name
    write_json(path, report)
    return path


def checkpoint_path(run_id: str) -> Path:
    return loop_root(run_id) / "checkpoint.json"


def leaderboard_path(run_id: str) -> Path:
    return loop_root(run_id) / "leaderboard.json"


def strategy_loop_registry_path() -> Path:
    return repo_paths.artifacts_root() / "factor_strategy_loop" / "run_registry.jsonl"


def iteration_dir(run_id: str, iteration: int) -> Path:
    return loop_root(run_id) / f"iter_{int(iteration):02d}"


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def load_checkpoint(run_id: str) -> tuple[StrategyLoopConfig, StrategyLoopState]:
    path = checkpoint_path(run_id)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found for run_id={run_id}: {path}")
    payload = load_json(path, {})
    return StrategyLoopConfig.from_dict(payload.get("config") or {}), StrategyLoopState.from_dict(payload.get("state") or {})


def save_checkpoint(config: StrategyLoopConfig, state: StrategyLoopState) -> Path:
    payload = {
        "version": "factor-strategy-loop-v1",
        "updated_at": time.time(),
        "run_id": state.run_id,
        "config": asdict(config),
        "state": asdict(state),
        "validation_protocol": validation_protocol_summary(config),
        "pareto_pool": state.pareto_pool,
        "final_blind_status": state.final_blind_status,
    }
    path = checkpoint_path(state.run_id)
    write_json(path, payload)
    return path


def _next_phase(phase: str) -> str:
    idx = PHASES.index(phase) if phase in PHASES else 0
    return PHASES[min(idx + 1, len(PHASES) - 1)]


def _as_repo_meta(path: Path) -> str:
    return repo_paths.relpath_for_meta(path)


def _resolve_factor_state(tag: str) -> tuple[Optional[Path], str]:
    candidates = [
        repo_paths.artifacts_root() / "factor_lab" / "mining" / tag / "latest.json",
        repo_paths.REPO_ROOT / "artifacts" / "factor_lab" / "mining" / tag / "latest.json",
        repo_paths.user_data_root() / f"freqai_expressions_{tag}.json",
        repo_paths.REPO_ROOT / "user_data" / f"freqai_expressions_{tag}.json",
    ]
    for path in candidates:
        if path.exists():
            return path.resolve(), _as_repo_meta(path)
    return None, ""


# Features that only exist in 4h-mined states; unavailable in shorter timeframes.
_MTF_INCOMPATIBLE_PREFIXES = ("mtf4h_", "funding_z_", "amihud_")


def _filter_state_for_timeframe(state_path: Optional[Path], timeframe: str, idir: Path) -> Optional[Path]:
    """Return a timeframe-compatible copy of the factor state.

    Removes survivors whose expressions reference features only available
    in 4h-mined data (mtf4h_*, funding_z_*, amihud_*) when running in
    sub-4h timeframes. Writes a filtered copy to idir/filtered_state.json
    so the original state is preserved.
    """
    if state_path is None or not state_path.exists():
        return state_path
    tf = str(timeframe or "").strip().lower()
    if tf in ("4h", "1d", ""):
        return state_path
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        survivors = state.get("survivors") or []
        if not survivors:
            return state_path
        filtered = [
            s for s in survivors
            if not any(pfx in str(s.get("expression", "")) for pfx in _MTF_INCOMPATIBLE_PREFIXES)
        ]
        if len(filtered) == len(survivors):
            return state_path
        filtered_state = dict(state)
        filtered_state["survivors"] = filtered
        filtered_path = idir / "filtered_state.json"
        write_json(filtered_path, filtered_state)
        return filtered_path
    except Exception:
        return state_path


def _summarize_json(path: Path, max_items: int = 8) -> dict[str, Any]:
    try:
        payload = load_json(path, {})
    except Exception as exc:
        return {"path": _as_repo_meta(path), "error": str(exc)}
    if isinstance(payload, list):
        return {"path": _as_repo_meta(path), "kind": "list", "count": len(payload), "sample": payload[:max_items]}
    if not isinstance(payload, dict):
        return {"path": _as_repo_meta(path), "kind": type(payload).__name__}
    summary: dict[str, Any] = {"path": _as_repo_meta(path), "keys": sorted(payload.keys())[:30]}
    if "survivors" in payload and isinstance(payload["survivors"], list):
        summary["survivor_count"] = len(payload["survivors"])
        summary["sample_factors"] = [
            {
                "expression": x.get("expression"),
                "origin": x.get("origin"),
                "oos_ic": x.get("oos_ic"),
                "neutralized_ic": x.get("neutralized_ic"),
                "fitness": x.get("fitness"),
            }
            for x in payload["survivors"][:max_items]
            if isinstance(x, dict)
        ]
    if "factors" in payload and isinstance(payload["factors"], list):
        summary["factor_count"] = len(payload["factors"])
        summary["sample_factors"] = [
            {
                "expression": x.get("expression"),
                "origin": x.get("origin"),
                "oos_ic": x.get("oos_ic"),
                "neutralized_ic": x.get("neutralized_ic"),
                "fitness": x.get("fitness"),
            }
            for x in payload["factors"][:max_items]
            if isinstance(x, dict)
        ]
    return summary


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() not in {"0", "false", "no", "off", ""}


def _load_dotenv_into(env: dict[str, str], path: Optional[Path] = None) -> None:
    env_path = path or repo_paths.REPO_ROOT / ".env"
    if not env_path.exists():
        return
    try:
        raw = env_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return
    for line in raw.splitlines():
        item = line.strip()
        if not item or item.startswith("#"):
            continue
        if item.lower().startswith("export "):
            item = item[7:].strip()
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and ((value[0] == value[-1] == "'") or (value[0] == value[-1] == '"')):
            value = value[1:-1]
        env.setdefault(key, value)


def _append_csv_unique(raw: str, values: Sequence[str]) -> str:
    items: list[str] = []
    seen: set[str] = set()
    for item in [*str(raw or "").split(","), *values]:
        cleaned = item.strip()
        if not cleaned or cleaned in seen:
            continue
        items.append(cleaned)
        seen.add(cleaned)
    return ",".join(items)


def _opencode_cli_env(base_env: Optional[Mapping[str, str]] = None, *, load_dotenv: bool = True) -> dict[str, str]:
    env = dict(base_env or os.environ)
    if load_dotenv:
        _load_dotenv_into(env)
    if not str(env.get("OPENAI_BASE_URL") or "").strip():
        api_base = str(env.get("OPENAI_API_BASE") or env.get("LLM_BASE_URL") or "").strip().rstrip("/")
        if api_base:
            env["OPENAI_BASE_URL"] = api_base if api_base.endswith("/v1") else f"{api_base}/v1"
    if not str(env.get("OPENAI_API_KEY") or "").strip() and str(env.get("LLM_API_KEY") or "").strip():
        env["OPENAI_API_KEY"] = str(env.get("LLM_API_KEY") or "")
    if not str(env.get("LLM_BASE_URL") or "").strip() and str(env.get("OPENAI_BASE_URL") or "").strip():
        env["LLM_BASE_URL"] = str(env.get("OPENAI_BASE_URL") or "")
    if not str(env.get("LLM_API_KEY") or "").strip() and str(env.get("OPENAI_API_KEY") or "").strip():
        env["LLM_API_KEY"] = str(env.get("OPENAI_API_KEY") or "")
    project_config = repo_paths.REPO_ROOT / ".opencode.json"
    if project_config.exists():
        env.setdefault("OPENCODE_CONFIG", str(project_config))
    no_proxy = _append_csv_unique(env.get("NO_PROXY") or env.get("no_proxy") or "", ("127.0.0.1", "localhost", "::1"))
    env["NO_PROXY"] = no_proxy
    env["no_proxy"] = no_proxy
    return env


def _hermes_cli_env(base_env: Optional[Mapping[str, str]] = None, *, load_dotenv: bool = True) -> dict[str, str]:
    env = dict(base_env or os.environ)
    if load_dotenv:
        _load_dotenv_into(env)
    if not str(env.get("OPENAI_BASE_URL") or "").strip():
        api_base = str(env.get("OPENAI_API_BASE") or env.get("LLM_BASE_URL") or "").strip().rstrip("/")
        if api_base:
            env["OPENAI_BASE_URL"] = api_base if api_base.endswith("/v1") else f"{api_base}/v1"
    if not str(env.get("OPENAI_API_KEY") or "").strip() and str(env.get("LLM_API_KEY") or "").strip():
        env["OPENAI_API_KEY"] = str(env.get("LLM_API_KEY") or "")
    if not str(env.get("LLM_BASE_URL") or "").strip() and str(env.get("OPENAI_BASE_URL") or "").strip():
        env["LLM_BASE_URL"] = str(env.get("OPENAI_BASE_URL") or "")
    if not str(env.get("LLM_API_KEY") or "").strip() and str(env.get("OPENAI_API_KEY") or "").strip():
        env["LLM_API_KEY"] = str(env.get("OPENAI_API_KEY") or "")
    no_proxy = _append_csv_unique(env.get("NO_PROXY") or env.get("no_proxy") or "", ("127.0.0.1", "localhost", "::1"))
    env["NO_PROXY"] = no_proxy
    env["no_proxy"] = no_proxy
    return env


def _openai_compatible_env(base_env: Optional[Mapping[str, str]] = None, *, load_dotenv: bool = True) -> dict[str, str]:
    return _hermes_cli_env(base_env, load_dotenv=load_dotenv)


def _openai_compatible_model(config_model: str, env: Mapping[str, str]) -> str:
    for value in (
        config_model,
        env.get("LLM_MODEL", ""),
        env.get("OPENAI_MODEL", ""),
        env.get("HERMES_MODEL", ""),
        env.get("OPENCODE_MODEL", ""),
    ):
        raw = str(value or "").strip()
        if raw:
            return raw.split("/", 1)[1] if raw.startswith("custom/") else raw
    return ""


@contextmanager
def _temporary_environ(overrides: Mapping[str, str]):
    previous: dict[str, Optional[str]] = {}
    for key, value in overrides.items():
        previous[key] = os.environ.get(key)
        os.environ[key] = str(value)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _json_object_from_text(text: str) -> Optional[dict[str, Any]]:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    for match in re.finditer(r"```(?:json)?\s*([\s\S]*?)```", raw, flags=re.IGNORECASE):
        try:
            obj = json.loads(match.group(1).strip())
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _prepare_hermes_run_home(run_id: str, env: dict[str, str]) -> Path:
    source_home = Path(str(env.get("HERMES_HOME") or Path.home() / ".hermes")).expanduser()
    hermes_home = loop_root(run_id) / "hermes_home"
    hermes_home.mkdir(parents=True, exist_ok=True)

    source_config = source_home / "config.yaml"
    target_config = hermes_home / "config.yaml"
    if source_config.exists() and not target_config.exists():
        shutil.copy2(source_config, target_config)

    source_auth = source_home / "auth.json"
    target_auth = hermes_home / "auth.json"
    if source_auth.exists() and not target_auth.exists():
        shutil.copy2(source_auth, target_auth)
        try:
            target_auth.chmod(0o600)
        except OSError:
            pass

    source_env = source_home / ".env"
    if source_env.exists() and source_home != hermes_home:
        _load_dotenv_into(env, source_env)

    env["HERMES_HOME"] = str(hermes_home)
    return hermes_home


def _hermes_model(config_model: str, env: Mapping[str, str]) -> str:
    for value in (
        config_model,
        env.get("HERMES_MODEL", ""),
        env.get("LLM_MODEL", ""),
        env.get("OPENAI_MODEL", ""),
    ):
        raw = str(value or "").strip()
        if raw:
            return raw.split("/", 1)[1] if raw.startswith("custom/") else raw
    return ""


def _default_baseline_profile_path(tag: str) -> Path:
    primary = repo_paths.artifacts_root() / "rank_portfolio" / tag / "optimized_profile.json"
    if primary.exists():
        return primary
    return repo_paths.REPO_ROOT / "artifacts" / "rank_portfolio" / tag / "optimized_profile.json"


def _resolve_baseline_profile_path(config: StrategyLoopConfig) -> Optional[Path]:
    raw = str(config.baseline_profile_path or "").strip()
    path = repo_paths.resolve_repo_path(raw) if raw else _default_baseline_profile_path(config.tag)
    return path if path.exists() else None


def _extract_optimized_rank_profile(payload: Mapping[str, Any]) -> dict[str, Any]:
    profile: dict[str, Any] = {}
    raw_profile = payload.get("rank_profile")
    if isinstance(raw_profile, Mapping):
        profile.update({str(k): v for k, v in raw_profile.items() if str(k) in RANK_PROFILE_KEYS or str(k) == "score_threshold"})

    candidate = payload.get("candidate")
    if isinstance(candidate, Mapping) and isinstance(candidate.get("rank_profile"), Mapping):
        profile.update({str(k): v for k, v in candidate["rank_profile"].items() if str(k) in RANK_PROFILE_KEYS or str(k) == "score_threshold"})

    risk = payload.get("risk")
    if isinstance(risk, Mapping):
        for key, value in risk.items():
            key_s = "min_abs_score_z" if str(key) == "score_threshold" else str(key)
            if key_s in RANK_PROFILE_KEYS:
                profile[key_s] = value

    selection = payload.get("selection")
    if isinstance(selection, Mapping):
        if selection.get("n") is not None:
            profile["n"] = selection.get("n")
        if selection.get("recompute_corr") is not None:
            profile["recompute_corr"] = selection.get("recompute_corr")

    if payload.get("candidate_state"):
        profile["candidate_state"] = payload.get("candidate_state")

    return normalize_rank_profile(profile) if profile else {}


def _load_optimized_baseline(config: StrategyLoopConfig) -> dict[str, Any]:
    path = _resolve_baseline_profile_path(config)
    if path is None:
        return {"available": False, "path": _as_repo_meta(_default_baseline_profile_path(config.tag))}
    payload = load_json(path, {})
    if not isinstance(payload, Mapping):
        return {"available": False, "path": _as_repo_meta(path), "error": "optimized_profile.json is not an object"}
    rank_profile = _extract_optimized_rank_profile(payload)
    return {
        "available": True,
        "path": _as_repo_meta(path),
        "label": "state_0149 + no_corr_recompute + filters" if "state_0149" in str(rank_profile.get("candidate_state", "")) else "optimized_profile.json",
        "rank_profile": rank_profile,
        "expected_research": payload.get("research_backtest") if isinstance(payload.get("research_backtest"), Mapping) else {},
        "expected_freqtrade": payload.get("freqtrade_backtest") if isinstance(payload.get("freqtrade_backtest"), Mapping) else {},
        "commands": payload.get("commands") if isinstance(payload.get("commands"), Mapping) else {},
        "notes": payload.get("notes") if isinstance(payload.get("notes"), list) else [],
    }


def _diff_values(expected: Mapping[str, Any], actual: Mapping[str, Any], keys: Sequence[str]) -> dict[str, Any]:
    diff: dict[str, Any] = {}
    for key in keys:
        if key not in expected and key not in actual:
            continue
        exp = expected.get(key)
        act = actual.get(key)
        if exp != act:
            diff[key] = {"baseline": exp, "latest": act}
    return diff


def _baseline_rank_profile(config: StrategyLoopConfig) -> dict[str, Any]:
    baseline = _load_optimized_baseline(config)
    profile = baseline.get("rank_profile")
    return dict(profile) if isinstance(profile, Mapping) else {}


def _compact_leaderboard_row(row: Mapping[str, Any]) -> dict[str, Any]:
    candidate = row.get("candidate") if isinstance(row.get("candidate"), Mapping) else {}
    metrics = row.get("metrics") if isinstance(row.get("metrics"), Mapping) else {}
    research_metrics = row.get("research_metrics") if isinstance(row.get("research_metrics"), Mapping) else metrics
    freqtrade_metrics = row.get("freqtrade_metrics") if isinstance(row.get("freqtrade_metrics"), Mapping) else {}
    lean_metrics = row.get("lean_metrics") if isinstance(row.get("lean_metrics"), Mapping) else {}
    score_components = row.get("score_components") if isinstance(row.get("score_components"), Mapping) else {}
    return {
        "iteration": row.get("iteration"),
        "name": candidate.get("name") if isinstance(candidate, Mapping) else None,
        "candidate_type": candidate.get("candidate_type") if isinstance(candidate, Mapping) else None,
        "candidate_path": row.get("candidate_path"),
        "rank_profile": candidate.get("rank_profile") if isinstance(candidate, Mapping) else row.get("parameters"),
        "parameter_signature": row.get("parameter_signature"),
        "score": row.get("score"),
        "score_components": {
            key: score_components.get(key)
            for key in (
                "score_mode",
                "research_score",
                "freqtrade_score",
                "composite_score",
                "selection_reason",
                "lean_score",
                "blended_score",
                "score_lean_weight",
            )
            if key in score_components
        },
        "constraints_ok": row.get("constraints_ok"),
        "metrics": {
            key: metrics.get(key)
            for key in (
                "profit_pct",
                "max_drawdown_pct",
                "trades",
                "profit_over_max_drawdown",
                "simulated_liquidations",
                "liquidation_rejects",
                "avg_turnover",
                "kill_mode_count",
            )
            if key in metrics
        },
        "research_metrics": {
            key: research_metrics.get(key)
            for key in ("profit_pct", "max_drawdown_pct", "trades", "profit_over_max_drawdown")
            if key in research_metrics
        },
        "freqtrade_metrics": {
            key: freqtrade_metrics.get(key)
            for key in ("ok", "profit_pct", "max_drawdown_pct", "trades", "profit_over_max_drawdown", "backtest_zip")
            if key in freqtrade_metrics
        },
        "lean_gate_status": row.get("lean_gate_status"),
        "lean_comparison_status": row.get("lean_comparison_status"),
        "lean_metrics": {
            key: lean_metrics.get(key)
            for key in ("final_equity", "max_drawdown", "trades", "orders", "turnover", "max_gross", "fee_cost", "ending_open_positions")
            if key in lean_metrics
        },
        "lean_score": row.get("lean_score"),
        "lean_analysis_summary": row.get("lean_analysis_summary") or {},
        "violations": row.get("violations") or [],
        "window_metrics": {
            key: value
            for key, value in (row.get("window_metrics") or {}).items()
            if key != "blind"
        } if isinstance(row.get("window_metrics"), Mapping) else {},
        "verification_status": row.get("verification_status"),
        "promotion_eligible": row.get("promotion_eligible"),
        "pareto_axes": row.get("pareto_axes") or [],
        "artifact_refs": row.get("artifact_refs") or {},
        "diagnostics": row.get("diagnostics") or (row.get("promotion") or {}).get("reason"),
    }


def _row_candidate(row: Mapping[str, Any]) -> Mapping[str, Any]:
    candidate = row.get("candidate")
    return candidate if isinstance(candidate, Mapping) else {}


def _row_rank_profile(row: Mapping[str, Any]) -> Mapping[str, Any]:
    candidate = _row_candidate(row)
    profile = candidate.get("rank_profile")
    if isinstance(profile, Mapping):
        return profile
    params = row.get("parameters")
    return params if isinstance(params, Mapping) else {}


def _row_signature(row: Mapping[str, Any]) -> str:
    existing = str(row.get("parameter_signature") or "").strip()
    if existing:
        return existing
    profile = _row_rank_profile(row)
    if not profile:
        return ""
    try:
        return rank_profile_signature(profile)
    except Exception:
        return ""


def _score_component(row: Mapping[str, Any], key: str) -> Optional[float]:
    components = row.get("score_components") if isinstance(row.get("score_components"), Mapping) else {}
    value = components.get(key)
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) else None


def _metric_value(row: Mapping[str, Any], stage: str, key: str) -> Optional[float]:
    metrics = row.get(f"{stage}_metrics") if isinstance(row.get(f"{stage}_metrics"), Mapping) else {}
    value = metrics.get(key)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _best_compact(rows: Sequence[Mapping[str, Any]], key_fn: Any) -> Optional[dict[str, Any]]:
    best_row: Optional[Mapping[str, Any]] = None
    best_value = float("-inf")
    for row in rows:
        value = key_fn(row)
        if value is None:
            continue
        if float(value) > best_value:
            best_value = float(value)
            best_row = row
    return _compact_leaderboard_row(best_row) if best_row is not None else None


def _row_identity(row: Mapping[str, Any]) -> str:
    signature = _row_signature(row)
    if signature:
        return f"sig:{signature}"
    path = str(row.get("candidate_path") or "").strip()
    if path:
        return f"path:{path}"
    return f"iter:{row.get('run_id')}:{row.get('iteration')}"


def _research_robustness_score(row: Mapping[str, Any]) -> Optional[float]:
    explicit = _score_component(row, "research_robustness_score")
    if explicit is not None:
        return explicit
    research_pdd = _metric_value(row, "research", "profit_over_max_drawdown")
    research_profit = _metric_value(row, "research", "profit_pct")
    freqtrade_profit = _metric_value(row, "freqtrade", "profit_pct")
    metrics = row.get("metrics") if isinstance(row.get("metrics"), Mapping) else {}
    turnover = metrics.get("avg_turnover")
    try:
        turnover_f = float(turnover or 0.0)
    except (TypeError, ValueError):
        turnover_f = 0.0
    if research_pdd is None:
        return None
    profit_gap = abs(float(research_profit or 0.0) - float(freqtrade_profit or 0.0))
    return float(research_pdd) * 100.0 - profit_gap - max(0.0, turnover_f - 2.0) * 10.0


def _regime_stability_score(row: Mapping[str, Any]) -> Optional[float]:
    explicit = _score_component(row, "regime_stability_score")
    if explicit is not None:
        return explicit
    window_metrics = row.get("window_metrics") if isinstance(row.get("window_metrics"), Mapping) else {}
    validation = window_metrics.get("validation") if isinstance(window_metrics.get("validation"), Mapping) else {}
    stability = validation.get("regime_stability") if isinstance(validation.get("regime_stability"), Mapping) else {}
    if stability.get("score") is not None:
        try:
            return float(stability["score"])
        except (TypeError, ValueError):
            return None
    positive_ratio = stability.get("positive_subwindow_ratio")
    worst_profit = stability.get("worst_subwindow_profit_pct")
    profit_std = stability.get("profit_std")
    if positive_ratio is None and worst_profit is None and profit_std is None:
        return None
    return float(positive_ratio or 0.0) * 100.0 + float(worst_profit or 0.0) - float(profit_std or 0.0)


def _axis_value(axis: str, row: Mapping[str, Any]) -> Optional[float]:
    if axis == "best_validation_composite":
        return _score_component(row, "composite_score")
    if axis == "best_validation_freqtrade_profit":
        return _metric_value(row, "freqtrade", "profit_pct")
    if axis == "best_validation_freqtrade_profit_over_drawdown":
        return _metric_value(row, "freqtrade", "profit_over_max_drawdown")
    if axis == "lowest_validation_drawdown_positive_profit":
        profit = _metric_value(row, "freqtrade", "profit_pct")
        if profit is None:
            profit = _metric_value(row, "research", "profit_pct")
        trades = _metric_value(row, "freqtrade", "trades")
        if trades is None:
            trades = _metric_value(row, "research", "trades")
        drawdown = _metric_value(row, "freqtrade", "max_drawdown_pct")
        if drawdown is None:
            drawdown = _metric_value(row, "research", "max_drawdown_pct")
        if profit is None or drawdown is None or profit <= 0:
            return None
        if row.get("constraints_ok") is False and (trades is None or trades < 1):
            return None
        return -float(drawdown)
    if axis == "best_research_robustness":
        return _research_robustness_score(row)
    if axis == "best_regime_stability":
        return _regime_stability_score(row)
    return None


def build_pareto_pool(
    rows: Sequence[Mapping[str, Any]],
    *,
    size_per_axis: int = 3,
    max_total: int = PARETO_MAX_TOTAL,
) -> dict[str, Any]:
    axis_rows: dict[str, list[dict[str, Any]]] = {}
    finalist_rows: dict[str, dict[str, Any]] = {}
    finalist_axes: dict[str, list[str]] = {}
    for axis in PARETO_AXES:
        scored: list[tuple[float, Mapping[str, Any]]] = []
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            value = _axis_value(axis, row)
            if value is None or not math.isfinite(float(value)):
                continue
            scored.append((float(value), row))
        scored.sort(key=lambda item: item[0], reverse=True)
        selected: list[dict[str, Any]] = []
        axis_seen: set[str] = set()
        for value, row in scored:
            ident = _row_identity(row)
            if ident in axis_seen:
                continue
            axis_seen.add(ident)
            compact = _compact_leaderboard_row(row)
            compact["axis_value"] = value
            selected.append(compact)
            if len(selected) >= int(size_per_axis):
                break
        axis_rows[axis] = selected
        for compact in selected:
            ident = str(compact.get("parameter_signature") or compact.get("candidate_path") or compact.get("iteration"))
            if ident not in finalist_rows and len(finalist_rows) >= int(max_total):
                continue
            finalist_rows.setdefault(ident, compact)
            finalist_axes.setdefault(ident, [])
            if axis not in finalist_axes[ident]:
                finalist_axes[ident].append(axis)

    finalists: list[dict[str, Any]] = []
    for ident, compact in finalist_rows.items():
        item = dict(compact)
        item["pareto_axes"] = finalist_axes.get(ident, [])
        finalists.append(item)
    return {
        "version": "factor-strategy-loop-pareto-pool-v1",
        "updated_at": time.time(),
        "axes": axis_rows,
        "finalists": finalists[: int(max_total)],
        "size_per_axis": int(size_per_axis),
        "max_total": int(max_total),
    }


def _loop_memory(run_id: str, iteration: int, *, recent_limit: int = 8) -> dict[str, Any]:
    memory: dict[str, Any] = {
        "best_candidate": None,
        "best_research_result": None,
        "best_freqtrade_result": None,
        "best_composite_candidate": None,
        "best_freqtrade_profit": None,
        "best_freqtrade_profit_over_drawdown": None,
        "best_research_profit_over_drawdown": None,
        "best_lean_candidate": None,
        "lean_metrics_history": [],
        "pareto_memory": {},
        "recent_score_history": [],
        "previous_failure": None,
        "avoid_repeating_rank_profiles": [],
        "avoid_repeating_rank_profile_signatures": [],
        "negative_feedback": [],
        "stagnation": {},
        "gate_repair_hints": {},
    }
    try:
        loaded_config, state = load_checkpoint(run_id)
    except Exception:
        loaded_config = None
        state = None

    if state is not None:
        if isinstance(state.best_candidate, Mapping):
            memory["best_candidate"] = _compact_leaderboard_row(state.best_candidate)
            memory["best_research_result"] = memory["best_candidate"].get("research_metrics")
            memory["best_freqtrade_result"] = memory["best_candidate"].get("freqtrade_metrics")
        recent = state.score_history[-recent_limit:]
        memory["recent_score_history"] = [_compact_leaderboard_row(row) for row in recent if isinstance(row, Mapping)]
        history = [row for row in state.score_history if isinstance(row, Mapping)]
        memory["best_composite_candidate"] = _best_compact(history, lambda r: _score_component(r, "composite_score"))
        memory["best_freqtrade_profit"] = _best_compact(history, lambda r: _metric_value(r, "freqtrade", "profit_pct"))
        memory["best_freqtrade_profit_over_drawdown"] = _best_compact(history, lambda r: _metric_value(r, "freqtrade", "profit_over_max_drawdown"))
        memory["best_research_profit_over_drawdown"] = _best_compact(history, lambda r: _metric_value(r, "research", "profit_over_max_drawdown"))
        memory["best_lean_candidate"] = _best_compact(history, lambda r: r.get("lean_score") if r.get("lean_score") is not None else float("-inf"))
        lean_history: list[dict[str, Any]] = []
        for row in state.score_history[-recent_limit:]:
            if not isinstance(row, Mapping):
                continue
            compact = _compact_leaderboard_row(row)
            la_summary = compact.get("lean_analysis_summary") or {}
            lm = compact.get("lean_metrics") or {}
            if lm or la_summary or compact.get("lean_score") is not None:
                lean_history.append({
                    "iteration": compact.get("iteration"),
                    "lean_score": compact.get("lean_score"),
                    "lean_gate_status": compact.get("lean_gate_status"),
                    "lean_metrics": {k: lm.get(k) for k in ("final_equity", "max_drawdown", "trades") if lm.get(k) is not None},
                    "monthly_worst": la_summary.get("monthly_worst"),
                    "monthly_best": la_summary.get("monthly_best"),
                    "drawdown_worst": la_summary.get("drawdown_worst"),
                    "consecutive_loss_months": la_summary.get("consecutive_loss_months"),
                })
        memory["lean_metrics_history"] = lean_history
        pareto_payload = state.pareto_pool if isinstance(state.pareto_pool, Mapping) and state.pareto_pool else build_pareto_pool(history)
        axes = pareto_payload.get("axes") if isinstance(pareto_payload.get("axes"), Mapping) else {}
        memory["pareto_memory"] = {
            axis: rows[:3] if isinstance(rows, list) else []
            for axis, rows in axes.items()
            if axis in PARETO_AXES
        }
        memory["pareto_memory"].setdefault("best_composite", memory["best_composite_candidate"])
        memory["pareto_memory"].setdefault("best_freqtrade_profit", memory["best_freqtrade_profit"])
        memory["pareto_memory"].setdefault("best_freqtrade_profit_over_drawdown", memory["best_freqtrade_profit_over_drawdown"])
        memory["pareto_memory"].setdefault("best_research_profit_over_drawdown", memory["best_research_profit_over_drawdown"])
        memory["stagnation"] = {
            "valid_candidate_count": state.valid_candidate_count,
            "no_composite_improvement_count": state.no_composite_improvement_count,
            "exploration_mode": state.exploration_mode,
            "switch_to_structured_after": STAGNATION_EXPLORE_AFTER,
            "stop_after": STAGNATION_STOP_AFTER,
            "status": state.status,
            "stopped_reason": state.stopped_reason,
        }
        seen_profiles: list[dict[str, Any]] = []
        seen_signatures: list[str] = []
        feedback: list[str] = []
        for row in state.score_history:
            if not isinstance(row, Mapping):
                continue
            compact = _compact_leaderboard_row(row)
            profile = compact.get("rank_profile")
            if not isinstance(profile, Mapping):
                continue
            signature = _row_signature(row)
            if signature and signature not in seen_signatures:
                seen_signatures.append(signature)
            normalized = dict(profile)
            if normalized not in seen_profiles:
                seen_profiles.append(normalized)
            metrics = compact.get("metrics") if isinstance(compact.get("metrics"), Mapping) else {}
            if float(metrics.get("profit_pct") or 0.0) < 0.0:
                feedback.append(f"Iteration {compact.get('iteration')} lost money; avoid broad rewrites of {normalized}.")
            if compact.get("constraints_ok") is False and compact.get("violations"):
                feedback.append(f"Iteration {compact.get('iteration')} failed gates: {compact.get('violations')}.")
        memory["avoid_repeating_rank_profiles"] = seen_profiles[-recent_limit:]
        memory["avoid_repeating_rank_profile_signatures"] = seen_signatures
        memory["negative_feedback"] = feedback[-recent_limit:]
        if loaded_config is not None:
            memory["gate_repair_hints"] = _search_gate_repair_hints(history, loaded_config)

    previous_iter = iteration_dir(run_id, iteration - 1) if iteration > 1 else None
    if previous_iter is not None and (previous_iter / "error.json").exists():
        error = load_json(previous_iter / "error.json", {})
        memory["previous_failure"] = {
            "iteration": iteration - 1,
            "phase": error.get("phase"),
            "error_type": error.get("error_type"),
            "message": error.get("message"),
        }
    return memory


def prepare_context(config: StrategyLoopConfig, run_id: str, iteration: int) -> dict[str, Any]:
    factor_state, factor_source = _resolve_factor_state(config.tag)
    rank_dir = repo_paths.artifacts_root() / "rank_portfolio" / config.tag
    fixed_rank_dir = repo_paths.REPO_ROOT / "artifacts" / "rank_portfolio" / config.tag
    if not rank_dir.exists() and fixed_rank_dir.exists():
        rank_dir = fixed_rank_dir

    previous_iter = iteration_dir(run_id, iteration - 1) if iteration > 1 else None
    previous: dict[str, Any] = {}
    if previous_iter is not None:
        for name in ("analysis.md", "backtest.json", "candidate.json", "evaluation.json", "error.json", "lean_analysis.json"):
            path = previous_iter / name
            if path.exists():
                if path.suffix == ".json":
                    previous[name] = load_json(path, {})
                else:
                    previous[name] = path.read_text(encoding="utf-8")[:12_000]
        lean_md_path = previous_iter / "lean_analysis.md"
        if lean_md_path.exists():
            previous["lean_analysis.md"] = lean_md_path.read_text(encoding="utf-8")[:12_000]

    venue_dir = repo_paths.user_data_root() / "data" / config.venue / "futures"
    if config.venue == "okx" and not venue_dir.exists():
        venue_dir = repo_paths.REPO_ROOT / "user_data" / "data" / "okx" / "futures"
    futures_files = sorted(venue_dir.glob(f"*-{config.timeframe}-futures.feather")) if venue_dir.exists() else []
    coverage = {
        "venue": config.venue,
        "path": _as_repo_meta(venue_dir) if venue_dir.exists() else str(venue_dir),
        "timeframe": config.timeframe,
        "data_venue": config.data_venue,
        "evaluation_lane": config.evaluation_lane,
        "futures_files": len(futures_files),
        "sample_files": [p.name for p in futures_files[:10]],
    }

    rank_artifacts: dict[str, Any] = {"path": _as_repo_meta(rank_dir)}
    for name in ("rank_export.json", "backtest.json", "selected_factors.json", "optimized_profile.json"):
        path = rank_dir / name
        if path.exists():
            if name.endswith(".json"):
                rank_artifacts[name] = load_json(path, {})
            else:
                rank_artifacts[name] = _as_repo_meta(path)
    optimized_baseline = _load_optimized_baseline(config)
    if optimized_baseline.get("available"):
        latest_backtest = rank_artifacts.get("backtest.json") if isinstance(rank_artifacts.get("backtest.json"), Mapping) else {}
        latest_profile = {
            key: latest_backtest.get(key)
            for key in sorted(RANK_PROFILE_KEYS)
            if isinstance(latest_backtest, Mapping) and latest_backtest.get(key) is not None
        }
        latest_profile["candidate_state"] = latest_backtest.get("candidate_source") if isinstance(latest_backtest, Mapping) else None
        optimized_baseline["latest_delta"] = {
            "profile": _diff_values(
                optimized_baseline.get("rank_profile") if isinstance(optimized_baseline.get("rank_profile"), Mapping) else {},
                latest_profile,
                sorted(RANK_PROFILE_KEYS),
            ),
            "research_metrics": _diff_values(
                optimized_baseline.get("expected_research") if isinstance(optimized_baseline.get("expected_research"), Mapping) else {},
                latest_backtest if isinstance(latest_backtest, Mapping) else {},
                ("total_return_pct", "max_drawdown_pct", "profit_over_max_drawdown", "trades", "simulated_liquidations", "liquidation_rejects"),
            ),
        }

    return {
        "version": "factor-strategy-loop-context-v1",
        "run_id": run_id,
        "iteration": int(iteration),
        "objective": {
            "mode": "profit_first_with_drawdown_controls",
            "candidate_type": config.candidate_type,
            "eval_mode": config.eval_mode,
            "score_mode": config.score_mode,
            "validation_protocol": validation_protocol_summary(config),
            "lean_gate": {
                "mode": config.lean_gate_mode,
                "required_status": config.lean_required_status,
                "fail_closed": _lean_gate_active(config),
            },
            "hard_gates": {
                "max_drawdown_pct": config.max_drawdown_pct,
                "min_trades": config.min_trades,
                "min_profit_over_dd": config.min_profit_over_dd,
                "simulated_liquidations": 0,
                "liquidation_rejects": 0,
            },
            "target_profit_pct": config.target_profit_pct,
        },
        "config": asdict(config),
        "optimized_baseline": optimized_baseline,
        "baseline_search_policy": {
            "default_candidate_type": CANDIDATE_RANK_PROFILE,
            "first_iteration": "reproduce optimized_baseline.rank_profile before any ablation",
            "step_size": "change one factor/risk/filter knob at a time near the baseline",
            "dedup_signature": "risk_per_trade quantized to 1e-5; momentum/ATR/z thresholds quantized to 1e-3; pair lists sorted and normalized",
            "stagnation_policy": {
                "structured_exploration_after_valid_non_improving_candidates": STAGNATION_EXPLORE_AFTER,
                "stop_after_valid_non_improving_candidates": STAGNATION_STOP_AFTER,
            },
            "search_modes": {
                "local_exploit": "small parameter changes around an existing anchor",
                "structured_explore": "must change at least one structural dimension: pair universe, side mode, cadence, edge/regime mode, factor state, top_k, or core risk structure",
            },
            "do_not_rewrite_freqtrade_strategy": config.candidate_type != CANDIDATE_FREQTRADE_STRATEGY,
        },
        "factor_source": factor_source,
        "factor_summary": _summarize_json(factor_state) if factor_state else {"missing": True, "tag": config.tag},
        "rank_artifacts": rank_artifacts,
        "loop_memory": _loop_memory(run_id, iteration),
        "futures_coverage": coverage,
        "okx_coverage": coverage if config.venue == "okx" else None,
        "previous_iteration": previous,
        "allowed_candidate_files": (
            ["candidate.json", "strategy.py", "analysis.md"]
            if config.candidate_type == CANDIDATE_FREQTRADE_STRATEGY
            else ["candidate.json", "analysis.md"]
        ),
        "allowed_rank_profile_keys": sorted(RANK_PROFILE_KEYS),
    }


def normalize_rank_profile(profile: Mapping[str, Any], *, default_n: int = 50) -> dict[str, Any]:
    out: dict[str, Any] = {"n": int(default_n)}
    for key, value in dict(profile).items():
        key = "min_abs_score_z" if key == "score_threshold" else key
        if key not in RANK_PROFILE_KEYS:
            raise ValueError(f"rank_profile contains unsupported key: {key}")
        if value is None:
            continue
        if key == "candidate_state":
            raw = str(value).strip()
            if not raw:
                continue
            if ".." in Path(raw).parts:
                raise ValueError("candidate_state must not contain '..' traversal")
            out[key] = raw
            continue
        if key == "recompute_corr":
            out[key] = _coerce_bool(value)
            continue
        if key == "timeframe":
            out[key] = normalize_timeframe(value)
            continue
        if key == "evaluation_lane":
            out[key] = normalize_lane(value, timeframe=out.get("timeframe")).lane
            continue
        if key in {"exclude_pairs"}:
            if isinstance(value, str):
                out[key] = [p.strip() for p in value.split(",") if p.strip()]
            elif isinstance(value, Sequence):
                out[key] = [str(p).strip() for p in value if str(p).strip()]
            else:
                raise ValueError("exclude_pairs must be a string or list")
            continue
        if key in ENUM_LIMITS:
            lowered = str(value).strip().lower()
            if lowered not in ENUM_LIMITS[key]:
                raise ValueError(f"{key} must be one of {sorted(ENUM_LIMITS[key])}, got {value!r}")
            out[key] = lowered
            continue
        if key == "pair_edge_leverage":
            out[key] = _coerce_bool(value)
            continue
        if key in NUMERIC_LIMITS:
            lo, hi = NUMERIC_LIMITS[key]
            if isinstance(lo, int) and isinstance(hi, int):
                num: Any = int(value)
            else:
                num = float(value)
            if num < lo or num > hi:
                raise ValueError(f"{key}={num!r} outside allowed range [{lo}, {hi}]")
            out[key] = num
            continue
        out[key] = value
    return out


def _normalize_pair_token(value: Any) -> str:
    raw = str(value or "").strip().upper()
    if not raw:
        return ""
    if ":" in raw:
        raw = raw.split(":", 1)[0]
    raw = raw.replace("-", "/")
    if "/" not in raw and raw.endswith("USDT"):
        raw = f"{raw[:-4]}/USDT"
    return raw


def _signature_number(key: str, value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and key in NUMERIC_LIMITS and isinstance(NUMERIC_LIMITS[key][0], int):
        return int(value)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return value
    if not math.isfinite(numeric):
        return value
    if key == "risk_per_trade":
        return round(numeric, 5)
    if (
        "mom" in key
        or "atr" in key
        or "score_z" in key
        or key in {"min_abs_score_z", "score_threshold"}
    ):
        return round(numeric, 3)
    return round(numeric, 6)


def rank_profile_signature(profile: Mapping[str, Any], *, default_n: int = 50) -> str:
    """Canonical near-duplicate signature for rank-profile search knobs."""
    normalized = normalize_rank_profile(profile, default_n=default_n)
    canonical: dict[str, Any] = {}
    for key in sorted(normalized):
        value = normalized[key]
        if key.endswith("pairs") or key in {"exclude_pairs"}:
            if isinstance(value, str):
                pairs = [_normalize_pair_token(p) for p in value.split(",")]
            elif isinstance(value, Sequence):
                pairs = [_normalize_pair_token(p) for p in value]
            else:
                pairs = [_normalize_pair_token(value)]
            canonical[key] = sorted({p for p in pairs if p})
        elif isinstance(value, float) or key in NUMERIC_LIMITS:
            canonical[key] = _signature_number(key, value)
        else:
            canonical[key] = value
    return json.dumps(canonical, sort_keys=True, separators=(",", ":"), default=str)


def _validate_candidate_state_matches_profile(profile: Mapping[str, Any]) -> None:
    raw_state = str(profile.get("candidate_state") or "").strip()
    if not raw_state:
        return
    path = repo_paths.resolve_repo_path(raw_state)
    if not path.exists():
        return
    try:
        payload = load_json(path, {})
    except Exception:
        return
    if not isinstance(payload, Mapping):
        return
    manifest = {
        "timeframe": payload.get("timeframe"),
        "evaluation_lane": payload.get("evaluation_lane") or payload.get("lane"),
    }
    if isinstance(payload.get("intraday"), Mapping):
        intraday = payload["intraday"]
        manifest["timeframe"] = manifest.get("timeframe") or intraday.get("timeframe")
        manifest["evaluation_lane"] = manifest.get("evaluation_lane") or intraday.get("evaluation_lane")
    ok, reason = manifest_matches_profile(manifest, profile)
    if not ok:
        raise ValueError(reason)


def _coerce_finite_float(value: Any, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if math.isfinite(numeric) else default


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clamp_numeric(key: str, value: Any) -> Any:
    if key not in NUMERIC_LIMITS:
        return value
    lo, hi = NUMERIC_LIMITS[key]
    if isinstance(lo, int) and isinstance(hi, int):
        return int(max(lo, min(hi, int(round(float(value))))))
    return float(max(float(lo), min(float(hi), float(value))))


def _stage_metrics_from_row(row: Mapping[str, Any], stage: str = "search") -> dict[str, Any]:
    windows = row.get("window_metrics") if isinstance(row.get("window_metrics"), Mapping) else {}
    payload = windows.get(stage) if isinstance(windows.get(stage), Mapping) else {}
    for key in ("research_metrics", "metrics"):
        metrics = payload.get(key) if isinstance(payload.get(key), Mapping) else {}
        if metrics:
            return dict(metrics)
    for key in ("research_metrics", "metrics"):
        metrics = row.get(key) if isinstance(row.get(key), Mapping) else {}
        if metrics:
            return dict(metrics)
    return {}


def _search_gate_repair_hints(rows: Sequence[Mapping[str, Any]], config: StrategyLoopConfig) -> dict[str, Any]:
    gates = scaled_gate_values(config, config.search_timerange)
    min_trades = int(gates["min_trades"])
    min_pdd = float(gates["min_profit_over_dd"])
    near_misses: list[dict[str, Any]] = []
    high_trade_low_quality: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        profile = _row_rank_profile(row)
        metrics = _stage_metrics_from_row(row, "search")
        if not profile or not metrics:
            continue
        trades = _coerce_int(metrics.get("trades"), 0)
        pdd = _coerce_finite_float(metrics.get("profit_over_max_drawdown"), 0.0)
        profit = _coerce_finite_float(metrics.get("profit_pct"), 0.0)
        drawdown = _coerce_finite_float(metrics.get("max_drawdown_pct"), 0.0)
        compact = {
            "iteration": row.get("iteration"),
            "name": (_row_candidate(row) or {}).get("name"),
            "trades": trades,
            "trades_gap": max(0, min_trades - trades),
            "profit_over_max_drawdown": pdd,
            "profit_pct": profit,
            "max_drawdown_pct": drawdown,
            "rank_profile": dict(profile),
        }
        if trades < min_trades and pdd >= min_pdd and profit > 0:
            near_misses.append(compact)
        if trades >= min_trades and pdd < min_pdd:
            high_trade_low_quality.append(compact)
    near_misses.sort(key=lambda item: (int(item["trades_gap"]), -float(item["profit_over_max_drawdown"])))
    high_trade_low_quality.sort(key=lambda item: (float(item["profit_over_max_drawdown"]), -int(item["trades"])))
    hints: list[str] = []
    if near_misses:
        hints.append(
            "Best search near-misses are profitable with adequate profit/drawdown but too few trades; prefer tiny participation increases."
        )
    if high_trade_low_quality:
        hints.append(
            "High-trade attempts cleared trade count but damaged profit/drawdown; combine participation repairs with quality controls instead of broad cadence cuts."
        )
    return {
        "search_gates": gates,
        "near_miss_trade_gate": near_misses[:5],
        "high_trade_low_quality": high_trade_low_quality[:5],
        "recommended_repair_order": [
            "lower min_abs_score_z by 0.01-0.03 from a high-P/DD anchor",
            "try top_k+1 with a conservative z threshold before cutting rebalance_hours",
            "loosen short momentum entry filters by only 0.002-0.006 when trades are just below gate",
            "avoid repeating pure rebalance_hours=4 if it already cleared trades but failed profit/drawdown",
        ],
        "notes": hints,
    }


def _candidate_name(raw: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", raw).strip("_").lower()
    return safe[:72] or "rank_profile_repair"


def _profile_with_changes(base: Mapping[str, Any], changes: Mapping[str, Any], *, default_n: int) -> dict[str, Any]:
    profile = dict(base)
    for key, value in changes.items():
        if value is None:
            continue
        profile[key] = _clamp_numeric(key, value)
    return normalize_rank_profile(profile, default_n=default_n)


def build_rank_profile_repair_queue(
    baseline_profile: Mapping[str, Any],
    config: StrategyLoopConfig,
    *,
    rows: Sequence[Mapping[str, Any]] = (),
    structured: bool = False,
) -> list[dict[str, Any]]:
    """Build deterministic rank-profile repairs for common near-gate failures."""
    if not baseline_profile:
        return []
    base = normalize_rank_profile(baseline_profile, default_n=config.n)
    z = _coerce_finite_float(base.get("min_abs_score_z"), 1.5)
    top_k = _coerce_int(base.get("top_k"), 2)
    rebalance = _coerce_int(base.get("rebalance_hours"), 8)
    short_max_24h = _coerce_finite_float(base.get("short_max_mom_24h"), 0.04)
    short_market_24h = _coerce_finite_float(base.get("short_max_market_mom_24h"), 0.05)
    exit_mom_24h = _coerce_finite_float(base.get("short_exit_mom_24h"), 0.04)
    exit_market_24h = _coerce_finite_float(base.get("short_exit_market_mom_24h"), 0.04)

    search_mode = "structured_explore" if structured else "local_exploit"
    queue_specs: list[tuple[str, str, dict[str, Any], str]] = [
        ("z150_micro_trade_repair", "entry_threshold_trade_count_repair", {"min_abs_score_z": z - 0.01}, "tiny participation increase near the baseline threshold"),
        ("z149_trade_repair", "entry_threshold_trade_count_repair", {"min_abs_score_z": z - 0.02}, "small participation increase intended to clear a 1-5 trade deficit"),
        ("z148_trade_repair", "entry_threshold_trade_count_repair", {"min_abs_score_z": z - 0.03}, "moderate z-threshold repair before changing cadence or leverage"),
        ("z147_trade_repair", "entry_threshold_trade_count_repair", {"min_abs_score_z": z - 0.04}, "stronger z-threshold repair while preserving all other baseline filters"),
        ("top3_z149_quality_repair", "topk_diversification_trade_gate", {"top_k": top_k + 1, "min_abs_score_z": z - 0.02}, "add one rank slot while keeping entry quality close to baseline"),
        ("top3_z148_quality_repair", "topk_diversification_trade_gate", {"top_k": top_k + 1, "min_abs_score_z": z - 0.03}, "combine mild diversification with a modest z repair"),
        ("rebalance5_z149_balanced_repair", "cadence_threshold_balance", {"rebalance_hours": max(1, rebalance - 1), "min_abs_score_z": z - 0.02}, "use a slight cadence increase plus conservative z repair instead of a 4h turnover jump"),
        ("top3_rebalance5_trade_repair", "topk_cadence_balance", {"top_k": top_k + 1, "rebalance_hours": max(1, rebalance - 1)}, "combine the safer top_k and 5h cadence variants"),
        ("short_mom040_entry_repair", "entry_momentum_filter_repair", {"short_max_mom_24h": short_max_24h + 0.002}, "loosen pair momentum entry filter minimally to add borderline shorts"),
        ("short_mom042_entry_repair", "entry_momentum_filter_repair", {"short_max_mom_24h": short_max_24h + 0.004}, "loosen pair momentum entry filter while preserving ATR and z controls"),
        ("market_mom055_entry_repair", "market_momentum_filter_repair", {"short_max_market_mom_24h": short_market_24h + 0.005}, "allow slightly more market momentum when pair-level score is strong"),
        ("exit_mom035_market035_repair", "exit_filter_tightening", {"short_exit_mom_24h": exit_mom_24h - 0.005, "short_exit_market_mom_24h": exit_market_24h - 0.005}, "close adverse short exposure sooner to create trades without loosening entry quality"),
    ]

    hints = _search_gate_repair_hints(rows, config) if rows else {}
    high_trade_low_quality = hints.get("high_trade_low_quality") if isinstance(hints, Mapping) else []
    if high_trade_low_quality:
        queue_specs.extend(
            [
                ("quality_z152_after_churn", "quality_repair_after_churn", {"min_abs_score_z": z + 0.01, "top_k": top_k + 1}, "restore quality after high-trade low-P/DD attempts"),
                ("rebalance5_quality_z150", "quality_cadence_balance", {"rebalance_hours": max(1, rebalance - 1), "min_abs_score_z": z - 0.01}, "avoid the 4h churn failure while nudging trades upward"),
            ]
        )

    candidates: list[dict[str, Any]] = []
    seen_profiles: set[str] = set()
    for raw_name, family, changes, tradeoff in queue_specs:
        try:
            profile = _profile_with_changes(base, changes, default_n=config.n)
            signature = rank_profile_signature(profile, default_n=config.n)
        except Exception:
            continue
        if signature in seen_profiles or profile == base:
            continue
        seen_profiles.add(signature)
        structural_change = any(key in STRUCTURAL_RANK_KEYS and profile.get(key) != base.get(key) for key in profile)
        candidates.append(
            {
                "candidate_type": CANDIDATE_RANK_PROFILE,
                "name": _candidate_name(raw_name),
                "description": f"Controller-generated repair from optimized baseline: {tradeoff}.",
                "metadata": {
                    "source": "controller_rank_profile_repair_queue",
                    "search_mode": "structured_explore" if structured or structural_change and raw_name.startswith("top3_") else search_mode,
                    "parent_anchor": "optimized_baseline",
                    "hypothesis_family": family,
                    "expected_tradeoff": tradeoff,
                    "changed_keys": sorted(changes),
                },
                "rank_profile": profile,
            }
        )
    return candidates


def _safe_relative_path(base: Path, raw: str | Path) -> Path:
    candidate = Path(raw)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (base / candidate).resolve()
    try:
        resolved.relative_to(base.resolve())
    except ValueError as exc:
        raise ValueError(f"path escapes iteration workspace: {raw}") from exc
    return resolved


def validate_strategy_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ValueError(f"strategy file does not exist: {path}")
    if path.suffix != ".py":
        raise ValueError(f"strategy file must be .py: {path}")
    source = path.read_text(encoding="utf-8")
    repo_root_s = str(repo_paths.REPO_ROOT)
    if repo_root_s in source:
        raise ValueError("strategy must not hard-code absolute repository paths")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        raise ValueError(f"strategy syntax error: {exc}") from exc

    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module.split(".", 1)[0])
    banned = sorted(set(imports) & BANNED_STRATEGY_IMPORTS)
    if banned:
        raise ValueError(f"strategy imports banned modules: {banned}")

    class_names: list[str] = []
    istrategy_classes: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        class_names.append(node.name)
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id == "IStrategy":
                istrategy_classes.append(node.name)
            elif isinstance(base, ast.Attribute) and base.attr == "IStrategy":
                istrategy_classes.append(node.name)
    if not istrategy_classes:
        raise ValueError("strategy must define a class inheriting from IStrategy")
    if "freqtrade" not in imports:
        raise ValueError("strategy must import freqtrade.strategy.IStrategy")
    if "read_feather" not in source:
        raise ValueError("strategy must read exported rank signal feather files")
    required_signal_cols = ("rp_target_weight", "rp_side")
    missing_cols = [col for col in required_signal_cols if col not in source]
    if missing_cols:
        raise ValueError(f"strategy must consume exported rank columns: {missing_cols}")
    if "RP_SIGNAL_DIR" not in source and "RP_TAG" not in source:
        raise ValueError("strategy must support RP_SIGNAL_DIR or RP_TAG signal discovery")
    return {
        "path": str(path),
        "class_names": class_names,
        "istrategy_classes": istrategy_classes,
        "imports": sorted(set(imports)),
    }


def validate_candidate(candidate_path: str | Path, *, default_n: int = 50) -> dict[str, Any]:
    path = Path(candidate_path).resolve()
    if not path.exists():
        raise ValueError(f"candidate.json does not exist: {path}")
    workspace = path.parent.resolve()
    payload = load_json(path, {})
    if not isinstance(payload, dict):
        raise ValueError("candidate.json must contain an object")

    ctype = str(payload.get("candidate_type") or payload.get("type") or "").strip().lower()
    if ctype not in CANDIDATE_TYPES:
        raise ValueError(f"candidate_type must be one of {sorted(CANDIDATE_TYPES)}, got {ctype!r}")
    name = str(payload.get("name") or path.parent.name).strip() or path.parent.name

    rank_profile_raw = payload.get("rank_profile") or payload.get("profile") or payload.get("params") or {}
    if not rank_profile_raw:
        rank_profile_raw = {k: v for k, v in payload.items() if k in RANK_PROFILE_KEYS}
    if ctype == CANDIDATE_RANK_PROFILE and not isinstance(rank_profile_raw, Mapping):
        raise ValueError("rank_profile candidate requires an object rank_profile")
    if rank_profile_raw and not isinstance(rank_profile_raw, Mapping):
        raise ValueError("rank_profile must be an object")
    rank_profile = normalize_rank_profile(rank_profile_raw, default_n=default_n) if rank_profile_raw or ctype == CANDIDATE_RANK_PROFILE else {}
    if ctype == CANDIDATE_RANK_PROFILE:
        _validate_candidate_state_matches_profile(rank_profile)

    strategy_info: Optional[dict[str, Any]] = None
    strategy_path_raw = payload.get("strategy_path") or ("strategy.py" if ctype == CANDIDATE_FREQTRADE_STRATEGY else "")
    if ctype == CANDIDATE_FREQTRADE_STRATEGY:
        strategy_path = _safe_relative_path(workspace, strategy_path_raw)
        strategy_info = validate_strategy_file(strategy_path)
    elif strategy_path_raw:
        raise ValueError("rank_profile candidates must not include strategy_path or write strategy.py")

    normalized = {
        "candidate_type": ctype,
        "name": name,
        "description": str(payload.get("description") or ""),
        "rank_profile": rank_profile,
        "strategy_path": strategy_path_raw if strategy_path_raw else None,
        "metadata": payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
        "path": str(path),
        "workspace": str(workspace),
        "strategy_validation": strategy_info,
    }
    return normalized


def _resolve_candidate_state_value(value: Any, fallback: Optional[Path]) -> Optional[str]:
    raw = str(value or "").strip()
    if raw:
        return str(repo_paths.resolve_repo_path(raw))
    return str(fallback) if fallback else None


def _rank_kwargs(
    profile: Mapping[str, Any],
    config: StrategyLoopConfig,
    *,
    candidate_state: Optional[Path],
    tag: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> dict[str, Any]:
    candidate_params = dict(profile)
    baseline_params = _baseline_rank_profile(config)
    params = dict(baseline_params)
    params.update(candidate_params)

    candidate_state_raw = (
        candidate_params.get("candidate_state")
        or str(config.candidate_state or "").strip()
        or baseline_params.get("candidate_state")
    )
    if "recompute_corr" in candidate_params:
        recompute_corr = _coerce_bool(candidate_params.get("recompute_corr"))
    elif config.recompute_corr is not None:
        recompute_corr = bool(config.recompute_corr)
    elif "recompute_corr" in baseline_params:
        recompute_corr = _coerce_bool(baseline_params.get("recompute_corr"))
    else:
        recompute_corr = True

    params.pop("candidate_state", None)
    params.pop("recompute_corr", None)
    n = int(params.pop("n", config.n))
    min_abs_score_z = params.pop("score_threshold", params.pop("min_abs_score_z", 1.5))
    return {
        "tag": tag,
        "venue": config.venue,
        "risk_profile": config.risk_profile,
        "n": n,
        "start": start or config.start,
        "end": end or config.end,
        "timeframe": params.pop("timeframe", config.timeframe),
        "data_venue": params.pop("data_venue", config.data_venue),
        "top_k": params.pop("top_k", 2),
        "gross_cap": params.pop("gross_cap", 2.0),
        "net_cap": params.pop("net_cap", None),
        "single_pair_cap": params.pop("single_pair_cap", None),
        "side_mode": params.pop("side_mode", "short"),
        "min_abs_score_z": min_abs_score_z,
        "rebalance_hours": params.pop("rebalance_hours", 8),
        "rebalance_minutes": params.pop("rebalance_minutes", None),
        "risk_per_trade": params.pop("risk_per_trade", 0.08),
        "leverage_cap": params.pop("leverage_cap", 5.0),
        "edge_mode": params.pop("edge_mode", "rolling_ic"),
        "edge_lookback_hours": params.pop("edge_lookback_hours", 336),
        "edge_min_periods": params.pop("edge_min_periods", 168),
        "edge_deadband": params.pop("edge_deadband", 0.005),
        "pair_edge_leverage": params.pop("pair_edge_leverage", None),
        "pair_edge_deadband": params.pop("pair_edge_deadband", None),
        "pair_edge_strong_ic": params.pop("pair_edge_strong_ic", None),
        "pair_edge_very_strong_ic": params.pop("pair_edge_very_strong_ic", None),
        "pair_edge_weak_cap": params.pop("pair_edge_weak_cap", None),
        "regime_mode": params.pop("regime_mode", None),
        "regime_min_edge_ic": params.pop("regime_min_edge_ic", None),
        "regime_min_pair_edge_ic": params.pop("regime_min_pair_edge_ic", None),
        "regime_min_pair_count": params.pop("regime_min_pair_count", None),
        "regime_short_max_market_mom_24h": params.pop("regime_short_max_market_mom_24h", None),
        "regime_short_max_market_mom_72h": params.pop("regime_short_max_market_mom_72h", None),
        "regime_max_market_atr_pct": params.pop("regime_max_market_atr_pct", None),
        "short_max_mom_24h": params.pop("short_max_mom_24h", None),
        "short_max_mom_72h": params.pop("short_max_mom_72h", None),
        "long_min_mom_24h": params.pop("long_min_mom_24h", None),
        "max_entry_atr_pct": params.pop("max_entry_atr_pct", None),
        "short_max_market_mom_24h": params.pop("short_max_market_mom_24h", None),
        "short_max_market_mom_72h": params.pop("short_max_market_mom_72h", None),
        "short_max_market_ma_gap": params.pop("short_max_market_ma_gap", None),
        "short_exit_mom_24h": params.pop("short_exit_mom_24h", None),
        "short_exit_mom_72h": params.pop("short_exit_mom_72h", None),
        "short_exit_market_mom_24h": params.pop("short_exit_market_mom_24h", None),
        "short_exit_market_ma_gap": params.pop("short_exit_market_ma_gap", None),
        "exclude_pairs": params.pop("exclude_pairs", None),
        "candidate_state": _resolve_candidate_state_value(candidate_state_raw, candidate_state),
        "recompute_corr": bool(recompute_corr),
    }


def score_backtest_result(
    metrics: Mapping[str, Any],
    *,
    min_trades: int = 80,
    max_drawdown_pct: float = 25.0,
    min_profit_over_dd: float = 1.2,
    target_profit_pct: float = 25.0,
) -> dict[str, Any]:
    profit_pct = float(metrics.get("total_return_pct") or metrics.get("profit_total_pct") or metrics.get("profit_pct") or 0.0)
    max_dd_pct = float(metrics.get("max_drawdown_pct") or 0.0)
    trades = int(metrics.get("trades") or metrics.get("total_trades") or 0)
    profit_over_dd = float(
        metrics.get("profit_over_max_drawdown")
        if metrics.get("profit_over_max_drawdown") is not None
        else profit_pct / max(max_dd_pct, 1e-9)
    )
    simulated_liquidations = int(metrics.get("simulated_liquidations") or 0)
    liquidation_rejects = int(metrics.get("liquidation_rejects") or 0)
    avg_turnover = float(metrics.get("avg_turnover") or 0.0)
    risk_mode_counts = metrics.get("risk_mode_counts") or {}
    kill_count = 0
    if isinstance(risk_mode_counts, Mapping):
        kill_count = sum(int(v or 0) for k, v in risk_mode_counts.items() if str(k) not in {"normal", "None", ""})
    concentration = float(metrics.get("dominant_pair_profit_share") or metrics.get("max_pair_profit_share") or 0.0)

    violations: list[str] = []
    if simulated_liquidations > 0:
        violations.append(f"simulated_liquidations={simulated_liquidations} > 0")
    if liquidation_rejects > 0:
        violations.append(f"liquidation_rejects={liquidation_rejects} > 0")
    if max_dd_pct > float(max_drawdown_pct):
        violations.append(f"max_drawdown_pct={max_dd_pct:.4g} > {max_drawdown_pct:.4g}")
    if trades < int(min_trades):
        violations.append(f"trades={trades} < {int(min_trades)}")
    if profit_over_dd < float(min_profit_over_dd):
        violations.append(f"profit_over_max_drawdown={profit_over_dd:.4g} < {min_profit_over_dd:.4g}")

    constraints_ok = not violations
    score = 1_000_000.0 if constraints_ok else 0.0
    score += profit_over_dd * 1000.0
    score += profit_pct * 10.0
    score += min(trades, 1000) * 0.05
    score -= max_dd_pct * 5.0
    score -= simulated_liquidations * 500.0
    score -= liquidation_rejects * 10.0
    score -= kill_count * 0.5
    score -= max(0.0, avg_turnover - 2.0) * 20.0
    score -= max(0.0, concentration - 0.5) * 200.0
    if profit_pct > 0:
        score += 1000.0
    elif profit_pct < 0:
        score -= 1000.0
    if profit_pct >= target_profit_pct:
        score += 25.0
    if violations:
        score -= len(violations) * 100.0

    return {
        "score": float(score),
        "constraints_ok": constraints_ok,
        "violations": violations,
        "metrics": {
            "profit_pct": profit_pct,
            "max_drawdown_pct": max_dd_pct,
            "trades": trades,
            "profit_over_max_drawdown": profit_over_dd,
            "simulated_liquidations": simulated_liquidations,
            "liquidation_rejects": liquidation_rejects,
            "avg_turnover": avg_turnover,
            "kill_mode_count": kill_count,
            "dominant_pair_profit_share": concentration,
        },
        "promotion_reason": "passes hard gates" if constraints_ok else "; ".join(violations),
    }


def _score_quality(scored: Mapping[str, Any]) -> float:
    score = float(scored.get("score") or 0.0)
    return score - (1_000_000.0 if bool(scored.get("constraints_ok")) else 0.0)


def _freqtrade_metrics_from_backtest(backtest: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    freqtrade = backtest.get("freqtrade_backtest") if isinstance(backtest.get("freqtrade_backtest"), Mapping) else {}
    if not isinstance(freqtrade, Mapping):
        return {}, {}
    raw_metrics = freqtrade.get("metrics") if isinstance(freqtrade.get("metrics"), Mapping) else {}
    metrics = dict(raw_metrics)
    summary = freqtrade.get("summary") if isinstance(freqtrade.get("summary"), Mapping) else {}
    if summary:
        metrics.setdefault("profit_pct", summary.get("profit_total_pct"))
        metrics.setdefault("max_drawdown_pct", summary.get("max_drawdown_pct") or summary.get("max_drawdown_account"))
        metrics.setdefault("trades", summary.get("trades") or summary.get("total_trades"))
        profit = metrics.get("profit_pct")
        drawdown = metrics.get("max_drawdown_pct")
        if metrics.get("profit_over_max_drawdown") is None and profit is not None and drawdown is not None:
            metrics["profit_over_max_drawdown"] = float(profit) / max(float(drawdown), 1e-9)
    if freqtrade.get("ok") is not None:
        metrics["ok"] = bool(freqtrade.get("ok"))
    return metrics, dict(freqtrade)


def _score_freqtrade_stage(
    backtest: Mapping[str, Any],
    *,
    min_trades: int,
    max_drawdown_pct: float,
    min_profit_over_dd: float,
    target_profit_pct: float,
) -> dict[str, Any]:
    metrics, freqtrade = _freqtrade_metrics_from_backtest(backtest)
    if freqtrade and freqtrade.get("skipped"):
        reason = str(freqtrade.get("reason") or "freqtrade_backtest skipped")
        return {
            "score": FAILED_ITERATION_SCORE,
            "constraints_ok": False,
            "violations": [reason],
            "metrics": {},
            "promotion_reason": reason,
        }
    scored = score_backtest_result(
        metrics,
        min_trades=min_trades,
        max_drawdown_pct=max_drawdown_pct,
        min_profit_over_dd=min_profit_over_dd,
        target_profit_pct=target_profit_pct,
    )
    if not freqtrade:
        scored["constraints_ok"] = False
        scored["violations"] = ["freqtrade_backtest missing", *list(scored.get("violations") or [])]
        scored["score"] = _score_quality(scored) - 1_000_000.0
        scored["promotion_reason"] = "; ".join(scored["violations"])
        return scored
    if not bool(freqtrade.get("ok")):
        reason = str(freqtrade.get("reason") or freqtrade.get("error") or "freqtrade_backtest failed")
        scored["constraints_ok"] = False
        scored["violations"] = [reason, *list(scored.get("violations") or [])]
        scored["score"] = _score_quality(scored) - 1_000_000.0
        scored["promotion_reason"] = "; ".join(scored["violations"])
    return scored


def score_strategy_loop_backtest(
    backtest: Mapping[str, Any],
    config: StrategyLoopConfig,
    *,
    gates: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    gate_values = gates or {}
    min_trades = int(gate_values.get("min_trades", config.min_trades))
    max_drawdown_pct = float(gate_values.get("max_drawdown_pct", config.max_drawdown_pct))
    min_profit_over_dd = float(gate_values.get("min_profit_over_dd", config.min_profit_over_dd))
    target_profit_pct = float(gate_values.get("target_profit_pct", config.target_profit_pct))
    research = score_backtest_result(
        backtest,
        min_trades=min_trades,
        max_drawdown_pct=max_drawdown_pct,
        min_profit_over_dd=min_profit_over_dd,
        target_profit_pct=target_profit_pct,
    )
    freqtrade = _score_freqtrade_stage(
        backtest,
        min_trades=min_trades,
        max_drawdown_pct=max_drawdown_pct,
        min_profit_over_dd=min_profit_over_dd,
        target_profit_pct=target_profit_pct,
    )

    research_ok = bool(research.get("constraints_ok"))
    freqtrade_ok = bool(freqtrade.get("constraints_ok"))
    research_score = float(research.get("score") or 0.0)
    freqtrade_score = float(freqtrade.get("score") or 0.0)
    composite_violations = [f"research: {v}" for v in research.get("violations") or []]
    composite_violations.extend(f"freqtrade: {v}" for v in freqtrade.get("violations") or [])
    composite_ok = research_ok and freqtrade_ok
    composite_score = (
        2_000_000.0 + _score_quality(freqtrade) + _score_quality(research) * 0.001
        if composite_ok
        else -1_000_000.0
        + _score_quality(freqtrade)
        + _score_quality(research) * 0.001
        - len(composite_violations) * 1000.0
    )

    if config.score_mode == SCORE_FREQTRADE:
        selected_score = freqtrade_score
        selected_ok = freqtrade_ok
        selected_violations = list(freqtrade.get("violations") or [])
        selection_reason = "score_mode=freqtrade selected fixed Freqtrade metrics"
        selected_metrics = dict(freqtrade.get("metrics") or {})
    elif config.score_mode == SCORE_COMPOSITE:
        selected_score = composite_score
        selected_ok = composite_ok
        selected_violations = composite_violations
        selection_reason = "score_mode=composite requires research and fixed Freqtrade gates; Freqtrade quality is primary"
        selected_metrics = dict(freqtrade.get("metrics") or {})
    else:
        selected_score = research_score
        selected_ok = research_ok
        selected_violations = list(research.get("violations") or [])
        selection_reason = "score_mode=research selected research rank-backtest metrics"
        selected_metrics = dict(research.get("metrics") or {})

    return {
        "score": float(selected_score),
        "constraints_ok": bool(selected_ok),
        "violations": selected_violations,
        "metrics": dict(research.get("metrics") or {}),
        "selected_metrics": selected_metrics,
        "research_metrics": dict(research.get("metrics") or {}),
        "freqtrade_metrics": dict(freqtrade.get("metrics") or {}),
        "research_evaluation": research,
        "freqtrade_evaluation": freqtrade,
        "score_components": {
            "score_mode": config.score_mode,
            "gate_min_trades": min_trades,
            "gate_target_profit_pct": target_profit_pct,
            "gate_window_days": gate_values.get("window_days"),
            "gate_full_days": gate_values.get("full_days"),
            "research_score": research_score,
            "freqtrade_score": freqtrade_score,
            "composite_score": float(composite_score),
            "research_constraints_ok": research_ok,
            "freqtrade_constraints_ok": freqtrade_ok,
            "composite_constraints_ok": composite_ok,
            "selection_reason": selection_reason,
        },
        "promotion_reason": "passes selected hard gates" if selected_ok else "; ".join(selected_violations),
    }


def _score_lean_result(
    lean_metrics: Mapping[str, Any],
    lean_analysis: Optional[Mapping[str, Any]],
    config: "StrategyLoopConfig",
) -> float:
    """Compute a score from LEAN backtest metrics and time-period analysis."""
    total_return = float(lean_metrics.get("total_return") or 0.0)
    profit_pct = total_return * 100.0
    max_dd = float(lean_metrics.get("max_drawdown") or 0.0)
    max_dd_pct = max_dd * 100.0
    profit_over_dd = float(lean_metrics.get("profit_over_max_drawdown") or 0.0)
    trades = float(lean_metrics.get("trades") or 0.0)

    score = 0.0
    score += profit_pct * 10.0
    score -= max_dd_pct * 5.0
    score += profit_over_dd * 1000.0
    score += min(trades, 1000) * 0.05
    if profit_pct > 0:
        score += 1000.0
    elif profit_pct < 0:
        score -= 1000.0
    if profit_pct >= float(getattr(config, "target_profit_pct", 25.0)):
        score += 25.0

    # Time-period adjustments from lean_analysis
    if isinstance(lean_analysis, Mapping):
        regime = lean_analysis.get("regime_segments") if isinstance(lean_analysis.get("regime_segments"), Mapping) else {}
        dd_episodes = lean_analysis.get("drawdown_episodes") if isinstance(lean_analysis.get("drawdown_episodes"), list) else []
        pair_contrib = lean_analysis.get("pair_contribution") if isinstance(lean_analysis.get("pair_contribution"), Mapping) else {}

        consecutive_loss = int(regime.get("consecutive_loss_months") or 0)
        if consecutive_loss >= 3:
            score -= 500.0
        worst_month_ret = float((regime.get("worst_month") or {}).get("return_pct") or 0.0)
        if worst_month_ret < -20.0:
            score -= 300.0
        pos_pct = float(regime.get("positive_month_pct") or 0.0)
        if pos_pct >= 60.0:
            score += 500.0

        herfindahl = pair_contrib.get("herfindahl_index")
        if herfindahl is not None and float(herfindahl) > 0.5:
            score -= 200.0

        total_dd_depth = sum(abs(float(ep.get("depth_pct") or 0.0)) for ep in dd_episodes)
        if total_dd_depth > 50.0:
            score -= 500.0

    return float(score)


def apply_lean_score_blend(
    evaluation: dict[str, Any],
    config: "StrategyLoopConfig",
) -> None:
    """Blend LEAN score into evaluation in-place. Must be called after lean_gate is loaded."""
    lean_gate = evaluation.get("lean_gate")
    if not isinstance(lean_gate, Mapping):
        # No LEAN data: penalize rank score to discourage LEAN-free candidates in leaderboard
        rank_score = float(evaluation.get("score") or float("-inf"))
        if math.isfinite(rank_score) and _lean_gate_active(config):
            evaluation["score"] = rank_score * 0.5
            (evaluation.setdefault("score_components", {}))["lean_penalty"] = "no_lean_data_0.5x"
        return

    lean_metrics = lean_gate.get("lean_metrics")
    if not isinstance(lean_metrics, Mapping):
        return

    lean_analysis = evaluation.get("lean_analysis")
    lean_score = _score_lean_result(lean_metrics, lean_analysis, config)
    rank_score = float(evaluation.get("score") or 0.0)
    w = float(getattr(config, "score_lean_weight", 0.7))
    w = max(0.0, min(1.0, w))
    blended = (1.0 - w) * rank_score + w * lean_score
    evaluation["score"] = float(blended)
    sc = evaluation.setdefault("score_components", {})
    sc["lean_score"] = float(lean_score)
    sc["rank_score_pre_blend"] = float(rank_score)
    sc["blended_score"] = float(blended)
    sc["score_lean_weight"] = float(w)


def _lean_analysis_summary(lean_analysis: Optional[Any]) -> dict[str, Any]:
    """Extract compact summary from lean_analysis dict for leaderboard rows."""
    if not isinstance(lean_analysis, Mapping):
        return {}
    regime = lean_analysis.get("regime_segments") if isinstance(lean_analysis.get("regime_segments"), Mapping) else {}
    dd_episodes = lean_analysis.get("drawdown_episodes") if isinstance(lean_analysis.get("drawdown_episodes"), list) else []
    pair_contrib = lean_analysis.get("pair_contribution") if isinstance(lean_analysis.get("pair_contribution"), Mapping) else {}
    worst_month = regime.get("worst_month") or {}
    best_month = regime.get("best_month") or {}
    top_dd = dd_episodes[0] if dd_episodes else {}
    return {
        "worst_month": {"period": worst_month.get("period"), "return_pct": worst_month.get("return_pct")},
        "best_month": {"period": best_month.get("period"), "return_pct": best_month.get("return_pct")},
        "consecutive_loss_months": regime.get("consecutive_loss_months"),
        "positive_month_pct": regime.get("positive_month_pct"),
        "max_drawdown_episode_pct": top_dd.get("depth_pct"),
        "max_drawdown_recovery_days": top_dd.get("recovery_days"),
        "herfindahl_index": pair_contrib.get("herfindahl_index"),
        "top_winners": (pair_contrib.get("top_winners") or [])[:3],
        "top_losers": (pair_contrib.get("top_losers") or [])[:3],
    }


def score_research_only_window(
    backtest: Mapping[str, Any],
    config: StrategyLoopConfig,
    *,
    gates: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Score a research-only discovery window without inventing Freqtrade failures."""
    gate_values = gates or {}
    min_trades = int(gate_values.get("min_trades", config.min_trades))
    max_drawdown_pct = float(gate_values.get("max_drawdown_pct", config.max_drawdown_pct))
    min_profit_over_dd = float(gate_values.get("min_profit_over_dd", config.min_profit_over_dd))
    target_profit_pct = float(gate_values.get("target_profit_pct", config.target_profit_pct))
    research = score_backtest_result(
        backtest,
        min_trades=min_trades,
        max_drawdown_pct=max_drawdown_pct,
        min_profit_over_dd=min_profit_over_dd,
        target_profit_pct=target_profit_pct,
    )
    research_score = float(research.get("score") or 0.0)
    research_ok = bool(research.get("constraints_ok"))
    return {
        "score": research_score,
        "constraints_ok": research_ok,
        "violations": list(research.get("violations") or []),
        "metrics": dict(research.get("metrics") or {}),
        "selected_metrics": dict(research.get("metrics") or {}),
        "research_metrics": dict(research.get("metrics") or {}),
        "freqtrade_metrics": {},
        "research_evaluation": research,
        "freqtrade_evaluation": {"skipped": True, "reason": "research-only search window"},
        "score_components": {
            "score_mode": SCORE_RESEARCH,
            "gate_min_trades": min_trades,
            "gate_target_profit_pct": target_profit_pct,
            "gate_window_days": gate_values.get("window_days"),
            "gate_full_days": gate_values.get("full_days"),
            "research_score": research_score,
            "freqtrade_score": None,
            "composite_score": research_score,
            "research_constraints_ok": research_ok,
            "freqtrade_constraints_ok": None,
            "composite_constraints_ok": research_ok,
            "selection_reason": "research-only discovery/search window; fixed Freqtrade is evaluated on validation/blind only",
        },
        "promotion_reason": "passes research search gates" if research_ok else "; ".join(research.get("violations") or []),
    }


def _stage_window_metrics(stage_result: Mapping[str, Any], evaluation: Mapping[str, Any]) -> dict[str, Any]:
    signals = stage_result.get("signals")
    signal_dir = ""
    if isinstance(signals, str):
        signal_dir = _as_repo_meta(Path(signals).parent) if signals else ""
    elif isinstance(signals, Mapping):
        all_path = signals.get("all")
        signal_dir = _as_repo_meta(Path(str(all_path)).parent) if all_path else ""
    return {
        "timerange": stage_result.get("timerange"),
        "start": stage_result.get("start"),
        "end": stage_result.get("end"),
        "constraints_ok": evaluation.get("constraints_ok"),
        "score": evaluation.get("score"),
        "metrics": evaluation.get("selected_metrics") or evaluation.get("metrics") or {},
        "research_metrics": evaluation.get("research_metrics") or evaluation.get("metrics") or {},
        "freqtrade_metrics": evaluation.get("freqtrade_metrics") or {},
        "violations": evaluation.get("violations") or [],
        "signal_dir": signal_dir,
    }


def score_triple_holdout_backtest(backtest: Mapping[str, Any], config: StrategyLoopConfig) -> dict[str, Any]:
    stages = backtest.get("stages") if isinstance(backtest.get("stages"), Mapping) else {}
    search = stages.get("search") if isinstance(stages.get("search"), Mapping) else {}
    validation = stages.get("validation") if isinstance(stages.get("validation"), Mapping) else {}

    search_gates = scaled_gate_values(config, config.search_timerange)
    search_eval = score_research_only_window(search, config, gates=search_gates) if search else {
        "score": FAILED_ITERATION_SCORE,
        "constraints_ok": False,
        "violations": ["search window did not run"],
        "metrics": {},
        "selected_metrics": {},
        "research_metrics": {},
        "freqtrade_metrics": {},
        "score_components": {"score_mode": config.score_mode, "composite_score": FAILED_ITERATION_SCORE},
        "promotion_reason": "search window did not run",
    }

    if not validation:
        try:
            search_near_miss_score = float(search_eval.get("score") or 0.0)
        except (TypeError, ValueError):
            search_near_miss_score = 0.0
        if not math.isfinite(search_near_miss_score):
            search_near_miss_score = 0.0
        search_near_miss_score = min(search_near_miss_score, abs(FAILED_ITERATION_SCORE) - 1_000.0)
        near_miss_score = FAILED_ITERATION_SCORE + search_near_miss_score
        result = {
            "score": near_miss_score,
            "constraints_ok": False,
            "violations": ["validation window skipped because search gates failed"],
            "metrics": dict(search_eval.get("metrics") or {}),
            "selected_metrics": dict(search_eval.get("selected_metrics") or search_eval.get("metrics") or {}),
            "research_metrics": dict(search_eval.get("research_metrics") or search_eval.get("metrics") or {}),
            "freqtrade_metrics": {},
            "research_evaluation": {},
            "freqtrade_evaluation": {},
            "score_components": {
                "score_mode": config.score_mode,
                "research_score": search_eval.get("score"),
                "freqtrade_score": FAILED_ITERATION_SCORE,
                "composite_score": near_miss_score,
                "search_constraints_ok": search_eval.get("constraints_ok"),
                "selection_reason": "triple_holdout uses validation for promotion; validation was skipped, so leaderboard keeps the search near-miss score below any validation pass",
            },
            "promotion_reason": "validation window skipped because search gates failed",
            "window_metrics": {"search": _stage_window_metrics(search, search_eval) if search else {}},
            "selected_window": "validation",
        }
        return result

    validation_gates = scaled_gate_values(config, config.validation_timerange)
    validation_eval = score_strategy_loop_backtest(validation, config, gates=validation_gates)
    validation_eval["window_metrics"] = {
        "search": _stage_window_metrics(search, search_eval) if search else {},
        "validation": _stage_window_metrics(validation, validation_eval),
    }
    validation_eval["selected_window"] = "validation"
    validation_eval["promotion_reason"] = (
        "validation window passed selected hard gates"
        if validation_eval.get("constraints_ok")
        else str(validation_eval.get("promotion_reason") or "validation window failed")
    )
    return validation_eval


def _lower_key_map(row: Mapping[str, Any]) -> dict[str, Any]:
    return {str(k).strip().lower().replace(" ", "_"): v for k, v in row.items()}


def _boolish_false(value: Any) -> bool:
    return str(value or "").strip().lower() in {"", "0", "false", "no", "none", "nan"}


def _numeric_zero(value: Any) -> bool:
    try:
        return abs(float(value or 0.0)) <= 1e-12
    except (TypeError, ValueError):
        return _boolish_false(value)


def parse_lookahead_csv(
    path: str | Path,
    *,
    strategy: str = FIXED_FREQTRADE_STRATEGY,
    min_trades: int = 0,
) -> dict[str, Any]:
    csv_path = Path(path)
    if not csv_path.exists():
        return {"status": VERIFICATION_INCONCLUSIVE, "violations": [f"lookahead csv missing: {csv_path}"], "rows": []}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = [_lower_key_map(row) for row in csv.DictReader(fh)]
    if not rows:
        return {"status": VERIFICATION_INCONCLUSIVE, "violations": ["lookahead csv has no rows"], "rows": []}

    strategy_l = str(strategy or "").strip().lower()
    target: Optional[dict[str, Any]] = None
    for row in rows:
        row_strategy = str(row.get("strategy") or row.get("strategy_name") or row.get("name") or "").strip().lower()
        if not strategy_l or row_strategy == strategy_l:
            target = row
            break
    if target is None:
        return {
            "status": VERIFICATION_FAILED,
            "violations": [f"lookahead csv has no target strategy row for {strategy}"],
            "rows": rows,
        }

    violations: list[str] = []
    has_bias = target.get("has_bias")
    if not _boolish_false(has_bias):
        violations.append(f"has_bias={has_bias}")
    for key in ("biased_entry_signals", "entry_biased_signals", "biased_entries"):
        if key in target and not _numeric_zero(target.get(key)):
            violations.append(f"{key}={target.get(key)}")
            break
    for key in ("biased_exit_signals", "exit_biased_signals", "biased_exits"):
        if key in target and not _numeric_zero(target.get(key)):
            violations.append(f"{key}={target.get(key)}")
            break
    indicators = target.get("biased_indicators") or target.get("biased_indicator") or ""
    if not _boolish_false(indicators):
        violations.append(f"biased_indicators={indicators}")
    if min_trades > 0:
        count_value = (
            target.get("total_signals")
            or target.get("signals")
            or target.get("trades")
            or target.get("total_trades")
            or target.get("entry_signal_count")
            or target.get("entries")
        )
        try:
            count = int(float(count_value or 0))
        except (TypeError, ValueError):
            count = 0
        if count < int(min_trades):
            violations.append(f"signals={count} < min_trades={int(min_trades)}")
    return {
        "status": VERIFICATION_FAILED if violations else VERIFICATION_PASSED,
        "violations": violations,
        "target_row": target,
        "rows": rows,
    }


def parse_recursive_output(path: str | Path) -> dict[str, Any]:
    raw_path = Path(path)
    if not raw_path.exists():
        return {"status": VERIFICATION_INCONCLUSIVE, "violations": [f"recursive artifact missing: {raw_path}"], "rows": []}
    text = raw_path.read_text(encoding="utf-8", errors="ignore")
    if raw_path.suffix.lower() == ".json":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            return {"status": VERIFICATION_INCONCLUSIVE, "violations": [f"recursive json parse failed: {exc}"], "rows": []}
        if isinstance(payload, Mapping):
            rows = payload.get("rows") or payload.get("analysis") or payload.get("results") or []
        else:
            rows = payload
        if not isinstance(rows, list):
            rows = [rows]
        normalized = [_lower_key_map(row) for row in rows if isinstance(row, Mapping)]
    elif raw_path.suffix.lower() == ".csv":
        with raw_path.open("r", encoding="utf-8-sig", newline="") as fh:
            normalized = [_lower_key_map(row) for row in csv.DictReader(fh)]
    else:
        if re.search(r"(not enough|insufficient|too few|sample)", text, flags=re.I):
            return {"status": VERIFICATION_INCONCLUSIVE, "violations": ["recursive log reports insufficient sample"], "rows": []}
        if re.search(r"(recursive|indicator).{0,80}(bias|difference|diff|drift)", text, flags=re.I):
            if re.search(r"(?<![A-Za-z])([1-9]\d*|0\.\d*[1-9]\d*)(?![A-Za-z])", text):
                return {"status": VERIFICATION_FAILED, "violations": ["recursive log reports non-zero differences"], "rows": []}
        return {"status": VERIFICATION_PASSED, "violations": [], "rows": []}

    violations: list[str] = []
    for idx, row in enumerate(normalized):
        for key, value in row.items():
            key_s = str(key).lower()
            if not any(token in key_s for token in ("diff", "difference", "variance", "drift")):
                continue
            if not _numeric_zero(value):
                violations.append(f"row {idx} {key}={value}")
    return {
        "status": VERIFICATION_FAILED if violations else VERIFICATION_PASSED,
        "violations": violations,
        "rows": normalized,
    }


def combine_verification_status(results: Mapping[str, Any]) -> str:
    statuses = []
    for value in results.values():
        if isinstance(value, Mapping):
            status = str(value.get("status") or VERIFICATION_PENDING).lower()
            if status in VERIFICATION_STATUSES:
                statuses.append(status)
    if not statuses:
        return VERIFICATION_PENDING
    if VERIFICATION_FAILED in statuses:
        return VERIFICATION_FAILED
    if VERIFICATION_INCONCLUSIVE in statuses:
        return VERIFICATION_INCONCLUSIVE
    if all(status == VERIFICATION_PASSED for status in statuses):
        return VERIFICATION_PASSED
    return VERIFICATION_PENDING


def _is_full_holdout(config: StrategyLoopConfig) -> bool:
    return config.start <= DEFAULT_START and config.end >= DEFAULT_END


def _lean_gate_active(config: StrategyLoopConfig) -> bool:
    return str(config.lean_gate_mode or LEAN_GATE_OFF).lower() != LEAN_GATE_OFF


def _lean_required_statuses(config: StrategyLoopConfig) -> set[str]:
    raw = str(config.lean_required_status or "ok").strip().lower()
    if raw in {"*", "any", "all"}:
        return {"ok", "partial", "drift"}
    statuses = {item.strip().lower() for item in raw.split(",") if item.strip()}
    result = statuses or {"ok"}
    # skip_signal_load=True always produces "partial" (orders field unavailable);
    # treat partial as equivalent to ok unless caller explicitly excluded it.
    if "ok" in result:
        result = result | {"partial"}
    return result


def _lean_gate_status(evaluation: Mapping[str, Any]) -> str:
    gate = evaluation.get("lean_gate") if isinstance(evaluation.get("lean_gate"), Mapping) else {}
    return str(gate.get("status") or "").strip().lower()


def _lean_gate_passed(evaluation: Mapping[str, Any]) -> bool:
    return _lean_gate_status(evaluation) == VERIFICATION_PASSED


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _expected_ending_open_positions(lean_project: Path) -> dict[str, Any]:
    signals_path = lean_project / "data" / "signals.csv"
    if not signals_path.exists():
        return {
            "expected": None,
            "reason": f"signals.csv missing: {_as_repo_meta(signals_path)}",
        }
    with signals_path.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return {"expected": 0, "latest_time": None, "nonzero_symbols": []}
    latest_time = max(str(row.get("time") or "") for row in rows)
    latest_rows = [row for row in rows if str(row.get("time") or "") == latest_time]
    symbols = [
        str(row.get("symbol") or row.get("pair") or "")
        for row in latest_rows
        if abs(_optional_float(row.get("lean_target_weight")) or 0.0) > 1e-12
    ]
    return {
        "expected": len(symbols),
        "latest_time": latest_time,
        "nonzero_symbols": sorted(symbols),
    }


def _evaluate_lean_gate_report(
    report: Mapping[str, Any],
    config: StrategyLoopConfig,
    *,
    gates: Mapping[str, Any],
    expected_positions: Mapping[str, Any],
) -> dict[str, Any]:
    violations: list[str] = []
    checks: dict[str, Any] = {}
    allowed_statuses = _lean_required_statuses(config)
    comparison_status = str(report.get("status") or "").strip().lower()
    if not comparison_status:
        violations.append("LEAN comparison status missing")
    elif comparison_status not in allowed_statuses:
        violations.append(
            f"LEAN comparison status={comparison_status!r} not in required statuses {sorted(allowed_statuses)}"
        )

    metrics = report.get("metrics") if isinstance(report.get("metrics"), Mapping) else {}
    for field in ("trades",):
        item = metrics.get(field) if isinstance(metrics.get(field), Mapping) else {}
        status = str(item.get("status") or "").strip().lower()
        checks[f"{field}_comparison"] = dict(item) if isinstance(item, Mapping) else {}
        if not item or status == "missing":
            violations.append(f"LEAN comparison metric missing: {field}")
        elif status == "drift" and "drift" not in allowed_statuses:
            violations.append(f"LEAN comparison metric drift: {field}")
    for field in ("orders", "turnover", "final_equity", "max_drawdown"):
        item = metrics.get(field) if isinstance(metrics.get(field), Mapping) else {}
        checks[f"{field}_comparison"] = dict(item) if isinstance(item, Mapping) else {}

    lean = report.get("lean") if isinstance(report.get("lean"), Mapping) else {}
    research = report.get("research") if isinstance(report.get("research"), Mapping) else {}
    final_equity = _optional_float(lean.get("final_equity"))
    max_drawdown_limit = float(config.max_drawdown_pct) / 100.0
    checks["final_equity"] = {"value": final_equity, "informational": True}
    checks["max_drawdown"] = {"value": _optional_float(lean.get("max_drawdown")), "max": max_drawdown_limit, "informational": True}
    if final_equity is None:
        violations.append("LEAN final_equity missing")

    trades = _optional_float(lean.get("trades"))
    min_trades = int(gates.get("min_trades", config.min_trades))
    checks["trades"] = {"value": trades, "min": min_trades}
    if trades is None:
        violations.append("LEAN trades missing")
    elif trades < min_trades:
        violations.append(f"LEAN trades={trades:.6g} < {min_trades}")

    orders = _optional_float(lean.get("orders"))
    turnover = _optional_float(lean.get("turnover"))
    max_gross = _optional_float(lean.get("max_gross"))
    fee_cost = _optional_float(lean.get("fee_cost"))
    research_max_gross = _optional_float(research.get("max_gross"))
    orders_comparison = metrics.get("orders") if isinstance(metrics.get("orders"), Mapping) else {}
    turnover_comparison = metrics.get("turnover") if isinstance(metrics.get("turnover"), Mapping) else {}
    checks["orders"] = {"value": orders, "comparison_threshold": orders_comparison.get("threshold")}
    checks["turnover"] = {"value": turnover, "comparison_threshold": turnover_comparison.get("threshold")}
    checks["max_gross"] = {"value": max_gross, "research": research_max_gross, "max_rel_drift": 0.10}
    checks["fee_cost"] = {"value": fee_cost, "min": 0.0}
    if orders is None:
        violations.append("LEAN orders missing")
    if turnover is None:
        violations.append("LEAN turnover missing")
    if max_gross is None:
        violations.append("LEAN max_gross missing")
    elif research_max_gross is not None and max_gross > research_max_gross * 1.10 + 1e-12:
        violations.append(
            f"LEAN max_gross={max_gross:.6g} > research max_gross {research_max_gross:.6g} by more than 10%"
        )
    if fee_cost is None:
        violations.append("LEAN fee_cost missing")
    elif fee_cost < 0.0:
        violations.append(f"LEAN fee_cost={fee_cost:.6g} < 0")

    expected_open = expected_positions.get("expected")
    actual_open = _optional_float(lean.get("ending_open_positions"))
    checks["ending_open_positions"] = {
        "value": actual_open,
        "expected": expected_open,
        "latest_time": expected_positions.get("latest_time"),
        "nonzero_symbols": expected_positions.get("nonzero_symbols") or [],
    }
    if expected_open is None:
        violations.append(str(expected_positions.get("reason") or "expected ending open positions unavailable"))
    elif actual_open is None:
        violations.append("LEAN ending_open_positions missing")
    elif actual_open > float(expected_open) + 1e-9:
        violations.append(f"LEAN ending_open_positions={actual_open:.6g} > expected {expected_open}")

    return {
        "status": VERIFICATION_FAILED if violations else VERIFICATION_PASSED,
        "comparison_status": comparison_status,
        "required_statuses": sorted(allowed_statuses),
        "violations": violations,
        "checks": checks,
    }


def _copytree_replace(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _cached_sha256(path: Path) -> str:
    resolved = path.resolve()
    stat = resolved.stat()
    cache_root = repo_paths.artifacts_root() / "data_manifest_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_name = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest() + ".json"
    cache_path = cache_root / cache_name
    cached = load_json(cache_path, {})
    if (
        isinstance(cached, Mapping)
        and cached.get("path") == str(resolved)
        and int(cached.get("size") or -1) == int(stat.st_size)
        and int(cached.get("mtime_ns") or -1) == int(stat.st_mtime_ns)
        and cached.get("sha256")
    ):
        return str(cached["sha256"])
    digest = _sha256_file(resolved)
    write_json(
        cache_path,
        {
            "path": str(resolved),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "sha256": digest,
        },
    )
    return digest


def _artifact_ref(path: Path) -> dict[str, Any]:
    resolved = repo_paths.resolve_repo_path(path) if not path.is_absolute() else path
    ref: dict[str, Any] = {"path": _as_repo_meta(resolved)}
    try:
        if resolved.is_file():
            ref["sha256"] = _cached_sha256(resolved)
            ref["bytes"] = int(resolved.stat().st_size)
        elif resolved.is_dir():
            ref["kind"] = "directory"
    except OSError as exc:
        ref["error"] = str(exc)
    return ref


def _artifact_refs_for_iteration(idir: Path) -> dict[str, Any]:
    refs: dict[str, Any] = {}
    for name in ("candidate.json", "signal_export.json", "backtest.json", "evaluation.json", "verification.json", "lean_gate.json", "manifest.json"):
        path = idir / name
        if path.exists():
            refs[name] = _artifact_ref(path)
    lean_gate = load_json(idir / "lean_gate.json", {})
    lean_artifacts = lean_gate.get("artifacts") if isinstance(lean_gate, Mapping) and isinstance(lean_gate.get("artifacts"), Mapping) else {}
    for key, raw in lean_artifacts.items():
        if key == "dir":
            continue
        if not raw:
            continue
        path = repo_paths.resolve_repo_path(str(raw))
        if path.exists():
            refs[f"lean_{key}"] = _artifact_ref(path)
    if lean_artifacts.get("dir"):
        refs["lean_gate_dir"] = {"path": str(lean_artifacts["dir"]), "kind": "directory"}
    backtest = load_json(idir / "backtest.json", {})
    stage_sources: list[tuple[str, Mapping[str, Any]]] = []
    if isinstance(backtest, Mapping):
        stages = backtest.get("stages") if isinstance(backtest.get("stages"), Mapping) else {}
        for stage_name, stage_payload in stages.items():
            if isinstance(stage_payload, Mapping):
                stage_sources.append((str(stage_name), stage_payload))
        if not stage_sources:
            stage_sources.append(("single", backtest))
    for stage_name, payload in stage_sources:
        signals = payload.get("signals")
        all_signal: Optional[str] = None
        if isinstance(signals, Mapping) and signals.get("all"):
            all_signal = str(signals.get("all"))
        elif isinstance(signals, str):
            all_signal = signals
        if all_signal:
            signal_path = repo_paths.resolve_repo_path(all_signal)
            refs[f"{stage_name}_signals"] = _artifact_ref(signal_path)
            refs[f"{stage_name}_signal_dir"] = {"path": _as_repo_meta(signal_path.parent), "kind": "directory"}
        freqtrade = payload.get("freqtrade_backtest") if isinstance(payload.get("freqtrade_backtest"), Mapping) else {}
        metrics = freqtrade.get("metrics") if isinstance(freqtrade.get("metrics"), Mapping) else {}
        if metrics.get("backtest_zip"):
            refs[f"{stage_name}_freqtrade_zip"] = _artifact_ref(repo_paths.resolve_repo_path(str(metrics["backtest_zip"])))
    return refs


def _run_capture(cmd: Sequence[str], *, cwd: Optional[Path] = None, timeout: float = 10.0) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            list(cmd),
            cwd=str(cwd or repo_paths.REPO_ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
    except Exception as exc:
        return {"ok": False, "error": str(exc), "command": list(cmd)}
    return {"ok": proc.returncode == 0, "returncode": proc.returncode, "stdout": (proc.stdout or "").strip(), "command": list(cmd)}


def _pair_universe_from_config(config_path: Path) -> list[str]:
    payload = load_json(config_path, {})
    if not isinstance(payload, Mapping):
        return []
    pairlists = payload.get("pairlists") if isinstance(payload.get("pairlists"), list) else []
    pairs: list[str] = []
    for item in pairlists:
        if isinstance(item, Mapping) and isinstance(item.get("pair_whitelist"), list):
            pairs.extend(str(p) for p in item["pair_whitelist"])
    if not pairs and isinstance(payload.get("exchange"), Mapping):
        exchange = payload["exchange"]
        if isinstance(exchange.get("pair_whitelist"), list):
            pairs.extend(str(p) for p in exchange["pair_whitelist"])
    return sorted(dict.fromkeys(pairs))


def _data_files_for_pairs(pairs: Sequence[str], *, timeframe: str = "1h", venue: str = "okx") -> list[Path]:
    venue_s = str(venue or "okx").strip().lower()
    root = repo_paths.user_data_root() / "data" / venue_s / "futures"
    if venue_s == "okx" and not root.exists():
        root = repo_paths.REPO_ROOT / "user_data" / "data" / "okx" / "futures"
    out: list[Path] = []
    tf = normalize_timeframe(timeframe)
    for pair in pairs:
        base = str(pair).split(":", 1)[0].replace("/", "_")
        futures_base = f"{base}_USDT" if base.endswith("_USDT") else base
        for pattern in (
            f"{futures_base}-{tf}-futures.feather",
            f"{base}-{tf}-futures.feather",
            f"{futures_base}-*-futures.feather",
            f"{base}-*-futures.feather",
            f"{futures_base}-funding_rate.feather",
            f"{base}-funding_rate.feather",
            f"{futures_base}-mark-*.feather",
            f"{base}-mark-*.feather",
        ):
            out.extend(sorted(root.glob(pattern)))
    return sorted(dict.fromkeys(out))


def build_run_manifest(config: StrategyLoopConfig) -> dict[str, Any]:
    config_path = repo_paths.resolve_repo_path(FIXED_FREQTRADE_CONFIG)
    if not config_path.exists():
        fallback = repo_paths.REPO_ROOT / FIXED_FREQTRADE_CONFIG
        config_path = fallback if fallback.exists() else config_path
    strategy_path = repo_paths.user_data_root() / "strategies" / f"{FIXED_FREQTRADE_STRATEGY}.py"
    if not strategy_path.exists():
        strategy_path = repo_paths.REPO_ROOT / "user_data" / "strategies" / f"{FIXED_FREQTRADE_STRATEGY}.py"
    pairs = _pair_universe_from_config(config_path)
    data_refs = [_artifact_ref(path) for path in _data_files_for_pairs(pairs, timeframe=config.timeframe, venue=config.venue)]
    return {
        "version": "factor-strategy-loop-run-manifest-v1",
        "created_at": time.time(),
        "run_id": config.run_id,
        "git": {
            "commit": _run_capture(["git", "rev-parse", "HEAD"], timeout=5.0).get("stdout"),
            "dirty_files": (_run_capture(["git", "status", "--short"], timeout=5.0).get("stdout") or "").splitlines(),
        },
        "cli_args": asdict(config),
        "validation_protocol": validation_protocol_summary(config),
        "lean_gate": {
            "mode": config.lean_gate_mode,
            "lean_bin": config.lean_bin,
            "lean_timeout": config.lean_timeout,
            "required_status": config.lean_required_status,
            "data_root": config.lean_data_root,
        },
        "pair_universe": pairs,
        "freqtrade_version": _run_capture([sys.executable, str(repo_paths.REPO_ROOT / "scripts" / "freqtrade_cli.py"), "--version"], timeout=15.0),
        "config_path": _artifact_ref(config_path) if config_path.exists() else {"path": _as_repo_meta(config_path), "missing": True},
        "strategy_path": _artifact_ref(strategy_path) if strategy_path.exists() else {"path": _as_repo_meta(strategy_path), "missing": True},
        "cost_settings": {
            "fee": "from freqtrade config/backtest command",
            "slippage": "rank_portfolio RiskConfig.slippage",
            "funding": "exchange data files when present",
        },
        "baseline_profile": _load_optimized_baseline(config),
        "data_files": data_refs,
    }


def build_iteration_manifest(
    idir: Path,
    config: StrategyLoopConfig,
    candidate: Mapping[str, Any],
    evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    backtest = load_json(idir / "backtest.json", {})
    refs = _artifact_refs_for_iteration(idir)
    return {
        "version": "factor-strategy-loop-iteration-manifest-v1",
        "created_at": time.time(),
        "run_id": config.run_id,
        "iteration": evaluation.get("iteration"),
        "candidate_signature": evaluation.get("parameter_signature"),
        "timeframe": config.timeframe,
        "data_venue": config.data_venue,
        "evaluation_lane": config.evaluation_lane,
        "rank_params": candidate.get("rank_profile") if isinstance(candidate.get("rank_profile"), Mapping) else {},
        "validation_protocol": validation_protocol_summary(config),
        "stage_signal_dirs": {
            key.removesuffix("_signal_dir"): ref
            for key, ref in refs.items()
            if key.endswith("_signal_dir")
        },
        "freqtrade_zips": {
            key.removesuffix("_freqtrade_zip"): ref
            for key, ref in refs.items()
            if key.endswith("_freqtrade_zip")
        },
        "lookahead_recursive_artifacts": {
            "verification": refs.get("verification.json"),
        },
        "lean_gate": {
            "status": (evaluation.get("lean_gate") or {}).get("status") if isinstance(evaluation.get("lean_gate"), Mapping) else None,
            "comparison_status": (evaluation.get("lean_gate") or {}).get("comparison_status") if isinstance(evaluation.get("lean_gate"), Mapping) else None,
            "comparison_json": refs.get("lean_comparison_json"),
            "lean_project": refs.get("lean_lean_project"),
            "lean_result": refs.get("lean_lean_result"),
        },
        "artifact_refs": refs,
        "window_metrics": evaluation.get("window_metrics") or {},
        "backtest_shape": {
            "keys": sorted(backtest.keys()) if isinstance(backtest, Mapping) else [],
        },
    }


def strategy_loop_retention_tier(run_id: str, *, final_promotion: Optional[Mapping[str, Any]] = None) -> str:
    root = loop_root(str(run_id))
    promotion = final_promotion or load_json(root / "final_promotion.json", {})
    if isinstance(promotion, Mapping) and bool(promotion.get("promoted")):
        return "keep_promoted"
    if not (root / "manifest.json").exists() or not (root / "checkpoint.json").exists():
        return "review_incomplete_manifest"
    final_status = load_json(root / "final_blind_status.json", {})
    if isinstance(final_status, Mapping) and final_status:
        selected = final_status.get("selected") if isinstance(final_status.get("selected"), Mapping) else {}
        if selected and selected.get("promotion_eligible"):
            return "keep_blind_passed_not_promoted"
        return "keep_audit"
    leaderboard = load_json(leaderboard_path(str(run_id)), {})
    rows = list(leaderboard.get("rows") or []) if isinstance(leaderboard, Mapping) else []
    if rows:
        return "review_unpromoted"
    return "delete_candidate"


def write_strategy_loop_registry_entry(
    config: StrategyLoopConfig,
    state: StrategyLoopState,
    *,
    final_promotion: Optional[Mapping[str, Any]] = None,
) -> Path:
    path = strategy_loop_registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    root = loop_root(config.run_id)
    final_status_path = root / "final_blind_status.json"
    doctor_path = root / "doctor_latest.json"
    promotion = dict(final_promotion or state.final_promotion or {})
    entry = {
        "version": "factor-strategy-loop-run-registry-v1",
        "ts": time.time(),
        "run_id": config.run_id,
        "tag": config.tag,
        "protocol": config.validation_protocol,
        "status": state.status,
        "best_score": state.best_score,
        "best_iteration": (state.best_candidate or {}).get("iteration") if isinstance(state.best_candidate, Mapping) else None,
        "promoted": bool(promotion.get("promoted")),
        "promotion": promotion,
        "verification_summary": {
            "verify_policy": config.verify_policy,
            "promote_policy": config.promote_policy,
            "score_mode": config.score_mode,
            "eval_mode": config.eval_mode,
            "lean_gate_mode": config.lean_gate_mode,
        },
        "artifacts": {
            "run_dir": _as_repo_meta(root),
            "manifest": _as_repo_meta(root / "manifest.json"),
            "checkpoint": _as_repo_meta(checkpoint_path(config.run_id)),
            "leaderboard": _as_repo_meta(leaderboard_path(config.run_id)),
            "pareto_pool": _as_repo_meta(root / "pareto_pool.json"),
            "final_blind_status": _as_repo_meta(final_status_path) if final_status_path.exists() else "",
            "doctor_latest": _as_repo_meta(doctor_path) if doctor_path.exists() else "",
        },
        "retention_tier": strategy_loop_retention_tier(config.run_id, final_promotion=promotion),
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, sort_keys=True, default=str) + "\n")
    return path


def _doctor_finding(severity: str, message: str, *, path: str = "", detail: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    item: dict[str, Any] = {"severity": severity, "message": message}
    if path:
        item["path"] = path
    if detail:
        item["detail"] = dict(detail)
    return item


def _doctor_config_from_payloads(root: Path) -> dict[str, Any]:
    checkpoint = load_json(root / "checkpoint.json", {})
    if isinstance(checkpoint, Mapping) and isinstance(checkpoint.get("config"), Mapping):
        return dict(checkpoint["config"])
    manifest = load_json(root / "manifest.json", {})
    if isinstance(manifest, Mapping) and isinstance(manifest.get("cli_args"), Mapping):
        return dict(manifest["cli_args"])
    return {}


def _doctor_window_order(config_payload: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    detail: dict[str, Any] = {}
    try:
        search_start, search_end = parse_timerange(str(config_payload.get("search_timerange") or DEFAULT_SEARCH_TIMERANGE))
        validation_start, validation_end = parse_timerange(str(config_payload.get("validation_timerange") or DEFAULT_VALIDATION_TIMERANGE))
        blind_start, blind_end = parse_timerange(str(config_payload.get("blind_timerange") or DEFAULT_BLIND_TIMERANGE))
        detail = {
            "search": {"start": search_start, "end": search_end},
            "validation": {"start": validation_start, "end": validation_end},
            "blind": {"start": blind_start, "end": blind_end},
        }
        ok = search_start < search_end <= validation_start < validation_end <= blind_start < blind_end
        return ok, detail
    except Exception as exc:
        return False, {"error": str(exc)}


def _doctor_manifest_hash_status(manifest: Mapping[str, Any]) -> dict[str, int]:
    refs = manifest.get("artifact_refs") if isinstance(manifest.get("artifact_refs"), Mapping) else {}
    files = 0
    hashed = 0
    missing_hash = 0
    for ref in refs.values():
        if not isinstance(ref, Mapping):
            continue
        if ref.get("kind") == "directory":
            continue
        if ref.get("missing"):
            continue
        files += 1
        if ref.get("sha256"):
            hashed += 1
        else:
            missing_hash += 1
    return {"files": files, "hashed": hashed, "missing_hash": missing_hash}


def doctor_strategy_loop_run(run_id: str, *, strict_formal: bool = True, write: bool = True) -> dict[str, Any]:
    """Audit a factor strategy-loop run and optionally persist doctor_latest.json."""
    root = loop_root(str(run_id))
    findings: list[dict[str, Any]] = []
    if not root.exists():
        return {
            "version": "factor-strategy-loop-doctor-v1",
            "run_id": str(run_id),
            "run_dir": str(root),
            "ok": False,
            "findings": [_doctor_finding("BLOCKER", "run directory does not exist", path=_as_repo_meta(root))],
            "summary": {},
        }

    config_payload = _doctor_config_from_payloads(root)
    protocol = str(config_payload.get("validation_protocol") or "").strip().lower()
    verify_policy = str(config_payload.get("verify_policy") or "").strip().lower()
    promote_policy = str(config_payload.get("promote_policy") or "").strip().lower()
    lean_gate_mode = str(config_payload.get("lean_gate_mode") or LEAN_GATE_OFF).strip().lower()
    if strict_formal:
        if protocol != VALIDATION_TRIPLE_HOLDOUT:
            findings.append(_doctor_finding("BLOCKER", "formal run must use validation_protocol=triple_holdout"))
        if verify_policy != VERIFY_PARETO:
            findings.append(_doctor_finding("BLOCKER", "formal run must use verify_policy=pareto"))
        if promote_policy != PROMOTE_FINAL:
            findings.append(_doctor_finding("BLOCKER", "formal run must use promote_policy=final"))
        if lean_gate_mode == LEAN_GATE_OFF:
            findings.append(_doctor_finding("BLOCKER", "formal run must enable lean_gate_mode"))

    windows_ok, windows_detail = _doctor_window_order(config_payload)
    if protocol in {VALIDATION_TRIPLE_HOLDOUT, VALIDATION_WALKFORWARD} and not windows_ok:
        findings.append(_doctor_finding("BLOCKER", "search/validation/blind windows are missing, invalid, or overlapping", detail=windows_detail))

    required_root_files = ("manifest.json", "checkpoint.json", "leaderboard.json")
    if protocol == VALIDATION_TRIPLE_HOLDOUT or strict_formal:
        required_root_files = (*required_root_files, "pareto_pool.json", "final_blind_status.json", "final_promotion.json")
    root_artifacts = {}
    for name in required_root_files:
        path = root / name
        root_artifacts[name] = _as_repo_meta(path)
        if not path.exists():
            findings.append(_doctor_finding("BLOCKER", f"missing root artifact: {name}", path=_as_repo_meta(path)))

    iteration_manifests = sorted(root.glob("iter_*/manifest.json"))
    blind_manifests = sorted(root.glob("blind_*/manifest.json"))
    manifest_hashes = [_doctor_manifest_hash_status(load_json(path, {})) for path in [*iteration_manifests, *blind_manifests]]
    missing_hash_total = sum(item.get("missing_hash", 0) for item in manifest_hashes)
    hashed_total = sum(item.get("hashed", 0) for item in manifest_hashes)
    if iteration_manifests and missing_hash_total:
        findings.append(_doctor_finding("MEDIUM", "some manifest artifact refs are missing sha256 hashes", detail={"missing_hash": missing_hash_total}))

    leaderboard = load_json(root / "leaderboard.json", {})
    rows = list(leaderboard.get("rows") or []) if isinstance(leaderboard, Mapping) else []
    non_blind_eligible = [
        row for row in rows
        if isinstance(row, Mapping) and row.get("promotion_eligible") is True and not bool(row.get("blind_final"))
    ]
    if protocol == VALIDATION_TRIPLE_HOLDOUT and non_blind_eligible:
        findings.append(_doctor_finding("BLOCKER", "leaderboard has promotion_eligible=true before blind finalization", detail={"count": len(non_blind_eligible)}))

    final_status = load_json(root / "final_blind_status.json", {})
    selected = final_status.get("selected") if isinstance(final_status, Mapping) and isinstance(final_status.get("selected"), Mapping) else {}
    promotion = final_status.get("promotion") if isinstance(final_status, Mapping) and isinstance(final_status.get("promotion"), Mapping) else {}
    if protocol == VALIDATION_TRIPLE_HOLDOUT or strict_formal:
        if not final_status:
            findings.append(_doctor_finding("BLOCKER", "final_blind_status.json is missing or invalid"))
        elif not selected:
            findings.append(_doctor_finding("HIGH", "no selected blind finalist"))
        else:
            if not bool(selected.get("blind_final")):
                findings.append(_doctor_finding("BLOCKER", "selected candidate is not marked blind_final"))
            if selected.get("promotion_eligible") and str(selected.get("verification_status") or "").lower() != VERIFICATION_PASSED:
                findings.append(_doctor_finding("BLOCKER", "promotion_eligible selected candidate did not pass verification"))
            if selected.get("promotion_eligible") and lean_gate_mode != LEAN_GATE_OFF and _lean_gate_status(selected) != VERIFICATION_PASSED:
                findings.append(_doctor_finding("BLOCKER", "promotion_eligible selected candidate did not pass LEAN gate"))
        if promotion.get("promoted") and not selected.get("promotion_eligible"):
            findings.append(_doctor_finding("BLOCKER", "promotion artifact says promoted but selected candidate is not promotion_eligible"))
        if promotion.get("promoted") and lean_gate_mode != LEAN_GATE_OFF and _lean_gate_status(selected) != VERIFICATION_PASSED:
            findings.append(_doctor_finding("BLOCKER", "promotion artifact says promoted without a passed LEAN gate"))

    verification_files = sorted([*root.glob("iter_*/verification.json"), *root.glob("blind_*/verification.json")])
    verification_counts: dict[str, int] = {}
    for path in verification_files:
        payload = load_json(path, {})
        status = str(payload.get("status") or VERIFICATION_PENDING).lower() if isinstance(payload, Mapping) else VERIFICATION_INCONCLUSIVE
        verification_counts[status] = verification_counts.get(status, 0) + 1
    if verify_policy != VERIFY_NONE and not verification_files and (protocol == VALIDATION_TRIPLE_HOLDOUT or strict_formal):
        findings.append(_doctor_finding("BLOCKER", "verify_policy requires lookahead/recursive artifacts but no verification.json files were found"))

    lean_gate_files = sorted([*root.glob("iter_*/lean_gate.json"), *root.glob("blind_*/lean_gate.json"), *root.glob("best/lean_gate.json")])
    lean_gate_counts: dict[str, int] = {}
    for path in lean_gate_files:
        payload = load_json(path, {})
        status = str(payload.get("status") or VERIFICATION_PENDING).lower() if isinstance(payload, Mapping) else VERIFICATION_INCONCLUSIVE
        lean_gate_counts[status] = lean_gate_counts.get(status, 0) + 1
    if lean_gate_mode != LEAN_GATE_OFF and not lean_gate_files and (protocol == VALIDATION_TRIPLE_HOLDOUT or strict_formal):
        findings.append(_doctor_finding("BLOCKER", "lean_gate_mode requires LEAN gate artifacts but no lean_gate.json files were found"))

    deepresearch = final_status.get("deepresearch") if isinstance(final_status, Mapping) and isinstance(final_status.get("deepresearch"), Mapping) else {}
    deep_artifacts = deepresearch.get("artifacts") if isinstance(deepresearch.get("artifacts"), Mapping) else {}
    if protocol == VALIDATION_TRIPLE_HOLDOUT or strict_formal:
        for key in ("context", "sources"):
            raw = str(deep_artifacts.get(key) or "").strip()
            if not raw:
                findings.append(_doctor_finding("HIGH", f"deepresearch artifact missing: {key}"))
                continue
            path = repo_paths.resolve_repo_path(raw)
            if not path.exists():
                findings.append(_doctor_finding("HIGH", f"deepresearch artifact path does not exist: {key}", path=raw))

    severity_rank = {"BLOCKER": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
    worst = max((severity_rank.get(str(item.get("severity")), 0) for item in findings), default=0)
    doctor_path = root / "doctor_latest.json"
    result = {
        "version": "factor-strategy-loop-doctor-v1",
        "run_id": str(run_id),
        "run_dir": _as_repo_meta(root),
        "ok": worst < severity_rank["HIGH"],
        "strict_formal": bool(strict_formal),
        "policy": {
            "validation_protocol": protocol,
            "verify_policy": verify_policy,
            "promote_policy": promote_policy,
            "lean_gate_mode": lean_gate_mode,
        },
        "windows": windows_detail,
        "artifacts": {**root_artifacts, "doctor_latest.json": _as_repo_meta(doctor_path)},
        "summary": {
            "leaderboard_rows": len(rows),
            "iteration_manifests": len(iteration_manifests),
            "blind_manifests": len(blind_manifests),
            "artifact_refs_hashed": hashed_total,
            "artifact_refs_missing_hash": missing_hash_total,
            "verification_files": len(verification_files),
            "verification_counts": verification_counts,
            "lean_gate_files": len(lean_gate_files),
            "lean_gate_counts": lean_gate_counts,
            "final_promoted": bool(promotion.get("promoted")) if promotion else False,
        },
        "findings": findings,
    }
    if write:
        write_json(doctor_path, result)
    return result


def promote_candidate(
    candidate: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    config: StrategyLoopConfig,
    *,
    iter_dir: Path,
    final: bool = False,
) -> dict[str, Any]:
    promoted = False
    artifacts: dict[str, str] = {}
    reason = str(evaluation.get("promotion_reason") or "")
    if not config.promote or config.promote_policy == PROMOTE_NONE:
        reason = "promotion disabled"
    elif not bool(evaluation.get("constraints_ok")):
        reason = str(evaluation.get("promotion_reason") or "constraints failed")
    elif config.validation_protocol != VALIDATION_SINGLE and not bool(evaluation.get("blind_final")):
        reason = f"{config.validation_protocol} promotion requires final blind evaluation"
    elif str(evaluation.get("verification_status") or VERIFICATION_PASSED).lower() != VERIFICATION_PASSED:
        reason = f"verification_status={evaluation.get('verification_status')} blocks promotion"
    elif evaluation.get("promotion_eligible") is False and _lean_gate_active(config) and not _lean_gate_passed(evaluation):
        status = _lean_gate_status(evaluation) or "missing"
        reason = f"lean_gate_status={status} blocks promotion"
    elif evaluation.get("promotion_eligible") is False:
        reason = "promotion_eligible=false"
    elif not _is_full_holdout(config):
        reason = f"not full holdout ({config.start} to {config.end}); formal promotion skipped"
    elif config.promote_policy == PROMOTE_FINAL and not final:
        reason = "promotion deferred until run completion by promote_policy=final"
    elif _lean_gate_active(config) and not _lean_gate_passed(evaluation):
        status = _lean_gate_status(evaluation) or "missing"
        reason = f"lean_gate_status={status} blocks promotion"
    else:
        ctype = str(candidate.get("candidate_type"))
        if ctype == CANDIDATE_RANK_PROFILE:
            out = repo_paths.artifacts_root() / "rank_portfolio" / config.tag / "optimized_profile.json"
            try:
                iteration = int(evaluation.get("iteration") or iter_dir.name.rsplit("_", 1)[-1])
            except Exception:
                iteration = 0
            payload = {
                "version": "factor-strategy-loop-optimized-profile-v1",
                "created_at": time.time(),
                "run_id": config.run_id,
                "iteration": iteration,
                "promotion_policy": config.promote_policy,
                "final_promotion": bool(final),
                "candidate": dict(candidate),
                "evaluation": dict(evaluation),
                "rank_profile": dict(candidate.get("rank_profile") or {}),
            }
            write_json(out, payload)
            artifacts["optimized_profile"] = _as_repo_meta(out)
            promoted = True
            reason = "rank profile passed full holdout and was written as optimized_profile.json"
        elif ctype == CANDIDATE_FREQTRADE_STRATEGY:
            strategy_raw = candidate.get("strategy_path") or "strategy.py"
            src = _safe_relative_path(iter_dir, strategy_raw)
            strategies_dir = repo_paths.user_data_root() / "strategies"
            strategies_dir.mkdir(parents=True, exist_ok=True)
            base = src.name
            dst = strategies_dir / base
            if dst.exists():
                dst = strategies_dir / f"{src.stem}_{config.run_id}_{iter_dir.name}{src.suffix}"
            shutil.copy2(src, dst)
            artifacts["strategy"] = _as_repo_meta(dst)
            promoted = True
            reason = "strategy passed full holdout and was copied to user_data/strategies"

    return {"promoted": promoted, "artifacts": artifacts, "reason": reason}


def render_agent_prompt(context_path: Path, *, candidate_type: str = "auto") -> str:
    forced = str(candidate_type or "auto").strip().lower()
    baseline_example: dict[str, Any] = {
        "top_k": 2,
        "gross_cap": 2.0,
        "net_cap": 2.0,
        "single_pair_cap": 2.0,
        "side_mode": "short",
        "min_abs_score_z": 1.51,
        "rebalance_hours": 6,
        "risk_per_trade": 0.01785,
        "leverage_cap": 5.0,
        "edge_mode": "rolling_ic",
        "candidate_state": "artifacts/factor_lab/mining/gpt54_purealpha_v2_full1000_fix1/state_0149.json",
        "recompute_corr": False,
        "short_max_mom_24h": 0.038,
        "short_max_mom_72h": 0.10,
        "max_entry_atr_pct": 0.05,
    }
    try:
        context_payload = load_json(context_path, {})
        optimized = context_payload.get("optimized_baseline") if isinstance(context_payload, Mapping) else {}
        profile = optimized.get("rank_profile") if isinstance(optimized, Mapping) and isinstance(optimized.get("rank_profile"), Mapping) else {}
        if profile:
            baseline_example = {
                key: profile[key]
                for key in sorted(profile)
                if key in RANK_PROFILE_KEYS
            }
        # When no optimized baseline exists, resolve candidate_state to an
        # available file so the LLM doesn't copy a non-existent hardcoded path.
        if not profile:
            hardcoded = baseline_example.get("candidate_state", "")
            hardcoded_path = repo_paths.REPO_ROOT / hardcoded if hardcoded else None
            if hardcoded_path is None or not hardcoded_path.exists():
                factor_source = context_payload.get("factor_source", "") if isinstance(context_payload, Mapping) else ""
                if factor_source:
                    src_path = repo_paths.REPO_ROOT / factor_source
                    if src_path.exists():
                        baseline_example = dict(baseline_example)
                        baseline_example["candidate_state"] = factor_source
                else:
                    # Find any available candidate_state.json under artifacts/factor_lab
                    lab_root = repo_paths.artifacts_root() / "factor_lab"
                    found = sorted(lab_root.rglob("candidate_state.json")) if lab_root.exists() else []
                    if found:
                        rel = found[0].relative_to(repo_paths.REPO_ROOT)
                        baseline_example = dict(baseline_example)
                        baseline_example["candidate_state"] = str(rel)
    except Exception:
        pass
    schema_example = {
        "candidate_type": "rank_profile",
        "name": "short_descriptive_name",
        "description": "what changed and why",
        "metadata": {
            "search_mode": "local_exploit",
            "parent_anchor": "optimized_baseline",
            "hypothesis_family": "risk_filter_ablation",
            "expected_tradeoff": "lower turnover and drawdown at the cost of fewer trades",
        },
        "rank_profile": baseline_example,
    }
    if forced == "auto":
        type_instruction = (
            "Default to a `rank_profile` candidate. Choose `freqtrade_strategy` only if the controller "
            "explicitly asks for execution-layer code validation."
        )
    elif forced == CANDIDATE_RANK_PROFILE:
        type_instruction = (
            "You must create a `rank_profile` candidate. Do not write `strategy.py`; the controller "
            f"validates execution with the fixed `{FIXED_FREQTRADE_STRATEGY}` loader after research backtests pass."
        )
    else:
        type_instruction = f"You must create a `{forced}` candidate. Do not choose any other candidate_type."
    return f"""You are modifying one candidate inside an isolated factor-strategy-loop workspace.

Read `context/prepare.json` first. Use these sections before proposing changes:

**LEAN P&L is the primary signal. Always read these LEAN sections FIRST:**
- `previous_iteration["lean_analysis.md"]`: LLM-generated analysis of the previous LEAN backtest —
  headline P&L, monthly stability, drawdown episodes, pair contributions, and rank vs LEAN divergence.
  This is the most important context for your next candidate.
- `previous_iteration["lean_analysis.json"]`: structured LEAN time-period metrics — monthly_returns,
  drawdown_episodes (top-3), pair_contribution (per-symbol P&L and herfindahl index), regime_segments.
- `loop_memory.lean_metrics_history`: last 8 iterations' LEAN scores, worst/best month, deepest
  drawdown, and consecutive-loss-month count. Identify patterns before proposing changes.
- `loop_memory.best_lean_candidate`: the iteration with the highest LEAN score so far.
- `previous_iteration["evaluation.json"].score_components.lean_score`: LEAN's contribution to the blended score.
- `previous_iteration["evaluation.json"].score_components.blended_score`: final score (0.7 LEAN + 0.3 rank).

**Score formula (DO NOT optimize for rank alone):**
The score is a weighted blend: `0.7 × lean_score + 0.3 × rank_score`.
LEAN score bonuses/penalties: monthly win rate ≥60% → +500; consecutive loss months ≥3 → -500;
worst monthly return < -20% → -300; pair concentration (herfindahl) > 0.5 → -200;
top-3 drawdown total depth > 50% → -500. Focus on monthly stability and balanced pair exposure.

**Then read these for search discipline:**
- `objective`: hard gates and target metric.
- `optimized_baseline`: expected reference, frozen candidate state,
  no-correlation-recompute setting, and the filters that must be preserved unless you are ablating one.
- `baseline_search_policy`: how close to the optimized baseline this iteration should stay.
- `loop_memory.best_candidate`: current best result to beat.
- `loop_memory.best_research_result` and `loop_memory.best_freqtrade_result`: best saved metrics by stage.
- `loop_memory.pareto_memory`: best composite, Freqtrade profit, Freqtrade profit/drawdown, and research profit/drawdown anchors.
- `loop_memory.stagnation`: whether local search has switched into structured exploration.
- `loop_memory.gate_repair_hints`: search-window near misses and targeted repairs for trade-count/PDD gates.
- `loop_memory.recent_score_history`: recent attempts, metrics, and violations.
- `loop_memory.previous_failure`: exact validation/runtime failure to fix first.
- `loop_memory.avoid_repeating_rank_profiles`: parameter sets that should not be repeated.
- `loop_memory.avoid_repeating_rank_profile_signatures`: full-run quantized signatures already tried.
- `loop_memory.negative_feedback`: parameter choices that recently lost money or failed gates.
- `previous_iteration`: full previous candidate, evaluation, backtest, analysis, and error details.

Write only these files in the current workspace root:
- `candidate.json`
- `strategy.py` only when `candidate_type` is explicitly `freqtrade_strategy`
- `analysis.md`

{type_instruction}

Do not write outside this workspace. Do not edit repository files. Do not run long backtests.
Do not use future data, and do not weaken risk controls just to increase headline return.

Search discipline:
- On iteration 1, reproduce `optimized_baseline.rank_profile` exactly before proposing new ablations.
- After the baseline is reproduced, change only one factor/risk/filter setting at a time near that baseline.
- If `loop_memory.stagnation.exploration_mode` is `structured`, leave tiny local tweaks behind and test one
  deliberate structural ablation such as a factor-state, side/filter, or regime/risk-control change.
- Always set `metadata.search_mode` to `local_exploit` or `structured_explore`, `metadata.parent_anchor`
  to the anchor you changed, `metadata.hypothesis_family` to the idea family, and
  `metadata.expected_tradeoff` to the risk/return tradeoff you expect.
- In structured exploration, `metadata.search_mode` must be `structured_explore` and at least one
  structural rank dimension must change: factor state, pair universe/exclusions, side mode, rebalance
  cadence, edge/regime mode, top_k, gross/net/single cap, or another core risk structure.
- Preserve `candidate_state`, `recompute_corr=false`, `short_max_mom_24h`, `short_max_mom_72h`,
  and `max_entry_atr_pct` unless your `analysis.md` clearly labels that one-field ablation.
- If `previous_failure` exists, fix that contract failure first and mention the fix in `analysis.md`.
- If recent valid candidates are unprofitable, make one to three targeted changes; do not randomly rewrite
  every knob at once.
- Do not repeat any rank profile listed in `avoid_repeating_rank_profiles`.
- Do not repeat any profile whose quantized signature is in `avoid_repeating_rank_profile_signatures`.
  The controller rounds `risk_per_trade` to 1e-5 and momentum/ATR/z thresholds to 1e-3 before deduping,
  so ultra-precise parameter nudges will be rejected as near-duplicates.
- Prefer changes that reduce negative expectancy, turnover, kill-mode exposure, or signal alignment errors
  before increasing leverage or widening risk.
- In composite scoring, research is the Stage A risk gate and fixed Freqtrade metrics are the primary
  ranking target. The hard goal is to improve fixed Freqtrade profit/drawdown while preserving research
  and Freqtrade `min_trades`, zero research liquidations, and max drawdown limits.
- **LEAN-guided changes**: if `lean_analysis.md` shows consecutive loss months, reduce leverage or tighten
  entry filters. If it shows high pair concentration (herfindahl > 0.5), expand top_k or diversify
  universe. If rank vs LEAN divergence is high, reduce rank-only signals that don't hold in execution.
  If the worst monthly return < -20%, add ATR/momentum filters to reduce tail exposure.

Candidate schema:
```json
{json.dumps(schema_example, indent=2, sort_keys=True)}
```

For a Freqtrade candidate, use `"candidate_type": "freqtrade_strategy"` and write
`strategy.py`. The strategy must inherit `freqtrade.strategy.IStrategy` and read
pre-generated rank signal files; do not reinvent cross-coin ranking inside Freqtrade.
It must support `RP_SIGNAL_DIR` and/or `RP_TAG` for signal discovery, must handle
Freqtrade futures pair names such as `ETH/USDT:USDT`, and must not hard-code
absolute repository paths. Prefer the signal-loading conventions from
`user_data/strategies/ELRankPortfolioLeverageStrategy.py` if you need a reference.
The strategy source must explicitly consume the exported rank columns
`rp_target_weight` and `rp_side`; use `rp_target_weight` for stake/position sizing
and `rp_side` for long/short/flat direction decisions.
It will be rejected unless `strategy.py` contains these exact implementation hooks:
- `pd.read_feather` or `.read_feather(` for loading exported signals.
- `rp_target_weight` used in stake/position sizing logic, not only listed in metadata.
- `rp_side` used in entry/exit direction logic.
- `RP_SIGNAL_DIR` or `RP_TAG` for signal discovery.
- pair normalization for futures names like `ETH/USDT:USDT`.
Its `rank_profile` must be either omitted or use only the same allowed numeric/risk
keys shown above, for example `top_k`, `gross_cap`, `net_cap`, `single_pair_cap`,
`side_mode`, `min_abs_score_z`, `rebalance_hours`, `risk_per_trade`, `leverage_cap`,
and `edge_mode`. Do not put signal-column mappings, version fields, or arbitrary
metadata inside `rank_profile`; use `metadata` for free-form notes.

Before finishing:
- Run only fast local checks if needed, such as `python3 -m py_compile strategy.py`.
- Re-open your own `candidate.json` and ensure it is valid JSON with no unsupported rank keys.
- If and only if you were forced to write `strategy.py`, re-open it and confirm the required hooks above are present.
- In `analysis.md`, list the previous best score/metrics, what changed, and why the change should
  improve the next evaluation.

The controller will validate schema, export rank signals, run backtests, score risk, and
handle promotion. Your job is to create the next candidate and concise reasoning.

Context path: {context_path.as_posix()}
"""


def _truncate_text(value: Any, *, limit: int = 1000) -> str:
    text = str(value or "")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + " ...[truncated]"


def _failure_message(phase: str, exc: Exception) -> str:
    if isinstance(exc, subprocess.TimeoutExpired):
        return f"{phase}: TimeoutExpired after {exc.timeout}s"
    text = _truncate_text(str(exc), limit=500)
    return f"{phase}: {exc.__class__.__name__}: {text}"


class StrategyLoopRunner:
    def __init__(self, config: StrategyLoopConfig) -> None:
        if not config.run_id:
            config.run_id = make_run_id(config.tag)
        self.config = config
        self.state = StrategyLoopState(run_id=config.run_id)
        if config.resume:
            loaded_config, loaded_state = load_checkpoint(config.run_id)
            for key in (
                "model",
                "agent",
                "max_iterations",
                "max_turns",
                "stale_timeout",
                "max_retries",
                "promote",
                "candidate_type",
                "opencode_mode",
                "hermes_provider",
                "hermes_toolsets",
                "hermes_reasoning_effort",
                "hermes_yolo",
                "eval_mode",
                "score_mode",
                "promote_policy",
                "validation_protocol",
                "search_timerange",
                "validation_timerange",
                "blind_timerange",
                "verify_policy",
                "pareto_size_per_axis",
            ):
                setattr(loaded_config, key, getattr(config, key))
            loaded_config.resume = True
            self.config = loaded_config
            self.state = loaded_state
            self.config.run_id = self.state.run_id

    def run(self) -> dict[str, Any]:
        root = loop_root(self.config.run_id)
        root.mkdir(parents=True, exist_ok=True)
        self._write_run_manifest()
        save_checkpoint(self.config, self.state)

        while self.state.iteration <= self.config.max_iterations:
            self._run_iteration()
            if self.state.status == LOOP_STOPPED_STAGNATED:
                save_checkpoint(self.config, self.state)
                break
            self.state.iteration += 1
            self.state.phase = PHASE_PREPARE
            save_checkpoint(self.config, self.state)
        if self.state.status == LOOP_RUNNING:
            self.state.status = LOOP_COMPLETED
        self.state.final_promotion = self._finalize_promotion()
        registry_path = ""
        try:
            registry_path = _as_repo_meta(
                write_strategy_loop_registry_entry(
                    self.config,
                    self.state,
                    final_promotion=self.state.final_promotion,
                )
            )
        except Exception:
            registry_path = ""
        save_checkpoint(self.config, self.state)
        return {
            "run_id": self.config.run_id,
            "checkpoint": _as_repo_meta(checkpoint_path(self.config.run_id)),
            "leaderboard": _as_repo_meta(leaderboard_path(self.config.run_id)),
            "best_candidate": self.state.best_candidate,
            "best_score": self.state.best_score,
            "status": self.state.status,
            "stopped_reason": self.state.stopped_reason,
            "validation_protocol": validation_protocol_summary(self.config),
            "pareto_pool": _as_repo_meta(loop_root(self.config.run_id) / "pareto_pool.json"),
            "final_promotion": self.state.final_promotion,
            "run_registry": registry_path,
        }

    def _run_iteration(self) -> None:
        idir = iteration_dir(self.config.run_id, self.state.iteration)
        idir.mkdir(parents=True, exist_ok=True)
        while self.state.phase != PHASE_COMPLETE:
            phase = self.state.phase
            try:
                if phase == PHASE_PREPARE:
                    self._prepare(idir)
                elif phase == PHASE_CODE_GEN:
                    self._code_gen(idir)
                elif phase == PHASE_SIGNAL_EXPORT:
                    self._signal_export(idir)
                elif phase == PHASE_BACKTEST:
                    self._backtest(idir)
                elif phase == PHASE_LEAN_ANALYSIS:
                    self._lean_analysis_phase(idir)
                elif phase == PHASE_EVALUATION:
                    self._evaluation(idir)
                elif phase == PHASE_ANALYSIS:
                    self._analysis(idir)
                else:
                    self.state.phase = PHASE_COMPLETE
            except Exception as exc:
                self._record_iteration_failure(idir, phase, exc)
                self.state.phase = PHASE_COMPLETE
            if self.state.phase != PHASE_COMPLETE:
                self.state.phase = _next_phase(self.state.phase)
            save_checkpoint(self.config, self.state)

    def _prepare(self, idir: Path) -> None:
        context = prepare_context(self.config, self.config.run_id, self.state.iteration)
        ctx_path = idir / "context" / "prepare.json"
        write_json(ctx_path, context)

    def _code_gen(self, idir: Path) -> None:
        candidate_path = idir / "candidate.json"
        if candidate_path.exists():
            candidate = validate_candidate(candidate_path, default_n=self.config.n)
            self._validate_unique_candidate(candidate)
            self._record_candidate_path(candidate_path)
            return

        if self._seed_initial_baseline_candidate(idir, candidate_path):
            return
        if self._seed_rank_profile_repair_candidate(idir, candidate_path):
            return

        prompt = render_agent_prompt(idir / "context" / "prepare.json", candidate_type=self.config.candidate_type)
        if self.config.agent == AGENT_HERMES:
            self._run_hermes_cli(idir, prompt)
            candidate = validate_candidate(candidate_path, default_n=self.config.n)
            self._validate_unique_candidate(candidate)
            self._record_candidate_path(candidate_path)
            return
        if self.config.agent == AGENT_OPENAI:
            self._run_openai_compatible_agent(idir, prompt)
            candidate = validate_candidate(candidate_path, default_n=self.config.n)
            self._validate_unique_candidate(candidate)
            self._record_candidate_path(candidate_path)
            return
        if self.config.agent != AGENT_OPENCODE:
            raise ValueError(f"unsupported strategy-loop agent: {self.config.agent!r}")

        opencode_env = _opencode_cli_env()
        if not (self.config.model or opencode_env.get("OPENCODE_MODEL")):
            raise RuntimeError("OpenCode unavailable: set --model or OPENCODE_MODEL before running strategy-loop")
        has_opencode_url = bool(opencode_env.get("OPENCODE_URL"))
        has_opencode_cli = shutil.which("opencode") is not None
        if self.config.opencode_mode == "cli" and not has_opencode_cli:
            raise RuntimeError("OpenCode CLI mode requires `opencode` on PATH")
        if not (has_opencode_url or has_opencode_cli):
            raise RuntimeError("OpenCode unavailable: `opencode` CLI is not on PATH and OPENCODE_URL is not set")

        if self.config.opencode_mode == "cli":
            self._run_opencode_cli(idir, prompt, env=opencode_env)
            candidate = validate_candidate(candidate_path, default_n=self.config.n)
            self._validate_unique_candidate(candidate)
            self._record_candidate_path(candidate_path)
            return

        try:
            from agent_market.strategy_miner.agent_adapter import StrategyAgent
        except Exception as exc:
            raise RuntimeError("StrategyAgent/OpenCode dependencies are unavailable") from exc

        agent = StrategyAgent(
            workspace=idir,
            provider="opencode",
            model=self.config.model,
            max_turns=self.config.max_turns,
            stale_timeout=self.config.stale_timeout,
            max_retries=self.config.max_retries,
        )
        try:
            try:
                result = agent.run_result(prompt)
            except Exception:
                if self.config.opencode_mode != "auto":
                    raise
                if not has_opencode_cli:
                    raise RuntimeError("OpenCode auto mode could not fall back to CLI because `opencode` is not on PATH")
                self._run_opencode_cli(idir, prompt, env=opencode_env)
                result = None
            if result is None:
                candidate = validate_candidate(candidate_path, default_n=self.config.n)
                self._validate_unique_candidate(candidate)
                self._record_candidate_path(candidate_path)
                return
            usage = result.usage or {}
            if usage:
                self.state.token_cost[str(self.state.iteration)] = usage
            (idir / "agent_response.txt").write_text(result.assistant_text or "", encoding="utf-8")
        finally:
            agent.close()

        if not candidate_path.exists():
            raise RuntimeError(f"agent did not write required candidate.json in {idir}")
        candidate = validate_candidate(candidate_path, default_n=self.config.n)
        self._validate_unique_candidate(candidate)
        self._record_candidate_path(candidate_path)

    def _seed_initial_baseline_candidate(self, idir: Path, candidate_path: Path) -> bool:
        if self.state.iteration != 1 or self.config.candidate_type != CANDIDATE_RANK_PROFILE:
            return False
        baseline = _load_optimized_baseline(self.config)
        profile = baseline.get("rank_profile")
        if not baseline.get("available") or not isinstance(profile, Mapping) or not profile:
            return False

        candidate = {
            "candidate_type": CANDIDATE_RANK_PROFILE,
            "name": "optimized_baseline_replay",
            "description": "First-iteration replay of optimized_profile.json before nearby ablations.",
            "rank_profile": dict(profile),
            "metadata": {
                "source": "optimized_baseline",
                "search_mode": "local_exploit",
                "parent_anchor": "optimized_baseline",
                "hypothesis_family": "baseline_replay",
                "expected_tradeoff": "no change; establishes reproducible baseline metrics",
                "baseline_profile": baseline.get("path"),
                "baseline_label": baseline.get("label"),
                "policy": "reproduce optimized baseline before one-factor-at-a-time ablation",
            },
        }
        write_json(candidate_path, candidate)
        (idir / "agent_response.txt").write_text(
            f"Seeded optimized baseline candidate before invoking {self.config.agent}.\n",
            encoding="utf-8",
        )
        normalized = validate_candidate(candidate_path, default_n=self.config.n)
        self._validate_unique_candidate(normalized)
        self._record_candidate_path(candidate_path)
        return True

    def _seed_rank_profile_repair_candidate(self, idir: Path, candidate_path: Path) -> bool:
        if self.state.iteration <= 1 or self.config.candidate_type != CANDIDATE_RANK_PROFILE:
            return False
        if not self.state.score_history:
            return False
        baseline = _baseline_rank_profile(self.config)
        if not baseline:
            return False
        candidates = build_rank_profile_repair_queue(
            baseline,
            self.config,
            rows=[row for row in self.state.score_history if isinstance(row, Mapping)],
            structured=self.state.exploration_mode == "structured",
        )
        for candidate in candidates:
            try:
                self._validate_unique_candidate(candidate)
            except ValueError:
                continue
            write_json(candidate_path, candidate)
            profile = candidate.get("rank_profile") if isinstance(candidate.get("rank_profile"), Mapping) else {}
            changes = (candidate.get("metadata") or {}).get("changed_keys") if isinstance(candidate.get("metadata"), Mapping) else []
            analysis = [
                f"# {candidate.get('name')}",
                "",
                "Controller-generated rank-profile repair candidate.",
                "",
                f"- Parent: optimized_baseline",
                f"- Changed keys: {changes}",
                f"- Expected tradeoff: {(candidate.get('metadata') or {}).get('expected_tradeoff') if isinstance(candidate.get('metadata'), Mapping) else ''}",
                f"- Signature: {rank_profile_signature(profile, default_n=self.config.n) if profile else ''}",
            ]
            (idir / "analysis.md").write_text("\n".join(analysis) + "\n", encoding="utf-8")
            (idir / "agent_response.txt").write_text(
                "Seeded controller rank-profile repair candidate before invoking the LLM.\n",
                encoding="utf-8",
            )
            normalized = validate_candidate(candidate_path, default_n=self.config.n)
            self._record_candidate_path(candidate_path)
            self._validate_unique_candidate(normalized)
            return True
        return False

    def _run_openai_compatible_agent(self, idir: Path, prompt: str) -> None:
        env = _openai_compatible_env()
        api_key = str(env.get("OPENAI_API_KEY") or env.get("LLM_API_KEY") or "").strip()
        if not api_key:
            raise RuntimeError("OpenAI-compatible agent requires OPENAI_API_KEY or LLM_API_KEY")
        model = _openai_compatible_model(self.config.model, env)
        if not model:
            raise RuntimeError("OpenAI-compatible agent requires --model, LLM_MODEL, or OPENAI_MODEL")
        base_url = (
            str(env.get("OPENAI_BASE_URL") or env.get("LLM_BASE_URL") or env.get("OPENAI_API_BASE") or "https://api.openai.com/v1")
            .strip()
            .rstrip("/")
        )
        if not base_url:
            base_url = "https://api.openai.com/v1"

        try:
            from agent_market.strategy_miner.agent_adapter import StrategyAgent
        except Exception as exc:
            raise RuntimeError("StrategyAgent/OpenAI-compatible dependencies are unavailable") from exc

        direct_prompt = (
            f"{prompt}\n\n"
            "You are running through the direct OpenAI-compatible strategy-loop adapter. "
            "You do not have filesystem tools. Return exactly one JSON object that can be "
            "saved as candidate.json. Do not include markdown fences or commentary."
        )
        with _temporary_environ(env):
            agent = StrategyAgent(
                workspace=idir,
                provider="openai",
                model=model,
                base_url=base_url,
                max_turns=self.config.max_turns,
                stale_timeout=self.config.stale_timeout,
                max_retries=self.config.max_retries,
            )
            try:
                result = agent.run_result(direct_prompt)
            finally:
                agent.close()

        assistant_text = getattr(result, "assistant_text", "") or ""
        usage = getattr(result, "usage", None) or {}
        if usage:
            self.state.token_cost[str(self.state.iteration)] = usage
        (idir / "agent_response.txt").write_text(assistant_text, encoding="utf-8")
        payload = _json_object_from_text(assistant_text)
        if payload is None:
            raise RuntimeError("OpenAI-compatible agent did not return a JSON candidate")
        if "candidate_type" not in payload and isinstance(payload.get("candidate"), Mapping):
            payload = dict(payload["candidate"])
        write_json(idir / "candidate.json", payload)

    def _run_hermes_cli(self, idir: Path, prompt: str, *, env: Optional[Mapping[str, str]] = None) -> None:
        if shutil.which("hermes") is None:
            raise RuntimeError("Hermes CLI mode requires `hermes` on PATH")
        hermes_env = _hermes_cli_env(env)
        hermes_home = _prepare_hermes_run_home(self.config.run_id, hermes_env)
        effort = str(self.config.hermes_reasoning_effort or hermes_env.get("HERMES_REASONING_EFFORT") or "").strip().lower()
        if effort:
            cfg_cmd = ["hermes", "config", "set", "agent.reasoning_effort", effort]
            cfg_proc = subprocess.run(
                cfg_cmd,
                cwd=str(idir),
                env=hermes_env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=60.0,
                check=False,
            )
            if cfg_proc.returncode != 0:
                (idir / "agent_response.txt").write_text(cfg_proc.stdout or "", encoding="utf-8")
                raise RuntimeError(f"Hermes config failed with exit code {cfg_proc.returncode}; see {idir / 'agent_response.txt'}")
        cmd = [
            "hermes",
            "chat",
            "-Q",
            "--toolsets",
            str(self.config.hermes_toolsets or "terminal,file"),
            "--max-turns",
            str(int(self.config.max_turns)),
            "--source",
            "strategy-loop",
        ]
        model = _hermes_model(self.config.model, hermes_env)
        if model:
            cmd.extend(["-m", model])
        provider = str(self.config.hermes_provider or hermes_env.get("HERMES_PROVIDER") or "").strip()
        if provider:
            cmd.extend(["--provider", provider])
        if self.config.hermes_yolo:
            cmd.append("--yolo")
        cmd.extend(["-q", prompt])
        proc = subprocess.run(
            cmd,
            cwd=str(idir),
            env=hermes_env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=max(120.0, float(self.config.stale_timeout) + 300.0),
            check=False,
        )
        prefix = f"Hermes HERMES_HOME={hermes_home}\n" + (f"Hermes reasoning_effort={effort}\n" if effort else "")
        (idir / "agent_response.txt").write_text(prefix + (proc.stdout or ""), encoding="utf-8")
        if proc.returncode != 0:
            raise RuntimeError(f"Hermes CLI failed with exit code {proc.returncode}; see {idir / 'agent_response.txt'}")

    def _run_opencode_cli(self, idir: Path, prompt: str, *, env: Optional[Mapping[str, str]] = None) -> None:
        opencode_env = _opencode_cli_env(env)
        cmd = ["opencode", "run", "-m", self.config.model or opencode_env.get("OPENCODE_MODEL", ""), prompt]
        proc = subprocess.run(
            cmd,
            cwd=str(idir),
            env=opencode_env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=max(120.0, float(self.config.stale_timeout) + 300.0),
            check=False,
        )
        (idir / "agent_response.txt").write_text(proc.stdout or "", encoding="utf-8")
        if proc.returncode != 0:
            raise RuntimeError(f"opencode CLI failed with exit code {proc.returncode}; see {idir / 'agent_response.txt'}")

    def _signal_export(self, idir: Path) -> None:
        out = idir / "signal_export.json"
        if out.exists():
            return
        candidate = validate_candidate(idir / "candidate.json", default_n=self.config.n)
        factor_state, _ = _resolve_factor_state(self.config.tag)
        factor_state = _filter_state_for_timeframe(factor_state, self.config.timeframe, idir)
        effective_tag = self._effective_rank_tag(idir)
        export_timerange = self.config.search_timerange if self.config.validation_protocol != VALIDATION_SINGLE else self.config.timerange
        start, end = parse_timerange(export_timerange)
        rank_profile = dict(candidate.get("rank_profile") or {})
        if factor_state is not None:
            rank_profile["candidate_state"] = str(factor_state)
        kwargs = _rank_kwargs(
            rank_profile,
            self.config,
            candidate_state=factor_state,
            tag=effective_tag,
            start=start,
            end=end,
        )
        summary = rank_portfolio.rank_export(**kwargs)
        summary["effective_tag"] = effective_tag
        summary["base_tag"] = self.config.tag
        summary["timerange"] = export_timerange
        summary["stage"] = "search" if self.config.validation_protocol != VALIDATION_SINGLE else "single"
        summary["rank_kwargs"] = kwargs
        write_json(out, summary)

    def _backtest(self, idir: Path) -> None:
        out = idir / "backtest.json"
        if out.exists():
            return
        candidate = validate_candidate(idir / "candidate.json", default_n=self.config.n)
        if self.config.validation_protocol != VALIDATION_SINGLE:
            result = self._run_triple_holdout_backtest(idir, candidate)
            write_json(out, result)
            return
        result = self._run_window_backtest(
            idir,
            candidate,
            stage="single",
            timerange=self.config.timerange,
            run_freqtrade=self.config.eval_mode == EVAL_FREQTRADE,
        )
        if self.config.eval_mode == EVAL_FREQTRADE:
            stage_a = {"constraints_ok": True, "score": 0, "violations": []}
        else:
            stage_a = score_backtest_result(
                result,
                min_trades=self.config.min_trades,
                max_drawdown_pct=self.config.max_drawdown_pct,
                min_profit_over_dd=self.config.min_profit_over_dd,
                target_profit_pct=self.config.target_profit_pct,
            )
        result["stage_a"] = {
            "constraints_ok": stage_a["constraints_ok"],
            "score": stage_a["score"],
            "violations": stage_a["violations"],
        }
        if self.config.eval_mode in {EVAL_TWO_STAGE, EVAL_FREQTRADE}:
            if self.config.eval_mode == EVAL_FREQTRADE or bool(stage_a.get("constraints_ok")):
                result["freqtrade_backtest"] = self._run_fixed_freqtrade_backtest(
                    idir,
                    result,
                    timerange=self.config.timerange,
                    stage="single",
                )
            else:
                result["freqtrade_backtest"] = {
                    "ok": False,
                    "skipped": True,
                    "reason": "Stage A research gates failed",
                    "stage_a_violations": stage_a.get("violations") or [],
                }
        write_json(out, result)

    def _run_window_backtest(
        self,
        idir: Path,
        candidate: Mapping[str, Any],
        *,
        stage: str,
        timerange: str,
        run_freqtrade: bool = False,
    ) -> dict[str, Any]:
        start, end = parse_timerange(timerange)
        factor_state, _ = _resolve_factor_state(self.config.tag)
        kwargs = _rank_kwargs(
            candidate.get("rank_profile") or {},
            self.config,
            candidate_state=factor_state,
            tag=f"{self._effective_rank_tag(idir)}_{stage}",
            start=start,
            end=end,
        )
        if self.config.eval_mode == EVAL_FREQTRADE:
            result: dict[str, Any] = {}
        else:
            result = rank_portfolio.rank_backtest(**kwargs)
        result["base_tag"] = self.config.tag
        result["stage"] = stage
        result["timerange"] = timerange
        result["candidate"] = candidate
        result["rank_kwargs"] = kwargs
        result["research_backtest"] = {
            key: result.get(key)
            for key in (
                "total_return_pct",
                "max_drawdown_pct",
                "profit_over_max_drawdown",
                "trades",
                "simulated_liquidations",
                "liquidation_rejects",
                "avg_turnover",
            )
            if key in result
        }
        if run_freqtrade:
            result["freqtrade_backtest"] = self._run_fixed_freqtrade_backtest(
                idir,
                result,
                timerange=timerange,
                stage=stage,
            )
        return result

    def _run_triple_holdout_backtest(self, idir: Path, candidate: Mapping[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {
            "validation_protocol": validation_protocol_summary(self.config),
            "base_tag": self.config.tag,
            "candidate": candidate,
            "stages": {},
        }
        search = self._run_window_backtest(
            idir,
            candidate,
            stage="search",
            timerange=self.config.search_timerange,
            run_freqtrade=False,
        )
        if self.config.eval_mode == EVAL_FREQTRADE:
            search_eval = {"constraints_ok": True, "score": 0, "violations": []}
        else:
            search_eval = score_backtest_result(
                search,
                **{
                    key: scaled_gate_values(self.config, self.config.search_timerange)[key]
                    for key in ("min_trades", "max_drawdown_pct", "min_profit_over_dd", "target_profit_pct")
                },
            )
        search["stage_a"] = {
            "constraints_ok": search_eval["constraints_ok"],
            "score": search_eval["score"],
            "violations": search_eval["violations"],
        }
        result["stages"]["search"] = search
        if not search_eval["constraints_ok"]:
            result["validation_skipped"] = {
                "reason": "search gates failed",
                "violations": search_eval["violations"],
            }
            return result

        validation = self._run_window_backtest(
            idir,
            candidate,
            stage="validation",
            timerange=self.config.validation_timerange,
            run_freqtrade=False,
        )
        validation_gate_values = scaled_gate_values(self.config, self.config.validation_timerange)
        if self.config.eval_mode == EVAL_FREQTRADE:
            validation_stage_a = {"constraints_ok": True, "score": 0, "violations": []}
        else:
            validation_stage_a = score_backtest_result(
                validation,
                min_trades=validation_gate_values["min_trades"],
                max_drawdown_pct=validation_gate_values["max_drawdown_pct"],
                min_profit_over_dd=validation_gate_values["min_profit_over_dd"],
                target_profit_pct=validation_gate_values["target_profit_pct"],
            )
        validation["stage_a"] = {
            "constraints_ok": validation_stage_a["constraints_ok"],
            "score": validation_stage_a["score"],
            "violations": validation_stage_a["violations"],
        }
        if self.config.eval_mode in {EVAL_TWO_STAGE, EVAL_FREQTRADE}:
            if self.config.eval_mode == EVAL_FREQTRADE or validation_stage_a["constraints_ok"]:
                validation["freqtrade_backtest"] = self._run_fixed_freqtrade_backtest(
                    idir,
                    validation,
                    timerange=self.config.validation_timerange,
                    stage="validation",
                )
            else:
                validation["freqtrade_backtest"] = {
                    "ok": False,
                    "skipped": True,
                    "reason": "Validation research gates failed",
                    "stage_a_violations": validation_stage_a.get("violations") or [],
                }
        result["stages"]["validation"] = validation
        return result

    def _run_fixed_freqtrade_backtest(
        self,
        idir: Path,
        research_result: Mapping[str, Any],
        *,
        timerange: Optional[str] = None,
        stage: str = "single",
    ) -> dict[str, Any]:
        signals_raw = str(research_result.get("signals") or "")
        # In EVAL_FREQTRADE mode research_backtest is skipped, so signals come from signal_export.json
        if not signals_raw:
            se_path = idir / "signal_export.json"
            if se_path.exists():
                try:
                    se = json.loads(se_path.read_text(encoding="utf-8"))
                    sigs = se.get("signals")
                    if isinstance(sigs, dict):
                        signals_raw = str(sigs.get("all") or "")
                    elif isinstance(sigs, str):
                        signals_raw = sigs
                except Exception:
                    pass
        signals_path = Path(signals_raw).expanduser() if signals_raw else Path()
        signal_dir = signals_path.parent if signals_path.exists() else None
        if signal_dir is None:
            return {"ok": False, "error": f"rank signals not found: {signals_raw}"}

        config_path = repo_paths.resolve_repo_path(FIXED_FREQTRADE_CONFIG)
        if not config_path.exists():
            fallback = repo_paths.REPO_ROOT / FIXED_FREQTRADE_CONFIG
            config_path = fallback if fallback.exists() else config_path
        strategy_dir = repo_paths.user_data_root() / "strategies"
        if not (strategy_dir / f"{FIXED_FREQTRADE_STRATEGY}.py").exists():
            fallback_dir = repo_paths.REPO_ROOT / "user_data" / "strategies"
            strategy_dir = fallback_dir if (fallback_dir / f"{FIXED_FREQTRADE_STRATEGY}.py").exists() else strategy_dir
        if not config_path.exists():
            return {"ok": False, "error": f"fixed Freqtrade config not found: {config_path}"}
        if not (strategy_dir / f"{FIXED_FREQTRADE_STRATEGY}.py").exists():
            return {"ok": False, "error": f"fixed Freqtrade strategy not found in: {strategy_dir}"}

        # Build venue-specific override config so freqtrade reads the right datadir/exchange.
        venue = str(self.config.venue or "okx").strip().lower()
        exchange_name = _VENUE_EXCHANGE.get(venue, "okx")
        venue_datadir = _VENUE_DATADIR.get(venue)
        override_path: Optional[Path] = None
        if venue != "okx" and venue_datadir is not None:
            pairs = _pairs_from_signal_dir(signal_dir, exchange_name=exchange_name)
            override = {
                "datadir": venue_datadir,
                "exchange": {"name": exchange_name, "pair_whitelist": pairs},
            }
            override_path = idir / f"freqtrade_override_{stage}.json"
            override_path.write_text(json.dumps(override, ensure_ascii=False, indent=2), encoding="utf-8")

        cmd = [
            sys.executable,
            str(repo_paths.REPO_ROOT / "scripts" / "freqtrade_cli.py"),
            "backtesting",
            "--cache",
            "none",
            "--config",
            str(config_path),
        ]
        if override_path is not None:
            cmd += ["--config", str(override_path)]
        cmd += [
            "--strategy",
            FIXED_FREQTRADE_STRATEGY,
            "--strategy-path",
            str(strategy_dir),
            "--timerange",
            str(timerange or self.config.timerange),
        ]
        env = dict(os.environ)
        env["RP_SIGNAL_DIR"] = str(signal_dir)
        env["RP_TAG"] = str(research_result.get("tag") or self.config.tag)
        start_time = time.time() - 5.0
        log_name = "freqtrade_backtest.log" if stage in {"", "single"} else f"freqtrade_{stage}.log"
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(repo_paths.REPO_ROOT),
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=max(300.0, float(self.config.stale_timeout) + 300.0),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            (idir / log_name).write_text(str(exc), encoding="utf-8")
            return {"ok": False, "error": f"freqtrade backtesting timed out after {exc.timeout}s"}
        (idir / log_name).write_text(proc.stdout or "", encoding="utf-8")
        command_meta = {
            "command": cmd,
            "signal_dir": _as_repo_meta(signal_dir),
            "timerange": str(timerange or self.config.timerange),
            "stage": stage,
            "strategy": FIXED_FREQTRADE_STRATEGY,
            "config": _as_repo_meta(config_path),
        }
        if proc.returncode != 0:
            return {
                "ok": False,
                "error": f"freqtrade backtesting failed with exit code {proc.returncode}",
                **command_meta,
            }

        results_dir = repo_paths.user_data_root() / "backtest_results"
        if not results_dir.exists():
            results_dir = repo_paths.REPO_ROOT / "user_data" / "backtest_results"
        zips = [
            p
            for p in results_dir.glob("backtest-result-*.zip")
            if p.stat().st_mtime >= start_time
        ] if results_dir.exists() else []
        zips.sort(key=lambda p: p.stat().st_mtime)
        if not zips:
            return {"ok": False, "error": f"no Freqtrade backtest result zip found in {results_dir}", **command_meta}

        zip_path = zips[-1]
        try:
            summary = build_backtest_summary(zip_path)
        except Exception as exc:
            return {"ok": False, "error": f"failed to parse Freqtrade result {zip_path}: {exc}", **command_meta}
        profit_pct = float(summary.get("profit_total_pct") or 0.0)
        max_dd_pct = float(summary.get("max_drawdown_pct") or summary.get("max_drawdown_account") or 0.0)
        metrics = {
            "ok": True,
            "profit_pct": profit_pct,
            "max_drawdown_pct": max_dd_pct,
            "profit_over_max_drawdown": profit_pct / max(max_dd_pct, 1e-9),
            "trades": int(summary.get("trades") or 0),
            "backtest_zip": _as_repo_meta(zip_path),
        }
        return {
            "ok": True,
            "metrics": metrics,
            "summary": summary,
            "monthly_profit": summary.get("monthly_profit"),
            "daily_profit": summary.get("daily_profit"),
            "drawdown_start": summary.get("drawdown_start"),
            "drawdown_end": summary.get("drawdown_end"),
            "drawdown_high": summary.get("drawdown_high"),
            "drawdown_low": summary.get("drawdown_low"),
            **command_meta,
        }

    def _should_run_lean_gate(self, stage: str, *, promotion_candidate: bool = False) -> bool:
        mode = str(self.config.lean_gate_mode or LEAN_GATE_OFF).lower()
        if mode == LEAN_GATE_OFF:
            return False
        if mode == LEAN_GATE_ALL:
            return True
        stage_s = str(stage or "").lower()
        if stage_s in {"blind", "final"} and mode in {LEAN_GATE_FINAL, LEAN_GATE_PARETO}:
            return True
        if promotion_candidate and mode == LEAN_GATE_FINAL and self.config.validation_protocol == VALIDATION_SINGLE:
            return True
        if stage_s == "pareto" and mode == LEAN_GATE_PARETO:
            return True
        return False

    def _lean_gate_values(self, timerange: Optional[str]) -> dict[str, Any]:
        if timerange:
            return scaled_gate_values(self.config, timerange)
        return {
            "min_trades": self.config.min_trades,
            "max_drawdown_pct": self.config.max_drawdown_pct,
            "min_profit_over_dd": self.config.min_profit_over_dd,
            "target_profit_pct": self.config.target_profit_pct,
        }

    def _lean_rank_artifact_path(self, idir: Path, gate_dir: Path, *, stage: str) -> Path:
        backtest_path = idir / "backtest.json"
        payload = load_json(backtest_path, {})
        if not isinstance(payload, Mapping):
            return backtest_path
        if payload.get("signals"):
            return backtest_path
        stages = payload.get("stages") if isinstance(payload.get("stages"), Mapping) else {}
        preferred = str(stage or "").lower()
        stage_payload = stages.get(preferred) if isinstance(stages.get(preferred), Mapping) else None
        if stage_payload is None and preferred == "iteration":
            stage_payload = stages.get("validation") if isinstance(stages.get("validation"), Mapping) else None
        if stage_payload is None:
            stage_payload = stages.get("blind") if isinstance(stages.get("blind"), Mapping) else None
        if stage_payload is None:
            stage_payload = stages.get("validation") if isinstance(stages.get("validation"), Mapping) else None
        if stage_payload is None:
            stage_payload = stages.get("search") if isinstance(stages.get("search"), Mapping) else None
        if stage_payload is None or not stage_payload.get("signals"):
            return backtest_path
        artifact = dict(stage_payload)
        artifact.setdefault("venue", self.config.venue)
        artifact.setdefault("timeframe", self.config.timeframe)
        artifact.setdefault("data_venue", self.config.data_venue)
        artifact.setdefault("tag", artifact.get("tag") or self.config.tag)
        artifact_path = gate_dir / "rank_artifact.json"
        write_json(artifact_path, artifact)
        return artifact_path

    def _run_lean_gate(
        self,
        idir: Path,
        *,
        stage: str,
        timerange: Optional[str],
    ) -> dict[str, Any]:
        gate_dir = idir / "lean_gate" / stage
        project = gate_dir / "project"
        summary_path = idir / "lean_gate.json"
        if summary_path.exists():
            cached = load_json(summary_path, {})
            if isinstance(cached, Mapping) and cached.get("stage") == stage:
                return dict(cached)

        gate_dir.mkdir(parents=True, exist_ok=True)
        rank_artifact = self._lean_rank_artifact_path(idir, gate_dir, stage=stage)
        started = time.time()
        lean_version = _run_capture([self.config.lean_bin, "--version"], timeout=10.0)
        base_summary: dict[str, Any] = {
            "version": "factor-strategy-loop-lean-gate-v1",
            "created_at": started,
            "run_id": self.config.run_id,
            "iteration_dir": _as_repo_meta(idir),
            "stage": stage,
            "timerange": timerange,
            "mode": self.config.lean_gate_mode,
            "required_status": self.config.lean_required_status,
            "lean_bin": self.config.lean_bin,
            "lean_timeout": self.config.lean_timeout,
            "lean_version": lean_version,
            "artifacts": {
                "dir": _as_repo_meta(gate_dir),
                "rank_artifact": _as_repo_meta(rank_artifact),
                "lean_project": _as_repo_meta(project),
            },
        }

        def _fail(reason: str, *, extra: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
            payload = {
                **base_summary,
                "status": VERIFICATION_FAILED,
                "reason": reason,
                "violations": [reason],
                "duration_sec": float(time.time() - started),
            }
            if extra:
                payload.update(dict(extra))
            write_json(summary_path, payload)
            return payload

        if not rank_artifact.exists():
            return _fail(f"rank artifact missing for LEAN gate: {_as_repo_meta(rank_artifact)}")

        try:
            export_manifest = lean_bridge.export_project(
                rank_artifact=rank_artifact,
                output=project,
                timeframe=self.config.timeframe,
                data_root=self.config.lean_data_root or None,
            )
        except Exception as exc:
            return _fail(f"LEAN export failed: {exc}", extra={"traceback": traceback.format_exc()})

        base_summary["export_manifest"] = export_manifest
        base_summary["artifacts"]["lean_manifest"] = _as_repo_meta(project / "manifest.json")
        try:
            backtest_run = lean_bridge.run_lean_backtest(
                lean_project=project,
                lean_bin=self.config.lean_bin,
                timeout=self.config.lean_timeout,
            )
        except Exception as exc:
            return _fail(
                f"LEAN backtest failed: {exc}",
                extra={
                    "export_manifest": export_manifest,
                    "traceback": traceback.format_exc(),
                    "artifacts": {
                        **base_summary["artifacts"],
                        "lean_backtest_run": _as_repo_meta(project / "lean_backtest_run.json"),
                    },
                },
            )

        result_raw = backtest_run.get("result_path") if isinstance(backtest_run, Mapping) else None
        if result_raw:
            result_path = Path(str(result_raw)).expanduser()
            if not result_path.is_absolute():
                result_path = (project / result_path).resolve()
        else:
            result_path = lean_bridge.find_latest_lean_result(project)
        if result_path is None or not result_path.exists():
            return _fail(
                "LEAN backtest completed but no result JSON was found",
                extra={
                    "export_manifest": export_manifest,
                    "backtest_run": backtest_run,
                    "artifacts": {
                        **base_summary["artifacts"],
                        "lean_backtest_run": _as_repo_meta(project / "lean_backtest_run.json"),
                    },
                },
            )

        comparison_path = gate_dir / "comparison.json"
        try:
            comparison = lean_bridge.compare_results(
                rank_artifact=rank_artifact,
                lean_result=result_path,
                output=comparison_path,
                timeframe=self.config.timeframe,
                skip_signal_load=True,
            )
        except Exception as exc:
            return _fail(
                f"LEAN comparison failed: {exc}",
                extra={
                    "export_manifest": export_manifest,
                    "backtest_run": backtest_run,
                    "lean_result": str(result_path),
                    "traceback": traceback.format_exc(),
                    "artifacts": {
                        **base_summary["artifacts"],
                        "lean_backtest_run": _as_repo_meta(project / "lean_backtest_run.json"),
                        "lean_result": _as_repo_meta(result_path),
                    },
                },
            )

        expected_positions = _expected_ending_open_positions(project)
        assessment = _evaluate_lean_gate_report(
            comparison,
            self.config,
            gates=self._lean_gate_values(timerange),
            expected_positions=expected_positions,
        )
        payload = {
            **base_summary,
            "status": assessment["status"],
            "reason": "LEAN gate passed" if assessment["status"] == VERIFICATION_PASSED else "; ".join(assessment["violations"]),
            "violations": assessment["violations"],
            "checks": assessment["checks"],
            "comparison_status": assessment["comparison_status"],
            "required_statuses": assessment["required_statuses"],
            "export_manifest": export_manifest,
            "backtest_run": backtest_run,
            "comparison": comparison,
            "lean_metrics": comparison.get("lean") if isinstance(comparison.get("lean"), Mapping) else {},
            "research_metrics": comparison.get("research") if isinstance(comparison.get("research"), Mapping) else {},
            "duration_sec": float(time.time() - started),
            "artifacts": {
                **base_summary["artifacts"],
                "lean_backtest_run": _as_repo_meta(project / "lean_backtest_run.json"),
                "lean_result": _as_repo_meta(result_path),
                "comparison_json": _as_repo_meta(comparison_path),
            },
        }
        write_json(summary_path, payload)
        return payload

    def _apply_lean_gate(
        self,
        idir: Path,
        evaluation: dict[str, Any],
        *,
        stage: str,
        timerange: Optional[str],
    ) -> dict[str, Any]:
        gate = self._run_lean_gate(idir, stage=stage, timerange=timerange)
        evaluation["lean_gate"] = gate
        if isinstance(gate.get("comparison"), Mapping):
            evaluation["lean_comparison"] = gate["comparison"]
        elif gate.get("comparison_status"):
            evaluation["lean_comparison"] = {"status": gate.get("comparison_status")}
        if not _lean_gate_passed(evaluation):
            evaluation["promotion_eligible"] = False
            reason = str(gate.get("reason") or f"lean_gate_status={gate.get('status')}")
            prior = str(evaluation.get("promotion_reason") or "").strip()
            evaluation["promotion_reason"] = f"{prior}; LEAN gate failed: {reason}" if prior else f"LEAN gate failed: {reason}"
        return gate

    def _evaluation(self, idir: Path) -> None:
        out = idir / "evaluation.json"
        if out.exists():
            evaluation = load_json(out, {})
        else:
            backtest = load_json(idir / "backtest.json", {})
            if self.config.validation_protocol == VALIDATION_SINGLE:
                evaluation = score_strategy_loop_backtest(backtest, self.config)
            else:
                evaluation = score_triple_holdout_backtest(backtest, self.config)
            candidate = validate_candidate(idir / "candidate.json", default_n=self.config.n)
            signature = self._candidate_signature(candidate)
            evaluation["iteration"] = self.state.iteration
            evaluation["candidate"] = candidate
            evaluation["candidate_path"] = _as_repo_meta(idir / "candidate.json")
            evaluation["parameter_signature"] = signature
            if self.config.validation_protocol == VALIDATION_SINGLE:
                freqtrade = backtest.get("freqtrade_backtest") if isinstance(backtest.get("freqtrade_backtest"), Mapping) else {}
            else:
                stages = backtest.get("stages") if isinstance(backtest.get("stages"), Mapping) else {}
                validation = stages.get("validation") if isinstance(stages.get("validation"), Mapping) else {}
                freqtrade = validation.get("freqtrade_backtest") if isinstance(validation.get("freqtrade_backtest"), Mapping) else {}
            evaluation["freqtrade_backtest"] = freqtrade
            evaluation["validation_protocol"] = validation_protocol_summary(self.config)
            evaluation["verification_status"] = VERIFICATION_PASSED if self.config.validation_protocol == VALIDATION_SINGLE else VERIFICATION_PENDING
            evaluation["promotion_eligible"] = bool(evaluation.get("constraints_ok")) and self.config.validation_protocol == VALIDATION_SINGLE
            # Merge lean_gate results already written by PHASE_LEAN_ANALYSIS
            lean_gate_path = idir / "lean_gate.json"
            if lean_gate_path.exists() and not isinstance(evaluation.get("lean_gate"), Mapping):
                lg = load_json(lean_gate_path, {})
                if isinstance(lg, Mapping):
                    evaluation["lean_gate"] = lg
                    if isinstance(lg.get("comparison"), Mapping):
                        evaluation["lean_comparison"] = lg["comparison"]
                    if not _lean_gate_passed(evaluation):
                        evaluation["promotion_eligible"] = False
                        reason = str(lg.get("reason") or "lean_gate failed")
                        prior = str(evaluation.get("promotion_reason") or "").strip()
                        evaluation["promotion_reason"] = f"{prior}; LEAN gate failed: {reason}" if prior else f"LEAN gate failed: {reason}"
            # Merge lean_analysis results written by PHASE_LEAN_ANALYSIS
            lean_analysis_path = idir / "lean_analysis.json"
            if lean_analysis_path.exists():
                la = load_json(lean_analysis_path, {})
                if isinstance(la, Mapping):
                    evaluation["lean_analysis"] = la
            # Apply LEAN score blend (0.7 LEAN + 0.3 rank by default)
            apply_lean_score_blend(evaluation, self.config)
            score = float(evaluation.get("score") or float("-inf"))
            promotion_candidate = (
                score > self.state.best_score
                and self.config.validation_protocol == VALIDATION_SINGLE
                and self.config.promote_policy != PROMOTE_FINAL
            )
            if score > self.state.best_score:
                if self.config.validation_protocol == VALIDATION_SINGLE:
                    promotion = promote_candidate(candidate, evaluation, self.config, iter_dir=idir)
                else:
                    promotion = {
                        "promoted": False,
                        "artifacts": {},
                        "reason": "triple_holdout promotion deferred until blind finalization",
                    }
                best_dir = loop_root(self.config.run_id) / "best"
                promotion.setdefault("artifacts", {})["best_dir"] = _as_repo_meta(best_dir)
            else:
                promotion = {
                    "promoted": False,
                    "artifacts": {},
                    "reason": f"score did not exceed current best ({self.state.best_score:.6g})",
                }
            evaluation["promotion"] = promotion
            write_json(out, evaluation)
            artifact_refs = _artifact_refs_for_iteration(idir)
            evaluation["artifact_refs"] = artifact_refs
            write_json(idir / "manifest.json", build_iteration_manifest(idir, self.config, candidate, evaluation))
            evaluation["artifact_refs"] = _artifact_refs_for_iteration(idir)
            write_json(out, evaluation)
            if score > self.state.best_score:
                _copytree_replace(idir, loop_root(self.config.run_id) / "best")

        row = {
            "run_id": self.config.run_id,
            "iteration": self.state.iteration,
            "candidate_path": _as_repo_meta(idir / "candidate.json"),
            "candidate": evaluation.get("candidate"),
            "parameters": (evaluation.get("candidate") or {}).get("rank_profile") if isinstance(evaluation.get("candidate"), dict) else {},
            "strategy_path": (evaluation.get("candidate") or {}).get("strategy_path") if isinstance(evaluation.get("candidate"), dict) else None,
            "score": evaluation.get("score"),
            "score_components": evaluation.get("score_components") or {},
            "constraints_ok": evaluation.get("constraints_ok"),
            "metrics": evaluation.get("metrics"),
            "selected_metrics": evaluation.get("selected_metrics") or {},
            "research_metrics": evaluation.get("research_metrics") or evaluation.get("metrics"),
            "freqtrade_metrics": evaluation.get("freqtrade_metrics") or {},
            "lean_gate_status": (evaluation.get("lean_gate") or {}).get("status") if isinstance(evaluation.get("lean_gate"), Mapping) else None,
            "lean_comparison_status": (evaluation.get("lean_gate") or {}).get("comparison_status") if isinstance(evaluation.get("lean_gate"), Mapping) else None,
            "lean_metrics": (evaluation.get("lean_gate") or {}).get("lean_metrics") if isinstance(evaluation.get("lean_gate"), Mapping) else {},
            "lean_gate": evaluation.get("lean_gate"),
            "lean_score": (evaluation.get("score_components") or {}).get("lean_score"),
            "lean_analysis_summary": _lean_analysis_summary(evaluation.get("lean_analysis")),
            "window_metrics": evaluation.get("window_metrics") or {},
            "verification_status": evaluation.get("verification_status") or VERIFICATION_PENDING,
            "promotion_eligible": evaluation.get("promotion_eligible"),
            "artifact_refs": evaluation.get("artifact_refs") or {},
            "parameter_signature": evaluation.get("parameter_signature"),
            "violations": evaluation.get("violations"),
            "diagnostics": evaluation.get("promotion_reason"),
            "promotion": evaluation.get("promotion"),
        }
        self._append_leaderboard(row)
        score = float(evaluation.get("score") or float("-inf"))
        if score > self.state.best_score:
            self.state.best_score = score
            self.state.best_candidate = row
        self.state.score_history = [
            r
            for r in self.state.score_history
            if not (r.get("run_id") == row.get("run_id") and int(r.get("iteration") or -1) == int(row.get("iteration") or -2))
        ]
        self.state.score_history.append(row)
        self._update_stagnation(evaluation)
        self._refresh_pareto_pool()
        self._maybe_verify_iteration_candidate(idir, row, evaluation)

    def _lean_analysis_phase(self, idir: Path) -> None:
        """PHASE_LEAN_ANALYSIS: run LEAN gate, compute time-period analysis, optionally run LLM analysis."""
        lean_gate_path = idir / "lean_gate.json"
        lean_analysis_out = idir / "lean_analysis.json"

        # Step 1: Run LEAN gate (every iteration when lean_gate_mode != off)
        if not lean_gate_path.exists() and _lean_gate_active(self.config):
            lean_timerange = (
                self.config.timerange
                if self.config.validation_protocol == VALIDATION_SINGLE
                else self.config.validation_timerange
            )
            gate_result = self._run_lean_gate(idir, stage="iteration", timerange=lean_timerange)
            write_json(lean_gate_path, gate_result)

        # Step 2: Program-compute time-period metrics
        if not lean_analysis_out.exists() and lean_gate_path.exists():
            lean_gate = load_json(lean_gate_path, {})
            lean_result_path = None
            if isinstance(lean_gate, Mapping):
                artifacts = lean_gate.get("artifacts") if isinstance(lean_gate.get("artifacts"), Mapping) else {}
                lean_res_raw = artifacts.get("lean_result")
                if isinstance(lean_res_raw, Mapping):
                    lean_result_path = lean_res_raw.get("path")
                elif isinstance(lean_res_raw, str):
                    lean_result_path = lean_res_raw
                if not lean_result_path:
                    lean_result_path = lean_gate.get("lean_result")

            if lean_result_path:
                rank_curve: list = []
                backtest = load_json(idir / "backtest.json", {})
                if isinstance(backtest, Mapping):
                    # Try to get rank curve from current backtest or validation stage
                    if self.config.validation_protocol == VALIDATION_SINGLE:
                        curve_src = backtest
                    else:
                        stages = backtest.get("stages") if isinstance(backtest.get("stages"), Mapping) else {}
                        curve_src = stages.get("validation") or backtest
                    raw_curve = curve_src.get("curve") if isinstance(curve_src, Mapping) else None
                    if isinstance(raw_curve, list):
                        rank_curve = raw_curve

                try:
                    compute_lean_analysis(
                        lean_result=lean_result_path,
                        output=lean_analysis_out,
                        timeframe=self.config.timeframe,
                        rank_curve=rank_curve or None,
                    )
                except Exception as exc:
                    print(f"[lean_analysis] program compute failed for {idir.name}: {exc}")

        # Step 3: LLM analysis of lean metrics + equity curve
        lean_llm_out = idir / "lean_analysis.md"
        if not lean_llm_out.exists() and lean_analysis_out.exists():
            try:
                self._lean_llm_analysis(idir, lean_analysis_out, lean_llm_out)
            except Exception as exc:
                print(f"[lean_analysis] LLM analysis failed for {idir.name}: {exc}, skipping")

    def _lean_llm_analysis(self, idir: Path, lean_analysis_path: Path, output_path: Path) -> None:
        """Run a Hermes LLM call to produce lean_analysis.md, with fallback to program-generated summary."""
        lean_analysis = load_json(lean_analysis_path, {})
        if not isinstance(lean_analysis, Mapping):
            return

        regime = lean_analysis.get("regime_segments") if isinstance(lean_analysis.get("regime_segments"), Mapping) else {}
        dd_episodes = lean_analysis.get("drawdown_episodes") if isinstance(lean_analysis.get("drawdown_episodes"), list) else []
        monthly = lean_analysis.get("monthly_returns") if isinstance(lean_analysis.get("monthly_returns"), list) else []
        pair_contrib = lean_analysis.get("pair_contribution") if isinstance(lean_analysis.get("pair_contribution"), Mapping) else {}
        vs_rank = lean_analysis.get("vs_rank_comparison") if isinstance(lean_analysis.get("vs_rank_comparison"), Mapping) else {}

        # Write a structured fallback first (always present even if LLM fails)
        fallback_lines = [
            f"# LEAN Analysis — Iteration {self.state.iteration}",
            "",
            "## Monthly Performance",
        ]
        for m in monthly[-6:]:  # last 6 months
            fallback_lines.append(f"- {m.get('period')}: {m.get('return_pct', '?'):.2f}% (max_dd_in_period: {m.get('max_dd_in_period', '?'):.2f}%)")
        fallback_lines.extend([
            "",
            "## Drawdown Episodes",
        ])
        for ep in dd_episodes:
            rec = f"{ep.get('recovery_days')}d" if ep.get("recovered") else "not recovered"
            fallback_lines.append(f"- {ep.get('start')} → trough {ep.get('trough')}: {ep.get('depth_pct', '?'):.2f}%, duration {ep.get('duration_days')}d, recovery {rec}")
        fallback_lines.extend([
            "",
            "## Regime",
            f"- Positive months: {regime.get('positive_month_pct', '?'):.1f}%",
            f"- Consecutive loss streak: {regime.get('consecutive_loss_months', '?')}",
            f"- Worst month: {(regime.get('worst_month') or {}).get('period')} ({(regime.get('worst_month') or {}).get('return_pct', '?'):.2f}%)",
            f"- Best month: {(regime.get('best_month') or {}).get('period')} ({(regime.get('best_month') or {}).get('return_pct', '?'):.2f}%)",
            "",
            "## Pair Contribution",
            f"- Herfindahl index: {pair_contrib.get('herfindahl_index')}",
            f"- Top winners: {pair_contrib.get('top_winners')}",
            f"- Top losers: {pair_contrib.get('top_losers')}",
        ])
        if vs_rank.get("available"):
            fallback_lines.extend([
                "",
                "## LEAN vs Rank Divergence",
                f"- Divergence score: {vs_rank.get('divergence_score')}",
                f"- Mean abs monthly diff: {vs_rank.get('mean_abs_diff_pct'):.2f}%",
                f"- Worst divergence month: {vs_rank.get('worst_divergence_month')} ({vs_rank.get('worst_divergence_pct'):.2f}%)",
            ])
        output_path.write_text("\n".join(fallback_lines) + "\n", encoding="utf-8")

        # Try LLM analysis if Hermes is configured
        if not self.config.agent == AGENT_HERMES:
            return
        try:
            import json as _json
            # Build a compact context payload for the LLM
            context_summary = {
                "iteration": self.state.iteration,
                "regime": regime,
                "drawdown_episodes": dd_episodes,
                "monthly_returns": monthly,
                "pair_contribution": {
                    k: v for k, v in pair_contrib.items() if k != "pairs"
                },
                "pair_contribution_top": (pair_contrib.get("pairs") or [])[:10],
                "vs_rank_comparison": vs_rank,
                "equity_curve_sample": (lean_analysis.get("equity_curve") or [])[-20:],
            }
            prompt_path = idir / "context" / "lean_analysis_prompt.json"
            prompt_path.parent.mkdir(parents=True, exist_ok=True)
            write_json(prompt_path, {
                "task": "lean_analysis",
                "instruction": (
                    "You are analyzing a LEAN backtest result for a crypto rank portfolio strategy.\n"
                    "Write a concise analysis in Markdown covering:\n"
                    "1. **Headline P&L**: single paragraph summary (total return, max drawdown, profitable months %)\n"
                    "2. **Time-period strengths/weaknesses**: best/worst months, consecutive loss streaks\n"
                    "3. **Drawdown diagnosis**: for each top drawdown episode — likely cause (leverage? sector? market?)\n"
                    "4. **Pair contribution**: top winners, top losers, concentration risk (Herfindahl)\n"
                    "5. **LEAN vs Rank divergence**: when and why did real execution differ from ideal?\n"
                    "6. **Next iteration suggestions**: what to keep, what to fix, what to explore\n\n"
                    "Be specific. Reference months and depths by name. Avoid generic advice.\n"
                    "Output ONLY the markdown analysis, no preamble."
                ),
                "data": context_summary,
            })
            hermes_out = idir / "lean_analysis_hermes.txt"
            self._run_hermes_cli(idir, f"Read {prompt_path} and follow the task/instruction. Write your analysis output to {output_path} (overwrite it).")
        except Exception as exc:
            print(f"[lean_analysis] LLM analysis failed for {idir.name}: {exc}, keeping fallback markdown")

    def _analysis(self, idir: Path) -> None:
        path = idir / "analysis.md"
        if path.exists():
            return
        evaluation = load_json(idir / "evaluation.json", {})
        lean_analysis = load_json(idir / "lean_analysis.json", {}) if (idir / "lean_analysis.json").exists() else {}
        lines = [
            f"# Iteration {self.state.iteration} Analysis",
            "",
            f"Score: {evaluation.get('score')}",
            f"Constraints OK: {evaluation.get('constraints_ok')}",
            f"Violations: {evaluation.get('violations') or []}",
            "",
            "Score Components:",
        ]
        components = evaluation.get("score_components") or {}
        if isinstance(components, Mapping):
            for key in sorted(components):
                lines.append(f"- {key}: {components[key]}")
        lines.extend([
            "",
            "Metrics (Rank Backtest):",
        ])
        metrics = evaluation.get("metrics") or {}
        if isinstance(metrics, Mapping):
            for key in sorted(metrics):
                lines.append(f"- {key}: {metrics[key]}")
        freqtrade = evaluation.get("freqtrade_backtest") if isinstance(evaluation.get("freqtrade_backtest"), Mapping) else {}
        if freqtrade:
            lines.extend(["", "Freqtrade Stage:"])
            lines.append(f"- ok: {freqtrade.get('ok')}")
            if freqtrade.get("skipped"):
                lines.append(f"- skipped: {freqtrade.get('reason')}")
            ft_metrics = freqtrade.get("metrics") if isinstance(freqtrade.get("metrics"), Mapping) else {}
            for key in sorted(ft_metrics):
                lines.append(f"- {key}: {ft_metrics[key]}")
        # LEAN analysis summary
        if isinstance(lean_analysis, Mapping) and lean_analysis:
            la_summary = _lean_analysis_summary(lean_analysis)
            lines.extend(["", "LEAN Performance Summary:"])
            sc = evaluation.get("score_components") or {}
            if sc.get("lean_score") is not None:
                lines.append(f"- lean_score: {sc['lean_score']:.2f}  (blended weight: {sc.get('score_lean_weight', 0.7)})")
            for key, val in la_summary.items():
                lines.append(f"- {key}: {val}")
            regime = lean_analysis.get("regime_segments") if isinstance(lean_analysis.get("regime_segments"), Mapping) else {}
            monthly = lean_analysis.get("monthly_returns") if isinstance(lean_analysis.get("monthly_returns"), list) else []
            if monthly:
                lines.extend(["", "Recent Monthly Returns (LEAN):"])
                for m in monthly[-6:]:
                    lines.append(f"  {m.get('period')}: {m.get('return_pct', 0):.2f}% (max_dd: {m.get('max_dd_in_period', 0):.2f}%)")
            dd_episodes = lean_analysis.get("drawdown_episodes") if isinstance(lean_analysis.get("drawdown_episodes"), list) else []
            if dd_episodes:
                lines.extend(["", "Top Drawdown Episodes (LEAN):"])
                for ep in dd_episodes:
                    rec = f"{ep.get('recovery_days')}d" if ep.get("recovered") else "not recovered"
                    lines.append(f"  {ep.get('start')} → {ep.get('trough')}: depth {ep.get('depth_pct', 0):.2f}%, {ep.get('duration_days')}d, recovery {rec}")
        lines.extend(
            [
                "",
                "Next iteration guidance:",
                "- Optimize for LEAN P&L stability (monthly consistency, low drawdown depth) — not just rank backtest profit.",
                "- Preserve hard risk gates before increasing leverage or turnover.",
                "- See lean_analysis.md for LLM-generated detailed analysis.",
            ]
        )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _effective_rank_tag(self, idir: Path) -> str:
        return f"{self.config.tag}__loop_{self.config.run_id}_{idir.name}"

    def _candidate_signature(self, candidate: Mapping[str, Any]) -> str:
        if str(candidate.get("candidate_type")) != CANDIDATE_RANK_PROFILE:
            return ""
        profile = candidate.get("rank_profile") if isinstance(candidate.get("rank_profile"), Mapping) else {}
        if not profile:
            return ""
        return rank_profile_signature(profile, default_n=self.config.n)

    def _validate_unique_candidate(self, candidate: Mapping[str, Any]) -> None:
        signature = self._candidate_signature(candidate)
        if not signature:
            return
        for row in self.state.score_history:
            if not isinstance(row, Mapping):
                continue
            if _row_signature(row) == signature:
                iteration = row.get("iteration")
                raise ValueError(f"near-duplicate rank_profile signature already tried in iteration {iteration}: {signature}")
        if self.state.exploration_mode == "structured":
            metadata = candidate.get("metadata") if isinstance(candidate.get("metadata"), Mapping) else {}
            if str(metadata.get("search_mode") or "").strip().lower() != "structured_explore":
                raise ValueError("structured exploration requires metadata.search_mode=structured_explore")
            profile = candidate.get("rank_profile") if isinstance(candidate.get("rank_profile"), Mapping) else {}
            if not self._has_structural_rank_change(profile):
                raise ValueError("insufficient structural change for structured exploration")

    def _has_structural_rank_change(self, profile: Mapping[str, Any]) -> bool:
        if not profile:
            return False
        anchors: list[Mapping[str, Any]] = []
        if isinstance(self.state.best_candidate, Mapping):
            best_profile = _row_rank_profile(self.state.best_candidate)
            if best_profile:
                anchors.append(best_profile)
        baseline = _baseline_rank_profile(self.config)
        if baseline:
            anchors.append(baseline)
        if not anchors:
            return any(key in profile for key in STRUCTURAL_RANK_KEYS)
        for anchor in anchors:
            for key in STRUCTURAL_RANK_KEYS:
                if key in profile and profile.get(key) != anchor.get(key):
                    return True
        return False

    def _record_candidate_path(self, path: Path) -> None:
        rendered = _as_repo_meta(path)
        if rendered not in self.state.candidate_paths:
            self.state.candidate_paths.append(rendered)

    def _write_run_manifest(self) -> None:
        manifest_path = loop_root(self.config.run_id) / "manifest.json"
        if manifest_path.exists():
            return
        write_json(manifest_path, build_run_manifest(self.config))

    def _refresh_pareto_pool(self) -> dict[str, Any]:
        pool = build_pareto_pool(self.state.score_history, size_per_axis=self.config.pareto_size_per_axis)
        self.state.pareto_pool = pool
        write_json(loop_root(self.config.run_id) / "pareto_pool.json", pool)
        axes_by_identity: dict[str, list[str]] = {}
        for finalist in pool.get("finalists") or []:
            if not isinstance(finalist, Mapping):
                continue
            ident = str(finalist.get("parameter_signature") or finalist.get("candidate_path") or finalist.get("iteration"))
            axes_by_identity[ident] = list(finalist.get("pareto_axes") or [])
        for row in self.state.score_history:
            ident = str(row.get("parameter_signature") or row.get("candidate_path") or row.get("iteration"))
            if ident in axes_by_identity:
                row["pareto_axes"] = axes_by_identity[ident]
        leaderboard = load_json(leaderboard_path(self.config.run_id), {"version": "factor-strategy-loop-leaderboard-v1", "rows": []})
        rows = list(leaderboard.get("rows") or [])
        for row in rows:
            ident = str(row.get("parameter_signature") or row.get("candidate_path") or row.get("iteration"))
            row["pareto_axes"] = axes_by_identity.get(ident, [])
        leaderboard["rows"] = rows
        write_json(leaderboard_path(self.config.run_id), leaderboard)
        return pool

    def _maybe_verify_iteration_candidate(self, idir: Path, row: dict[str, Any], evaluation: dict[str, Any]) -> None:
        if self.config.validation_protocol == VALIDATION_SINGLE or self.config.verify_policy == VERIFY_NONE:
            return
        should_verify = self.config.verify_policy == VERIFY_ALL
        if self.config.verify_policy == VERIFY_BEST and self.state.best_candidate is row:
            should_verify = True
        if self.config.verify_policy == VERIFY_PARETO and row.get("pareto_axes"):
            should_verify = True
        if not should_verify:
            return
        backtest = load_json(idir / "backtest.json", {})
        stages = backtest.get("stages") if isinstance(backtest, Mapping) and isinstance(backtest.get("stages"), Mapping) else {}
        validation = stages.get("validation") if isinstance(stages.get("validation"), Mapping) else {}
        if not validation:
            return
        verification = self._run_validation_gates(
            idir,
            validation,
            timerange=self.config.validation_timerange,
            gate_label="validation",
        )
        status = str(verification.get("status") or VERIFICATION_PENDING)
        evaluation["verification"] = verification
        evaluation["verification_status"] = status
        evaluation["promotion_eligible"] = False
        evaluation["artifact_refs"] = _artifact_refs_for_iteration(idir)
        write_json(idir / "evaluation.json", evaluation)
        write_json(idir / "manifest.json", build_iteration_manifest(idir, self.config, evaluation.get("candidate") or {}, evaluation))
        evaluation["artifact_refs"] = _artifact_refs_for_iteration(idir)
        write_json(idir / "evaluation.json", evaluation)
        row["verification_status"] = status
        row["promotion_eligible"] = False
        row["artifact_refs"] = evaluation["artifact_refs"]
        self._append_leaderboard(row)

    def _update_stagnation(self, evaluation: Mapping[str, Any]) -> None:
        components = evaluation.get("score_components") if isinstance(evaluation.get("score_components"), Mapping) else {}
        if not components:
            return
        try:
            composite = float(components.get("composite_score"))
        except (TypeError, ValueError):
            return
        if not math.isfinite(composite):
            return
        self.state.valid_candidate_count += 1
        if composite > self.state.best_composite_score + 1e-9:
            self.state.best_composite_score = composite
            self.state.no_composite_improvement_count = 0
            self.state.exploration_mode = "local"
            return
        self.state.no_composite_improvement_count += 1
        if self.state.no_composite_improvement_count >= STAGNATION_EXPLORE_AFTER:
            self.state.exploration_mode = "structured"
        if self.state.no_composite_improvement_count >= STAGNATION_STOP_AFTER:
            self.state.status = LOOP_STOPPED_STAGNATED
            self.state.stopped_reason = (
                f"{self.state.no_composite_improvement_count} valid candidates without composite improvement"
            )

    def _freqtrade_validation_base(self, stage_result: Mapping[str, Any]) -> tuple[Optional[Path], Optional[Path], Optional[Path]]:
        signals_raw = str(stage_result.get("signals") or "")
        signal_dir: Optional[Path] = None
        if signals_raw:
            signals_path = Path(signals_raw).expanduser()
            if signals_path.exists():
                signal_dir = signals_path.parent
        config_path = repo_paths.resolve_repo_path(FIXED_FREQTRADE_CONFIG)
        if not config_path.exists():
            fallback = repo_paths.REPO_ROOT / FIXED_FREQTRADE_CONFIG
            config_path = fallback if fallback.exists() else config_path
        strategy_dir = repo_paths.user_data_root() / "strategies"
        if not (strategy_dir / f"{FIXED_FREQTRADE_STRATEGY}.py").exists():
            fallback_dir = repo_paths.REPO_ROOT / "user_data" / "strategies"
            strategy_dir = fallback_dir if (fallback_dir / f"{FIXED_FREQTRADE_STRATEGY}.py").exists() else strategy_dir
        return signal_dir, config_path if config_path.exists() else None, strategy_dir if (strategy_dir / f"{FIXED_FREQTRADE_STRATEGY}.py").exists() else None

    def _run_validation_gates(
        self,
        idir: Path,
        stage_result: Mapping[str, Any],
        *,
        timerange: str,
        gate_label: str,
    ) -> dict[str, Any]:
        gate_dir = idir / "validation_gates" / gate_label
        gate_dir.mkdir(parents=True, exist_ok=True)
        signal_dir, config_path, strategy_dir = self._freqtrade_validation_base(stage_result)
        if signal_dir is None or config_path is None or strategy_dir is None:
            result = {
                "status": VERIFICATION_INCONCLUSIVE,
                "lookahead": {"status": VERIFICATION_INCONCLUSIVE, "violations": ["missing signal/config/strategy inputs"]},
                "recursive": {"status": VERIFICATION_INCONCLUSIVE, "violations": ["missing signal/config/strategy inputs"]},
                "artifacts": {"dir": _as_repo_meta(gate_dir)},
            }
            write_json(idir / "verification.json", result)
            return result

        env = dict(os.environ)
        env["RP_SIGNAL_DIR"] = str(signal_dir)
        env["RP_TAG"] = str(stage_result.get("tag") or self.config.tag)
        lookahead_csv = gate_dir / "lookahead.csv"
        lookahead_cmd = [
            sys.executable,
            str(repo_paths.REPO_ROOT / "scripts" / "freqtrade_cli.py"),
            "lookahead-analysis",
            "--config",
            str(config_path),
            "--strategy",
            FIXED_FREQTRADE_STRATEGY,
            "--strategy-path",
            str(strategy_dir),
            "--timerange",
            timerange,
            "--minimum-trade-amount",
            str(scaled_gate_values(self.config, timerange)["min_trades"]),
            "--lookahead-analysis-exportfilename",
            str(lookahead_csv),
        ]
        recursive_log = gate_dir / "recursive.log"
        recursive_cmd = [
            sys.executable,
            str(repo_paths.REPO_ROOT / "scripts" / "freqtrade_cli.py"),
            "recursive-analysis",
            "--config",
            str(config_path),
            "--strategy",
            FIXED_FREQTRADE_STRATEGY,
            "--strategy-path",
            str(strategy_dir),
            "--timerange",
            timerange,
        ]

        try:
            lookahead_proc = subprocess.run(
                lookahead_cmd,
                cwd=str(repo_paths.REPO_ROOT),
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=max(300.0, float(self.config.stale_timeout) + 300.0),
                check=False,
            )
            (gate_dir / "lookahead.log").write_text(lookahead_proc.stdout or "", encoding="utf-8")
        except subprocess.TimeoutExpired as exc:
            lookahead_proc = subprocess.CompletedProcess(lookahead_cmd, 124, stdout=str(exc))
            (gate_dir / "lookahead.log").write_text(str(exc), encoding="utf-8")
        min_gate_trades = int(scaled_gate_values(self.config, timerange)["min_trades"])
        lookahead = parse_lookahead_csv(lookahead_csv, strategy=FIXED_FREQTRADE_STRATEGY, min_trades=min_gate_trades)
        if lookahead_proc.returncode != 0:
            lookahead["status"] = VERIFICATION_FAILED
            lookahead.setdefault("violations", []).append(f"lookahead command exited {lookahead_proc.returncode}")

        try:
            recursive_proc = subprocess.run(
                recursive_cmd,
                cwd=str(repo_paths.REPO_ROOT),
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=max(300.0, float(self.config.stale_timeout) + 300.0),
                check=False,
            )
            recursive_log.write_text(recursive_proc.stdout or "", encoding="utf-8")
        except subprocess.TimeoutExpired as exc:
            recursive_proc = subprocess.CompletedProcess(recursive_cmd, 124, stdout=str(exc))
            recursive_log.write_text(str(exc), encoding="utf-8")
        recursive = parse_recursive_output(recursive_log)
        if recursive_proc.returncode != 0:
            recursive["status"] = VERIFICATION_INCONCLUSIVE
            recursive.setdefault("violations", []).append(f"recursive command exited {recursive_proc.returncode}")

        result = {
            "status": combine_verification_status({"lookahead": lookahead, "recursive": recursive}),
            "lookahead": lookahead,
            "recursive": recursive,
            "commands": {"lookahead": lookahead_cmd, "recursive": recursive_cmd},
            "artifacts": {
                "dir": _as_repo_meta(gate_dir),
                "lookahead_csv": _as_repo_meta(lookahead_csv),
                "lookahead_log": _as_repo_meta(gate_dir / "lookahead.log"),
                "recursive_log": _as_repo_meta(recursive_log),
            },
        }
        write_json(idir / "verification.json", result)
        return result

    def _deepresearch_sidecar(self, final_status: Mapping[str, Any]) -> dict[str, Any]:
        root = repo_paths.artifacts_root() / "strategy_deepresearch" / self.config.run_id
        root.mkdir(parents=True, exist_ok=True)
        findings: list[dict[str, Any]] = []
        if self.config.validation_protocol != VALIDATION_TRIPLE_HOLDOUT:
            findings.append({"severity": "HIGH", "message": "formal promotion did not use triple_holdout"})
        if str((final_status.get("selected") or {}).get("verification_status") or "") != VERIFICATION_PASSED:
            findings.append({"severity": "BLOCKER", "message": "selected candidate did not pass lookahead/recursive gates"})
        if not bool((final_status.get("selected") or {}).get("blind_final")):
            findings.append({"severity": "BLOCKER", "message": "selected candidate is not backed by blind evaluation"})
        if _lean_gate_active(self.config) and _lean_gate_status(final_status.get("selected") or {}) != VERIFICATION_PASSED:
            findings.append({"severity": "BLOCKER", "message": "selected candidate did not pass LEAN promotion gate"})
        status = VERIFICATION_FAILED if any(f.get("severity") in {"BLOCKER", "HIGH"} for f in findings) else VERIFICATION_PASSED
        context = {
            "run_id": self.config.run_id,
            "validation_protocol": validation_protocol_summary(self.config),
            "leaderboard": _as_repo_meta(leaderboard_path(self.config.run_id)),
            "pareto_pool": _as_repo_meta(loop_root(self.config.run_id) / "pareto_pool.json"),
            "final_status": final_status,
            "findings": findings,
            "status": status,
        }
        write_json(root / "context.json", context)
        write_json(root / "sources.json", {"sources": [], "note": "local artifact audit only; no external sources used"})
        review = [
            "# Strategy Research Review",
            "",
            f"Run id: `{self.config.run_id}`",
            f"Status: `{status}`",
            "",
            "## Findings",
        ]
        if findings:
            review.extend(f"- {item['severity']}: {item['message']}" for item in findings)
        else:
            review.append("- No BLOCKER/HIGH findings in the controller-side audit artifacts.")
        review.extend([
            "",
            "## Evidence",
            f"- Leaderboard: `{_as_repo_meta(leaderboard_path(self.config.run_id))}`",
            f"- Pareto pool: `{_as_repo_meta(loop_root(self.config.run_id) / 'pareto_pool.json')}`",
            f"- Final blind status: `{_as_repo_meta(loop_root(self.config.run_id) / 'final_blind_status.json')}`",
        ])
        (repo_paths.REPO_ROOT / "docs" / "strategy_research_review.md").write_text("\n".join(review) + "\n", encoding="utf-8")
        protocol = validation_protocol_summary(self.config)
        protocol_doc = [
            "# Validation Protocol",
            "",
            f"Run id: `{self.config.run_id}`",
            "",
            "## Windows",
        ]
        for stage, payload in (protocol.get("windows") or {}).items():
            protocol_doc.append(f"- {stage}: `{payload.get('timerange')}` gates={payload.get('gates')}")
        protocol_doc.extend([
            "",
            "## Promotion Rule",
            "- Search generates and filters candidates.",
            "- Validation ranks leaderboard/Pareto candidates.",
            "- Blind holdout is run only for Pareto finalists.",
            "- Promotion requires blind selected gates plus lookahead/recursive verification status `passed`.",
            "- When `lean_gate_mode` is enabled, promotion also requires LEAN gate status `passed`.",
        ])
        (repo_paths.REPO_ROOT / "docs" / "validation_protocol.md").write_text("\n".join(protocol_doc) + "\n", encoding="utf-8")
        return {"status": status, "artifacts": {"context": _as_repo_meta(root / "context.json"), "sources": _as_repo_meta(root / "sources.json")}, "findings": findings}

    def _finalize_promotion(self) -> Optional[dict[str, Any]]:
        if not self.config.promote or self.config.promote_policy != PROMOTE_FINAL:
            return None
        if self.config.validation_protocol != VALIDATION_SINGLE:
            return self._finalize_triple_holdout()
        best_dir = loop_root(self.config.run_id) / "best"
        candidate_path = best_dir / "candidate.json"
        evaluation_path = best_dir / "evaluation.json"
        if not candidate_path.exists() or not evaluation_path.exists():
            return {"promoted": False, "artifacts": {}, "reason": "no run-local best candidate available"}
        candidate = validate_candidate(candidate_path, default_n=self.config.n)
        evaluation = load_json(evaluation_path, {})
        if not isinstance(evaluation, Mapping):
            return {"promoted": False, "artifacts": {}, "reason": "best evaluation.json is invalid"}
        evaluation = dict(evaluation)
        if self._should_run_lean_gate("final", promotion_candidate=True) and not isinstance(evaluation.get("lean_gate"), Mapping):
            self._apply_lean_gate(best_dir, evaluation, stage="final", timerange=self.config.timerange)
            evaluation["artifact_refs"] = _artifact_refs_for_iteration(best_dir)
            write_json(evaluation_path, evaluation)
            write_json(best_dir / "manifest.json", build_iteration_manifest(best_dir, self.config, candidate, evaluation))
            evaluation["artifact_refs"] = _artifact_refs_for_iteration(best_dir)
            write_json(evaluation_path, evaluation)
        promotion = promote_candidate(candidate, evaluation, self.config, iter_dir=best_dir, final=True)
        final_path = loop_root(self.config.run_id) / "final_promotion.json"
        write_json(final_path, promotion)
        updated = dict(evaluation)
        updated["final_promotion"] = promotion
        write_json(evaluation_path, updated)
        if isinstance(self.state.best_candidate, dict):
            self.state.best_candidate["final_promotion"] = promotion
        return promotion

    def _finalize_triple_holdout(self) -> dict[str, Any]:
        pool = self._refresh_pareto_pool()
        finalists = pool.get("finalists") if isinstance(pool.get("finalists"), list) else []
        if not finalists:
            promotion = {"promoted": False, "artifacts": {}, "reason": "no Pareto finalists available"}
            status = {
                "promoted": False,
                "promotion": promotion,
                "selected": None,
                "finalists": [],
            }
            self.state.final_blind_status = status
            write_json(loop_root(self.config.run_id) / "final_blind_status.json", status)
            write_json(loop_root(self.config.run_id) / "final_promotion.json", promotion)
            return promotion

        final_rows: list[dict[str, Any]] = []
        for finalist in finalists:
            if not isinstance(finalist, Mapping):
                continue
            candidate_path_raw = str(finalist.get("candidate_path") or "")
            if not candidate_path_raw:
                continue
            candidate_path = repo_paths.resolve_repo_path(candidate_path_raw)
            if not candidate_path.exists():
                final_rows.append({"finalist": finalist, "ok": False, "reason": f"candidate missing: {candidate_path_raw}"})
                continue
            candidate = validate_candidate(candidate_path, default_n=self.config.n)
            iteration_value = finalist.get("iteration") or candidate_path.parent.name
            blind_dir = loop_root(self.config.run_id) / f"blind_{str(iteration_value).replace('/', '_')}"
            blind_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate_path, blind_dir / "candidate.json")
            blind_result = self._run_window_backtest(
                blind_dir,
                candidate,
                stage="blind",
                timerange=self.config.blind_timerange,
                run_freqtrade=False,
            )
            blind_gate_values = scaled_gate_values(self.config, self.config.blind_timerange)
            blind_stage_a = score_backtest_result(
                blind_result,
                min_trades=blind_gate_values["min_trades"],
                max_drawdown_pct=blind_gate_values["max_drawdown_pct"],
                min_profit_over_dd=blind_gate_values["min_profit_over_dd"],
                target_profit_pct=blind_gate_values["target_profit_pct"],
            )
            blind_result["stage_a"] = {
                "constraints_ok": blind_stage_a["constraints_ok"],
                "score": blind_stage_a["score"],
                "violations": blind_stage_a["violations"],
            }
            if self.config.eval_mode in {EVAL_TWO_STAGE, EVAL_FREQTRADE}:
                if self.config.eval_mode == EVAL_FREQTRADE or blind_stage_a["constraints_ok"]:
                    blind_result["freqtrade_backtest"] = self._run_fixed_freqtrade_backtest(
                        blind_dir,
                        blind_result,
                        timerange=self.config.blind_timerange,
                        stage="blind",
                    )
                else:
                    blind_result["freqtrade_backtest"] = {
                        "ok": False,
                        "skipped": True,
                        "reason": "Blind research gates failed",
                        "stage_a_violations": blind_stage_a.get("violations") or [],
                    }
            write_json(blind_dir / "backtest.json", blind_result)
            blind_eval = score_strategy_loop_backtest(blind_result, self.config, gates=blind_gate_values)
            blind_eval["iteration"] = finalist.get("iteration")
            blind_eval["candidate"] = candidate
            blind_eval["candidate_path"] = _as_repo_meta(blind_dir / "candidate.json")
            blind_eval["source_candidate_path"] = candidate_path_raw
            blind_eval["parameter_signature"] = finalist.get("parameter_signature") or self._candidate_signature(candidate)
            blind_eval["window_metrics"] = {
                "blind": _stage_window_metrics(blind_result, blind_eval),
            }
            blind_eval["selected_window"] = "blind"
            blind_eval["blind_final"] = True
            validation_stage: Mapping[str, Any] = blind_result
            source_backtest = load_json(candidate_path.parent / "backtest.json", {})
            if isinstance(source_backtest, Mapping):
                stages = source_backtest.get("stages") if isinstance(source_backtest.get("stages"), Mapping) else {}
                if isinstance(stages.get("validation"), Mapping):
                    validation_stage = stages["validation"]

            if self.config.verify_policy == VERIFY_NONE:
                verification = {"status": VERIFICATION_PENDING, "reason": "verify_policy=none"}
            else:
                verification = self._run_validation_gates(
                    blind_dir,
                    validation_stage,
                    timerange=self.config.validation_timerange,
                    gate_label="final_validation",
                )
            verification_status = str(verification.get("status") or VERIFICATION_PENDING)
            blind_eval["verification"] = verification
            blind_eval["verification_status"] = verification_status
            base_promotion_eligible = bool(blind_eval.get("constraints_ok")) and verification_status == VERIFICATION_PASSED
            blind_eval["promotion_eligible"] = base_promotion_eligible
            if self._should_run_lean_gate("blind", promotion_candidate=base_promotion_eligible):
                self._apply_lean_gate(blind_dir, blind_eval, stage="blind", timerange=self.config.blind_timerange)
                blind_eval["promotion_eligible"] = base_promotion_eligible and _lean_gate_passed(blind_eval)
            lean_status = _lean_gate_status(blind_eval) if _lean_gate_active(self.config) else ""
            blind_eval["promotion_reason"] = (
                "blind window, verification gates, and LEAN gate passed"
                if blind_eval["promotion_eligible"]
                else (
                    f"blind/verification/LEAN failed: {blind_eval.get('violations') or []}; "
                    f"verification={verification_status}; lean={lean_status or 'off'}"
                )
            )
            blind_eval["artifact_refs"] = _artifact_refs_for_iteration(blind_dir)
            write_json(blind_dir / "evaluation.json", blind_eval)
            write_json(blind_dir / "manifest.json", build_iteration_manifest(blind_dir, self.config, candidate, blind_eval))
            blind_eval["artifact_refs"] = _artifact_refs_for_iteration(blind_dir)
            write_json(blind_dir / "evaluation.json", blind_eval)
            final_rows.append(
                {
                    "finalist": finalist,
                    "blind_dir": _as_repo_meta(blind_dir),
                    "score": blind_eval.get("score"),
                    "constraints_ok": blind_eval.get("constraints_ok"),
                    "verification_status": verification_status,
                    "lean_gate_status": (blind_eval.get("lean_gate") or {}).get("status") if isinstance(blind_eval.get("lean_gate"), Mapping) else None,
                    "lean_comparison_status": (blind_eval.get("lean_gate") or {}).get("comparison_status") if isinstance(blind_eval.get("lean_gate"), Mapping) else None,
                    "promotion_eligible": blind_eval["promotion_eligible"],
                    "blind_final": True,
                    "evaluation": blind_eval,
                }
            )

        eligible = [row for row in final_rows if row.get("promotion_eligible")]
        selected: Optional[dict[str, Any]] = None
        if eligible:
            eligible.sort(key=lambda row: float(row.get("score") or float("-inf")), reverse=True)
            selected = eligible[0]
            promotion = {
                "promoted": False,
                "artifacts": {},
                "reason": "pending deepresearch audit",
            }
        else:
            promotion = {
                "promoted": False,
                "artifacts": {},
                "reason": "no Pareto finalist passed blind holdout and verification gates",
            }
        final_status = {
            "promoted": promotion.get("promoted"),
            "promotion": promotion,
            "selected": selected["evaluation"] if selected else None,
            "finalists": final_rows,
        }
        audit = self._deepresearch_sidecar(final_status)
        final_status["deepresearch"] = audit
        if selected and audit.get("status") == VERIFICATION_PASSED:
            selected_eval = selected["evaluation"]
            selected_candidate = selected_eval["candidate"]
            selected_dir = repo_paths.resolve_repo_path(str(selected["blind_dir"]))
            final_status["promotion"] = promote_candidate(selected_candidate, selected_eval, self.config, iter_dir=selected_dir, final=True)
            final_status["promoted"] = bool(final_status["promotion"].get("promoted"))
        elif selected:
            final_status["promotion"] = {"promoted": False, "artifacts": {}, "reason": "deepresearch BLOCKER/HIGH finding blocks promotion"}
            final_status["promoted"] = False
        self.state.final_blind_status = final_status
        write_json(loop_root(self.config.run_id) / "final_blind_status.json", final_status)
        write_json(loop_root(self.config.run_id) / "final_promotion.json", final_status["promotion"])
        return final_status["promotion"]

    def _append_leaderboard(self, row: Mapping[str, Any]) -> None:
        path = leaderboard_path(self.config.run_id)
        payload = load_json(path, {"version": "factor-strategy-loop-leaderboard-v1", "rows": []})
        rows = list(payload.get("rows") or [])
        rows = [
            r
            for r in rows
            if not (r.get("run_id") == row.get("run_id") and int(r.get("iteration") or -1) == int(row.get("iteration") or -2))
        ]
        rows.append(dict(row))
        payload["rows"] = rows
        write_json(path, payload)

    def _record_iteration_failure(self, idir: Path, phase: str, exc: Exception) -> None:
        message = _failure_message(phase, exc)
        traceback_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        error = {
            "run_id": self.config.run_id,
            "iteration": self.state.iteration,
            "phase": phase,
            "error_type": exc.__class__.__name__,
            "message": _truncate_text(str(exc), limit=1000),
            "traceback": _truncate_text(traceback_text, limit=4000),
        }
        write_json(idir / "error.json", error)

        raw_candidate = load_json(idir / "candidate.json", None)
        if (idir / "candidate.json").exists():
            self._record_candidate_path(idir / "candidate.json")
        parameter_signature = ""
        if isinstance(raw_candidate, Mapping) and isinstance(raw_candidate.get("rank_profile"), Mapping):
            try:
                parameter_signature = rank_profile_signature(raw_candidate["rank_profile"], default_n=self.config.n)
            except Exception:
                parameter_signature = ""
        evaluation = {
            "score": FAILED_ITERATION_SCORE,
            "constraints_ok": False,
            "metrics": {},
            "research_metrics": {},
            "freqtrade_metrics": {},
            "window_metrics": {},
            "verification_status": VERIFICATION_FAILED,
            "promotion_eligible": False,
            "artifact_refs": _artifact_refs_for_iteration(idir),
            "score_components": {
                "score_mode": self.config.score_mode,
                "research_score": FAILED_ITERATION_SCORE,
                "freqtrade_score": FAILED_ITERATION_SCORE,
                "composite_score": FAILED_ITERATION_SCORE,
                "selection_reason": message,
            },
            "violations": [message],
            "promotion_reason": message,
            "candidate": raw_candidate if isinstance(raw_candidate, Mapping) else None,
            "candidate_path": _as_repo_meta(idir / "candidate.json") if (idir / "candidate.json").exists() else None,
            "parameter_signature": parameter_signature,
            "promotion": {"promoted": False, "artifacts": {}, "reason": message},
        }
        write_json(idir / "evaluation.json", evaluation)

        row = {
            "run_id": self.config.run_id,
            "iteration": self.state.iteration,
            "candidate_path": evaluation.get("candidate_path"),
            "candidate": evaluation.get("candidate"),
            "parameters": (evaluation.get("candidate") or {}).get("rank_profile") if isinstance(evaluation.get("candidate"), dict) else {},
            "strategy_path": (evaluation.get("candidate") or {}).get("strategy_path") if isinstance(evaluation.get("candidate"), dict) else None,
            "score": FAILED_ITERATION_SCORE,
            "score_components": evaluation["score_components"],
            "constraints_ok": False,
            "metrics": {},
            "research_metrics": {},
            "freqtrade_metrics": {},
            "window_metrics": {},
            "verification_status": VERIFICATION_FAILED,
            "promotion_eligible": False,
            "artifact_refs": evaluation.get("artifact_refs") or {},
            "parameter_signature": parameter_signature,
            "violations": [message],
            "diagnostics": message,
            "promotion": evaluation["promotion"],
        }
        self._append_leaderboard(row)
        self.state.score_history = [
            r
            for r in self.state.score_history
            if not (r.get("run_id") == row.get("run_id") and int(r.get("iteration") or -1) == int(row.get("iteration") or -2))
        ]
        self.state.score_history.append(row)


def run_strategy_loop(**kwargs: Any) -> dict[str, Any]:
    config = StrategyLoopConfig.from_args(**kwargs)
    runner = StrategyLoopRunner(config)
    return runner.run()


def evaluate_candidate(
    candidate_path: str | Path,
    *,
    tag: str = rank_portfolio.DEFAULT_TAG,
    venue: str = "okx",
    risk_profile: str = "aggressive",
    timeframe: str = "1h",
    data_venue: str = "auto",
    evaluation_lane: str = "auto",
    timerange: Optional[str] = None,
    n: int = 50,
    run_id: Optional[str] = None,
    promote: bool = False,
    candidate_state: Optional[str] = None,
    recompute_corr: Optional[bool] = None,
    baseline_profile: Optional[str] = None,
    eval_mode: str = EVAL_RESEARCH,
    score_mode: str = SCORE_RESEARCH,
    promote_policy: str = PROMOTE_IMMEDIATE,
    validation_protocol: str = VALIDATION_SINGLE,
    search_timerange: Optional[str] = None,
    validation_timerange: Optional[str] = None,
    blind_timerange: Optional[str] = None,
    verify_policy: Optional[str] = None,
    pareto_size_per_axis: int = 3,
    lean_gate_mode: str = LEAN_GATE_OFF,
    lean_bin: str = "lean",
    lean_timeout: Optional[int] = None,
    lean_required_status: str = "ok",
    lean_data_root: Optional[str] = None,
) -> dict[str, Any]:
    config = StrategyLoopConfig.from_args(
        tag=tag,
        venue=venue,
        risk_profile=risk_profile,
        timeframe=timeframe,
        data_venue=data_venue,
        evaluation_lane=evaluation_lane,
        timerange=timerange,
        n=n,
        run_id=run_id or make_run_id(f"{tag}_eval"),
        max_iterations=1,
        promote=promote,
        candidate_state=candidate_state,
        recompute_corr=recompute_corr,
        baseline_profile=baseline_profile,
        eval_mode=eval_mode,
        score_mode=score_mode,
        promote_policy=promote_policy,
        validation_protocol=validation_protocol,
        search_timerange=search_timerange,
        validation_timerange=validation_timerange,
        blind_timerange=blind_timerange,
        verify_policy=verify_policy,
        pareto_size_per_axis=pareto_size_per_axis,
        lean_gate_mode=lean_gate_mode,
        lean_bin=lean_bin,
        lean_timeout=lean_timeout,
        lean_required_status=lean_required_status,
        lean_data_root=lean_data_root,
    )
    candidate = validate_candidate(candidate_path, default_n=config.n)
    idir = Path(candidate_path).resolve().parent
    runner = StrategyLoopRunner(config)
    if config.validation_protocol == VALIDATION_SINGLE:
        backtest = runner._run_window_backtest(
            idir,
            candidate,
            stage="single",
            timerange=config.timerange,
            run_freqtrade=False,
        )
        if config.eval_mode == EVAL_FREQTRADE:
            stage_a = {"constraints_ok": True, "score": 0, "violations": []}
        else:
            stage_a = score_backtest_result(
                backtest,
                min_trades=config.min_trades,
                max_drawdown_pct=config.max_drawdown_pct,
                min_profit_over_dd=config.min_profit_over_dd,
                target_profit_pct=config.target_profit_pct,
            )
        backtest["stage_a"] = {
            "constraints_ok": stage_a["constraints_ok"],
            "score": stage_a["score"],
            "violations": stage_a["violations"],
        }
        if config.eval_mode in {EVAL_TWO_STAGE, EVAL_FREQTRADE}:
            if config.eval_mode == EVAL_FREQTRADE or bool(stage_a.get("constraints_ok")):
                backtest["freqtrade_backtest"] = runner._run_fixed_freqtrade_backtest(idir, backtest, timerange=config.timerange, stage="single")
            else:
                backtest["freqtrade_backtest"] = {
                    "ok": False,
                    "skipped": True,
                    "reason": "Stage A research gates failed",
                    "stage_a_violations": stage_a.get("violations") or [],
                }
        evaluation = score_strategy_loop_backtest(backtest, config)
        freqtrade = backtest.get("freqtrade_backtest") if isinstance(backtest.get("freqtrade_backtest"), Mapping) else {}
    else:
        backtest = runner._run_triple_holdout_backtest(idir, candidate)
        evaluation = score_triple_holdout_backtest(backtest, config)
        stages = backtest.get("stages") if isinstance(backtest.get("stages"), Mapping) else {}
        validation = stages.get("validation") if isinstance(stages.get("validation"), Mapping) else {}
        freqtrade = validation.get("freqtrade_backtest") if isinstance(validation.get("freqtrade_backtest"), Mapping) else {}
    write_json(idir / "backtest.json", backtest)
    evaluation["iteration"] = 1
    evaluation["candidate"] = candidate
    evaluation["candidate_path"] = _as_repo_meta(Path(candidate_path))
    evaluation["parameter_signature"] = (
        rank_profile_signature(candidate.get("rank_profile") or {}, default_n=config.n)
        if str(candidate.get("candidate_type")) == CANDIDATE_RANK_PROFILE
        else ""
    )
    evaluation["freqtrade_backtest"] = freqtrade
    evaluation["validation_protocol"] = validation_protocol_summary(config)
    evaluation["verification_status"] = VERIFICATION_PASSED if config.validation_protocol == VALIDATION_SINGLE else VERIFICATION_PENDING
    evaluation["promotion_eligible"] = bool(evaluation.get("constraints_ok")) and config.validation_protocol == VALIDATION_SINGLE
    if runner._should_run_lean_gate(
        "iteration",
        promotion_candidate=bool(promote) and config.validation_protocol == VALIDATION_SINGLE and config.promote_policy != PROMOTE_FINAL,
    ):
        runner._apply_lean_gate(idir, evaluation, stage="iteration", timerange=config.timerange)
    evaluation["promotion"] = promote_candidate(candidate, evaluation, config, iter_dir=idir)
    evaluation["artifact_refs"] = _artifact_refs_for_iteration(idir)
    write_json(idir / "manifest.json", build_iteration_manifest(idir, config, candidate, evaluation))
    evaluation["artifact_refs"] = _artifact_refs_for_iteration(idir)
    write_json(idir / "evaluation.json", evaluation)
    evaluation["artifact_refs"] = _artifact_refs_for_iteration(idir)
    write_json(idir / "evaluation.json", evaluation)
    return evaluation


def replay_optimized_profile(
    *,
    tag: str = rank_portfolio.DEFAULT_TAG,
    baseline_profile: Optional[str] = None,
    venue: str = "okx",
    risk_profile: str = "aggressive",
    timerange: Optional[str] = None,
    include_freqtrade: bool = True,
) -> dict[str, Any]:
    config = StrategyLoopConfig.from_args(
        tag=tag,
        venue=venue,
        risk_profile=risk_profile,
        timerange=timerange,
        baseline_profile=baseline_profile,
        eval_mode=EVAL_TWO_STAGE if include_freqtrade else EVAL_RESEARCH,
        candidate_type=CANDIDATE_RANK_PROFILE,
    )
    baseline = _load_optimized_baseline(config)
    if not baseline.get("available"):
        raise FileNotFoundError(f"optimized baseline profile not found for tag={tag}: {baseline.get('path')}")
    profile = baseline.get("rank_profile") if isinstance(baseline.get("rank_profile"), Mapping) else {}
    run_id = make_run_id(f"{tag}_replay")
    config.run_id = run_id
    factor_state, _ = _resolve_factor_state(config.tag)
    kwargs = _rank_kwargs(profile, config, candidate_state=factor_state, tag=config.tag)
    actual_research = rank_portfolio.rank_backtest(**kwargs)
    expected_research = baseline.get("expected_research") if isinstance(baseline.get("expected_research"), Mapping) else {}
    research_keys = ("total_return_pct", "max_drawdown_pct", "profit_over_max_drawdown", "trades", "simulated_liquidations", "liquidation_rejects")
    result: dict[str, Any] = {
        "tag": tag,
        "baseline_profile": baseline.get("path"),
        "rank_kwargs": kwargs,
        "expected": {
            "research": expected_research,
            "freqtrade": baseline.get("expected_freqtrade") if isinstance(baseline.get("expected_freqtrade"), Mapping) else {},
        },
        "actual": {
            "research": {key: actual_research.get(key) for key in research_keys if key in actual_research},
        },
        "delta": {
            "research": {
                key: {
                    "expected": expected_research.get(key),
                    "actual": actual_research.get(key),
                    "delta": (float(actual_research.get(key) or 0.0) - float(expected_research.get(key) or 0.0))
                    if isinstance(actual_research.get(key), (int, float)) or isinstance(expected_research.get(key), (int, float))
                    else None,
                }
                for key in research_keys
                if key in expected_research or key in actual_research
            }
        },
    }
    if include_freqtrade:
        runner = StrategyLoopRunner(config)
        replay_dir = iteration_dir(run_id, 1)
        replay_dir.mkdir(parents=True, exist_ok=True)
        actual_freqtrade = runner._run_fixed_freqtrade_backtest(replay_dir, actual_research)
        result["actual"]["freqtrade"] = actual_freqtrade.get("metrics") if isinstance(actual_freqtrade.get("metrics"), Mapping) else actual_freqtrade
    return result

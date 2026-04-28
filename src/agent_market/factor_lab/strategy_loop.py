"""Agent-driven factor strategy loop orchestration.

The loop keeps OpenCode confined to an iteration workspace.  Expensive and
stateful actions such as signal export, research backtest, scoring, and
promotion stay in this Python controller so candidate artifacts are auditable
and resumable.
"""
from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from agent_market import paths as repo_paths
from agent_market.factor_lab import rank_portfolio


PHASE_PREPARE = "PREPARE"
PHASE_CODE_GEN = "CODE_GEN"
PHASE_SIGNAL_EXPORT = "SIGNAL_EXPORT"
PHASE_BACKTEST = "BACKTEST"
PHASE_EVALUATION = "EVALUATION"
PHASE_ANALYSIS = "ANALYSIS"
PHASE_COMPLETE = "COMPLETE"

PHASES = (
    PHASE_PREPARE,
    PHASE_CODE_GEN,
    PHASE_SIGNAL_EXPORT,
    PHASE_BACKTEST,
    PHASE_EVALUATION,
    PHASE_ANALYSIS,
    PHASE_COMPLETE,
)

CANDIDATE_RANK_PROFILE = "rank_profile"
CANDIDATE_FREQTRADE_STRATEGY = "freqtrade_strategy"
CANDIDATE_TYPES = {CANDIDATE_RANK_PROFILE, CANDIDATE_FREQTRADE_STRATEGY}

DEFAULT_START = "2025-12-01"
DEFAULT_END = "2026-04-12"

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
    agent: str = "opencode"
    model: str = ""
    risk_profile: str = "aggressive"
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
    candidate_type: str = "auto"
    opencode_mode: str = "server"

    @classmethod
    def from_args(
        cls,
        *,
        tag: str = rank_portfolio.DEFAULT_TAG,
        venue: str = "okx",
        agent: str = "opencode",
        model: str = "",
        risk_profile: str = "aggressive",
        max_iterations: int = 30,
        timerange: Optional[str] = None,
        run_id: Optional[str] = None,
        resume: bool = False,
        n: int = 50,
        max_turns: int = 30,
        stale_timeout: float = 180.0,
        max_retries: int = 2,
        promote: bool = True,
        candidate_type: str = "auto",
        opencode_mode: str = "server",
    ) -> "StrategyLoopConfig":
        start, end = parse_timerange(timerange or f"{DEFAULT_START.replace('-', '')}-{DEFAULT_END.replace('-', '')}")
        ctype = str(candidate_type or "auto").strip().lower()
        if ctype not in {"auto", CANDIDATE_RANK_PROFILE, CANDIDATE_FREQTRADE_STRATEGY}:
            raise ValueError(f"candidate_type must be auto, {CANDIDATE_RANK_PROFILE}, or {CANDIDATE_FREQTRADE_STRATEGY}")
        mode = str(opencode_mode or "server").strip().lower()
        if mode not in {"server", "cli", "auto"}:
            raise ValueError("opencode_mode must be server, cli, or auto")
        return cls(
            tag=tag,
            venue=venue,
            agent=agent,
            model=model,
            risk_profile=risk_profile,
            max_iterations=int(max_iterations),
            timerange=timerange or f"{start.replace('-', '')}-{end.replace('-', '')}",
            run_id=str(run_id or ""),
            resume=bool(resume),
            n=int(n),
            start=start,
            end=end,
            max_turns=int(max_turns),
            stale_timeout=float(stale_timeout),
            max_retries=int(max_retries),
            promote=bool(promote),
            candidate_type=ctype,
            opencode_mode=mode,
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
        )


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


def make_run_id(tag: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", tag).strip("._") or "strategy_loop"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return f"{safe}_{stamp}_{uuid.uuid4().hex[:8]}"


def loop_root(run_id: str) -> Path:
    return repo_paths.artifacts_root() / "factor_strategy_loop" / str(run_id)


def checkpoint_path(run_id: str) -> Path:
    return loop_root(run_id) / "checkpoint.json"


def leaderboard_path(run_id: str) -> Path:
    return loop_root(run_id) / "leaderboard.json"


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


def prepare_context(config: StrategyLoopConfig, run_id: str, iteration: int) -> dict[str, Any]:
    factor_state, factor_source = _resolve_factor_state(config.tag)
    rank_dir = repo_paths.artifacts_root() / "rank_portfolio" / config.tag
    fixed_rank_dir = repo_paths.REPO_ROOT / "artifacts" / "rank_portfolio" / config.tag
    if not rank_dir.exists() and fixed_rank_dir.exists():
        rank_dir = fixed_rank_dir

    previous_iter = iteration_dir(run_id, iteration - 1) if iteration > 1 else None
    previous: dict[str, Any] = {}
    if previous_iter is not None:
        for name in ("analysis.md", "backtest.json", "candidate.json", "evaluation.json"):
            path = previous_iter / name
            if path.exists():
                if path.suffix == ".json":
                    previous[name] = load_json(path, {})
                else:
                    previous[name] = path.read_text(encoding="utf-8")[:12_000]

    okx_dir = repo_paths.user_data_root() / "data" / "okx" / "futures"
    if not okx_dir.exists():
        okx_dir = repo_paths.REPO_ROOT / "user_data" / "data" / "okx" / "futures"
    okx_files = sorted(okx_dir.glob("*-1h-futures.feather")) if okx_dir.exists() else []
    coverage = {
        "path": _as_repo_meta(okx_dir) if okx_dir.exists() else str(okx_dir),
        "futures_1h_files": len(okx_files),
        "sample_files": [p.name for p in okx_files[:10]],
    }

    rank_artifacts: dict[str, Any] = {"path": _as_repo_meta(rank_dir)}
    for name in ("rank_export.json", "backtest.json", "selected_factors.json", "optimized_profile.json"):
        path = rank_dir / name
        if path.exists():
            if name.endswith(".json"):
                rank_artifacts[name] = load_json(path, {})
            else:
                rank_artifacts[name] = _as_repo_meta(path)

    return {
        "version": "factor-strategy-loop-context-v1",
        "run_id": run_id,
        "iteration": int(iteration),
        "objective": {
            "mode": "profit_first_with_drawdown_controls",
            "candidate_type": config.candidate_type,
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
        "factor_source": factor_source,
        "factor_summary": _summarize_json(factor_state) if factor_state else {"missing": True, "tag": config.tag},
        "rank_artifacts": rank_artifacts,
        "okx_coverage": coverage,
        "previous_iteration": previous,
        "allowed_candidate_files": ["candidate.json", "strategy.py", "analysis.md"],
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
            out[key] = bool(value)
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

    strategy_info: Optional[dict[str, Any]] = None
    strategy_path_raw = payload.get("strategy_path") or ("strategy.py" if ctype == CANDIDATE_FREQTRADE_STRATEGY else "")
    if ctype == CANDIDATE_FREQTRADE_STRATEGY:
        strategy_path = _safe_relative_path(workspace, strategy_path_raw)
        strategy_info = validate_strategy_file(strategy_path)
    elif strategy_path_raw:
        _safe_relative_path(workspace, strategy_path_raw)

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


def _rank_kwargs(profile: Mapping[str, Any], config: StrategyLoopConfig, *, candidate_state: Optional[Path], tag: str) -> dict[str, Any]:
    params = dict(profile)
    n = int(params.pop("n", config.n))
    min_abs_score_z = params.pop("score_threshold", params.pop("min_abs_score_z", 1.5))
    return {
        "tag": tag,
        "venue": config.venue,
        "risk_profile": config.risk_profile,
        "n": n,
        "start": config.start,
        "end": config.end,
        "top_k": params.pop("top_k", 2),
        "gross_cap": params.pop("gross_cap", 2.0),
        "net_cap": params.pop("net_cap", None),
        "single_pair_cap": params.pop("single_pair_cap", None),
        "side_mode": params.pop("side_mode", "short"),
        "min_abs_score_z": min_abs_score_z,
        "rebalance_hours": params.pop("rebalance_hours", 8),
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
        "candidate_state": str(candidate_state) if candidate_state else None,
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

    score = 0.0
    score += profit_over_dd * 100.0
    score += profit_pct * 2.0
    score += min(trades, 1000) * 0.05
    score -= max(0.0, max_dd_pct - 10.0) * 3.0
    score -= simulated_liquidations * 500.0
    score -= liquidation_rejects * 10.0
    score -= kill_count * 0.5
    score -= max(0.0, avg_turnover - 2.0) * 20.0
    score -= max(0.0, concentration - 0.5) * 200.0
    if profit_pct >= target_profit_pct:
        score += 25.0
    if violations:
        score -= 1000.0 + len(violations) * 100.0

    constraints_ok = not violations
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


def _is_full_holdout(config: StrategyLoopConfig) -> bool:
    return config.start <= DEFAULT_START and config.end >= DEFAULT_END


def _copytree_replace(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def promote_candidate(
    candidate: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    config: StrategyLoopConfig,
    *,
    iter_dir: Path,
) -> dict[str, Any]:
    promoted = False
    artifacts: dict[str, str] = {}
    reason = str(evaluation.get("promotion_reason") or "")
    if not config.promote:
        reason = "promotion disabled"
    elif not bool(evaluation.get("constraints_ok")):
        reason = str(evaluation.get("promotion_reason") or "constraints failed")
    elif not _is_full_holdout(config):
        reason = f"not full holdout ({config.start} to {config.end}); formal promotion skipped"
    else:
        ctype = str(candidate.get("candidate_type"))
        if ctype == CANDIDATE_RANK_PROFILE:
            out = repo_paths.artifacts_root() / "rank_portfolio" / config.tag / "optimized_profile.json"
            try:
                iteration = int(iter_dir.name.rsplit("_", 1)[-1])
            except Exception:
                iteration = 0
            payload = {
                "version": "factor-strategy-loop-optimized-profile-v1",
                "created_at": time.time(),
                "run_id": config.run_id,
                "iteration": iteration,
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
    type_instruction = (
        "You may choose either `rank_profile` or `freqtrade_strategy`."
        if forced == "auto"
        else f"You must create a `{forced}` candidate. Do not choose any other candidate_type."
    )
    return f"""You are modifying one candidate inside an isolated factor-strategy-loop workspace.

Read `context/prepare.json` first. If present, also read the previous iteration's `analysis.md`
and `backtest.json` embedded in that context before proposing changes.

Write only these files in the current workspace root:
- `candidate.json`
- optionally `strategy.py`
- `analysis.md`

{type_instruction}

Do not write outside this workspace. Do not edit repository files. Do not run long backtests.
Do not use future data, and do not weaken risk controls just to increase headline return.

Candidate schema:
```json
{{
  "candidate_type": "rank_profile",
  "name": "short_descriptive_name",
  "description": "what changed and why",
  "rank_profile": {{
    "top_k": 2,
    "gross_cap": 2.0,
    "net_cap": 2.0,
    "single_pair_cap": 1.0,
    "side_mode": "short",
    "min_abs_score_z": 1.5,
    "rebalance_hours": 8,
    "risk_per_trade": 0.08,
    "leverage_cap": 5.0,
    "edge_mode": "rolling_ic"
  }}
}}
```

For a Freqtrade candidate, use `"candidate_type": "freqtrade_strategy"` and write
`strategy.py`. The strategy must inherit `freqtrade.strategy.IStrategy` and read
pre-generated rank signal files; do not reinvent cross-coin ranking inside Freqtrade.
It must support `RP_SIGNAL_DIR` and/or `RP_TAG` for signal discovery, must handle
Freqtrade futures pair names such as `ETH/USDT:USDT`, and must not hard-code
absolute repository paths. Prefer the signal-loading conventions from
`user_data/strategies/ELRankPortfolioLeverageStrategy.py` if you need a reference.
Its `rank_profile` must be either omitted or use only the same allowed numeric/risk
keys shown above, for example `top_k`, `gross_cap`, `net_cap`, `single_pair_cap`,
`side_mode`, `min_abs_score_z`, `rebalance_hours`, `risk_per_trade`, `leverage_cap`,
and `edge_mode`. Do not put signal-column mappings, version fields, or arbitrary
metadata inside `rank_profile`; use `metadata` for free-form notes.

The controller will validate schema, export rank signals, run backtests, score risk, and
handle promotion. Your job is to create the next candidate and concise reasoning.

Context path: {context_path.as_posix()}
"""


class StrategyLoopRunner:
    def __init__(self, config: StrategyLoopConfig) -> None:
        if not config.run_id:
            config.run_id = make_run_id(config.tag)
        self.config = config
        self.state = StrategyLoopState(run_id=config.run_id)
        if config.resume:
            loaded_config, loaded_state = load_checkpoint(config.run_id)
            for key in ("model", "agent", "max_iterations", "max_turns", "stale_timeout", "max_retries", "promote"):
                setattr(loaded_config, key, getattr(config, key))
            loaded_config.resume = True
            self.config = loaded_config
            self.state = loaded_state
            self.config.run_id = self.state.run_id

    def run(self) -> dict[str, Any]:
        root = loop_root(self.config.run_id)
        root.mkdir(parents=True, exist_ok=True)
        save_checkpoint(self.config, self.state)

        while self.state.iteration <= self.config.max_iterations:
            self._run_iteration()
            self.state.iteration += 1
            self.state.phase = PHASE_PREPARE
            save_checkpoint(self.config, self.state)
        return {
            "run_id": self.config.run_id,
            "checkpoint": _as_repo_meta(checkpoint_path(self.config.run_id)),
            "leaderboard": _as_repo_meta(leaderboard_path(self.config.run_id)),
            "best_candidate": self.state.best_candidate,
            "best_score": self.state.best_score,
        }

    def _run_iteration(self) -> None:
        idir = iteration_dir(self.config.run_id, self.state.iteration)
        idir.mkdir(parents=True, exist_ok=True)
        while self.state.phase != PHASE_COMPLETE:
            phase = self.state.phase
            if phase == PHASE_PREPARE:
                self._prepare(idir)
            elif phase == PHASE_CODE_GEN:
                self._code_gen(idir)
            elif phase == PHASE_SIGNAL_EXPORT:
                self._signal_export(idir)
            elif phase == PHASE_BACKTEST:
                self._backtest(idir)
            elif phase == PHASE_EVALUATION:
                self._evaluation(idir)
            elif phase == PHASE_ANALYSIS:
                self._analysis(idir)
            else:
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
            validate_candidate(candidate_path, default_n=self.config.n)
            self._record_candidate_path(candidate_path)
            return

        if self.config.agent != "opencode":
            raise ValueError(f"unsupported strategy-loop agent: {self.config.agent!r}")
        if not (self.config.model or os.environ.get("OPENCODE_MODEL")):
            raise RuntimeError("OpenCode unavailable: set --model or OPENCODE_MODEL before running strategy-loop")
        has_opencode_url = bool(os.environ.get("OPENCODE_URL"))
        has_opencode_cli = shutil.which("opencode") is not None
        if self.config.opencode_mode == "cli" and not has_opencode_cli:
            raise RuntimeError("OpenCode CLI mode requires `opencode` on PATH")
        if not (has_opencode_url or has_opencode_cli):
            raise RuntimeError("OpenCode unavailable: `opencode` CLI is not on PATH and OPENCODE_URL is not set")
        prompt = render_agent_prompt(idir / "context" / "prepare.json", candidate_type=self.config.candidate_type)

        if self.config.opencode_mode == "cli":
            self._run_opencode_cli(idir, prompt)
            validate_candidate(candidate_path, default_n=self.config.n)
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
                self._run_opencode_cli(idir, prompt)
                result = None
            if result is None:
                validate_candidate(candidate_path, default_n=self.config.n)
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
        validate_candidate(candidate_path, default_n=self.config.n)
        self._record_candidate_path(candidate_path)

    def _run_opencode_cli(self, idir: Path, prompt: str) -> None:
        cmd = ["opencode", "run", "-m", self.config.model or os.environ.get("OPENCODE_MODEL", ""), prompt]
        env = dict(os.environ)
        proc = subprocess.run(
            cmd,
            cwd=str(idir),
            env=env,
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
        effective_tag = self._effective_rank_tag(idir)
        kwargs = _rank_kwargs(
            candidate.get("rank_profile") or {},
            self.config,
            candidate_state=factor_state,
            tag=effective_tag,
        )
        summary = rank_portfolio.rank_export(recompute_corr=True, **kwargs)
        summary["effective_tag"] = effective_tag
        summary["base_tag"] = self.config.tag
        write_json(out, summary)

    def _backtest(self, idir: Path) -> None:
        out = idir / "backtest.json"
        if out.exists():
            return
        candidate = validate_candidate(idir / "candidate.json", default_n=self.config.n)
        factor_state, _ = _resolve_factor_state(self.config.tag)
        kwargs = _rank_kwargs(
            candidate.get("rank_profile") or {},
            self.config,
            candidate_state=factor_state,
            tag=self._effective_rank_tag(idir),
        )
        result = rank_portfolio.rank_backtest(recompute_corr=True, **kwargs)
        result["base_tag"] = self.config.tag
        result["candidate"] = candidate
        write_json(out, result)

    def _evaluation(self, idir: Path) -> None:
        out = idir / "evaluation.json"
        if out.exists():
            evaluation = load_json(out, {})
        else:
            backtest = load_json(idir / "backtest.json", {})
            evaluation = score_backtest_result(
                backtest,
                min_trades=self.config.min_trades,
                max_drawdown_pct=self.config.max_drawdown_pct,
                min_profit_over_dd=self.config.min_profit_over_dd,
                target_profit_pct=self.config.target_profit_pct,
            )
            candidate = validate_candidate(idir / "candidate.json", default_n=self.config.n)
            evaluation["candidate"] = candidate
            evaluation["candidate_path"] = _as_repo_meta(idir / "candidate.json")
            score = float(evaluation.get("score") or float("-inf"))
            if score > self.state.best_score:
                promotion = promote_candidate(candidate, evaluation, self.config, iter_dir=idir)
                best_dir = loop_root(self.config.run_id) / "best"
                _copytree_replace(idir, best_dir)
                promotion.setdefault("artifacts", {})["best_dir"] = _as_repo_meta(best_dir)
            else:
                promotion = {
                    "promoted": False,
                    "artifacts": {},
                    "reason": f"score did not exceed current best ({self.state.best_score:.6g})",
                }
            evaluation["promotion"] = promotion
            write_json(out, evaluation)

        row = {
            "run_id": self.config.run_id,
            "iteration": self.state.iteration,
            "candidate_path": _as_repo_meta(idir / "candidate.json"),
            "candidate": evaluation.get("candidate"),
            "parameters": (evaluation.get("candidate") or {}).get("rank_profile") if isinstance(evaluation.get("candidate"), dict) else {},
            "strategy_path": (evaluation.get("candidate") or {}).get("strategy_path") if isinstance(evaluation.get("candidate"), dict) else None,
            "score": evaluation.get("score"),
            "constraints_ok": evaluation.get("constraints_ok"),
            "metrics": evaluation.get("metrics"),
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

    def _analysis(self, idir: Path) -> None:
        path = idir / "analysis.md"
        if path.exists():
            return
        evaluation = load_json(idir / "evaluation.json", {})
        lines = [
            f"# Iteration {self.state.iteration} Analysis",
            "",
            f"Score: {evaluation.get('score')}",
            f"Constraints OK: {evaluation.get('constraints_ok')}",
            f"Violations: {evaluation.get('violations') or []}",
            "",
            "Metrics:",
        ]
        metrics = evaluation.get("metrics") or {}
        if isinstance(metrics, Mapping):
            for key in sorted(metrics):
                lines.append(f"- {key}: {metrics[key]}")
        lines.extend(
            [
                "",
                "Next iteration guidance:",
                "- Preserve hard risk gates before increasing leverage or turnover.",
                "- Prefer changes that improve profit/drawdown without reducing trade count.",
            ]
        )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _effective_rank_tag(self, idir: Path) -> str:
        return f"{self.config.tag}__loop_{self.config.run_id}_{idir.name}"

    def _record_candidate_path(self, path: Path) -> None:
        rendered = _as_repo_meta(path)
        if rendered not in self.state.candidate_paths:
            self.state.candidate_paths.append(rendered)

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
    timerange: Optional[str] = None,
    n: int = 50,
    run_id: Optional[str] = None,
    promote: bool = False,
) -> dict[str, Any]:
    config = StrategyLoopConfig.from_args(
        tag=tag,
        venue=venue,
        risk_profile=risk_profile,
        timerange=timerange,
        n=n,
        run_id=run_id or make_run_id(f"{tag}_eval"),
        max_iterations=1,
        promote=promote,
    )
    candidate = validate_candidate(candidate_path, default_n=config.n)
    idir = Path(candidate_path).resolve().parent
    factor_state, _ = _resolve_factor_state(config.tag)
    kwargs = _rank_kwargs(
        candidate.get("rank_profile") or {},
        config,
        candidate_state=factor_state,
        tag=f"{config.tag}__loop_eval_{config.run_id}",
    )
    backtest = rank_portfolio.rank_backtest(recompute_corr=True, **kwargs)
    backtest["base_tag"] = config.tag
    backtest["candidate"] = candidate
    write_json(idir / "backtest.json", backtest)
    evaluation = score_backtest_result(
        backtest,
        min_trades=config.min_trades,
        max_drawdown_pct=config.max_drawdown_pct,
        min_profit_over_dd=config.min_profit_over_dd,
        target_profit_pct=config.target_profit_pct,
    )
    evaluation["candidate"] = candidate
    evaluation["promotion"] = promote_candidate(candidate, evaluation, config, iter_dir=idir)
    write_json(idir / "evaluation.json", evaluation)
    return evaluation

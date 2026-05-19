"""Shared utility helpers extracted from phases.py."""
from __future__ import annotations

import ast
import json
import re
from typing import Any, Dict, Optional

from agent_market import paths

from .dtypes import MinerConfig, MinerState, Phase, StrategyCandidate, VALID_TRANSITIONS, VALID_CANDIDATE_TRANSITIONS

# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def _truncate_text(text: str, limit: int = 2000) -> str:
    if not isinstance(text, str):
        return ""
    s = text.strip()
    if len(s) <= limit:
        return s
    return s[:limit] + "\u2026"


# ---------------------------------------------------------------------------
# Freqtrade config helpers
# ---------------------------------------------------------------------------


def _freqtrade_config_defaults(freqtrade_config_path: str) -> tuple[str, bool]:
    """Return (timeframe, enforce_can_short_false) from a freqtrade config file."""
    payload = _load_freqtrade_payload(freqtrade_config_path)
    timeframe = str(payload.get("timeframe") or "1h").strip() or "1h"
    trading_mode = str(payload.get("trading_mode") or "spot").strip().lower() or "spot"
    return timeframe, trading_mode == "spot"


def _load_freqtrade_payload(freqtrade_config_path: str) -> dict[str, Any]:
    try:
        ft_path = paths.resolve_repo_path(freqtrade_config_path)
        return json.loads(ft_path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


def _freqtrade_market_context(freqtrade_config_path: str) -> tuple[str, list[str], str, str]:
    payload = _load_freqtrade_payload(freqtrade_config_path)
    exchange = str(((payload.get("exchange") or {}).get("name")) or "gate")
    pairs = list(((payload.get("exchange") or {}).get("pair_whitelist")) or [])
    timeframe = str(payload.get("timeframe") or "1h")
    datadir = str(payload.get("datadir") or "user_data/data")
    return exchange, pairs, timeframe, datadir


def _freqtrade_trading_mode(freqtrade_config_path: str) -> str:
    payload = _load_freqtrade_payload(freqtrade_config_path)
    return str(payload.get("trading_mode") or "spot").strip().lower() or "spot"


# ---------------------------------------------------------------------------
# Timerange helpers
# ---------------------------------------------------------------------------


def _split_timerange(timerange: str, train_ratio: float = 0.7) -> tuple[str, str]:
    """Split a YYYYMMDD-YYYYMMDD timerange into train and validation segments."""
    try:
        parts = timerange.split("-")
        if len(parts) != 2:
            return timerange, ""
        from datetime import datetime, timedelta
        start = datetime.strptime(parts[0], "%Y%m%d")
        end = datetime.strptime(parts[1], "%Y%m%d")
        total_days = (end - start).days
        if total_days < 7:
            return timerange, ""
        split_point = start + timedelta(days=int(total_days * train_ratio))
        train_range = f"{parts[0]}-{split_point.strftime('%Y%m%d')}"
        val_range = f"{split_point.strftime('%Y%m%d')}-{parts[1]}"
        return train_range, val_range
    except Exception:
        return timerange, ""


# ---------------------------------------------------------------------------
# Timeframe helpers
# ---------------------------------------------------------------------------


def _get_leverage_factor(freqtrade_config_path: str) -> float:
    """Return the default leverage factor from a freqtrade config (1.0 for spot)."""
    try:
        payload = _load_freqtrade_payload(freqtrade_config_path)
        trading_mode = str(payload.get("trading_mode") or "spot").strip().lower()
        if trading_mode != "futures":
            return 1.0
        leverage = payload.get("leverage")
        if isinstance(leverage, dict):
            return float(leverage.get("default", 1.0))
        if isinstance(leverage, (int, float)):
            return float(leverage)
        return 1.0
    except Exception:
        return 1.0


def _timeframe_to_minutes(timeframe: str) -> int | None:
    s = str(timeframe or "").strip().lower()
    m = re.match(r"^(\d+)([mhdw])$", s)
    if not m:
        return None
    value = int(m.group(1))
    unit = m.group(2)
    scale = {
        "m": 1,
        "h": 60,
        "d": 1440,
        "w": 10080,
    }.get(unit)
    if scale is None:
        return None
    return value * scale


def _prompt_objective_profile(config: MinerConfig) -> Dict[str, Any]:
    base_timeframe, _ = _freqtrade_config_defaults(config.freqtrade_config)
    max_tf = str(getattr(config, "max_strategy_timeframe", "") or base_timeframe).strip() or base_timeframe
    max_minutes = _timeframe_to_minutes(max_tf)

    allowed_informative: list[str] = []
    for raw_tf in list(getattr(config, "allowed_informative_timeframes", []) or []):
        tf = str(raw_tf or "").strip()
        if not tf or tf == base_timeframe or tf in allowed_informative:
            continue
        tf_minutes = _timeframe_to_minutes(tf)
        if max_minutes is not None and tf_minutes is not None and tf_minutes > max_minutes:
            continue
        allowed_informative.append(tf)

    min_trades = int(getattr(config, "min_trades", 0) or 0)
    target_trades = int(getattr(config, "target_trades", 0) or 0)
    target_trades = max(target_trades, min_trades, 20)

    min_acceptable_trades = int(getattr(config, "min_acceptable_trades", 0) or 0)
    if min_acceptable_trades <= 0:
        min_acceptable_trades = max(1, min_trades or max(10, target_trades // 2))

    return {
        "base_timeframe": base_timeframe,
        "max_strategy_timeframe": max_tf,
        "allowed_informative_timeframes": allowed_informative,
        "target_trades": target_trades,
        "min_acceptable_trades": min_acceptable_trades,
        "roi_target_min_pct": float(getattr(config, "roi_target_min_pct", 5.0) or 5.0),
        "roi_target_max_pct": float(getattr(config, "roi_target_max_pct", 10.0) or 10.0),
        "stoploss_min_pct": float(getattr(config, "stoploss_min_pct", 5.0) or 5.0),
        "stoploss_max_pct": float(getattr(config, "stoploss_max_pct", 10.0) or 10.0),
        "preferred_patterns": list(getattr(config, "preferred_patterns", []) or []),
        "avoid_patterns": list(getattr(config, "avoid_patterns", []) or []),
    }


def _extract_strategy_timeframes(code: str) -> list[tuple[str, str]]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    found: list[tuple[str, str]] = []

    def _add(source: str, value: Any) -> None:
        if not isinstance(value, str):
            return
        tf = value.strip()
        if re.match(r"^\d+[mhdw]$", tf):
            found.append((source, tf))

    class _Visitor(ast.NodeVisitor):
        def visit_Assign(self, node: ast.Assign) -> None:
            if any(isinstance(t, ast.Name) and t.id == "timeframe" for t in node.targets):
                if isinstance(node.value, ast.Constant):
                    _add("timeframe", node.value.value)
            self.generic_visit(node)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            if isinstance(node.target, ast.Name) and node.target.id == "timeframe":
                if isinstance(node.value, ast.Constant):
                    _add("timeframe", node.value.value)
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            func_name = None
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr

            if func_name == "informative" and node.args:
                arg0 = node.args[0]
                if isinstance(arg0, ast.Constant):
                    _add("informative", arg0.value)
            elif func_name == "merge_informative_pair":
                for arg in node.args[2:4]:
                    if isinstance(arg, ast.Constant):
                        _add("merge_informative_pair", arg.value)

            self.generic_visit(node)

    _Visitor().visit(tree)
    return found


def _validate_timeframe_policy(code: str, config: MinerConfig) -> tuple[bool, str]:
    profile = _prompt_objective_profile(config)
    allowed = {
        str(profile["base_timeframe"]).strip(),
        *(str(tf).strip() for tf in profile["allowed_informative_timeframes"]),
    }
    max_tf = str(profile["max_strategy_timeframe"]).strip()
    max_minutes = _timeframe_to_minutes(max_tf)

    violations: list[str] = []
    for source, tf in _extract_strategy_timeframes(code):
        tf_minutes = _timeframe_to_minutes(tf)
        if max_minutes is not None and tf_minutes is not None and tf_minutes > max_minutes:
            violations.append(f"{source}:{tf}>{max_tf}")
        elif tf not in allowed:
            violations.append(f"{source}:{tf} not in {sorted(allowed)}")

    if violations:
        return False, f"Disallowed timeframe usage: {', '.join(violations[:4])}"
    return True, "Timeframe policy passed"


# ---------------------------------------------------------------------------
# Candidate type helpers
# ---------------------------------------------------------------------------

_CANDIDATE_TYPE_FAMILIES: dict[str, list[str]] = {
    "rule": [],
    "ml": ["lightgbm", "xgboost"],
    "dl": ["pytorch_mlp"],
    "rl": ["ppo"],
}


def _normalize_candidate_type(value: Any) -> str:
    raw = str(value or "rule").strip().lower()
    return raw if raw in _CANDIDATE_TYPE_FAMILIES else "rule"


def _configured_candidate_types(config: MinerConfig) -> list[str]:
    items = list(getattr(config, "candidate_types", None) or ["rule"])
    enabled = {"rule", "ml"}
    if bool(getattr(config, "enable_dl", False)):
        enabled.add("dl")
    if bool(getattr(config, "enable_rl", False)):
        enabled.add("rl")
    out: list[str] = []
    for item in items:
        normalized = _normalize_candidate_type(item)
        if normalized not in enabled:
            continue
        if normalized not in out:
            out.append(normalized)
    if out:
        return out
    fallback = [item for item in ("rule", "ml") if item in enabled]
    return fallback or ["rule"]


def _candidate_type_for_slot(
    config: MinerConfig,
    *,
    iteration: int,
    candidate_idx: int,
    candidates_per_iteration: int,
) -> str:
    choices = _configured_candidate_types(config)
    position = max(0, int(iteration)) * max(1, int(candidates_per_iteration)) + max(0, int(candidate_idx))
    return choices[position % len(choices)]


def _allowed_model_families(candidate_type: str) -> list[str]:
    return list(_CANDIDATE_TYPE_FAMILIES.get(_normalize_candidate_type(candidate_type), []))


def _candidate_requires_training(candidate: Optional[StrategyCandidate]) -> bool:
    if candidate is None:
        return False
    return _normalize_candidate_type(getattr(candidate, "candidate_type", "rule")) != "rule" and not bool(
        getattr(candidate, "training_summary", None)
    )


def _phase_for_candidate(candidate: Optional[StrategyCandidate]) -> Phase:
    if _candidate_requires_training(candidate):
        return Phase.TRAIN_MODEL
    return Phase.BACKTEST


# ---------------------------------------------------------------------------
# Name / sanitisation helpers
# ---------------------------------------------------------------------------


def _sanitize_candidate_name(value: str, *, fallback: str) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z_]+", " ", str(value or "")).strip()
    parts = [part for part in re.split(r"[\s_]+", cleaned) if part]
    if parts:
        base = "".join(part[:1].upper() + part[1:] for part in parts)
    else:
        base = fallback
    if not base[:1].isalpha():
        base = fallback + base
    return base


def _rewrite_strategy_class_name(code: str, old: str, new: str) -> str:
    """Best-effort rename `class <old>(...)` to `class <new>(...)` (first match only)."""
    pat = re.compile(rf"(\bclass\s+){re.escape(old)}(\s*\()")

    def _repl(m: re.Match[str]) -> str:
        return f"{m.group(1)}{new}{m.group(2)}"

    out, n = pat.subn(_repl, code, count=1)
    return out if n else code


# ---------------------------------------------------------------------------
# JSON / coercion helpers
# ---------------------------------------------------------------------------


def _parse_json_object(text: str) -> dict[str, Any] | None:
    if not isinstance(text, str) or not text.strip():
        return None

    s = text.strip()

    # Fast path: pure JSON.
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    # Best-effort extraction from mixed output.
    m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        return None

    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _json_block_or_none(text: str) -> dict[str, Any] | None:
    return _parse_json_object(text)


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _coerce_int(value: Any, default: int, *, minimum: int = 0, maximum: Optional[int] = None) -> int:
    try:
        out = int(value)
    except Exception:
        out = int(default)
    out = max(int(minimum), out)
    if maximum is not None:
        out = min(int(maximum), out)
    return out


def _normalize_roi_map(value: Any, default: dict[str, float]) -> dict[str, float]:
    if not isinstance(value, dict):
        return dict(default)
    out: dict[str, float] = {}
    for key, raw in value.items():
        try:
            out[str(key)] = float(raw)
        except Exception:
            continue
    if not out:
        return dict(default)

    def _sort_key(item: tuple[str, float]) -> tuple[int, str]:
        k = item[0]
        return (int(k), k) if str(k).isdigit() else (10**9, str(k))

    return dict(sorted(out.items(), key=_sort_key))


# ---------------------------------------------------------------------------
# Validation / classification helpers
# ---------------------------------------------------------------------------


def _classify_validation_failure(msg: str) -> str:
    s = str(msg or "").strip().lower()
    if not s:
        return "validation.unknown"
    if "syntax error" in s:
        return "validation.syntax"
    if "forbidden import" in s:
        return "validation.forbidden_import"
    if "forbidden call" in s:
        return "validation.forbidden_call"
    if "look-ahead" in s or "look ahead" in s:
        return "validation.lookahead"
    if "must inherit" in s and "istrategy" in s:
        return "validation.inheritance"
    if "missing required methods" in s:
        return "validation.missing_methods"
    return "validation.other"


# ---------------------------------------------------------------------------
# State management helpers
# ---------------------------------------------------------------------------


def _pick_active_candidate(state: MinerState) -> Optional[StrategyCandidate]:
    if not state.candidates:
        return None

    idx = state.active_candidate_idx
    if idx is None and state.pending_candidate_idxs:
        idx = state.pending_candidate_idxs[0]
        state.active_candidate_idx = idx

    if idx is None:
        return state.candidates[-1]

    if 0 <= idx < len(state.candidates):
        return state.candidates[idx]

    # Out-of-range (corrupt checkpoint) fallback
    state.active_candidate_idx = None
    return state.candidates[-1]


def _mark_candidate_done(state: MinerState) -> None:
    """Pop active candidate from queue and update next phase."""
    idx = state.active_candidate_idx
    if idx is not None:
        try:
            state.pending_candidate_idxs = [i for i in state.pending_candidate_idxs if i != idx]
        except Exception:
            state.pending_candidate_idxs = []

    state.active_candidate_idx = state.pending_candidate_idxs[0] if state.pending_candidate_idxs else None


def _advance_after_candidate(state: MinerState) -> None:
    _mark_candidate_done(state)
    next_candidate = _pick_active_candidate(state) if state.active_candidate_idx is not None else None
    target = _phase_for_candidate(next_candidate) if next_candidate is not None else Phase.ANALYSIS
    safe_transition(state, target)


# ---------------------------------------------------------------------------
# Safe phase transition (D2: explicit transition table)
# ---------------------------------------------------------------------------

import logging as _logging

_phase_logger = _logging.getLogger(__name__)


def safe_transition(state: MinerState, target: Phase) -> None:
    """Set state.phase with transition-table validation.

    Raises ValueError for invalid transitions to enforce FSM integrity.
    """
    allowed = VALID_TRANSITIONS.get(state.phase, [])
    if target not in allowed:
        _phase_logger.error(
            "Blocked invalid phase transition %s -> %s (allowed: %s)",
            state.phase.value, target.value,
            [p.value for p in allowed],
        )
        raise ValueError(
            f"Invalid phase transition {state.phase.value} -> {target.value}; "
            f"allowed: {[p.value for p in allowed]}"
        )
    state.phase = target


# ---------------------------------------------------------------------------
# Idempotent history append (D2: no duplicate rows on re-entry)
# ---------------------------------------------------------------------------

def history_has_row(state: MinerState, iteration: int, name: str) -> bool:
    """Return True if *history* already contains a row for (iteration, name)."""
    return any(
        r.get("iteration") == iteration and r.get("name") == name
        for r in state.history
    )


def upsert_history(state: MinerState, row: Dict[str, Any]) -> None:
    """Append *row* to history, or update in-place if (iteration, name) exists."""
    it = row.get("iteration")
    nm = row.get("name")
    for i, existing in enumerate(state.history):
        if existing.get("iteration") == it and existing.get("name") == nm:
            state.history[i] = row
            return
    state.history.append(row)


# ---------------------------------------------------------------------------
# Candidate stage tracking (D2: per-candidate lifecycle)
# ---------------------------------------------------------------------------

import time as _time


def update_candidate_stage(candidate: StrategyCandidate, stage: str) -> None:
    """Update candidate stage with transition validation and history tracking.

    D2: Fail-closed — raises ValueError on invalid transitions.
    Any stage can transition to 'failed' (always allowed).
    """
    current = getattr(candidate, "stage", "generated") or "generated"

    # Self-transition (idempotent) is always allowed
    if stage == current:
        candidate.stage_history.append({
            "stage": stage,
            "from": current,
            "timestamp": _time.time(),
        })
        return

    # Validate transition
    allowed = VALID_CANDIDATE_TRANSITIONS.get(current, [])
    if stage not in allowed:
        raise ValueError(
            f"Invalid candidate stage transition: {current} -> {stage}. "
            f"Allowed: {allowed}"
        )

    candidate.stage = stage
    candidate.stage_history.append({
        "stage": stage,
        "from": current,
        "timestamp": _time.time(),
    })


def _walkforward_timeranges(
    timerange: str,
    *,
    folds: int = 3,
    train_ratio: float = 0.6,
) -> list[tuple[str, str]]:
    """Split YYYYMMDD-YYYYMMDD into expanding-window walk-forward folds.

    Returns list of (train_range, test_range) tuples. Each fold uses an
    expanding training window and a fixed-size test window.
    """
    from datetime import datetime, timedelta

    parts = timerange.split("-")
    if len(parts) != 2:
        return []
    try:
        start = datetime.strptime(parts[0], "%Y%m%d")
        end = datetime.strptime(parts[1], "%Y%m%d")
    except ValueError:
        return []

    total_days = (end - start).days
    if total_days < 14 or folds < 1:
        return []

    # Each fold's test window
    test_days = max(1, int(total_days * (1.0 - train_ratio) / folds))
    results: list[tuple[str, str]] = []

    for i in range(folds):
        # Expanding train: from start to train_end
        test_start = start + timedelta(days=int(total_days * train_ratio) + i * test_days)
        test_end = test_start + timedelta(days=test_days)
        if test_end > end:
            test_end = end
        if test_start >= test_end:
            break
        train_end = test_start

        train_range = f"{start.strftime('%Y%m%d')}-{train_end.strftime('%Y%m%d')}"
        test_range = f"{test_start.strftime('%Y%m%d')}-{test_end.strftime('%Y%m%d')}"
        results.append((train_range, test_range))

    return results

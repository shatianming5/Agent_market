"""Backtest and training phase handlers extracted from phases.py."""
from __future__ import annotations

import ast
import hashlib
import json
import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from .knowledge_base import KnowledgeBase

from agent_market import paths
from agent_market.backtest_results import build_backtest_summary, find_latest_backtest_zip

from .agent_adapter import StrategyAgent
from .agent_factory import build_strategy_agent
from .dtypes import MinerConfig, MinerState, Phase, StrategyCandidate
from .prompts import build_repair_prompt
from .sandbox import (
    auto_fix_strategy_code,
    auto_fix_strategy_file,
    ensure_freqtrade_strategy_compliance_file,
    find_strategy_files,
    infer_strategy_class_name,
    prepare_sandbox,
    validate_strategy_code,
)
from ._helpers import (
    _truncate_text,
    _freqtrade_config_defaults,
    _freqtrade_trading_mode,
    _prompt_objective_profile,
    _normalize_candidate_type,
    _candidate_requires_training,
    _phase_for_candidate,
    _pick_active_candidate,
    _mark_candidate_done,
    _advance_after_candidate,
    _validate_timeframe_policy,
    _classify_validation_failure,
    _coerce_float,
    _coerce_int,
    _normalize_roi_map,
    safe_transition,
    update_candidate_stage,
)
from ._rendering import (
    _render_ml_strategy_code,
    _render_rl_signal_strategy_code,
    _restore_trained_wrapper,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Repair helper
# ---------------------------------------------------------------------------


def _update_repair_ledger_result(candidate: StrategyCandidate, result: str) -> None:
    """Update the most recent repair ledger entry's verification_result.

    Called after re-backtest to record whether the repair actually fixed the issue.
    result: 'pass', 'fail', or 'error'
    """
    ledger = getattr(candidate, "repair_ledger", None)
    if ledger and isinstance(ledger, list) and ledger:
        ledger[-1]["verification_result"] = result


def _repair_agent_explicitly_configured(config: MinerConfig) -> bool:
    """Return True when repair is explicitly enabled by caller/config.

    Bare `MinerConfig()` should stay fail-closed during static validation tests
    instead of opportunistically reaching for ambient LLM credentials.
    """
    model = str(getattr(config, "model", "") or "").strip()
    base_url = str(getattr(config, "base_url", "") or "").strip()
    provider = str(getattr(config, "provider", "auto") or "auto").strip().lower()
    return bool(model or base_url or provider not in {"", "auto"})


def _resolve_quick_gate_thresholds(config: MinerConfig) -> dict[str, float]:
    """Resolve quick-funnel thresholds.

    Discovery overrides are intended for early futures exploration only. The
    final evaluation gate remains unchanged in `_evaluation.py`.
    """
    thresholds = {
        "min_trades": float(int(getattr(config, "quick_min_trades", 3) or 3)),
        "min_profit_factor": float(getattr(config, "quick_min_profit_factor", 0.5) or 0.5),
        "min_profit_pct": float(getattr(config, "quick_min_profit_pct", -5.0) or -5.0),
        "max_drawdown_pct": float(getattr(config, "quick_max_drawdown_pct", 100.0) or 100.0),
    }
    if _freqtrade_trading_mode(config.freqtrade_config) != "futures":
        return thresholds

    overrides = {
        "min_trades": getattr(config, "discovery_quick_min_trades", 0),
        "min_profit_factor": getattr(config, "discovery_quick_min_profit_factor", 0.0),
        "min_profit_pct": getattr(config, "discovery_quick_min_profit_pct", 0.0),
        "max_drawdown_pct": getattr(config, "discovery_quick_max_drawdown_pct", 0.0),
    }
    for key, raw in overrides.items():
        if raw in (None, "", 0, 0.0):
            continue
        thresholds[key] = float(raw)
    return thresholds


def _ensure_configured_strategy_compliance(
    config: MinerConfig, strategy_path: Path
) -> tuple[bool, list[str]]:
    timeframe, enforce_can_short_false = _freqtrade_config_defaults(
        config.freqtrade_config
    )
    return ensure_freqtrade_strategy_compliance_file(
        strategy_path,
        timeframe=timeframe,
        enforce_can_short_false=enforce_can_short_false,
    )


def _repair_candidate(
    *,
    agent: StrategyAgent,
    config: MinerConfig,
    run_dir: Path,
    sandbox: Path,
    candidate: StrategyCandidate,
    failure: str,
    attempt: int,
    max_attempts: int,
) -> bool:
    try:
        objective_profile = _prompt_objective_profile(config)
        original_path = candidate.strategy_path
        before_hash = None
        try:
            if original_path.exists():
                before_hash = hashlib.sha256(original_path.read_bytes()).hexdigest()
        except Exception:
            before_hash = None

        rel = original_path
        try:
            rel = original_path.resolve().relative_to(sandbox.resolve())
        except Exception:
            rel = Path("user_data") / "strategies" / original_path.name

        prompt = build_repair_prompt(
            sandbox_path=str(sandbox),
            strategy_rel_path=str(rel),
            freqtrade_config=config.freqtrade_config,
            timerange=config.timerange,
            failure=failure,
            attempt=attempt,
            max_attempts=max_attempts,
            tool_allowlist=list(config.tool_allowlist or []),
            bash_allow=bool(config.bash_allow),
            bash_timeout=int(config.bash_timeout or 60),
            bash_allowlist=list(config.bash_allowlist or []),
            provider=config.provider,
            **objective_profile,
        )

        repair_provider = ""
        repair_model = None
        attempts: list[dict[str, Any]] = []

        def _on_result(r: Any) -> None:
            nonlocal repair_provider, repair_model
            repair_provider = str(getattr(r, "provider", "") or "")
            repair_model = getattr(r, "model", None)
            attempts.append(
                {
                    "provider": repair_provider,
                    "model": repair_model,
                    "assistant_text": (getattr(r, "assistant_text", "") or "")[:4000],
                    "tool_trace": getattr(r, "tool_trace", None),
                }
            )

        try:
            repaired_path = agent.generate_strategy(
                prompt,
                filename_hint=original_path.name,
                on_result=_on_result,
            )
        except TypeError:
            repaired_path = agent.generate_strategy(prompt, filename_hint=original_path.name)

        if repaired_path is None or not repaired_path.exists():
            return False

        after_hash = None
        try:
            after_hash = hashlib.sha256(repaired_path.read_bytes()).hexdigest()
        except Exception:
            after_hash = None

        same_file = False
        try:
            same_file = repaired_path.resolve() == original_path.resolve()
        except Exception:
            same_file = str(repaired_path) == str(original_path)

        if same_file and before_hash is not None and after_hash is not None and before_hash == after_hash:
            logger.info(
                "Repair produced no changes for %s (attempt %d/%d)",
                original_path.name,
                int(attempt),
                int(max_attempts),
            )
            return False

        # Update candidate to point at repaired artifact.
        candidate.strategy_path = repaired_path
        if repair_provider:
            candidate.source_provider = repair_provider
            # Preserve original generation provider if present.
            if not getattr(candidate, "generation_provider", ""):
                candidate.generation_provider = repair_provider
        if repair_model is not None:
            candidate.source_model = repair_model
            if getattr(candidate, "generation_model", None) is None:
                candidate.generation_model = repair_model

        candidate.code = repaired_path.read_text(encoding="utf-8", errors="replace")

        # Auto-fix any lingering tool tags / fences after repair.
        did_fix, auto_fixes = auto_fix_strategy_file(candidate.strategy_path)
        if did_fix:
            candidate.code = candidate.strategy_path.read_text(encoding="utf-8", errors="replace")
        else:
            auto_fixes = []

        # Ensure freqtrade sanity settings regardless of LLM output.
        compliance_fixes: list[str] = []
        try:
            did_comp, compliance_fixes = _ensure_configured_strategy_compliance(
                config, candidate.strategy_path
            )
            if did_comp:
                candidate.code = candidate.strategy_path.read_text(encoding="utf-8", errors="replace")
                logger.info(
                    "Compliance auto-fix applied (repair) for %s: %s",
                    candidate.strategy_path.name,
                    ",".join(compliance_fixes),
                )
        except Exception:
            compliance_fixes = []

        candidate.name = infer_strategy_class_name(candidate.code) or repaired_path.stem
        # D6: Capture failure info BEFORE clearing for repair ledger
        _repair_failure_type = str(candidate.failure_category or "unknown")
        _repair_diagnosis = str(candidate.diagnosis or "")
        candidate.backtest_summary = None
        candidate.reward = None
        candidate.failure_category = ""
        candidate.diagnosis = ""

        # D6: Append structured repair ledger entry
        import time as _rl_time
        _ledger_entry = {
            "attempt": int(attempt),
            "max_attempts": int(max_attempts),
            "failure_type": _repair_failure_type,
            "root_cause": _repair_diagnosis[:500] if _repair_diagnosis else "",
            "patch_scope": "full_strategy",
            "code_hash_before": before_hash,
            "code_hash_after": after_hash,
            "verification_result": "pending",  # updated after next backtest
            "provider": repair_provider,
            "model": repair_model,
            "auto_fixes": list(auto_fixes or []),
            "compliance_fixes": list(compliance_fixes or []),
            "timestamp": _rl_time.time(),
        }
        if not hasattr(candidate, "repair_ledger") or candidate.repair_ledger is None:
            candidate.repair_ledger = []
        candidate.repair_ledger.append(_ledger_entry)

        # Record repair trace (best-effort).
        try:
            from .artifacts import write_agent_trace

            slot = int(getattr(candidate, "candidate_slot", 0) or 0)
            role = f"repair_{int(attempt):02d}.{_repair_failure_type}"
            p = write_agent_trace(
                run_dir,
                iteration=int(candidate.iteration),
                candidate_idx=slot,
                role=role,
                payload={
                    "failure_category": _repair_failure_type,
                    "root_cause": _repair_diagnosis[:500] if _repair_diagnosis else "",
                    "failure": failure,
                    "attempt": int(attempt),
                    "max_attempts": int(max_attempts),
                    "provider": repair_provider,
                    "model": repair_model,
                    "repaired_path": str(repaired_path),
                    "attempts": attempts,
                    "same_file": bool(same_file),
                    "before_hash": before_hash,
                    "after_hash": after_hash,
                    "auto_fixes": list(auto_fixes or []),
                    "compliance_fixes": list(compliance_fixes or []),
                    "repair_ledger_entry": _ledger_entry,
                },
            )
            candidate.agent_traces = dict(getattr(candidate, "agent_traces", None) or {})
            candidate.agent_traces[role] = str(p)
        except Exception:
            logger.debug("Agent trace write failed (repair)", exc_info=True)

        try:
            from .artifacts import write_candidate_snapshot

            write_candidate_snapshot(run_dir, candidate)
        except Exception:
            logger.debug("Candidate snapshot write failed", exc_info=True)

        return True
    except Exception:
        logger.debug("Repair attempt failed", exc_info=True)
        return False


# ---------------------------------------------------------------------------
# Failure classifiers
# ---------------------------------------------------------------------------


def _classify_backtest_failure(stderr: str, stdout: str, *, rc: int | None = None) -> tuple[str, str]:
    blob = (stderr or "") + "\n" + (stdout or "")
    blob_l = blob.lower()

    if "no module named 'filelock'" in blob_l or ("module not found" in blob_l and "filelock" in blob_l):
        return (
            "backtest.dependency_missing.filelock",
            "Backtest/Hyperopt failed: dependency_missing(filelock). Install with: pip install filelock or pip install -r requirements-full.txt",
        )

    if "no module named 'freqtrade'" in blob_l or ("module not found" in blob_l and "freqtrade" in blob_l):
        return (
            "backtest.dependency_missing.freqtrade",
            "Backtest failed: dependency_missing(freqtrade). Install with: pip install -r requirements-full.txt",
        )

    if "no module named 'ccxt.static_dependencies" in blob_l:
        return (
            "backtest.dependency_missing.ccxt_static_dependencies",
            "Backtest failed: dependency_missing(ccxt_static_dependencies). Pin ccxt==4.5.4 (known-good) or reinstall ccxt.",
        )

    if "no module named 'talib'" in blob_l or ("importerror" in blob_l and "talib" in blob_l) or "talib" in blob_l:
        return (
            "backtest.dependency_missing.talib",
            "Backtest failed: dependency_missing(talib). TA-Lib is NOT installed. "
            "Replace `import talib.abstract as ta` with `import pandas_ta as ta` and update ALL API calls: "
            "e.g. ta.EMA(dataframe, timeperiod=N) → ta.ema(dataframe['close'], length=N), "
            "ta.BBANDS(dataframe, ...) → bbands_df = ta.bbands(dataframe['close'], length=20, std=2); "
            "upper=bbands_df['BBU_20_2.0'], middle=bbands_df['BBM_20_2.0'], lower=bbands_df['BBL_20_2.0'].",
        )

    if "no module named 'pandas_ta'" in blob_l:
        return (
            "backtest.dependency_missing.pandas_ta",
            "Backtest failed: dependency_missing(pandas_ta). Install via requirements.txt or switch to manual indicators.",
        )

    if "no data" in blob_l and "found" in blob_l:
        return (
            "backtest.data_missing",
            "Backtest failed: data_missing. Ensure OHLCV exists for pairs/timeframe and timerange. If you only have 1h data, set strategy timeframe to 1h.",
        )

    if "order-types mapping is incomplete" in blob_l or "order_types mapping is incomplete" in blob_l:
        return (
            "backtest.strategy_config_incomplete",
            "Backtest failed: strategy_config_incomplete. Define complete order_types and order_time_in_force dicts in the strategy.",
        )

    if "unrecognized arguments" in blob_l or "invalid choice" in blob_l:
        return (
            "backtest.parameter_error",
            "Backtest failed: parameter_error. Check freqtrade args/config/timerange.",
        )

    if "cannot allocate memory for thread-local data" in blob_l:
        return (
            "backtest.runtime_thread_allocation",
            "Backtest failed: runtime_thread_allocation. Shared-server thread allocation failed; lower BLAS/OpenMP thread caps (for example AGENT_MARKET_SANDBOX_THREADS=1) and retry.",
        )

    if "strategy" in blob_l and ("not found" in blob_l or "could not" in blob_l) and "strategy" in blob_l:
        return (
            "backtest.strategy_load_error",
            "Backtest failed: strategy_load_error. Strategy class name/path may be wrong.",
        )

    if "filenotfounderror" in blob_l and "config" in blob_l:
        return (
            "backtest.config_path_error",
            "Backtest failed: config_path_error. freqtrade_config path is invalid.",
        )

    tail = (stderr or "")[-2000:] or (stdout or "")[-2000:]
    rc_s = "" if rc is None else f"rc={rc} "
    return "backtest.unknown", f"Backtest failed ({rc_s}tail={tail})"


def _classify_train_failure(exc: Exception, candidate_type: str) -> tuple[str, str]:
    msg = str(exc or "").strip()
    blob = msg.lower()
    if "stable-baselines3" in blob or "stable_baselines3" in blob:
        return "train_model.dependency_missing.stable_baselines3", msg
    if "gymnasium" in blob:
        return "train_model.dependency_missing.gymnasium", msg
    if "lightgbm" in blob:
        return "train_model.dependency_missing.lightgbm", msg
    if "xgboost" in blob:
        return "train_model.dependency_missing.xgboost", msg
    if "filelock" in blob:
        return "train_model.dependency_missing.filelock", msg
    if candidate_type == "dl" and "model adapter" in blob and "not registered" in blob:
        return "train_model.dependency_missing.torch", msg
    if "torch" in blob:
        return "train_model.dependency_missing.torch", msg
    return f"train_model.{candidate_type}.failed", msg or f"{candidate_type} training failed"


def _candidate_runtime_mem_mb(candidate: StrategyCandidate, *, default_mb: int = 4096) -> int:
    """Return a sandbox memory budget tuned to the candidate runtime profile."""
    candidate_type = _normalize_candidate_type(getattr(candidate, "candidate_type", "rule"))
    if candidate_type == "dl":
        return max(default_mb, 12288)
    if candidate_type in {"ml", "rl"}:
        return max(default_mb, 6144)
    return default_mb


def _rewrite_float_assignment(path: Path, key: str, value: float) -> bool:
    """Rewrite a `key = <float>` assignment in a strategy file (best-effort)."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return False
    pat = re.compile(rf"^(\s*{re.escape(key)}\s*=\s*)([-+0-9.eE]+)(\s*)$", flags=re.MULTILINE)
    new_text, n = pat.subn(rf"\g<1>{value:.6g}\g<3>", text, count=1)
    if not n or new_text == text:
        return False
    try:
        path.write_text(new_text, encoding="utf-8")
    except Exception:
        return False
    return True


def _apply_trade_calibration(
    *,
    candidate: StrategyCandidate,
    config: MinerConfig,
    trades: int,
) -> bool:
    """Adjust model wrapper thresholds to increase trade count (best-effort).

    This is a deterministic self-heal for model candidates when the backtest
    succeeds but violates `min_trades`, avoiding a hard fail due to overly
    conservative signal thresholds.
    """
    min_trades = int(getattr(config, "min_trades", 0) or 0)
    if min_trades <= 0 or int(trades) >= min_trades:
        return False

    cand_type = _normalize_candidate_type(getattr(candidate, "candidate_type", "rule"))
    if cand_type == "rule":
        return False

    payload = dict(getattr(candidate, "candidate_payload", None) or {})
    signal = dict(payload.get("signal") or {})

    # ML/DL wrappers share the same `ml_*` threshold attributes.
    if cand_type in {"ml", "dl"}:
        enter = _coerce_float(signal.get("enter_threshold"), 0.0025)
        exit_ = _coerce_float(signal.get("exit_threshold"), -0.0005)
        scale = float(getattr(config, "trade_calibration_scale", 0.7) or 0.7)
        scale = 0.7 if not (0.2 <= scale <= 0.95) else scale

        new_enter = max(1e-4, enter * scale)
        new_exit = min(0.0, exit_ * scale) if exit_ is not None else -0.0005

        did_enter = _rewrite_float_assignment(candidate.strategy_path, "ml_enter_threshold", new_enter)
        did_exit = _rewrite_float_assignment(candidate.strategy_path, "ml_exit_threshold", new_exit)
        if not (did_enter or did_exit):
            return False

        signal["enter_threshold"] = float(new_enter)
        signal["exit_threshold"] = float(new_exit)
        payload["signal"] = signal
        candidate.candidate_payload = payload
        try:
            candidate.code = candidate.strategy_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            pass
        logger.info(
            "Trade calibration applied for %s: enter_threshold %.6g -> %.6g, exit_threshold %.6g -> %.6g (trades=%d<min_trades=%d)",
            candidate.name,
            float(enter),
            float(new_enter),
            float(exit_),
            float(new_exit),
            int(trades),
            int(min_trades),
        )
        return True

    if cand_type == "rl":
        enter = _coerce_float(signal.get("enter_prob_threshold"), 0.55)
        exit_ = _coerce_float(signal.get("exit_prob_threshold"), 0.35)
        step = float(getattr(config, "trade_calibration_prob_step", 0.05) or 0.05)
        step = 0.05 if not (0.01 <= step <= 0.2) else step

        new_enter = max(0.30, min(0.95, enter - step))
        new_exit = max(0.05, min(0.90, exit_ + step))

        did_enter = _rewrite_float_assignment(candidate.strategy_path, "rl_enter_prob_threshold", new_enter)
        did_exit = _rewrite_float_assignment(candidate.strategy_path, "rl_exit_prob_threshold", new_exit)
        if not (did_enter or did_exit):
            return False

        signal["enter_prob_threshold"] = float(new_enter)
        signal["exit_prob_threshold"] = float(new_exit)
        payload["signal"] = signal
        candidate.candidate_payload = payload
        try:
            candidate.code = candidate.strategy_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            pass
        logger.info(
            "Trade calibration applied for %s: enter_prob_threshold %.4f -> %.4f, exit_prob_threshold %.4f -> %.4f (trades=%d<min_trades=%d)",
            candidate.name,
            float(enter),
            float(new_enter),
            float(exit_),
            float(new_exit),
            int(trades),
            int(min_trades),
        )
        return True

    return False


# ---------------------------------------------------------------------------
# Phase: train model
# ---------------------------------------------------------------------------


def phase_train_model(
    state: MinerState,
    config: MinerConfig,
    run_dir: Path,
    kb: Optional["KnowledgeBase"] = None,
) -> None:
    candidate = _pick_active_candidate(state)
    if candidate is None:
        safe_transition(state, Phase.ANALYSIS)
        return

    candidate_type = _normalize_candidate_type(getattr(candidate, "candidate_type", "rule"))
    if candidate_type == "rule":
        safe_transition(state, Phase.BACKTEST)
        return
    if candidate_type == "rl" and not bool(getattr(config, "enable_rl", False)):
        candidate.failure_category = "train_model.rl.disabled"
        candidate.diagnosis = "[train_model.rl.disabled] RL is disabled in the main miner."
        _advance_after_candidate(state)
        return
    if candidate_type == "dl" and not bool(getattr(config, "enable_dl", False)):
        candidate.failure_category = "train_model.dl.quarantined"
        candidate.diagnosis = "[train_model.dl.quarantined] DL is quarantined from the main miner."
        _advance_after_candidate(state)
        return

    payload = dict(getattr(candidate, "candidate_payload", None) or {})
    training_config = getattr(candidate, "training_config", None) or payload.get("training_config")
    if not isinstance(training_config, dict):
        candidate.failure_category = f"train_model.{candidate_type}.invalid_config"
        candidate.diagnosis = f"[{candidate.failure_category}] Missing training configuration"
        if kb is not None:
            kb.add_failure(
                name=candidate.name,
                iteration=state.iteration,
                failure_type="train_model",
                detail=candidate.diagnosis,
            )
        _advance_after_candidate(state)
        return

    sandbox = candidate.strategy_path.parent.parent.parent
    summary_path = Path(str(((training_config.get("output") or {}).get("model_dir")) or "")) / "training_summary.json"
    from ._helpers import _get_leverage_factor
    _lev = _get_leverage_factor(config.freqtrade_config)

    training_evidence_files: list[Path] = []
    training_evidence_extra: dict[str, Any] = {}

    try:
        if candidate_type in {"ml", "dl"}:
            from agent_market.freqai.training.pipeline import TrainingPipeline  # noqa: WPS433

            TrainingPipeline(training_config).run()
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            risk = dict(payload.get("risk") or {})
            signal = dict(payload.get("signal") or {})
            wrapper_code = _render_ml_strategy_code(
                class_name=candidate.name,
                timeframe=str(payload.get("timeframe") or _prompt_objective_profile(config).get("base_timeframe") or "1h"),
                summary_path=summary_path,
                enter_threshold=_coerce_float(signal.get("enter_threshold"), 0.0025),
                exit_threshold=_coerce_float(signal.get("exit_threshold"), -0.0005),
                minimal_roi=_normalize_roi_map(risk.get("minimal_roi"), {"0": 0.006, "30": 0.003, "90": 0.0}),
                stoploss=_coerce_float(risk.get("stoploss"), -0.012),
                max_hold_minutes=_coerce_int(risk.get("max_hold_minutes"), 180, minimum=15, maximum=24 * 60),
                startup_candle_count=_coerce_int(risk.get("startup_candle_count"), 120, minimum=30, maximum=2000),
                leverage_factor=_lev,
                position_adjustment_enable=bool(getattr(config, "position_adjustment_enable", False)),
                max_entry_position_adjustment=int(getattr(config, "max_entry_position_adjustment", 0) or 0),
            )
            training_evidence_extra = {
                "signal": signal,
                "risk": risk,
            }
        elif candidate_type == "rl":
            from agent_market.freqai.rl.trainer import RLTrainer  # noqa: WPS433

            RLTrainer(training_config).train()
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            signals_root = (run_dir / "rl_signals" / f"iter_{int(candidate.iteration):04d}" / candidate.name).resolve()
            feature_file = payload.get("feature_file") or getattr(config, "model_feature_file", "user_data/freqai_features_real.json")
            gen_cmd = [
                sys.executable,
                str(paths.REPO_ROOT / "scripts" / "rl_generate_signals.py"),
                "--config",
                str(paths.resolve_repo_path(config.freqtrade_config)),
                "--timeframe",
                str(payload.get("timeframe") or _prompt_objective_profile(config).get("base_timeframe") or "1h"),
                "--feature-file",
                str(paths.resolve_repo_path(str(feature_file))),
                "--rl-summary",
                str(summary_path),
                "--out-dir",
                str(signals_root),
            ]
            if payload.get("expressions_file"):
                gen_cmd += ["--expressions-file", str(paths.resolve_repo_path(str(payload.get("expressions_file"))))]
            proc = subprocess.run(
                gen_cmd,
                cwd=str(paths.REPO_ROOT),
                capture_output=True,
                text=True,
                timeout=max(120, int(config.backtest_timeout or 300)),
                check=False,
            )
            if proc.returncode != 0:
                raise RuntimeError((proc.stderr or proc.stdout or "RL signal generation failed").strip())
            risk = dict(payload.get("risk") or {})
            signal = dict(payload.get("signal") or {})
            signals_exchange_dir = signals_root / str(payload.get("exchange") or "gate")
            if signals_exchange_dir.exists():
                training_evidence_files.extend(sorted(signals_exchange_dir.glob("*.feather")))
            wrapper_code = _render_rl_signal_strategy_code(
                class_name=candidate.name,
                timeframe=str(payload.get("timeframe") or _prompt_objective_profile(config).get("base_timeframe") or "1h"),
                signals_exchange_dir=signals_exchange_dir,
                enter_prob_threshold=_coerce_float(signal.get("enter_prob_threshold"), 0.55),
                exit_prob_threshold=_coerce_float(signal.get("exit_prob_threshold"), 0.35),
                minimal_roi=_normalize_roi_map(risk.get("minimal_roi"), {"0": 0.006, "30": 0.003, "90": 0.0}),
                stoploss=_coerce_float(risk.get("stoploss"), -0.012),
                max_hold_minutes=_coerce_int(risk.get("max_hold_minutes"), 180, minimum=15, maximum=24 * 60),
                startup_candle_count=_coerce_int(risk.get("startup_candle_count"), 120, minimum=30, maximum=2000),
                leverage_factor=_lev,
            )
            training_evidence_extra = {
                "signal": signal,
                "risk": risk,
                "signals_exchange_dir": str(signals_exchange_dir),
            }
        else:
            raise ValueError(f"Unsupported candidate_type={candidate_type}")
    except Exception as exc:
        category, detail = _classify_train_failure(exc, candidate_type)
        candidate.failure_category = category
        candidate.diagnosis = f"[{category}] {detail}"
        logger.warning("%s", candidate.diagnosis)
        if kb is not None:
            kb.add_failure(
                name=candidate.name,
                iteration=state.iteration,
                failure_type="train_model",
                detail=candidate.diagnosis,
            )
        _advance_after_candidate(state)
        return

    strategies_dir = sandbox / "user_data" / "strategies"
    strategies_dir.mkdir(parents=True, exist_ok=True)
    wrapper_path = strategies_dir / f"{candidate.name}.py"
    wrapper_path.write_text(wrapper_code, encoding="utf-8")

    try:
        _ensure_configured_strategy_compliance(config, wrapper_path)
        wrapper_code = wrapper_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        logger.debug("Compliance auto-fix failed for trained model wrapper", exc_info=True)

    candidate.training_summary = summary
    candidate.strategy_path = wrapper_path
    candidate.code = wrapper_code
    candidate.validation_passed = False
    candidate.failure_category = ""
    candidate.diagnosis = ""

    try:
        from .artifacts import write_candidate_snapshot, write_training_evidence

        write_candidate_snapshot(run_dir, candidate)
        write_training_evidence(
            run_dir,
            candidate,
            training_summary_path=summary_path,
            extra_files=training_evidence_files,
            extra_payload=training_evidence_extra,
        )
    except Exception:
        logger.debug("Candidate/training evidence write failed", exc_info=True)

    update_candidate_stage(candidate, "trained")
    safe_transition(state, Phase.BACKTEST)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Hyperopt integration
# ---------------------------------------------------------------------------


def _run_hyperopt(
    *,
    candidate: "StrategyCandidate",
    config: MinerConfig,
    sandbox: Path,
    strategies_dir: Path,
) -> bool:
    """Run Freqtrade Hyperopt to optimize candidate parameters.

    Returns True if hyperopt ran successfully (even if no improvement found).
    """
    if not bool(getattr(config, "hyperopt_enabled", False)):
        return False

    # Check if strategy has HyperOptable parameters
    code = candidate.code or ""
    has_params = any(kw in code for kw in ("IntParameter", "DecimalParameter", "CategoricalParameter", "BooleanParameter"))
    if not has_params:
        logger.debug("Skipping hyperopt for %s: no HyperOptable parameters", candidate.name)
        return False

    epochs = int(getattr(config, "hyperopt_epochs", 80) or 80)
    spaces = list(getattr(config, "hyperopt_spaces", ["buy", "sell", "roi", "stoploss"]) or ["buy", "sell"])
    loss_fn = str(getattr(config, "hyperopt_loss", "SharpeHyperOptLoss") or "SharpeHyperOptLoss")
    jobs = int(getattr(config, "hyperopt_jobs", 2) or 2)
    min_trades = int(getattr(config, "hyperopt_min_trades", 10) or 10)

    ft_config = paths.resolve_repo_path(config.freqtrade_config)
    timerange = str(getattr(config, "selection_timerange", "") or getattr(config, "timerange", "") or "")

    wrapper = paths.REPO_ROOT / "scripts" / "freqtrade_cli.py"
    if wrapper.exists():
        cmd = [sys.executable, str(wrapper), "hyperopt"]
    else:
        cmd = [sys.executable, "-m", "freqtrade", "hyperopt"]

    cmd += [
        "--config", str(ft_config),
        "--strategy", candidate.name,
        "--strategy-path", str(strategies_dir),
        "--timerange", timerange,
        "--userdir", str(sandbox / "user_data"),
        "--epochs", str(epochs),
        "--spaces", *spaces,
        "--hyperopt-loss", loss_fn,
        "-j", str(jobs),
        "--min-trades", str(min_trades),
    ]

    logger.info("Running hyperopt for %s: %d epochs, spaces=%s", candidate.name, epochs, spaces)

    try:
        from ._sandbox_exec import run_sandboxed
        cmd_cwd = sandbox / "user_data"
        if not cmd_cwd.exists():
            cmd_cwd = sandbox
        proc = run_sandboxed(
            cmd,
            cwd=cmd_cwd,
            timeout=max(300, config.backtest_timeout * 3),
            cpu_seconds=max(600, config.backtest_timeout * 4),
            mem_mb=8192,
        )
        if proc.returncode != 0:
            logger.warning("Hyperopt failed for %s (rc=%d): %s", candidate.name, proc.returncode, (proc.stderr or "")[-500:])
            return False

        # Parse best params from hyperopt output and apply to strategy file
        stdout = proc.stdout or ""
        # Look for exported params JSON
        param_file = strategies_dir / f"{candidate.name}.json"
        if param_file.exists():
            logger.info("Hyperopt params exported to %s", param_file)
            return True

        # Hyperopt auto-exports params to <strategy_name>.json; next backtest will pick them up
        logger.info("Hyperopt completed for %s (no params file yet — will be created on next run)", candidate.name)
        return True
    except subprocess.TimeoutExpired:
        logger.warning("Hyperopt timed out for %s", candidate.name)
        return False
    except Exception as exc:
        logger.warning("Hyperopt error for %s: %s", candidate.name, exc)
        return False


# Phase: backtest
# ---------------------------------------------------------------------------


def phase_backtest(
    state: MinerState,
    config: MinerConfig,
    run_dir: Path,
    agent: Optional[StrategyAgent] = None,
    kb: Optional["KnowledgeBase"] = None,
) -> None:
    """Validate and backtest the active candidate.

    When ``config.repair_attempts`` > 0, failures will trigger:
    1) local auto-fix (syntax/tool-tag cleanup)
    2) optional agent-guided repair loop

    Multi-candidate: failures advance to the next candidate without stopping the iteration.
    """

    candidate = _pick_active_candidate(state)
    if candidate is None:
        logger.warning("No candidates to backtest")
        safe_transition(state, Phase.ANALYSIS)
        return

    sandbox = candidate.strategy_path.parent.parent.parent  # sandbox root
    cmd_cwd = sandbox / "user_data"
    if not cmd_cwd.exists():
        cmd_cwd = sandbox
    candidate_type = _normalize_candidate_type(getattr(candidate, "candidate_type", "rule"))

    raw_repairs = int(getattr(config, "repair_attempts", 0) or 0)
    if raw_repairs <= 0:
        max_repairs = 0
    else:
        # Configurable repair rounds: clamp to [1, 8] when enabled.
        max_repairs = max(1, min(8, raw_repairs))
    if candidate_type != "rule":
        max_repairs = 0
    backtest_mem_mb = _candidate_runtime_mem_mb(candidate)

    trade_calibration_attempts = int(getattr(config, "trade_calibration_attempts", 0) or 0)
    trade_calibration_attempts = max(0, min(12, trade_calibration_attempts))
    max_backtest_attempts = max_repairs + 1
    # For model candidates, allow deterministic re-backtest attempts that only
    # adjust wrapper thresholds (no LLM repair). This is strictly opt-in.
    if candidate_type in {"ml", "dl", "rl"} and trade_calibration_attempts > 0:
        max_backtest_attempts = max(1, trade_calibration_attempts + 1)

    for attempt_idx in range(max_backtest_attempts):
        # D6: If previous repair iteration failed (we're back in the loop),
        # mark the last repair_ledger entry as 'fail'
        if attempt_idx > 0:
            _update_repair_ledger_result(candidate, "fail")

        if candidate_type != "rule":
            _restore_trained_wrapper(candidate, config, run_dir)

        # Always refresh code from disk if possible (repairs may have edited it).
        try:
            if candidate.strategy_path.exists():
                candidate.code = candidate.strategy_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            pass

        # Static validation
        strict_validation = candidate_type == "rule"
        if strict_validation:
            passed, msg = validate_strategy_code(candidate.code)
        else:
            try:
                ast.parse(candidate.code)
                passed, msg = True, "Internal model wrapper syntax passed"
            except SyntaxError as exc:
                passed, msg = False, f"syntax error: {exc}"
        if passed:
            passed, msg = _validate_timeframe_policy(candidate.code, config)
        candidate.validation_passed = passed
        if not passed:
            category = _classify_validation_failure(msg)
            candidate.failure_category = category
            failure = f"[{category}] Validation failed: {msg}"
            candidate.diagnosis = failure

            # Local auto-fix first for any deterministic validator failure.
            did, fixes = auto_fix_strategy_file(candidate.strategy_path)
            if did:
                try:
                    candidate.code = candidate.strategy_path.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    pass
                passed2, msg2 = validate_strategy_code(candidate.code)
                candidate.validation_passed = passed2
                if passed2:
                    logger.info("Auto-fix succeeded for %s: %s", candidate.name, ",".join(fixes))
                    candidate.diagnosis = ""
                    candidate.failure_category = ""
                else:
                    category2 = _classify_validation_failure(msg2)
                    candidate.failure_category = category2
                    failure = f"[{category2}] Validation failed after auto-fix({','.join(fixes)}): {msg2}"
                    candidate.diagnosis = failure

            if candidate.validation_passed:
                # Proceed to backtest without consuming an LLM repair attempt.
                pass
            else:
                if attempt_idx < max_repairs:
                    local_agent = agent
                    if local_agent is None:
                        if _repair_agent_explicitly_configured(config):
                            try:
                                local_agent = build_strategy_agent(config, sandbox)
                            except Exception as exc:
                                logger.warning("Repair skipped (agent unavailable): %s", exc)
                                local_agent = None
                        else:
                            logger.info(
                                "Repair skipped for %s: no explicit repair agent configured",
                                candidate.name,
                            )

                    try:
                        failure_for_repair = failure
                        if getattr(candidate, "backtester_notes", ""):
                            failure_for_repair += "\n\nBacktester preflight:\n" + _truncate_text(
                                str(candidate.backtester_notes),
                                limit=1500,
                            )

                        ok = (
                            _repair_candidate(
                                agent=local_agent,
                                config=config,
                                run_dir=run_dir,
                                sandbox=sandbox,
                                candidate=candidate,
                                failure=failure_for_repair,
                                attempt=attempt_idx + 1,
                                max_attempts=max_repairs,
                            )
                            if local_agent is not None
                            else False
                        )
                    finally:
                        if local_agent is not agent:
                            try:
                                local_agent.close()
                            except Exception:
                                pass

                    if ok:
                        continue

                if kb is not None:
                    kb.add_failure(
                        name=candidate.name,
                        iteration=state.iteration,
                        failure_type="validation",
                        detail=candidate.diagnosis,
                    )

                _advance_after_candidate(state)
                return

        # Preflight: ensure freqtrade sanity-required settings.
        try:
            did_comp, comp_fixes = _ensure_configured_strategy_compliance(
                config, candidate.strategy_path
            )
            if did_comp:
                try:
                    candidate.code = candidate.strategy_path.read_text(
                        encoding="utf-8", errors="replace"
                    )
                except Exception:
                    pass
                logger.info(
                    "Compliance auto-fix applied for %s: %s",
                    candidate.name,
                    ",".join(comp_fixes),
                )
        except Exception:
            logger.debug("Compliance auto-fix failed", exc_info=True)


        # Preflight: verify strategy is loadable via freqtrade list-strategies
        # Skip on repair attempts (attempt_idx > 0) since local validation passed
        strategies_dir = sandbox / "user_data" / "strategies"
        if attempt_idx == 0:
            try:
                ls_cmd = [
                    sys.executable,
                    "-m",
                    "freqtrade",
                    "list-strategies",
                    "--strategy-path",
                    str(strategies_dir),
                ]
                wrapper = paths.REPO_ROOT / "scripts" / "freqtrade_cli.py"
                if wrapper.exists():
                    ls_cmd = [
                        sys.executable,
                        str(wrapper),
                        "list-strategies",
                        "--strategy-path",
                        str(strategies_dir),
                    ]
                from ._sandbox_exec import run_sandboxed
                ls_proc = run_sandboxed(
                    ls_cmd, cwd=cmd_cwd, timeout=30, cpu_seconds=60, mem_mb=2048,
                )
                if ls_proc.returncode == 0 and (ls_proc.stdout or "").strip() and candidate.name not in ls_proc.stdout:
                    logger.warning(
                        "Preflight: %s not found by freqtrade list-strategies — skipping to repair",
                        candidate.name,
                    )
                    candidate.failure_category = "backtest.preflight_not_found"
                    candidate.diagnosis = (
                        f"[backtest.preflight_not_found] Strategy {candidate.name} not loadable by freqtrade. "
                        f"stderr: {(ls_proc.stderr or '')[:500]}"
                    )
                    if attempt_idx < max_repairs:
                        local_agent = agent
                        if local_agent is None:
                            try:
                                local_agent = build_strategy_agent(config, sandbox)
                            except Exception:
                                local_agent = None
                        if local_agent is not None:
                            ok = _repair_candidate(
                                agent=local_agent,
                                config=config,
                                run_dir=run_dir,
                                sandbox=sandbox,
                                candidate=candidate,
                                failure=candidate.diagnosis,
                                attempt=attempt_idx + 1,
                                max_attempts=max_repairs,
                            )
                            if ok:
                                continue
                    _advance_after_candidate(state)
                    return
            except Exception:
                pass  # preflight is best-effort; continue to backtest

        # Re-sync filename with class name before backtest (safety net)
        try:
            actual_class = infer_strategy_class_name(
                candidate.strategy_path.read_text(encoding="utf-8", errors="replace")
            )
            if actual_class and actual_class != candidate.name:
                candidate.name = actual_class
            desired = candidate.strategy_path.parent / f"{candidate.name}.py"
            if desired != candidate.strategy_path and not desired.exists():
                candidate.strategy_path.rename(desired)
                candidate.strategy_path = desired
                logger.debug("Re-synced file: %s -> %s", candidate.strategy_path.name, desired.name)
        except Exception:
            pass

        # Stage 1: 2-pair smoke backtest (cheap, ~10s) — reject obvious failures before hyperopt
        if attempt_idx == 0 and bool(getattr(config, "enable_quick_funnel", True)):
            smoke_pairs = list(getattr(config, "quick_backtest_pairs", []) or [])[:3] or ["BTC/USDT", "ETH/USDT"]
            smoke_timeout = int(getattr(config, "quick_backtest_timeout", 60) or 60)

            # Build smoke backtest command
            ft_config_path = paths.resolve_repo_path(config.freqtrade_config)
            smoke_cmd = [
                sys.executable,
                "-m",
                "freqtrade",
                "backtesting",
                "--config",
                str(ft_config_path),
                "--strategy",
                candidate.name,
                "--strategy-path",
                str(strategies_dir),
                "--timerange",
                str(getattr(config, "quick_backtest_timerange", "") or getattr(config, "selection_timerange", "") or config.timerange),
                "--userdir",
                str(sandbox / "user_data"),
                "-p",
                *smoke_pairs,
            ]
            wrapper = paths.REPO_ROOT / "scripts" / "freqtrade_cli.py"
            if wrapper.exists():
                smoke_cmd = [
                    sys.executable,
                    str(wrapper),
                    "backtesting",
                    "--config",
                    str(ft_config_path),
                    "--strategy",
                    candidate.name,
                    "--strategy-path",
                    str(strategies_dir),
                    "--timerange",
                    str(getattr(config, "quick_backtest_timerange", "") or getattr(config, "selection_timerange", "") or config.timerange),
                    "--userdir",
                    str(sandbox / "user_data"),
                    "-p",
                    *smoke_pairs,
                ]

            try:
                from ._sandbox_exec import run_sandboxed
                smoke_proc = run_sandboxed(
                    smoke_cmd, cwd=cmd_cwd,
                    timeout=smoke_timeout, cpu_seconds=smoke_timeout + 30, mem_mb=backtest_mem_mb,
                )
                if smoke_proc.returncode != 0:
                    logger.info("Smoke backtest crashed for %s, skipping hyperopt", candidate.name)
                else:
                    # Parse smoke results for quality gate
                    smoke_results_dir = sandbox / "user_data" / "backtest_results"
                    smoke_zip = find_latest_backtest_zip(smoke_results_dir)
                    smoke_passed = False
                    if smoke_zip:
                        try:
                            smoke_summary = build_backtest_summary(smoke_zip)
                            smoke_trades = int(smoke_summary.get("trades", 0) or 0)
                            smoke_pf = float(smoke_summary.get("profit_factor", 0) or 0)
                            smoke_profit = float(smoke_summary.get("profit_total_pct", 0) or 0)
                            candidate.quick_backtest_summary = smoke_summary

                            quick_gate = _resolve_quick_gate_thresholds(config)
                            quick_min_trades = int(quick_gate["min_trades"])
                            quick_min_pf = float(quick_gate["min_profit_factor"])
                            quick_min_profit = float(quick_gate["min_profit_pct"])
                            quick_max_drawdown = float(quick_gate["max_drawdown_pct"])
                            smoke_drawdown = float(
                                smoke_summary.get("max_drawdown_pct", smoke_summary.get("max_drawdown_account", 0)) or 0
                            )

                            smoke_passed = (
                                smoke_trades >= quick_min_trades
                                and smoke_pf >= quick_min_pf
                                and smoke_profit >= quick_min_profit
                                and smoke_drawdown <= quick_max_drawdown
                            )
                            logger.info(
                                "Smoke quality: trades=%d pf=%.2f profit=%.2f%% drawdown=%.2f%% → %s",
                                smoke_trades, smoke_pf, smoke_profit, smoke_drawdown,
                                "PASS" if smoke_passed else "FAIL",
                            )
                        except Exception:
                            smoke_passed = True  # parse failure → let full backtest decide
                    else:
                        smoke_passed = True  # no zip → let full backtest decide

                    if smoke_passed:
                        logger.info("Smoke passed for %s, proceeding to hyperopt", candidate.name)
                        update_candidate_stage(candidate, "smoke_passed")
                        _ho_ok = _run_hyperopt(
                            candidate=candidate,
                            config=config,
                            sandbox=sandbox,
                            strategies_dir=strategies_dir,
                        )
                        if _ho_ok:
                            update_candidate_stage(candidate, "hyperopt_done")
                    else:
                        logger.info("Smoke quality gate failed for %s, skipping hyperopt", candidate.name)
            except Exception:
                logger.debug("Smoke backtest error, skipping hyperopt", exc_info=True)

        logger.info(
            "Phase BACKTEST: running freqtrade backtesting for %s (attempt %d/%d)",
            candidate.name,
            attempt_idx,
            max_backtest_attempts - 1,
        )

        # Build backtest command
        ft_config = paths.resolve_repo_path(config.freqtrade_config)
        backtest_timerange = str(getattr(config, "selection_timerange", "") or config.timerange)
        results_dir = sandbox / "user_data" / "backtest_results"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Clear stale results before each attempt so we only see this run's output
        for stale_zip in results_dir.glob("backtest-result-*.zip"):
            try:
                stale_zip.unlink()
            except Exception:
                logger.debug("Failed to remove stale backtest zip: %s", stale_zip, exc_info=True)
        stale_last = results_dir / ".last_result.json"
        if stale_last.exists():
            try:
                stale_last.unlink()
            except Exception:
                logger.debug("Failed to remove stale .last_result.json", exc_info=True)

        cmd = [
            sys.executable,
            "-m",
            "freqtrade",
            "backtesting",
            "--config",
            str(ft_config),
            "--strategy",
            candidate.name,
            "--strategy-path",
            str(strategies_dir),
            "--timerange",
            backtest_timerange,
            "--userdir",
            str(sandbox / "user_data"),
        ]

        # Try wrapper script first
        wrapper = paths.REPO_ROOT / "scripts" / "freqtrade_cli.py"
        if wrapper.exists():
            cmd = [
                sys.executable,
                str(wrapper),
                "backtesting",
                "--config",
                str(ft_config),
                "--strategy",
                candidate.name,
                "--strategy-path",
                str(strategies_dir),
                "--timerange",
                backtest_timerange,
                "--userdir",
                str(sandbox / "user_data"),
            ]

        try:
            from ._sandbox_exec import run_sandboxed
            import time as _bt_time
            _bt_start = _bt_time.time()
            proc = run_sandboxed(
                cmd,
                cwd=cmd_cwd,
                timeout=config.backtest_timeout,
                cpu_seconds=config.backtest_timeout + 60,
                mem_mb=backtest_mem_mb,
            )
            _bt_elapsed = _bt_time.time() - _bt_start
            # D4/D13: Track backtest wall time on candidate
            if not hasattr(candidate, 'candidate_payload') or candidate.candidate_payload is None:
                candidate.candidate_payload = {}
            candidate.candidate_payload["backtest_seconds"] = round(_bt_elapsed, 1)
            # D13: Aggregate into state economics immediately
            if hasattr(state, "economics"):
                state.economics["total_backtest_seconds"] = (
                    state.economics.get("total_backtest_seconds", 0.0) + _bt_elapsed
                )
        except subprocess.TimeoutExpired:
            _bt_elapsed = _bt_time.time() - _bt_start
            # D13: Record backtest time even on timeout
            if not hasattr(candidate, 'candidate_payload') or candidate.candidate_payload is None:
                candidate.candidate_payload = {}
            candidate.candidate_payload["backtest_seconds"] = round(_bt_elapsed, 1)
            # D13: Aggregate into state economics immediately
            if hasattr(state, "economics"):
                state.economics["total_backtest_seconds"] = (
                    state.economics.get("total_backtest_seconds", 0.0) + _bt_elapsed
                )
            candidate.failure_category = "backtest.timeout"
            candidate.diagnosis = f"[backtest.timeout] Backtest timed out after {config.backtest_timeout}s"
            logger.warning("%s", candidate.diagnosis)

            if attempt_idx < max_repairs:
                local_agent = agent
                if local_agent is None:
                    try:
                        local_agent = build_strategy_agent(config, sandbox)
                    except Exception as exc:
                        logger.warning("Repair skipped (agent unavailable): %s", exc)
                        local_agent = None
                try:
                    failure_for_repair = candidate.diagnosis
                    if getattr(candidate, "backtester_notes", ""):
                        failure_for_repair += "\n\nBacktester preflight:\n" + _truncate_text(
                            str(candidate.backtester_notes),
                            limit=1500,
                        )
                    ok = (
                        _repair_candidate(
                            agent=local_agent,
                            config=config,
                            run_dir=run_dir,
                            sandbox=sandbox,
                            candidate=candidate,
                            failure=failure_for_repair,
                            attempt=attempt_idx + 1,
                            max_attempts=max_repairs,
                        )
                        if local_agent is not None
                        else False
                    )
                finally:
                    if local_agent is not agent:
                        try:
                            local_agent.close()
                        except Exception:
                            pass
                if ok:
                    continue

            if kb is not None:
                kb.add_failure(
                    name=candidate.name,
                    iteration=state.iteration,
                    failure_type="backtest",
                    detail=candidate.diagnosis,
                )

            _advance_after_candidate(state)
            return

        if proc.returncode != 0:
            category, diag = _classify_backtest_failure(proc.stderr or "", proc.stdout or "", rc=proc.returncode)
            candidate.failure_category = category
            candidate.diagnosis = f"[{category}] {diag}"
            logger.warning("%s", candidate.diagnosis)

            if attempt_idx < max_repairs:
                local_agent = agent
                if local_agent is None:
                    try:
                        local_agent = build_strategy_agent(config, sandbox)
                    except Exception as exc:
                        logger.warning("Repair skipped (agent unavailable): %s", exc)
                        local_agent = None
                try:
                    failure_for_repair = candidate.diagnosis
                    if getattr(candidate, "backtester_notes", ""):
                        failure_for_repair += "\n\nBacktester preflight:\n" + _truncate_text(
                            str(candidate.backtester_notes),
                            limit=1500,
                        )
                    ok = (
                        _repair_candidate(
                            agent=local_agent,
                            config=config,
                            run_dir=run_dir,
                            sandbox=sandbox,
                            candidate=candidate,
                            failure=failure_for_repair,
                            attempt=attempt_idx + 1,
                            max_attempts=max_repairs,
                        )
                        if local_agent is not None
                        else False
                    )
                finally:
                    if local_agent is not agent:
                        try:
                            local_agent.close()
                        except Exception:
                            pass
                if ok:
                    continue

            if kb is not None:
                kb.add_failure(
                    name=candidate.name,
                    iteration=state.iteration,
                    failure_type="backtest",
                    detail=candidate.diagnosis,
                )

            _advance_after_candidate(state)
            return

        # Parse results (stale artifacts cleared before each attempt)
        try:
            zip_path = find_latest_backtest_zip(results_dir)
            if zip_path is None:
                candidate.failure_category = "backtest.result_missing_zip"
                candidate.diagnosis = "[backtest.result_missing_zip] No backtest result zip found"
                logger.warning("%s", candidate.diagnosis)

                if attempt_idx < max_repairs:
                    local_agent = agent
                    if local_agent is None:
                        try:
                            local_agent = build_strategy_agent(config, sandbox)
                        except Exception as exc:
                            logger.warning("Repair skipped (agent unavailable): %s", exc)
                            local_agent = None
                    try:
                        failure_for_repair = candidate.diagnosis
                        if getattr(candidate, "backtester_notes", ""):
                            failure_for_repair += "\n\nBacktester preflight:\n" + _truncate_text(
                                str(candidate.backtester_notes),
                                limit=1500,
                            )
                        ok = (
                            _repair_candidate(
                                agent=local_agent,
                                config=config,
                                run_dir=run_dir,
                                sandbox=sandbox,
                                candidate=candidate,
                                failure=failure_for_repair,
                                attempt=attempt_idx + 1,
                                max_attempts=max_repairs,
                            )
                            if local_agent is not None
                            else False
                        )
                    finally:
                        if local_agent is not agent:
                            try:
                                local_agent.close()
                            except Exception:
                                pass
                    if ok:
                        continue

                if kb is not None:
                    kb.add_failure(
                        name=candidate.name,
                        iteration=state.iteration,
                        failure_type="backtest",
                        detail=candidate.diagnosis,
                    )

                _advance_after_candidate(state)
                return

            summary = build_backtest_summary(zip_path)
            trades = int(summary.get("trades") or 0)
            min_trades = int(getattr(config, "min_trades", 0) or 0)
            if (
                candidate_type in {"ml", "dl", "rl"}
                and trade_calibration_attempts > 0
                and min_trades > 0
                and trades < min_trades
                and attempt_idx < max_backtest_attempts - 1
            ):
                did = _apply_trade_calibration(candidate=candidate, config=config, trades=trades)
                if did:
                    # Record evidence for artifacts/debugging (best-effort).
                    payload = dict(getattr(candidate, "candidate_payload", None) or {})
                    tc = dict(payload.get("trade_calibration") or {})
                    tc["rounds"] = int(tc.get("rounds") or 0) + 1
                    tc.setdefault("trades_before", []).append(int(trades))
                    tc["min_trades"] = int(min_trades)
                    payload["trade_calibration"] = tc
                    candidate.candidate_payload = payload
                    logger.info(
                        "Re-running backtest after trade calibration for %s (%d/%d)",
                        candidate.name,
                        attempt_idx + 1,
                        max_backtest_attempts - 1,
                    )
                    continue
            candidate.backtest_summary = summary
            candidate.failure_category = ""
            candidate.diagnosis = ""
            # D6: Mark last repair as verified-pass (if any repair occurred)
            _update_repair_ledger_result(candidate, "pass")
            safe_transition(state, Phase.EVALUATION)
            update_candidate_stage(candidate, "backtested")

            try:
                from .artifacts import write_backtest_summary

                write_backtest_summary(run_dir, candidate, zip_path=zip_path)
            except Exception:
                logger.debug("Backtest summary artifact write failed", exc_info=True)

            logger.info(
                "Backtest completed: profit=%.2f%% trades=%s",
                summary.get("profit_total_pct", 0) or 0,
                summary.get("trades"),
            )
            return
        except Exception as e:
            candidate.failure_category = "backtest.result_parse_error"
            candidate.diagnosis = f"[backtest.result_parse_error] Backtest result parsing failed: {e}"
            logger.warning("%s", candidate.diagnosis)

            if attempt_idx < max_repairs:
                local_agent = agent
                if local_agent is None:
                    try:
                        local_agent = build_strategy_agent(config, sandbox)
                    except Exception as exc:
                        logger.warning("Repair skipped (agent unavailable): %s", exc)
                        local_agent = None
                try:
                    failure_for_repair = candidate.diagnosis
                    if getattr(candidate, "backtester_notes", ""):
                        failure_for_repair += "\n\nBacktester preflight:\n" + _truncate_text(
                            str(candidate.backtester_notes),
                            limit=1500,
                        )
                    ok = (
                        _repair_candidate(
                            agent=local_agent,
                            config=config,
                            run_dir=run_dir,
                            sandbox=sandbox,
                            candidate=candidate,
                            failure=failure_for_repair,
                            attempt=attempt_idx + 1,
                            max_attempts=max_repairs,
                        )
                        if local_agent is not None
                        else False
                    )
                finally:
                    if local_agent is not agent:
                        try:
                            local_agent.close()
                        except Exception:
                            pass
                if ok:
                    continue

            if kb is not None:
                kb.add_failure(
                    name=candidate.name,
                    iteration=state.iteration,
                    failure_type="backtest",
                    detail=candidate.diagnosis,
                )

            _advance_after_candidate(state)
            return

    if kb is not None:
        kb.add_failure(
            name=candidate.name,
            iteration=state.iteration,
            failure_type="backtest",
            detail=candidate.diagnosis or "Backtest failed",
        )

    _advance_after_candidate(state)

from __future__ import annotations

import json, logging
from pathlib import Path
from typing import Any, Dict, Optional

from agent_market import paths
from ._helpers import (
    _coerce_float,
    _coerce_int,
    _normalize_roi_map,
    _freqtrade_config_defaults,
    _freqtrade_market_context,
    _normalize_candidate_type,
    _prompt_objective_profile,
    _allowed_model_families,
    _sanitize_candidate_name,
)
from .dtypes import MinerConfig, StrategyCandidate
from .sandbox import ensure_freqtrade_strategy_compliance_file

logger = logging.getLogger(__name__)


def _sanitize_model_params(
    *,
    candidate_type: str,
    model_family: str,
    raw_params: dict[str, Any],
) -> dict[str, Any]:
    params = dict(raw_params or {})

    if candidate_type == "ml":
        if model_family == "lightgbm":
            boost_round = _coerce_int(
                params.pop("num_boost_round", params.pop("n_estimators", None)),
                240,
                minimum=50,
                maximum=4000,
            )
            for forbidden in (
                "class_weight",
                "is_unbalance",
                "scale_pos_weight",
                "pos_bagging_fraction",
                "neg_bagging_fraction",
            ):
                params.pop(forbidden, None)
            params["objective"] = "regression"
            params["metric"] = "rmse"
            params["num_boost_round"] = boost_round
        elif model_family == "xgboost":
            boost_round = _coerce_int(
                params.pop("num_boost_round", params.pop("n_estimators", None)),
                240,
                minimum=50,
                maximum=4000,
            )
            for forbidden in ("class_weight", "scale_pos_weight", "max_delta_step"):
                params.pop(forbidden, None)
            params["objective"] = "reg:squarederror"
            params["eval_metric"] = "rmse"
            params["num_boost_round"] = boost_round
    elif candidate_type == "dl" and model_family == "pytorch_mlp":
        params["hidden_dims"] = [int(max(16, _coerce_int(v, 64, minimum=16, maximum=2048))) for v in list(params.get("hidden_dims") or [128, 64])[:4]]
        params["dropout"] = max(0.0, min(0.5, _coerce_float(params.get("dropout"), 0.1)))
        params["epochs"] = _coerce_int(params.get("epochs"), 24, minimum=4, maximum=400)
        params["batch_size"] = _coerce_int(params.get("batch_size"), 128, minimum=16, maximum=4096)
        params["learning_rate"] = max(1e-5, min(5e-2, _coerce_float(params.get("learning_rate"), 1e-3)))
        params["use_cuda"] = bool(params.get("use_cuda", False))

    return params


def _normalize_model_candidate_payload(
    *,
    raw_spec: dict[str, Any],
    candidate_type: str,
    config: MinerConfig,
    run_dir: Path,
    iteration: int,
    candidate_idx: int,
) -> dict[str, Any]:
    normalized_type = _normalize_candidate_type(candidate_type)
    allowed_families = _allowed_model_families(normalized_type)
    market_context = _freqtrade_market_context(config.freqtrade_config)
    exchange = market_context.exchange
    base_timeframe = market_context.timeframe
    datadir = market_context.datadir
    training_pairs = list(getattr(config, "model_training_pairs", None) or []) or market_context.pairs

    raw_family = str(raw_spec.get("model_family") or "").strip().lower()
    if raw_family not in allowed_families:
        if allowed_families:
            raw_family = allowed_families[(iteration + candidate_idx) % len(allowed_families)]
        else:
            raw_family = ""

    fallback_name = f"{normalized_type.title()}CandidateCand{candidate_idx}"
    name_hint = _sanitize_candidate_name(str(raw_spec.get("name_hint") or ""), fallback=fallback_name)
    name_hint = f"{name_hint}_cand{candidate_idx}"

    feature_file = str(raw_spec.get("feature_file") or getattr(config, "model_feature_file", "") or "user_data/freqai_features_real.json")
    expressions_file_raw = raw_spec.get("expressions_file", getattr(config, "model_expressions_file", None))
    expressions_file = str(expressions_file_raw).strip() if expressions_file_raw else None

    label_period_default = 6 if normalized_type in {"ml", "dl", "rl"} else 12
    label_period = _coerce_int(raw_spec.get("label_period"), label_period_default, minimum=1, maximum=96)

    training_cfg = raw_spec.get("training") if isinstance(raw_spec.get("training"), dict) else {}
    model_params = raw_spec.get("model_params") if isinstance(raw_spec.get("model_params"), dict) else {}
    signal_cfg = raw_spec.get("signal") if isinstance(raw_spec.get("signal"), dict) else {}
    risk_cfg = raw_spec.get("risk") if isinstance(raw_spec.get("risk"), dict) else {}
    reward_cfg = raw_spec.get("reward") if isinstance(raw_spec.get("reward"), dict) else {}

    if normalized_type == "ml":
        params_default: dict[str, Any] = {
            "objective": "regression",
            "learning_rate": 0.05,
            "num_boost_round": 200,
        }
        if raw_family == "lightgbm":
            params_default.update({"num_leaves": 63, "feature_fraction": 0.9, "bagging_fraction": 0.9, "bagging_freq": 1})
        elif raw_family == "xgboost":
            params_default = {
                "objective": "reg:squarederror",
                "eta": 0.05,
                "max_depth": 6,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "num_boost_round": 240,
            }
    elif normalized_type == "dl":
        params_default = {
            "hidden_dims": [128, 64],
            "dropout": 0.1,
            "epochs": 24,
            "batch_size": 128,
            "learning_rate": 0.001,
            "use_cuda": False,
        }
    else:
        params_default = {
            "learning_rate": 3e-4,
            "n_steps": 256,
            "batch_size": 128,
            "gamma": 0.99,
        }

    params_default.update(model_params)
    params_default = _sanitize_model_params(
        candidate_type=normalized_type,
        model_family=raw_family,
        raw_params=params_default,
    )

    default_roi = {"0": 0.006, "30": 0.003, "90": 0.0}
    default_signal = {
        "enter_threshold": 0.0025,
        "exit_threshold": -0.0005,
        "enter_prob_threshold": 0.55,
        "exit_prob_threshold": 0.35,
    }
    default_risk = {
        "minimal_roi": default_roi,
        "stoploss": -0.012,
        "max_hold_minutes": 180,
        "startup_candle_count": 120,
    }
    merged_signal = dict(default_signal)
    merged_signal.update(signal_cfg)
    merged_risk = dict(default_risk)
    merged_risk.update(risk_cfg)
    merged_risk["minimal_roi"] = _normalize_roi_map(merged_risk.get("minimal_roi"), default_roi)
    merged_risk["stoploss"] = _coerce_float(merged_risk.get("stoploss"), -0.012)
    merged_risk["max_hold_minutes"] = _coerce_int(merged_risk.get("max_hold_minutes"), 180, minimum=15, maximum=24 * 60)
    merged_risk["startup_candle_count"] = _coerce_int(merged_risk.get("startup_candle_count"), 120, minimum=30, maximum=2000)

    output_root_raw = str(getattr(config, "model_output_root", "") or "").strip()
    if output_root_raw:
        output_root = paths.resolve_repo_path(output_root_raw)
    else:
        output_root = run_dir / "models"
    model_dir = (output_root / f"iter_{int(iteration):04d}" / f"cand_{int(candidate_idx):02d}_{name_hint}").resolve()
    data_cfg: dict[str, Any] = {
        "feature_file": feature_file,
        "data_dir": datadir,
        "exchange": exchange,
        "timeframe": base_timeframe,
        "pairs": training_pairs,
        "label_period": label_period,
        "train_timerange": str(getattr(config, "train_timerange", "") or "").strip(),
        "test_timerange": str(getattr(config, "timerange", "") or "").strip(),
    }
    if expressions_file:
        data_cfg["expressions_file"] = expressions_file

    normalized_reward = {
        "fee_bps": _coerce_float(reward_cfg.get("fee_bps"), 10.0),
        "holding_penalty_bps": _coerce_float(reward_cfg.get("holding_penalty_bps"), 0.2),
        "drawdown_penalty": _coerce_float(reward_cfg.get("drawdown_penalty"), 0.05),
        "invalid_action_penalty": _coerce_float(reward_cfg.get("invalid_action_penalty"), 0.0005),
    }

    if normalized_type == "rl":
        training_payload = {
            "data": data_cfg,
            "policy": "MlpPolicy",
            "algo": params_default,
            "training": {
                "total_timesteps": _coerce_int(
                    training_cfg.get("total_timesteps"),
                    int(getattr(config, "rl_total_timesteps", 5000) or 5000),
                    minimum=1000,
                    maximum=200_000,
                ),
            },
            "output": {
                "model_dir": str(model_dir),
            },
            "reward": normalized_reward,
        }
    else:
        training_payload = {
            "data": data_cfg,
            "model": {
                "name": raw_family,
                "params": params_default,
            },
            "training": {
                "validation_ratio": max(0.05, min(0.5, _coerce_float(training_cfg.get("validation_ratio"), getattr(config, "training_validation_ratio", 0.2) or 0.2))),
                "rolling_splits": _coerce_int(training_cfg.get("rolling_splits"), int(getattr(config, "training_rolling_splits", 3) or 3), minimum=0, maximum=8),
                "scaler": str(training_cfg.get("scaler") or getattr(config, "training_scaler", "robust") or "none"),
                "purge": _coerce_int(training_cfg.get("purge"), label_period, minimum=0, maximum=96),
                "embargo": _coerce_int(training_cfg.get("embargo"), max(0, label_period // 3), minimum=0, maximum=96),
            },
            "output": {
                "model_dir": str(model_dir),
            },
        }

    return {
        "name_hint": name_hint,
        "candidate_type": normalized_type,
        "model_family": raw_family,
        "rationale": str(raw_spec.get("rationale") or "").strip(),
        "feature_file": feature_file,
        "expressions_file": expressions_file,
        "label_period": label_period,
        "training_config": training_payload,
        "signal": {
            "enter_threshold": _coerce_float(merged_signal.get("enter_threshold"), 0.0025),
            "exit_threshold": _coerce_float(merged_signal.get("exit_threshold"), -0.0005),
            "enter_prob_threshold": _coerce_float(merged_signal.get("enter_prob_threshold"), 0.55),
            "exit_prob_threshold": _coerce_float(merged_signal.get("exit_prob_threshold"), 0.35),
        },
        "risk": merged_risk,
        "reward": normalized_reward,
        "pairs": training_pairs,
        "exchange": exchange,
        "timeframe": base_timeframe,
        "model_dir": str(model_dir),
    }


def _format_runtime_literal(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=4, sort_keys=True)


def _render_ml_strategy_code(
    *,
    class_name: str,
    timeframe: str,
    summary_path: Path,
    enter_threshold: float,
    exit_threshold: float,
    minimal_roi: dict[str, float],
    stoploss: float,
    max_hold_minutes: int,
    startup_candle_count: int,
    leverage_factor: float = 1.0,
    position_adjustment_enable: bool = False,
    max_entry_position_adjustment: int = 0,
) -> str:
    leverage_method = ""
    if leverage_factor > 1.0:
        leverage_method = f"""
    def leverage(self, pair: str, current_time, current_rate: float,
                 proposed_leverage: float, max_leverage: float,
                 entry_tag, side: str, **kwargs) -> float:
        return {leverage_factor}
"""

    dca_method = ""
    if position_adjustment_enable and max_entry_position_adjustment > 0:
        dca_method = f"""
    position_adjustment_enable = True
    max_entry_position_adjustment = {max_entry_position_adjustment}

    def adjust_trade_position(self, trade, current_time, current_rate: float,
                              current_profit: float, min_stake: float | None,
                              max_stake: float, current_entry_rate: float,
                              current_exit_rate: float, current_entry_profit: float,
                              current_exit_profit: float, **kwargs) -> float | None:
        if current_profit > -0.02:
            return None
        # DCA: add to position when underwater by >2%, up to {max_entry_position_adjustment} times
        if trade.nr_of_successful_entries >= self.max_entry_position_adjustment:
            return None  # max entries reached
        return trade.stake_amount  # add same size

    def custom_stake_amount(self, current_time, current_rate: float, proposed_stake: float,
                            min_stake: float | None, max_stake: float,
                            leverage: float, entry_tag: str | None, side: str, **kwargs) -> float:
        # Martingale: increase stake for each consecutive loss
        import math
        recent_trades = self.dp.get_trades() if hasattr(self, 'dp') and self.dp else []
        consecutive_losses = 0
        for t in reversed(recent_trades[-10:] if recent_trades else []):
            if hasattr(t, 'profit_ratio') and t.profit_ratio < 0:
                consecutive_losses += 1
            else:
                break
        # Cap martingale at 3x base stake
        multiplier = min(3.0, 1.0 + 0.5 * consecutive_losses)
        return proposed_stake * multiplier
"""
    return f'''from __future__ import annotations

import json
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from pandas import DataFrame

REPO_ROOT = Path(r"{paths.REPO_ROOT}")
for _path in [str(REPO_ROOT / "src"), str(REPO_ROOT)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from freqtrade.strategy import IStrategy


class {class_name}(IStrategy):
    order_types = {{"entry": "market", "exit": "market", "stoploss": "market", "stoploss_on_exchange": False}}
    order_time_in_force = {{"entry": "GTC", "exit": "GTC"}}
    timeframe = "{timeframe}"
    can_short = False
    minimal_roi = {_format_runtime_literal(minimal_roi)}
    stoploss = {stoploss}
    trailing_stop = False
    use_exit_signal = True
    process_only_new_candles = True
    startup_candle_count = {startup_candle_count}
    ml_enter_threshold = {enter_threshold}
    ml_exit_threshold = {exit_threshold}
    max_hold_minutes = {max_hold_minutes}
{leverage_method}{dca_method}
    _summary_path = Path(r"{summary_path}")
    _model = None
    _model_name: Optional[str] = None
    _model_features: Optional[List[str]] = None
    _feature_cfg: Optional[Dict[str, Any]] = None
    _expression_specs = None
    _scaler = None
    _loaded = False

    def _load_model(self) -> None:
        if self._loaded:
            return

        summary = json.loads(self._summary_path.read_text(encoding="utf-8"))
        feature_file = summary.get("feature_snapshot") or summary.get("feature_file")
        if feature_file:
            fp = Path(feature_file)
            if fp.exists():
                self._feature_cfg = json.loads(fp.read_text(encoding="utf-8-sig"))

        expr_file = summary.get("expressions_snapshot") or summary.get("expressions_file")
        if expr_file:
            ep = Path(expr_file)
            if ep.exists():
                from agent_market.freqai.expression_engine import load_expression_file
                self._expression_specs = load_expression_file(ep)

        scaler_path = self._summary_path.with_name("scaler.pkl")
        if scaler_path.exists():
            with open(scaler_path, "rb") as handle:
                scaler_payload = pickle.load(handle)
            self._scaler = scaler_payload.get("scaler")

        self._model_features = [str(item) for item in (summary.get("features") or []) if str(item).strip()]
        model_path = Path(summary.get("model_path", ""))
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {{model_path}}")

        model_name = str(summary.get("model", "lightgbm") or "lightgbm").strip().lower()
        self._model_name = model_name
        if model_name == "lightgbm":
            import lightgbm as lgb
            self._model = lgb.Booster(model_file=str(model_path))
        elif model_name in ("xgboost", "xgb"):
            import xgboost as xgb
            booster = xgb.Booster()
            booster.load_model(str(model_path))
            self._model = booster
        elif model_name == "pytorch_mlp":
            import torch
            checkpoint = torch.load(model_path, map_location="cpu")
            from agent_market.freqai.model.torch_models import FeedForwardNet
            cfg = checkpoint.get("config", {{}})
            model = FeedForwardNet(
                checkpoint.get("input_dim", len(self._model_features)),
                cfg.get("hidden_dims", [64, 32]),
                cfg.get("dropout", 0.0),
            )
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            self._model = model
        else:
            import pickle as _pickle
            with open(model_path, "rb") as handle:
                self._model = _pickle.load(handle)

        self._loaded = True

    def _predict(self, matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=np.float32)
        if (self._model_name or "") in ("xgboost", "xgb"):
            import xgboost as xgb
            return self._model.predict(xgb.DMatrix(matrix))
        if hasattr(self._model, "predict"):
            return self._model.predict(matrix)
        if hasattr(self._model, "forward"):
            import torch
            with torch.no_grad():
                tensor = torch.from_numpy(matrix.astype(np.float32))
                return self._model(tensor).cpu().numpy()
        raise RuntimeError(f"Unsupported model type: {{type(self._model)}}")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        self._load_model()
        if self._feature_cfg:
            from agent_market.freqai.features import apply_configured_features
            dataframe = apply_configured_features(dataframe, self._feature_cfg)
        if self._expression_specs:
            from agent_market.freqai.expression_engine import apply_expressions
            dataframe, _ = apply_expressions(dataframe, self._expression_specs, on_error="raise")

        for col in self._model_features or []:
            if col not in dataframe.columns:
                dataframe[col] = 0.0

        matrix = (
            dataframe[self._model_features]
            .astype(float)
            .replace([np.inf, -np.inf], np.nan)
            .ffill()
            .fillna(0.0)
        )
        if self._scaler is not None:
            matrix_values = self._scaler.transform(matrix.to_numpy(dtype=np.float32))
        else:
            matrix_values = matrix.to_numpy(dtype=np.float32)
        preds = self._predict(matrix_values)
        dataframe["ml_pred"] = np.asarray(preds).reshape(-1)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe.loc[
            (dataframe["volume"] > 0) & (dataframe["ml_pred"] >= self.ml_enter_threshold),
            ["enter_long", "enter_tag"],
        ] = (1, "ml_long")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe.loc[
            (dataframe["volume"] > 0) & (dataframe["ml_pred"] <= self.ml_exit_threshold),
            "exit_long",
        ] = 1
        return dataframe

    def custom_exit(self, pair: str, trade, current_time: datetime, current_rate: float, current_profit: float, **kwargs) -> Optional[str]:
        if trade.open_date_utc is None:
            return None
        held_seconds = (current_time - trade.open_date_utc).total_seconds()
        if held_seconds >= self.max_hold_minutes * 60:
            return "time_exit"
        return None
'''


def _render_rl_signal_strategy_code(
    *,
    class_name: str,
    timeframe: str,
    signals_exchange_dir: Path,
    enter_prob_threshold: float,
    exit_prob_threshold: float,
    minimal_roi: dict[str, float],
    stoploss: float,
    max_hold_minutes: int,
    startup_candle_count: int,
    leverage_factor: float = 1.0,
) -> str:
    leverage_method = ""
    if leverage_factor > 1.0:
        leverage_method = f"""
    def leverage(self, pair: str, current_time, current_rate: float,
                 proposed_leverage: float, max_leverage: float,
                 entry_tag, side: str, **kwargs) -> float:
        return {leverage_factor}
"""
    return f'''from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from pandas import DataFrame

from freqtrade.strategy import IStrategy


class {class_name}(IStrategy):
    order_types = {{"entry": "market", "exit": "market", "stoploss": "market", "stoploss_on_exchange": False}}
    order_time_in_force = {{"entry": "GTC", "exit": "GTC"}}
    timeframe = "{timeframe}"
    can_short = False
    minimal_roi = {_format_runtime_literal(minimal_roi)}
    stoploss = {stoploss}
    trailing_stop = False
    use_exit_signal = True
    process_only_new_candles = True
    startup_candle_count = {startup_candle_count}
    rl_enter_prob_threshold = {enter_prob_threshold}
    rl_exit_prob_threshold = {exit_prob_threshold}
    max_hold_minutes = {max_hold_minutes}
{leverage_method}
    _signals_dir = Path(r"{signals_exchange_dir}")

    def _signals_path(self, pair: str) -> Path:
        sanitized = pair.replace("/", "_")
        return self._signals_dir / f"{{sanitized}}-{{self.timeframe}}.feather"

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = dataframe.copy()
        dataframe["date"] = pd.to_datetime(dataframe["date"], utc=True)

        signal_path = self._signals_path(metadata["pair"])
        if signal_path.exists():
            sig = pd.read_feather(signal_path)
            sig["date"] = pd.to_datetime(sig["date"], utc=True)
            cols = ["date", "rl_hold_prob", "rl_long_prob", "rl_short_prob", "rl_action"]
            dataframe = dataframe.merge(sig[cols], on="date", how="left")
        else:
            dataframe["rl_hold_prob"] = 1.0
            dataframe["rl_long_prob"] = 0.0
            dataframe["rl_short_prob"] = 0.0
            dataframe["rl_action"] = 0

        dataframe["rl_hold_prob"] = dataframe.get("rl_hold_prob", 1.0).fillna(1.0)
        dataframe["rl_long_prob"] = dataframe.get("rl_long_prob", 0.0).fillna(0.0)
        dataframe["rl_short_prob"] = dataframe.get("rl_short_prob", 0.0).fillna(0.0)
        dataframe["rl_action"] = dataframe.get("rl_action", 0).fillna(0).astype(int)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        cond = (dataframe["volume"] > 0) & (
            (dataframe["rl_action"] == 1) | (dataframe["rl_long_prob"] >= self.rl_enter_prob_threshold)
        )
        dataframe.loc[cond, ["enter_long", "enter_tag"]] = (1, "rl_long")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        cond = (dataframe["volume"] > 0) & (
            (dataframe["rl_action"] == 2) | (dataframe["rl_long_prob"] <= self.rl_exit_prob_threshold)
        )
        dataframe.loc[cond, "exit_long"] = 1
        return dataframe

    def custom_exit(self, pair: str, trade, current_time: datetime, current_rate: float, current_profit: float, **kwargs) -> Optional[str]:
        if trade.open_date_utc is None:
            return None
        held_seconds = (current_time - trade.open_date_utc).total_seconds()
        if held_seconds >= self.max_hold_minutes * 60:
            return "time_exit"
        return None
'''


def _restore_trained_wrapper(candidate: StrategyCandidate, config: MinerConfig, run_dir: Path) -> bool:
    candidate_type = _normalize_candidate_type(getattr(candidate, "candidate_type", "rule"))
    if candidate_type == "rule":
        return False

    payload = dict(getattr(candidate, "candidate_payload", None) or {})
    training_config = getattr(candidate, "training_config", None) or payload.get("training_config")
    if not isinstance(training_config, dict):
        return False

    output_cfg = training_config.get("output") if isinstance(training_config.get("output"), dict) else {}
    model_dir_raw = str(output_cfg.get("model_dir") or "").strip()
    if not model_dir_raw:
        return False

    summary_path = Path(model_dir_raw) / "training_summary.json"
    if not summary_path.exists():
        return False

    risk = dict(payload.get("risk") or {})
    signal = dict(payload.get("signal") or {})
    timeframe = str(payload.get("timeframe") or _prompt_objective_profile(config).get("base_timeframe") or "1h")
    from ._helpers import _get_leverage_factor
    _lev = _get_leverage_factor(config.freqtrade_config)

    if candidate_type in {"ml", "dl"}:
        wrapper_code = _render_ml_strategy_code(
            class_name=candidate.name,
            timeframe=timeframe,
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
    elif candidate_type == "rl":
        exchange = str(payload.get("exchange") or "gate")
        signals_root = (run_dir / "rl_signals" / f"iter_{int(candidate.iteration):04d}" / candidate.name).resolve()
        signals_exchange_dir = signals_root / exchange
        wrapper_code = _render_rl_signal_strategy_code(
            class_name=candidate.name,
            timeframe=timeframe,
            signals_exchange_dir=signals_exchange_dir,
            enter_prob_threshold=_coerce_float(signal.get("enter_prob_threshold"), 0.55),
            exit_prob_threshold=_coerce_float(signal.get("exit_prob_threshold"), 0.35),
            minimal_roi=_normalize_roi_map(risk.get("minimal_roi"), {"0": 0.006, "30": 0.003, "90": 0.0}),
            stoploss=_coerce_float(risk.get("stoploss"), -0.012),
            max_hold_minutes=_coerce_int(risk.get("max_hold_minutes"), 180, minimum=15, maximum=24 * 60),
            startup_candle_count=_coerce_int(risk.get("startup_candle_count"), 120, minimum=30, maximum=2000),
            leverage_factor=_lev,
        )
    else:
        return False

    candidate.strategy_path.parent.mkdir(parents=True, exist_ok=True)
    candidate.strategy_path.write_text(wrapper_code, encoding="utf-8")

    try:
        tf, enforce_short = _freqtrade_config_defaults(config.freqtrade_config)
        ensure_freqtrade_strategy_compliance_file(
            candidate.strategy_path,
            timeframe=tf,
            enforce_can_short_false=enforce_short,
        )
        wrapper_code = candidate.strategy_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        logger.debug("Compliance auto-fix failed while restoring trained wrapper", exc_info=True)

    candidate.code = wrapper_code
    return True

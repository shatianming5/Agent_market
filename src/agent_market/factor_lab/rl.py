"""RL strategy layer — PPO trading agent that uses g-factors-13 (or any Hub
deployment) as state features and learns discrete open/hold/close decisions.

The TradingEnv + PPO trainer already exists under agent_market.freqai.rl; this
module is the glue that ties an Hub deployment + a timeframe into a ready-to-
train RL config, then dispatches training + inference.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .paths import (ROOT, USER_DATA, MODELS_DIR,
                    DEFAULT_PAIRS, DEFAULT_TRAIN3, DEFAULT_REAL_TEST3,
                    DEFAULT_LABEL_PERIOD, DEFAULT_CLASS_THRESHOLD)


DEFAULT_RL_ALGO = {
    "learning_rate": 3e-4,
    "n_steps": 1024,
    "batch_size": 64,
    "n_epochs": 8,
    "gamma": 0.995,
    "gae_lambda": 0.95,
    "ent_coef": 0.005,
    "vf_coef": 0.5,
    "clip_range": 0.2,
    "max_grad_norm": 0.5,
    "policy_kwargs": {
        "net_arch": {"pi": [128, 128], "vf": [128, 128]},
        "activation_fn": "tanh",
    },
}


DEFAULT_REWARD = {
    "fee_bps": 8.0,                # 0.08% taker
    "holding_penalty_bps": 0.3,    # small per-bar cost to prevent over-holding
    "drawdown_penalty": 0.08,      # discourage deep drawdowns
    "invalid_action_penalty": 0.0005,
    "reward_horizon": 1,
    "window_size": 0,              # 0 = no CNN OHLCV window
}

# "Strict" reward preset — discourages over-trading, used when you see PPO
# churning every bar and burning on fees. Raises fee impact ×3, lowers hold
# penalty ×3, penalizes invalid actions ×20.
STRICT_REWARD = {
    "fee_bps": 25.0,               # overstate real fee to push hold
    "holding_penalty_bps": 0.1,    # cheap to hold
    "drawdown_penalty": 0.20,      # strong DD deterrent
    "invalid_action_penalty": 0.01,
    "reward_horizon": 1,
    "window_size": 0,
}

REWARD_PRESETS = {"default": DEFAULT_REWARD, "strict": STRICT_REWARD}


def build_config(*, expressions_file: str, timeframe: str = "1h",
                 total_timesteps: int = 50_000,
                 policy: str = "MlpPolicy",
                 algo_overrides: Optional[Dict[str, Any]] = None,
                 reward_overrides: Optional[Dict[str, Any]] = None,
                 reward_profile: str = "default",
                 env_class: str = "trading",
                 algo_class: str = "ppo",
                 train_start: str = DEFAULT_TRAIN3[0],
                 train_end: str = DEFAULT_TRAIN3[1],
                 pairs: Optional[List[str]] = None,
                 model_dir: Optional[str] = None) -> Dict[str, Any]:
    """Produce the config dict that RLTrainer expects."""
    algo = dict(DEFAULT_RL_ALGO)
    if algo_overrides:
        algo.update(algo_overrides)
    base_reward = REWARD_PRESETS.get(reward_profile, DEFAULT_REWARD)
    reward = dict(base_reward)
    if reward_overrides:
        reward.update(reward_overrides)
    return {
        "data": {
            "feature_file": "user_data/freqai_features_real.json",
            "expressions_file": expressions_file,
            "data_dir": "user_data/data",
            "exchange": "kucoin",
            "timeframe": timeframe,
            "label_period": DEFAULT_LABEL_PERIOD,
            "task": "classify_3way",
            "class_threshold": DEFAULT_CLASS_THRESHOLD,
            "pairs": pairs or DEFAULT_PAIRS,
            "train_timerange": f"{train_start.replace('-','')}-{train_end.replace('-','')}",
        },
        "algo": algo,
        "algo_class": algo_class,
        "policy": policy,
        "reward": reward,
        "env_class": env_class,
        "training": {"total_timesteps": int(total_timesteps)},
        "output": {"model_dir": str(Path(model_dir) if model_dir else (MODELS_DIR / "rl"))},
    }


def train(*, tag: str, expressions_file: str = "user_data/freqai_expressions.json",
          timeframe: str = "1h", total_timesteps: int = 50_000,
          window_size: int = 0, reward_profile: str = "default",
          env_class: str = "trading", algo_class: str = "ppo",
          pairs: Optional[List[str]] = None,
          policy: str = "MlpPolicy") -> Dict[str, Any]:
    """Train a PPO agent on the chosen factor library. Returns a summary dict."""
    import sys
    _SRC = str(ROOT / "src")
    if _SRC not in sys.path: sys.path.insert(0, _SRC)
    from agent_market.freqai.rl.trainer import RLTrainer

    model_dir = MODELS_DIR / f"rl_{tag}"
    reward_over = {"window_size": int(window_size)} if window_size > 0 else None
    cfg = build_config(
        expressions_file=expressions_file, timeframe=timeframe,
        total_timesteps=total_timesteps, reward_overrides=reward_over,
        reward_profile=reward_profile, env_class=env_class,
        algo_class=algo_class, policy=policy, pairs=pairs,
        model_dir=str(model_dir),
    )
    t0 = time.time()
    print(f"[rl] training PPO tag={tag}  expr={expressions_file}  "
          f"tf={timeframe}  steps={total_timesteps}  window={window_size}")
    print(f"[rl] train range: {cfg['data']['train_timerange']}")

    _hub_event("rl.started", tag=tag, timeframe=timeframe,
               timesteps=total_timesteps, expressions=expressions_file)

    trainer = RLTrainer(cfg)
    result = trainer.train()
    elapsed = time.time() - t0

    summary_path = model_dir / "training_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    summary.update({
        "tag": tag, "elapsed_min": round(elapsed / 60, 2),
        "model_path": str(result.model_path),
    })
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _hub_event("rl.finished", tag=tag, elapsed_min=round(elapsed / 60, 2),
               model_path=str(result.model_path), timesteps=result.timesteps)
    print(f"[rl] done in {elapsed/60:.1f}m → {result.model_path}")
    return summary


def _hub_event(event_type: str, **payload) -> None:
    try:
        from agent_market.factor_hub import Client
        Client().log(event_type, payload=payload or None)
    except Exception:
        pass


# ============================================================
# Evaluation — rollout a trained PPO on a specific timerange
# ============================================================

def evaluate(*, tag: str, timerange_start: str, timerange_end: str,
             expressions_file: str = "user_data/freqai_expressions.json",
             timeframe: str = "1h",
             model_dir: Optional[str] = None,
             window_size: int = 0,
             env_class: str = "trading",
             reward_profile: str = "strict") -> Dict[str, Any]:
    """Deterministically rollout the saved PPO on a specified OOS window.

    Returns a dict with equity curve, trade count, per-bar return etc. — the
    same shape as a single walk-forward window result (but via env rollout,
    not freqtrade).
    """
    import sys
    _SRC = str(ROOT / "src")
    if _SRC not in sys.path: sys.path.insert(0, _SRC)
    from agent_market.freqai.rl.env import TradingEnv, TradingEnvConfig
    from agent_market.freqai.rl.target_pos_env import TargetPositionEnv, TargetPosEnvConfig
    from agent_market.freqai.training.pipeline import FeatureDatasetBuilder
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise ImportError("stable-baselines3 required for rl.evaluate") from exc

    model_dir_p = Path(model_dir) if model_dir else (MODELS_DIR / f"rl_{tag}")
    model_path = model_dir_p / "ppo_trading_env.zip"
    if not model_path.exists():
        raise FileNotFoundError(f"no PPO model at {model_path}")

    data_cfg = {
        "feature_file": "user_data/freqai_features_real.json",
        "expressions_file": expressions_file,
        "data_dir": "user_data/data",
        "exchange": "kucoin",
        "timeframe": timeframe,
        "label_period": DEFAULT_LABEL_PERIOD,
        "task": "classify_3way",
        "class_threshold": DEFAULT_CLASS_THRESHOLD,
        "pairs": DEFAULT_PAIRS,
        "train_timerange": f"{timerange_start.replace('-','')}-{timerange_end.replace('-','')}",
    }
    print(f"[rl-eval] rollout {tag} on {timerange_start} → {timerange_end}")

    builder = FeatureDatasetBuilder(data_cfg)
    dataset = builder.build()
    reward_preset = REWARD_PRESETS.get(reward_profile, STRICT_REWARD)
    if env_class == "target_position":
        tp_cfg = TargetPosEnvConfig(
            data=data_cfg,
            fee_bps=float(reward_preset.get("fee_bps", 25.0)),
            holding_penalty_bps=float(reward_preset.get("holding_penalty_bps", 0.1)),
            drawdown_penalty=float(reward_preset.get("drawdown_penalty", 0.15)),
        )
        env = TargetPositionEnv(dataset, tp_cfg)
    elif env_class == "threshold":
        from agent_market.freqai.rl.threshold_env import ThresholdPolicyEnv, ThresholdEnvConfig
        th_cfg = ThresholdEnvConfig(
            data=data_cfg,
            fee_bps=float(reward_preset.get("fee_bps", 25.0)),
            holding_penalty_bps=float(reward_preset.get("holding_penalty_bps", 0.1)),
            drawdown_penalty=float(reward_preset.get("drawdown_penalty", 0.15)),
        )
        env = ThresholdPolicyEnv(dataset, th_cfg)
    elif env_class == "trade":
        from agent_market.freqai.rl.trade_env import TradeLevelEnv, TradeEnvConfig
        te_cfg = TradeEnvConfig(
            data=data_cfg, fee_bps=8.0,
            stop_loss_pct=0.015, take_profit_pct=0.03, max_hold_bars=24,
        )
        env = TradeLevelEnv(dataset, te_cfg)
    else:
        env_cfg = TradingEnvConfig(
            data=data_cfg,
            fee_bps=float(reward_preset.get("fee_bps", 8.0)),
            holding_penalty_bps=float(reward_preset.get("holding_penalty_bps", 0.3)),
            drawdown_penalty=float(reward_preset.get("drawdown_penalty", 0.08)),
            window_size=int(window_size),
        )
        env = TradingEnv(dataset, env_cfg)
    # Recurrent PPO models need RecurrentPPO.load not PPO.load
    summary_path = model_dir_p / "training_summary.json"
    is_recurrent = False
    if summary_path.exists():
        try:
            s = json.loads(summary_path.read_text())
            is_recurrent = str(s.get("data", {}).get("algo_class", "")).lower() == "recurrent_ppo" \
                or str(s.get("policy", "")).lower().endswith("lstmpolicy")
        except Exception:
            pass
    if is_recurrent:
        from sb3_contrib import RecurrentPPO
        model = RecurrentPPO.load(str(model_path))
    else:
        model = PPO.load(str(model_path))

    obs, _ = env.reset()
    actions, positions, returns_per_bar, equities = [], [], [], []
    trades = 0
    last_position = 0
    done = False
    steps = 0
    # For RecurrentPPO we must carry lstm_states across steps
    lstm_states = None
    episode_starts = np.array([True])
    while not done:
        if is_recurrent:
            action, lstm_states = model.predict(
                obs, state=lstm_states,
                episode_start=episode_starts, deterministic=True,
            )
            episode_starts = np.array([False])
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
        actions.append(int(action))
        positions.append(int(info.get("position", 0)))
        equities.append(float(info.get("equity", 1.0)))
        returns_per_bar.append(float(reward))
        if info.get("position", 0) != last_position:
            trades += 1
            last_position = info.get("position", 0)
        steps += 1
        if steps > len(dataset.features) + 10:
            break

    eq = np.asarray(equities, dtype=np.float64)
    peak = np.maximum.accumulate(eq)
    max_dd = float(((peak - eq) / np.maximum(peak, 1e-9)).max()) if len(eq) else 0.0
    total_return = float(eq[-1] - 1.0) if len(eq) else 0.0
    n_bars = len(eq)
    trade_dist = {0: 0, 1: 0, 2: 0}
    for a in actions:
        trade_dist[a] = trade_dist.get(a, 0) + 1
    summary = {
        "tag": tag,
        "timerange": f"{timerange_start} → {timerange_end}",
        "train_size": int(dataset.features.shape[0]),
        "n_bars": n_bars,
        "final_equity": float(eq[-1]) if len(eq) else 1.0,
        "total_return_pct": total_return * 100.0,
        "max_drawdown_pct": max_dd * 100.0,
        "trades": trades,
        "actions": {"hold": int(trade_dist.get(0, 0)),
                     "buy": int(trade_dist.get(1, 0)),
                     "sell": int(trade_dist.get(2, 0))},
    }
    print(f"[rl-eval] {tag}: return={summary['total_return_pct']:+.2f}%  "
          f"bars={n_bars}  trades={trades}  max_dd={summary['max_drawdown_pct']:.2f}%")
    _hub_event("rl.evaluated", **summary)
    return summary

"""Behavior-cloning pretrain → PPO fine-tune.

Pipeline (E3):
    1. Load the stacked-walk-forward production Ridge model (or any
       predict_proba-compatible model).
    2. Replay every bar in TRAIN3, collect (state, teacher_action) pairs
       where teacher_action = argmax(Ridge.predict_proba(state)) mapped to
       TradingEnv actions {hold, buy, sell}.
    3. Pretrain a PPO-compatible MLP policy on supervised cross-entropy.
    4. Transfer weights into a fresh PPO model and continue with reward-
       based RL for N more steps.

This gives the RL agent an initial "act like the teacher" prior before
exposing it to the noisy reward landscape — a proven technique for
sparse-reward domains where pure exploration fails.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn

from .paths import ROOT, MODELS_DIR, DEFAULT_PAIRS


def _collect_teacher_actions(data_cfg: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Fit a fresh Ridge classifier inline on TRAIN3 data, then label every
    bar with argmax(predict_proba). Mapping: class 0/1/2 → action 2/0/1
    (down→sell, flat→hold, up→buy)."""
    import sys
    _SRC = str(ROOT / "src")
    if _SRC not in sys.path: sys.path.insert(0, _SRC)
    from agent_market.freqai.training.pipeline import FeatureDatasetBuilder
    from agent_market.freqai.model.ridge import RidgeOnlyAdapter

    builder = FeatureDatasetBuilder(data_cfg)
    dataset = builder.build()
    X = dataset.features
    y = dataset.labels.astype(np.int64)
    print(f"[rl-bc] fitting teacher Ridge on {X.shape[0]} bars / {X.shape[1]} features")

    adapter = RidgeOnlyAdapter({"num_class": 3, "alpha": 1.0,
                                 "model_dir": str(MODELS_DIR / "rl_bc_teacher")})
    adapter.fit(X, y)

    probs = adapter.predict(X)
    raw_labels = np.argmax(probs, axis=1)
    # Map 3-way classification {0=down, 1=flat, 2=up} to env actions
    # {0=hold, 1=buy, 2=sell}: flat→hold, up→buy, down→sell
    action_map = np.array([2, 0, 1], dtype=np.int64)
    labels = action_map[raw_labels]
    print(f"[rl-bc] raw class dist : {np.bincount(raw_labels, minlength=3)}")
    print(f"[rl-bc] action dist    : {np.bincount(labels, minlength=3)}  "
          f"(0=hold / 1=buy / 2=sell)")
    return {"X": X, "y": labels}


class _TeacherMLP(nn.Module):
    """Matches PPO default MlpPolicy architecture [64, 64] → 3 actions."""
    def __init__(self, input_dim: int, hidden: List[int] = None, n_actions: int = 3):
        super().__init__()
        h = hidden or [64, 64]
        layers = []
        prev = input_dim
        for width in h:
            layers += [nn.Linear(prev, width), nn.Tanh()]
            prev = width
        layers.append(nn.Linear(prev, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def pretrain(*, tag: str,
              expressions_file: str = "user_data/freqai_expressions.json",
              timeframe: str = "1h", epochs: int = 5, batch_size: int = 512,
              lr: float = 1e-3) -> Dict[str, Any]:
    """Run BC pretraining. Saves a torch state_dict that PPO can partially
    initialize from via custom policy_kwargs."""
    from agent_market.factor_lab.paths import DEFAULT_TRAIN3, DEFAULT_LABEL_PERIOD, DEFAULT_CLASS_THRESHOLD
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
        "train_timerange": f"{DEFAULT_TRAIN3[0].replace('-','')}-{DEFAULT_TRAIN3[1].replace('-','')}",
    }

    buf = _collect_teacher_actions(data_cfg)
    X = buf["X"].astype(np.float32)
    y = buf["y"].astype(np.int64)
    n = X.shape[0]

    device = torch.device("cpu")
    model = _TeacherMLP(input_dim=X.shape[1], n_actions=3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print(f"[rl-bc] pretraining {epochs} epochs × batch {batch_size} on {n} samples")
    idx = np.arange(n)
    for ep in range(epochs):
        np.random.shuffle(idx)
        total_loss = 0.0
        correct = 0
        batches = 0
        for start in range(0, n, batch_size):
            bi = idx[start:start + batch_size]
            xb = torch.from_numpy(X[bi]).to(device)
            yb = torch.from_numpy(y[bi]).to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()
            correct += int((logits.argmax(1) == yb).sum().item())
            batches += 1
        acc = correct / n
        print(f"  epoch {ep+1}: loss={total_loss/max(1,batches):.4f}  acc={acc:.3f}")

    out_dir = MODELS_DIR / f"rl_bc_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    bc_path = out_dir / "bc_policy.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "input_dim": int(X.shape[1]),
        "hidden": [64, 64],
        "n_actions": 3,
        "epochs": epochs,
        "final_acc": acc,
    }, bc_path)
    print(f"[rl-bc] saved → {bc_path}")
    return {"tag": tag, "bc_path": str(bc_path), "final_acc": acc,
            "n_samples": n, "input_dim": int(X.shape[1])}


def evaluate_bc(*, tag: str, timerange_start: str, timerange_end: str,
                expressions_file: str = "user_data/freqai_expressions.json",
                timeframe: str = "1h") -> Dict[str, Any]:
    """Roll out the BC-pretrained policy in TradingEnv on the given range."""
    import sys
    _SRC = str(ROOT / "src")
    if _SRC not in sys.path: sys.path.insert(0, _SRC)
    from agent_market.freqai.rl.env import TradingEnv, TradingEnvConfig
    from agent_market.freqai.training.pipeline import FeatureDatasetBuilder
    from agent_market.factor_lab.paths import (DEFAULT_LABEL_PERIOD, DEFAULT_CLASS_THRESHOLD)

    bc_path = MODELS_DIR / f"rl_bc_{tag}" / "bc_policy.pt"
    if not bc_path.exists():
        raise FileNotFoundError(f"BC policy not found at {bc_path}")
    ckpt = torch.load(bc_path, map_location="cpu", weights_only=False)
    policy = _TeacherMLP(input_dim=int(ckpt["input_dim"]),
                         hidden=list(ckpt["hidden"]),
                         n_actions=int(ckpt["n_actions"]))
    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()

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
    builder = FeatureDatasetBuilder(data_cfg)
    dataset = builder.build()
    env_cfg = TradingEnvConfig(data=data_cfg, fee_bps=25.0,
                                holding_penalty_bps=0.1, drawdown_penalty=0.15,
                                invalid_action_penalty=0.01)
    env = TradingEnv(dataset, env_cfg)

    obs, _ = env.reset()
    actions = []
    equities = []
    done = False
    steps = 0
    bc_input_dim = int(ckpt["input_dim"])
    with torch.no_grad():
        while not done:
            obs_arr = np.asarray(obs, dtype=np.float32)
            # BC policy was trained on feature-only observations (no position/unrealized).
            # Env obs = features + [position, unrealized]; take only the feature prefix.
            if obs_arr.shape[-1] > bc_input_dim:
                obs_arr = obs_arr[:bc_input_dim]
            x = torch.from_numpy(obs_arr).unsqueeze(0)
            logits = policy(x)
            action = int(logits.argmax(1).item())
            obs, _r, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            actions.append(action)
            equities.append(float(info.get("equity", 1.0)))
            steps += 1
            if steps > len(dataset.features) + 10:
                break
    eq = np.asarray(equities)
    total = float(eq[-1] - 1.0) if len(eq) else 0.0
    peak = np.maximum.accumulate(eq) if len(eq) else np.array([1.0])
    dd = float(((peak - eq) / np.maximum(peak, 1e-9)).max()) if len(eq) else 0.0
    dist = {0: 0, 1: 0, 2: 0}
    for a in actions:
        dist[a] = dist.get(a, 0) + 1
    print(f"[bc-eval] {tag} {timerange_start}→{timerange_end}: "
          f"return={total*100:+.2f}%  bars={len(eq)}  max_dd={dd*100:.2f}%")
    return {
        "tag": tag,
        "timerange": f"{timerange_start} → {timerange_end}",
        "total_return_pct": total * 100.0,
        "max_drawdown_pct": dd * 100.0,
        "n_bars": len(eq),
        "actions": {"hold": dist.get(0, 0),
                     "buy": dist.get(1, 0),
                     "sell": dist.get(2, 0)},
    }

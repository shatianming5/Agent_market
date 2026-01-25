#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def _ensure_import_paths() -> Path:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return root


ROOT = _ensure_import_paths()


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _infer_feature_file(config_path: Path) -> Path:
    base = config_path.parent
    candidates = [
        base / "freqai_features_real.json",
        base / "freqai_features.json",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError(
        "No feature file found. Tried: " + ", ".join(str(p) for p in candidates)
    )


def _matrix_from_df(df: pd.DataFrame, columns: List[str]) -> np.ndarray:
    mat = df[columns].astype(float).replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
    return mat.to_numpy(dtype=np.float32)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate RL (PPO) signals and save them for freqtrade strategies to consume without importing torch."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Freqtrade JSON config (used to resolve exchange/pairs/timeframe/datadir).",
    )
    parser.add_argument("--timeframe", default=None, help="Optional timeframe override (e.g. 1h).")
    parser.add_argument(
        "--feature-file",
        default=None,
        help="Optional freqai_features*.json. If omitted, inferred from config directory.",
    )
    parser.add_argument(
        "--rl-summary",
        default="artifacts/models/rl_real/training_summary.json",
        help="RL training_summary.json (contains model_path + feature list).",
    )
    parser.add_argument(
        "--out-dir",
        default="artifacts/signals/rl_real",
        help="Output base directory for signals.",
    )
    args = parser.parse_args(argv)

    from agent_market.config import FreqAISettings  # noqa: WPS433
    from agent_market.freqai.features import apply_configured_features  # noqa: WPS433

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (ROOT / cfg_path).resolve()
    settings = FreqAISettings.from_file(cfg_path, timeframe_override=args.timeframe)
    settings.validate_dataset(settings.timeframe)

    feature_path = Path(args.feature_file) if args.feature_file else _infer_feature_file(cfg_path)
    if not feature_path.is_absolute():
        feature_path = (ROOT / feature_path).resolve()
    feature_cfg = _read_json(feature_path)

    rl_summary_path = Path(args.rl_summary)
    if not rl_summary_path.is_absolute():
        rl_summary_path = (ROOT / rl_summary_path).resolve()
    summary = _read_json(rl_summary_path)
    model_path = Path(summary.get("model_path") or "")
    if not model_path.is_absolute():
        model_path = (ROOT / model_path).resolve()
    features = [str(x) for x in (summary.get("features") or []) if str(x).strip()]
    if not features:
        raise SystemExit(f"[rl_signals] missing features in {rl_summary_path}")
    if not model_path.exists():
        raise SystemExit(f"[rl_signals] RL model not found: {model_path}")

    from stable_baselines3 import PPO  # noqa: WPS433
    import torch  # noqa: WPS433

    model = PPO.load(model_path, device="cpu")

    out_base = Path(args.out_dir)
    if not out_base.is_absolute():
        out_base = (ROOT / out_base).resolve()
    out_exchange = out_base / settings.exchange
    out_exchange.mkdir(parents=True, exist_ok=True)

    for pair in settings.pairs:
        sanitized = pair.replace("/", "_")
        data_file = settings.data_dir / f"{sanitized}-{settings.timeframe}.feather"
        df = pd.read_feather(data_file)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df = df.sort_values("date").reset_index(drop=True)

        df_feat = apply_configured_features(df.copy(), feature_cfg)
        missing = [col for col in features if col not in df_feat.columns]
        if missing:
            raise SystemExit(
                f"[rl_signals] {pair} missing feature columns: {', '.join(missing[:10])}"
            )

        matrix = _matrix_from_df(df_feat, features)
        obs = torch.as_tensor(matrix, dtype=torch.float32, device=model.policy.device)
        dist = model.policy.get_distribution(obs)
        if hasattr(dist.distribution, "probs"):
            probs = dist.distribution.probs.detach().cpu().numpy()
            action = np.argmax(probs, axis=1).astype(int)
        else:  # pragma: no cover
            action, _ = model.predict(matrix, deterministic=False)
            probs = np.zeros((matrix.shape[0], model.action_space.n), dtype=np.float32)
            probs[np.arange(matrix.shape[0]), action] = 1.0

        out_df = pd.DataFrame(
            {
                "date": df["date"],
                "rl_hold_prob": probs[:, 0].astype(np.float32),
                "rl_long_prob": probs[:, 1].astype(np.float32) if probs.shape[1] > 1 else 0.0,
                "rl_short_prob": probs[:, 2].astype(np.float32) if probs.shape[1] > 2 else 0.0,
                "rl_action": action.astype(np.int8),
            }
        )
        out_path = out_exchange / f"{sanitized}-{settings.timeframe}.feather"
        out_df.to_feather(out_path)
        print(f"[rl_signals] wrote: {out_path} rows={len(out_df)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


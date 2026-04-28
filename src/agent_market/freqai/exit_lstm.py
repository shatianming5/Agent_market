"""LSTM-based exit signal predictor.

Input:  sliding window of last `window_size` OHLCV candles (normalized)
Output: P(reversal) — probability that the next `horizon` candles will return < -threshold

Architecture:
  LayerNorm → LSTM(hidden=128, layers=2) → Linear(128→1) → sigmoid
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn


@dataclass
class LSTMExitConfig:
    window_size: int = 24
    horizon: int = 4           # predict reversal in next `horizon` candles
    threshold: float = 0.008   # reversal = future_return < -threshold
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    n_channels: int = 5        # OHLCV channels


class LSTMExitModel(nn.Module):
    def __init__(self, cfg: LSTMExitConfig):
        super().__init__()
        self.cfg = cfg
        self.norm = nn.LayerNorm(cfg.n_channels)
        self.lstm = nn.LSTM(
            input_size=cfg.n_channels,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, window_size, n_channels) — normalised OHLCV"""
        x = self.norm(x)
        out, _ = self.lstm(x)
        last = out[:, -1, :]        # (batch, hidden)
        return self.head(last).squeeze(-1)  # (batch,) logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))

    def save(self, path: Path, cfg: LSTMExitConfig) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.state_dict(), "cfg": vars(cfg)}, path)

    @classmethod
    def load(cls, path: Path) -> "LSTMExitModel":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        cfg = LSTMExitConfig(**payload["cfg"])
        model = cls(cfg)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model

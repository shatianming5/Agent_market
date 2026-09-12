from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

try:
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    import gymnasium as gym  # type: ignore
    _HAS_SB3 = True
except ImportError:  # pragma: no cover
    _HAS_SB3 = False
    BaseFeaturesExtractor = object  # type: ignore
    gym = None  # type: ignore


if _HAS_SB3:
    class CandleCNNExtractor(BaseFeaturesExtractor):
        """1D CNN over a sliding OHLCV window, concatenated with engineered indicator features.

        Observation layout expected from TradingEnv (window_size > 0):
            obs[:indicator_dim]  — indicator features + [position, unrealized_pnl]
            obs[indicator_dim:]  — OHLCV window flattened, shape (window_size * n_channels,)
                                   order per candle: open, high, low, close, volume (normalised)

        The CNN processes the OHLCV window as a (n_channels, window_size) signal and
        produces a 64-d embedding; this is concatenated with the indicator vector.
        """

        def __init__(
            self,
            observation_space: gym.spaces.Box,
            indicator_dim: int,
            window_size: int = 24,
            n_channels: int = 5,
        ) -> None:
            cnn_out_dim = 64
            features_dim = indicator_dim + cnn_out_dim
            super().__init__(observation_space, features_dim=features_dim)
            self.indicator_dim = int(indicator_dim)
            self.window_size = int(window_size)
            self.n_channels = int(n_channels)
            self.cnn = nn.Sequential(
                nn.Conv1d(n_channels, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )

        def forward(self, observations: torch.Tensor) -> torch.Tensor:
            indicators = observations[:, : self.indicator_dim]
            ohlcv_flat = observations[:, self.indicator_dim :]
            batch = ohlcv_flat.shape[0]
            # (batch, window*channels) → (batch, channels, window) for Conv1d
            ohlcv = ohlcv_flat.view(batch, self.window_size, self.n_channels).permute(0, 2, 1)
            cnn_out = self.cnn(ohlcv).squeeze(-1)  # (batch, 64)
            return torch.cat([indicators, cnn_out], dim=1)

else:  # pragma: no cover
    class CandleCNNExtractor:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("stable-baselines3 and gymnasium are required for CandleCNNExtractor")

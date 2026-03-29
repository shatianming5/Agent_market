from __future__ import annotations
import pickle
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
from agent_market.freqai.model.base import BaseModelAdapter, TrainResult

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _rmse(preds: np.ndarray, targets: np.ndarray) -> float:
    preds = preds.reshape(-1).astype(np.float64, copy=False)
    targets = targets.reshape(-1).astype(np.float64, copy=False)
    return float(np.sqrt(np.mean((preds - targets) ** 2)))


def _mae(preds: np.ndarray, targets: np.ndarray) -> float:
    preds = preds.reshape(-1).astype(np.float64, copy=False)
    targets = targets.reshape(-1).astype(np.float64, copy=False)
    return float(np.mean(np.abs(preds - targets)))


class _ResidualBlock(nn.Module):
    def __init__(self, dim: int, hidden_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden_dim = max(dim * hidden_mult, dim)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class _Regressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        model_dim: int = 64,
        n_blocks: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        blocks = []
        for _ in range(int(n_blocks)):
            blocks.append(_ResidualBlock(model_dim, hidden_mult=4, dropout=dropout))
        self.blocks = nn.Sequential(*blocks)
        self.out = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        h = self.blocks(h)
        return self.out(h).squeeze(-1)


class AutoDL_v2(BaseModelAdapter):
    registry_name = "auto_dl_v2"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        use_cuda = bool(config.get("use_cuda", False))
        self.device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
        self.model: Optional[nn.Module] = None
        self.scaler: Optional[Dict[str, np.ndarray]] = None
        self._model_cfg: Optional[Dict[str, Any]] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: Optional[np.ndarray] = None,
        y_valid: Optional[np.ndarray] = None,
    ) -> TrainResult:
        seed = int(self.config.get("seed", 42))
        torch.manual_seed(seed)
        np.random.seed(seed)

        if X_train is None or y_train is None:
            raise ValueError("X_train / y_train must not be None")
        if X_train.ndim != 2:
            raise ValueError(f"X_train must be 2D (N, F), got shape {getattr(X_train, 'shape', None)}")
        if y_train.ndim != 1:
            y_train = y_train.reshape(-1)

        X_train = X_train.astype(np.float32, copy=False)
        y_train = y_train.astype(np.float32, copy=False)

        input_dim = int(X_train.shape[1])

        mean = X_train.mean(axis=0, dtype=np.float64).astype(np.float32)
        std = X_train.std(axis=0, dtype=np.float64).astype(np.float32)
        std = np.where(std < 1e-6, np.float32(1.0), std)

        self.scaler = {"mean": mean, "std": std}

        Xtr = (X_train - mean) / std

        has_valid = (
            X_valid is not None
            and y_valid is not None
            and getattr(X_valid, "size", 0) != 0
            and getattr(y_valid, "size", 0) != 0
        )
        if has_valid:
            X_valid = X_valid.astype(np.float32, copy=False)
            y_valid = y_valid.astype(np.float32, copy=False).reshape(-1)
            Xva = (X_valid - mean) / std
        else:
            Xva = None

        model_dim = int(self.config.get("model_dim", 64))
        n_blocks = int(self.config.get("n_blocks", 3))
        dropout = float(self.config.get("dropout", 0.1))

        self._model_cfg = {
            "input_dim": input_dim,
            "model_dim": model_dim,
            "n_blocks": n_blocks,
            "dropout": dropout,
        }

        model = _Regressor(input_dim=input_dim, model_dim=model_dim, n_blocks=n_blocks, dropout=dropout).to(self.device)

        lr = float(self.config.get("learning_rate", 2e-3))
        weight_decay = float(self.config.get("weight_decay", 1e-2))
        batch_size = int(self.config.get("batch_size", 128))
        epochs = int(self.config.get("epochs", 80))
        patience = int(self.config.get("patience", 12))
        clip_grad = float(self.config.get("clip_grad_norm", 1.0))

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.SmoothL1Loss(beta=float(self.config.get("huber_beta", 0.05)))

        train_loader = self._build_loader(Xtr, y_train, batch_size=batch_size, shuffle=True)
        valid_loader = self._build_loader(Xva, y_valid, batch_size=batch_size, shuffle=False) if has_valid else None

        best_state = None
        best_metric = float("inf")
        bad_epochs = 0

        for _ in range(epochs):
            model.train()
            for xb, yb in train_loader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                preds = model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                if clip_grad > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
                optimizer.step()

            if valid_loader is not None:
                val_rmse = self._evaluate_rmse(model, valid_loader)
                metric = val_rmse
            else:
                metric = self._evaluate_rmse(model, train_loader)

            if metric + 1e-10 < best_metric:
                best_metric = metric
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        self.model = model

        metrics: Dict[str, float] = {}
        metrics["rmse_train"] = self._evaluate_rmse(model, train_loader)
        metrics["mae_train"] = self._evaluate_mae(model, train_loader)
        if valid_loader is not None:
            metrics["rmse_valid"] = self._evaluate_rmse(model, valid_loader)
            metrics["mae_valid"] = self._evaluate_mae(model, valid_loader)

        model_dir = Path(self.config.get("model_dir", "artifacts/models/auto_dl_v2"))
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "auto_dl_v2.pt"
        self.save(model_path)

        return TrainResult(model_path=model_path, metrics=metrics)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model not trained/loaded")
        if X is None:
            raise ValueError("X must not be None")
        if X.ndim != 2:
            raise ValueError(f"X must be 2D (N, F), got shape {getattr(X, 'shape', None)}")

        X = X.astype(np.float32, copy=False)
        mean = self.scaler["mean"]
        std = self.scaler["std"]
        Xn = (X - mean) / std

        self.model.eval()
        with torch.no_grad():
            xb = torch.from_numpy(Xn).to(self.device)
            preds = self.model(xb).detach().cpu().numpy().reshape(-1)

        if preds.shape[0] != X.shape[0]:
            preds = preds[: X.shape[0]]
        return preds

    def save(self, path: Path) -> None:
        if self.model is None or self.scaler is None or self._model_cfg is None:
            raise RuntimeError("Nothing to save (train/load first)")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "state_dict": self.model.state_dict(),
            "model_cfg": self._model_cfg,
            "scaler": self.scaler,
            "config": dict(self.config),
        }
        torch.save(checkpoint, path)

    def load(self, path: Path) -> None:
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device)

        model_cfg = checkpoint.get("model_cfg")
        if not isinstance(model_cfg, dict):
            raise ValueError("Invalid checkpoint: missing model_cfg")

        self._model_cfg = dict(model_cfg)
        input_dim = int(self._model_cfg["input_dim"])
        model_dim = int(self._model_cfg.get("model_dim", 64))
        n_blocks = int(self._model_cfg.get("n_blocks", 3))
        dropout = float(self._model_cfg.get("dropout", 0.1))

        model = _Regressor(input_dim=input_dim, model_dim=model_dim, n_blocks=n_blocks, dropout=dropout).to(self.device)
        model.load_state_dict(checkpoint["state_dict"])
        self.model = model

        scaler = checkpoint.get("scaler")
        if not isinstance(scaler, dict) or "mean" not in scaler or "std" not in scaler:
            raise ValueError("Invalid checkpoint: missing scaler")
        self.scaler = {"mean": np.asarray(scaler["mean"], dtype=np.float32), "std": np.asarray(scaler["std"], dtype=np.float32)}

    def _build_loader(
        self,
        X: Optional[np.ndarray],
        y: Optional[np.ndarray],
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        if X is None or y is None:
            raise ValueError("X/y must not be None for DataLoader")
        dataset = TensorDataset(
            torch.from_numpy(X.astype(np.float32, copy=False)),
            torch.from_numpy(y.astype(np.float32, copy=False).reshape(-1)),
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)

    def _predict_loader(self, model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        model.eval()
        preds_list = []
        targets_list = []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(self.device, non_blocking=True)
                preds = model(xb).detach().cpu().numpy().reshape(-1)
                preds_list.append(preds)
                targets_list.append(yb.numpy().reshape(-1))
        preds = np.concatenate(preds_list, axis=0) if preds_list else np.zeros((0,), dtype=np.float32)
        targets = np.concatenate(targets_list, axis=0) if targets_list else np.zeros((0,), dtype=np.float32)
        return preds, targets

    def _evaluate_rmse(self, model: nn.Module, loader: DataLoader) -> float:
        preds, targets = self._predict_loader(model, loader)
        return _rmse(preds, targets)

    def _evaluate_mae(self, model: nn.Module, loader: DataLoader) -> float:
        preds, targets = self._predict_loader(model, loader)
        return _mae(preds, targets)
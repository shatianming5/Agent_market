from __future__ import annotations
import pickle
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
from agent_market.freqai.model.base import BaseModelAdapter, TrainResult

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class _MLPRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        dropout: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d_in = input_dim
        for d_out in hidden_dims:
            layers.append(nn.Linear(d_in, d_out))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d_in = d_out
        layers.append(nn.Linear(d_in, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class AutoDL_v2(BaseModelAdapter):
    registry_name = "auto_dl_v2"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model: Optional[nn.Module] = None
        self.device: Optional[torch.device] = None

        self.x_mean: Optional[np.ndarray] = None
        self.x_std: Optional[np.ndarray] = None

        self._model_hparams: Dict[str, Any] = {}
        self._is_fitted: bool = False

    def _get_device(self) -> torch.device:
        if self.device is not None:
            return self.device
        cfg_device = str(self.config.get("device", "auto")).lower()
        if cfg_device in ("cuda", "gpu"):
            d = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif cfg_device in ("cpu",):
            d = torch.device("cpu")
        else:
            d = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = d
        return d

    def _seed_everything(self) -> None:
        seed = self.config.get("seed", None)
        if seed is None:
            return
        try:
            seed_int = int(seed)
        except Exception:
            return
        np.random.seed(seed_int)
        torch.manual_seed(seed_int)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_int)

    def _sanitize_inputs(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> tuple[np.ndarray, Optional[np.ndarray]]:
        Xn = np.asarray(X, dtype=np.float32)
        Xn = np.nan_to_num(Xn, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        yn = None
        if y is not None:
            yn = np.asarray(y, dtype=np.float32).reshape(-1)
            yn = np.nan_to_num(yn, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        return Xn, yn

    def _fit_scaler(self, X_train: np.ndarray) -> None:
        mean = X_train.mean(axis=0, keepdims=False).astype(np.float32)
        std = X_train.std(axis=0, keepdims=False).astype(np.float32)
        std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
        self.x_mean = mean
        self.x_std = std

    def _transform_X(self, X: np.ndarray) -> np.ndarray:
        if self.x_mean is None or self.x_std is None:
            raise RuntimeError("Scaler not fitted. Call fit() or load() first.")
        return ((X - self.x_mean) / self.x_std).astype(np.float32, copy=False)

    def _build_model(self, input_dim: int) -> nn.Module:
        hidden_dims = self.config.get("hidden_dims", [128, 64, 32])
        if isinstance(hidden_dims, (tuple, list)):
            hidden_dims_list = [int(x) for x in hidden_dims]
        else:
            hidden_dims_list = [128, 64, 32]

        dropout = float(self.config.get("dropout", 0.10))
        model = _MLPRegressor(input_dim=input_dim, hidden_dims=hidden_dims_list, dropout=dropout)

        self._model_hparams = {
            "input_dim": int(input_dim),
            "hidden_dims": hidden_dims_list,
            "dropout": float(dropout),
        }
        return model

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: Optional[np.ndarray] = None,
        y_valid: Optional[np.ndarray] = None,
    ) -> TrainResult:
        self._seed_everything()

        model_dir_raw = self.config.get("model_dir", None)
        if not model_dir_raw:
            raise ValueError("config['model_dir'] is required")
        model_dir = Path(model_dir_raw)
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "auto_dl_v2.pkl"

        X_train, y_train = self._sanitize_inputs(X_train, y_train)
        if X_train.ndim != 2:
            raise ValueError(f"X_train must be 2D, got shape={X_train.shape}")
        if y_train is None:
            raise ValueError("y_train is required")
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError(f"X_train and y_train length mismatch: {X_train.shape[0]} vs {y_train.shape[0]}")

        if X_valid is not None and y_valid is not None:
            X_valid, y_valid = self._sanitize_inputs(X_valid, y_valid)
            if X_valid.ndim != 2:
                raise ValueError(f"X_valid must be 2D, got shape={X_valid.shape}")
            if y_valid is None:
                raise ValueError("y_valid is None")
            if X_valid.shape[0] != y_valid.shape[0]:
                raise ValueError(f"X_valid and y_valid length mismatch: {X_valid.shape[0]} vs {y_valid.shape[0]}")
        else:
            X_valid, y_valid = None, None

        self._fit_scaler(X_train)
        Xtr = self._transform_X(X_train)

        Xva = None
        yva = None
        if X_valid is not None and y_valid is not None:
            Xva = self._transform_X(X_valid)
            yva = y_valid

        device = self._get_device()
        self.model = self._build_model(input_dim=Xtr.shape[1]).to(device)

        lr = float(self.config.get("lr", 3e-4))
        weight_decay = float(self.config.get("weight_decay", 1e-3))
        batch_size = int(self.config.get("batch_size", 128))
        max_epochs = int(self.config.get("max_epochs", 200))
        patience = int(self.config.get("patience", 20))
        grad_clip = float(self.config.get("grad_clip", 1.0))

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.SmoothL1Loss(beta=float(self.config.get("huber_beta", 0.5)))

        train_ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(y_train))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

        valid_loader = None
        if Xva is not None and yva is not None:
            valid_ds = TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva))
            valid_loader = DataLoader(valid_ds, batch_size=max(256, batch_size), shuffle=False, drop_last=False)

        best_val = float("inf")
        best_state: Optional[dict[str, torch.Tensor]] = None
        best_epoch = -1
        no_improve = 0

        last_train_loss = float("inf")
        last_val_loss = float("inf")

        for epoch in range(max_epochs):
            self.model.train()
            train_losses = []

            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                optimizer.zero_grad(set_to_none=True)
                pred = self.model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()

                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip)

                optimizer.step()
                train_losses.append(float(loss.detach().cpu().item()))

            last_train_loss = float(np.mean(train_losses)) if train_losses else float("inf")

            if valid_loader is None:
                continue

            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in valid_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    pred = self.model(xb)
                    loss = loss_fn(pred, yb)
                    val_losses.append(float(loss.detach().cpu().item()))
            last_val_loss = float(np.mean(val_losses)) if val_losses else float("inf")

            if last_val_loss + 1e-9 < best_val:
                best_val = last_val_loss
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.model.eval()
        self._is_fitted = True

        self.save(model_path)

        metrics: Dict[str, float] = {
            "train_loss": float(last_train_loss),
        }
        if valid_loader is not None:
            metrics["val_loss"] = float(best_val if best_state is not None else last_val_loss)
            metrics["best_epoch"] = float(best_epoch)

        return TrainResult(model_path=model_path, metrics=metrics)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None or not self._is_fitted:
            raise RuntimeError("Model not fitted/loaded. Call fit() or load() first.")

        X, _ = self._sanitize_inputs(X, None)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape={X.shape}")

        Xt = self._transform_X(X)
        device = self._get_device()

        self.model.eval()
        with torch.no_grad():
            xb = torch.from_numpy(Xt).to(device)
            pred = self.model(xb).detach().cpu().numpy().astype(np.float32, copy=False)

        pred = pred.reshape(-1)
        if pred.shape[0] != X.shape[0]:
            raise RuntimeError("Prediction shape mismatch.")
        return pred

    def save(self, path: Path) -> None:
        if self.model is None or not self._is_fitted:
            raise RuntimeError("Nothing to save. Call fit() first.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "registry_name": self.registry_name,
            "model_hparams": dict(self._model_hparams),
            "state_dict": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
            "x_mean": self.x_mean,
            "x_std": self.x_std,
            "config": dict(self.config),
        }

        with path.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: Path) -> None:
        path = Path(path)
        if path.is_dir():
            path = path / "auto_dl_v2.pkl"

        with path.open("rb") as f:
            payload = pickle.load(f)

        model_hparams = payload.get("model_hparams", None)
        if not isinstance(model_hparams, dict):
            raise ValueError("Invalid model file: missing model_hparams")

        self.x_mean = payload.get("x_mean", None)
        self.x_std = payload.get("x_std", None)

        self.device = None
        device = self._get_device()

        input_dim = int(model_hparams["input_dim"])
        hidden_dims = list(model_hparams["hidden_dims"])
        dropout = float(model_hparams["dropout"])

        self._model_hparams = {
            "input_dim": input_dim,
            "hidden_dims": hidden_dims,
            "dropout": dropout,
        }

        self.model = _MLPRegressor(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout).to(device)
        state_dict = payload.get("state_dict", None)
        if not isinstance(state_dict, dict):
            raise ValueError("Invalid model file: missing state_dict")
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self._is_fitted = True
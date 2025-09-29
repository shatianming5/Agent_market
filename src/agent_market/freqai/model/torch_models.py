from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from agent_market.freqai.model.base import BaseModelAdapter, ModelRegistry, TrainResult


def _rmse(preds: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((preds - target) ** 2)))


class FeedForwardNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Iterable[int], dropout: float = 0.0):
        super().__init__()
        dims = [input_dim, *hidden_dims]
        layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class PyTorchMLPAdapter(BaseModelAdapter):
    registry_name = 'pytorch_mlp'

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model: Optional[nn.Module] = None
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.get('use_cuda', False) else 'cpu')

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: Optional[np.ndarray] = None,
        y_valid: Optional[np.ndarray] = None,
    ) -> TrainResult:
        torch.manual_seed(int(self.config.get('seed', 42)))
        input_dim = X_train.shape[1]
        hidden_dims = self.config.get('hidden_dims', [64, 32])
        dropout = float(self.config.get('dropout', 0.0))
        epochs = int(self.config.get('epochs', 20))
        batch_size = int(self.config.get('batch_size', 64))
        lr = float(self.config.get('learning_rate', 1e-3))

        model = FeedForwardNet(input_dim, hidden_dims, dropout).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        train_loader = self._build_loader(X_train, y_train, batch_size, shuffle=True)
        valid_loader = None
        if X_valid is not None and X_valid.size:
            valid_loader = self._build_loader(X_valid, y_valid, batch_size, shuffle=False)

        model.train()
        for _ in range(epochs):
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                preds = model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()

        self.model = model
        metrics = {'rmse_train': self._evaluate(train_loader)}
        if valid_loader is not None:
            metrics['rmse_valid'] = self._evaluate(valid_loader)

        model_dir = Path(self.config.get('model_dir', 'artifacts/models/pytorch'))
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / 'pytorch_mlp.pt'
        torch.save({'state_dict': model.state_dict(), 'config': self.config, 'input_dim': input_dim}, model_path)
        return TrainResult(model_path=model_path, metrics=metrics)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError('PyTorch model not trained')
        self.model.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(X.astype(np.float32)).to(self.device)
            preds = self.model(tensor).cpu().numpy()
        return preds

    def save(self, path: Path) -> None:
        if self.model is None:
            raise RuntimeError('PyTorch model not trained')
        torch.save({'state_dict': self.model.state_dict(), 'config': self.config}, path)

    def load(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        config = dict(self.config)
        input_dim = checkpoint.get('input_dim')
        hidden_dims = config.get('hidden_dims', checkpoint.get('config', {}).get('hidden_dims', [64, 32]))
        dropout = float(config.get('dropout', checkpoint.get('config', {}).get('dropout', 0.0)))
        model = FeedForwardNet(input_dim, hidden_dims, dropout).to(self.device)
        model.load_state_dict(checkpoint['state_dict'])
        self.model = model

    def _build_loader(self, X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
        dataset = TensorDataset(
            torch.from_numpy(X.astype(np.float32)),
            torch.from_numpy(y.astype(np.float32)),
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)

    def _evaluate(self, loader: DataLoader) -> float:
        assert self.model is not None
        self.model.eval()
        preds_list = []
        targets_list = []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(self.device)
                preds = self.model(xb).cpu().numpy()
                preds_list.append(preds)
                targets_list.append(yb.numpy())
        preds = np.concatenate(preds_list)
        targets = np.concatenate(targets_list)
        return _rmse(preds, targets)


ModelRegistry.register(PyTorchMLPAdapter.registry_name, PyTorchMLPAdapter)

__all__ = ['PyTorchMLPAdapter']

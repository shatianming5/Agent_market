from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class TrainResult:
    model_path: Path
    metrics: Dict[str, float]


class BaseModelAdapter(ABC):
    registry_name: str = "base"

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: Optional[np.ndarray] = None,
        y_valid: Optional[np.ndarray] = None,
    ) -> TrainResult:
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: Path) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, path: Path) -> None:
        raise NotImplementedError


class ModelRegistry:
    _registry: Dict[str, type[BaseModelAdapter]] = {}

    @classmethod
    def register(cls, name: str, adapter_cls: type[BaseModelAdapter]) -> None:
        cls._registry[name] = adapter_cls

    @classmethod
    def create(cls, name: str, config: Dict[str, Any]) -> BaseModelAdapter:
        if name not in cls._registry:
            raise KeyError(f"model adapter '{name}' not registered")
        return cls._registry[name](config)


__all__ = ["BaseModelAdapter", "ModelRegistry", "TrainResult"]

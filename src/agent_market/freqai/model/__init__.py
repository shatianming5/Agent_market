from . import gradient_boosting  # noqa: F401

try:
    # 仅在可用时才导入 torch 模型，避免纯读取场景强依赖 torch
    from . import torch_models  # noqa: F401
except Exception:  # pragma: no cover - 环境无 torch 时跳过
    torch_models = None  # type: ignore

try:
    from . import stacked  # noqa: F401
except Exception:  # pragma: no cover - 需要 scikit-learn + lightgbm
    stacked = None  # type: ignore

try:
    from . import ridge as ridge_model  # noqa: F401
except Exception:  # pragma: no cover - 需要 scikit-learn
    ridge_model = None  # type: ignore

__all__ = [
    'gradient_boosting',
    'torch_models',
    'stacked',
]


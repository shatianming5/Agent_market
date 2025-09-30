from . import gradient_boosting  # noqa: F401

try:
    # 仅在可用时才导入 torch 模型，避免纯读取场景强依赖 torch
    from . import torch_models  # noqa: F401
except Exception:  # pragma: no cover - 环境无 torch 时跳过
    torch_models = None  # type: ignore

__all__ = [
    'gradient_boosting',
    'torch_models',
]


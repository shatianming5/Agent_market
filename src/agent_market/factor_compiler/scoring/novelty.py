from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


def corr_to_library_max(factor: pd.Series, library: Dict[str, pd.Series]) -> Optional[float]:
    """
    Compute the maximum absolute correlation to a factor library (best-effort).
    """

    if factor is None or factor.empty:
        return None
    best: Optional[float] = None
    for _, other in (library or {}).items():
        if other is None or getattr(other, "empty", True):
            continue
        aligned = pd.concat([factor.rename("x"), other.rename("y")], axis=1).dropna()
        if aligned.shape[0] < 3:
            continue
        value = float(abs(aligned["x"].corr(aligned["y"])))
        if not np.isfinite(value):
            continue
        best = value if best is None else max(best, value)
    return float(best) if best is not None else None


def ast_similarity_max(expr_sha256: str, library_sha256: Iterable[str]) -> float:
    """
    Minimal AST similarity proxy:
      - 1.0 if sha256 already exists in the library, else 0.0
    """

    sha = str(expr_sha256 or "").strip().lower()
    library = {str(x).strip().lower() for x in library_sha256 or [] if str(x).strip()}
    return 1.0 if sha and sha in library else 0.0


__all__ = ["ast_similarity_max", "corr_to_library_max"]

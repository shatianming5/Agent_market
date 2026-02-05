from __future__ import annotations

from typing import Any, Dict, Sequence


def capture_bybit_ws(*, symbols: Sequence[str], channels: Sequence[str], duration_sec: float, out_dir: str) -> Dict[str, Any]:
    raise NotImplementedError("Bybit ws capture adapter is not implemented yet (use KuCoin adapter).")


__all__ = ["capture_bybit_ws"]


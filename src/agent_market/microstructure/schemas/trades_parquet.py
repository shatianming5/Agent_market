from __future__ import annotations

from typing import Any, Dict, List


TRADES_COLUMNS: List[str] = ["ts", "symbol", "side", "price", "size"]


def validate_trades_df(df: Any) -> Dict[str, Any]:
    cols = list(getattr(df, "columns", []))
    missing = [c for c in TRADES_COLUMNS if c not in cols]
    return {"ok": not missing, "missing": missing, "columns": cols}


__all__ = ["TRADES_COLUMNS", "validate_trades_df"]

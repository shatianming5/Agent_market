from __future__ import annotations

from typing import Any, Dict, List


LOB_STATE_COLUMNS: List[str] = [
    "ts",
    "symbol",
    "event",
    "bid1",
    "ask1",
    "mid",
    "spread",
    "rel_spread",
    "bid_sz",
    "ask_sz",
]


def validate_lob_state_df(df: Any) -> Dict[str, Any]:
    """
    Validate that a dataframe looks like `lob_state.parquet` (best-effort).
    """

    cols = list(getattr(df, "columns", []))
    missing = [c for c in LOB_STATE_COLUMNS if c not in cols]
    return {"ok": not missing, "missing": missing, "columns": cols}


__all__ = ["LOB_STATE_COLUMNS", "validate_lob_state_df"]

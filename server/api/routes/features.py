from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter

from ..errors import error
from ...runtime import ROOT

router = APIRouter()


@router.get("/features/top")
def features_top(file: str = "user_data/freqai_features.json", limit: int = 20):
    """Return top-N features by score/correlation/mutual_info."""
    path = Path(file)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    if not path.exists():
        return error("FEATURE_FILE_NOT_FOUND", f"Feature file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return error("PARSE_ERROR", f"Failed to parse: {exc}")
    feats = payload.get("features") or []
    rows = []
    for it in feats:
        name = it.get("name")
        if not name:
            continue
        score = it.get("score")
        if score is None:
            score = it.get("correlation")
        if score is None:
            score = it.get("mutual_info")
        try:
            val = float(score) if score is not None else None
        except Exception:
            val = None
        rows.append(
            {
                "name": name,
                "type": it.get("type"),
                "period": it.get("period"),
                "score": val,
                "correlation": it.get("correlation"),
                "mutual_info": it.get("mutual_info"),
                "description": it.get("description"),
            }
        )
    rows.sort(
        key=lambda r: abs(r["score"]) if isinstance(r["score"], (int, float)) else 0.0,
        reverse=True,
    )
    return {
        "file": str(path),
        "total": len(rows),
        "items": rows[: max(1, int(limit))],
    }


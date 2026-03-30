from __future__ import annotations
import logging

import json
import os

from fastapi import APIRouter, Body

from ..errors import error
from ..validators import validate_timeframe
from ...runtime import SETTINGS_PATH

router = APIRouter()


def _load_settings() -> dict:
    try:
        if SETTINGS_PATH.exists():
            return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except Exception as _exc:
        logging.getLogger(__name__).debug("Suppressed: %s", _exc)
    return {
        "llm_base_url": os.environ.get("LLM_BASE_URL"),
        "llm_model": os.environ.get("LLM_MODEL"),
        "default_timeframe": "4h",
        "note": "Server settings for Agent Market",
    }


def _save_settings(obj: dict) -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8"
    )


@router.get("/settings")
def get_settings():
    return _load_settings()


@router.post("/settings")
def post_settings(payload: dict = Body(...)):
    if not isinstance(payload, dict):
        return error("INVALID_BODY", "settings body must be a JSON object")
    allowed = {"llm_base_url", "llm_model", "default_timeframe", "note"}
    updates = {k: v for k, v in payload.items() if k in allowed}
    if not updates:
        return error("NO_ALLOWED_KEYS", f"allowed keys: {sorted(allowed)}")
    cur = _load_settings()
    cur.update(updates)
    if cur.get("default_timeframe") and not validate_timeframe(
        str(cur["default_timeframe"])
    ):
        return error(
            "INVALID_TIMEFRAME",
            f"Invalid default_timeframe: {cur['default_timeframe']}",
        )
    _save_settings(cur)
    return {"status": "ok", "settings": cur}


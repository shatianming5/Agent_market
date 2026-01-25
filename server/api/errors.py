from __future__ import annotations

from typing import Any


def error(code: str, message: str, **extra: Any) -> dict:
    payload = {"status": "error", "code": code, "message": message}
    if extra:
        payload.update(extra)
    return payload


__all__ = ["error"]


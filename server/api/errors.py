from __future__ import annotations

from typing import Any

from fastapi.responses import JSONResponse


def error_dict(code: str, message: str, **extra: Any) -> dict[str, Any]:
    """Return a plain error dict (safe for json.dumps in SSE/WS contexts)."""
    payload: dict[str, Any] = {"status": "error", "code": code, "message": message}
    if extra:
        payload.update(extra)
    return payload


def error(code: str, message: str, *, status_code: int = 400, **extra: Any) -> JSONResponse:
    """Return a structured error response with proper HTTP status code."""
    return JSONResponse(content=error_dict(code, message, **extra), status_code=status_code)


def not_found(code: str, message: str, **extra: Any) -> JSONResponse:
    return error(code, message, status_code=404, **extra)


def bad_request(code: str, message: str, **extra: Any) -> JSONResponse:
    return error(code, message, status_code=400, **extra)


__all__ = ["error", "error_dict", "not_found", "bad_request"]


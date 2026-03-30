"""API Key authentication for protected routes.

The API key is read from the environment variable ``AGENT_MARKET_API_KEY``.
If the variable is empty or unset, authentication is **disabled** (open access)
so that local development remains frictionless.

Usage in routes::

    from server.api.auth import require_api_key

    router = APIRouter(dependencies=[Depends(require_api_key)])
"""
from __future__ import annotations

import os
import secrets
from typing import Optional

from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader

_header_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)


def _get_configured_key() -> Optional[str]:
    raw = os.environ.get("AGENT_MARKET_API_KEY", "").strip()
    return raw if raw else None


def require_api_key(
    api_key: Optional[str] = Security(_header_scheme),
) -> None:
    """FastAPI dependency that enforces API key when configured."""
    expected = _get_configured_key()
    if expected is None:
        return  # auth disabled — open access
    if api_key is None:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")
    if not secrets.compare_digest(api_key, expected):
        raise HTTPException(status_code=403, detail="Invalid API key")

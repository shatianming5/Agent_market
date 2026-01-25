from __future__ import annotations

# Backward-compatible entrypoint for `uvicorn server.main:app`.
from .app import app

__all__ = ["app"]


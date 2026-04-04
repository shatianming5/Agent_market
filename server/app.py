from __future__ import annotations

import os

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request
from starlette.responses import JSONResponse

from .api.auth import require_api_key
from .api.routes.features import router as features_router
from .api.routes.flow import router as flow_router
from .api.routes.jobs import router as jobs_router
from .api.routes.results import router as results_router
from .api.routes.root import router as root_router
from .api.routes.run import router as run_router
from .api.routes.settings import router as settings_router
from .api.routes.strategy_miner import router as strategy_miner_router
from .runtime import ROOT

_PROTECTED_PREFIXES = ("/run", "/flow/run")


def _cors_origins() -> list[str]:
    raw = os.environ.get("AGENT_MARKET_CORS_ORIGINS", "").strip()
    if raw:
        return [o.strip() for o in raw.split(",") if o.strip()]
    return ["http://localhost:3000", "http://127.0.0.1:3000",
            "http://localhost:8000", "http://127.0.0.1:8000"]


def create_app() -> FastAPI:
    app = FastAPI(title="Agent Market Server")

    # CORS — restricted to configured origins (default: localhost only)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API key auth middleware — inside factory so every instance is protected
    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        # Skip auth for OPTIONS (CORS preflight) and non-protected paths
        if request.method == "OPTIONS":
            return await call_next(request)
        # Read API key at request time (not import time) so env changes take effect
        api_key = os.environ.get("AGENT_MARKET_API_KEY", "")
        if api_key and any(request.url.path.startswith(p) for p in _PROTECTED_PREFIXES):
            key = request.headers.get("X-API-Key", "")
            if key != api_key:
                return JSONResponse({"error": "unauthorized"}, status_code=401)
        return await call_next(request)

    # Static web UI (optional)
    try:
        app.mount("/web", StaticFiles(directory=str(ROOT / "web")), name="web")
    except (FileNotFoundError, RuntimeError):
        pass  # web/ dir may not exist in headless/workspace mode

    # Public routes (no auth required)
    app.include_router(root_router)

    # Protected routes (require API key when AGENT_MARKET_API_KEY is set)
    _auth = [Depends(require_api_key)]
    app.include_router(settings_router, dependencies=_auth)
    app.include_router(jobs_router, dependencies=_auth)
    app.include_router(features_router, dependencies=_auth)
    app.include_router(results_router, dependencies=_auth)
    app.include_router(run_router, dependencies=_auth)
    app.include_router(flow_router, dependencies=_auth)
    app.include_router(strategy_miner_router, dependencies=_auth)
    return app


app = create_app()

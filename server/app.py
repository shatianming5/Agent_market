from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request
from starlette.responses import JSONResponse

from .api.routes.features import router as features_router
from .api.routes.flow import router as flow_router
from .api.routes.jobs import router as jobs_router
from .api.routes.results import router as results_router
from .api.routes.root import router as root_router
from .api.routes.run import router as run_router
from .api.routes.settings import router as settings_router
from .runtime import ROOT

_PROTECTED_PREFIXES = ("/run", "/flow/run")


def create_app() -> FastAPI:
    app = FastAPI(title="Agent Market Server")

    # CORS — restricted to configured origins (default: localhost only)
    allowed_origins = os.environ.get(
        "CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
    ).split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
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

    app.include_router(root_router)
    app.include_router(settings_router)
    app.include_router(jobs_router)
    app.include_router(features_router)
    app.include_router(results_router)
    app.include_router(run_router)
    app.include_router(flow_router)
    return app


app = create_app()

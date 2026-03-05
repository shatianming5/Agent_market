from __future__ import annotations

import os

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

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


def _cors_origins() -> list[str]:
    raw = os.environ.get("AGENT_MARKET_CORS_ORIGINS", "").strip()
    if raw:
        return [o.strip() for o in raw.split(",") if o.strip()]
    return ["http://localhost:3000", "http://127.0.0.1:3000",
            "http://localhost:8000", "http://127.0.0.1:8000"]


def create_app() -> FastAPI:
    app = FastAPI(title="Agent Market Server")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Serve static web UI (optional): http://host:8000/web/index.html
    try:
        app.mount("/web", StaticFiles(directory=str(ROOT / "web")), name="web")
    except Exception:
        pass

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

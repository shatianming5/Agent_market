from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .api.routes.features import router as features_router
from .api.routes.flow import router as flow_router
from .api.routes.jobs import router as jobs_router
from .api.routes.results import router as results_router
from .api.routes.root import router as root_router
from .api.routes.run import router as run_router
from .api.routes.settings import router as settings_router
from .runtime import ROOT


def create_app() -> FastAPI:
    app = FastAPI(title="Agent Market Server")
    import os
    allowed_origins = os.environ.get("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Serve static web UI (optional): http://host:8000/web/index.html
    try:
        app.mount("/web", StaticFiles(directory=str(ROOT / "web")), name="web")
    except Exception:
        pass

    app.include_router(root_router)
    app.include_router(settings_router)
    app.include_router(jobs_router)
    app.include_router(features_router)
    app.include_router(results_router)
    app.include_router(run_router)
    app.include_router(flow_router)
    return app


app = create_app()


# Simple API key auth middleware
import os as _os
_API_KEY = _os.environ.get("AGENT_MARKET_API_KEY", "")

@app.middleware("http")
async def auth_middleware(request, call_next):
    if _API_KEY and (request.url.path.startswith("/run") or request.url.path.startswith("/flow/run")):
        key = request.headers.get("X-API-Key", "")
        if key != _API_KEY:
            from starlette.responses import JSONResponse
            return JSONResponse({"error": "unauthorized"}, status_code=401)
    return await call_next(request)

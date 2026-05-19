"""FastAPI REST + WebSocket server for Factor Hub.

Run:
    python -m agent_market.factor_hub.server --host 0.0.0.0 --port 8765

Endpoints:
    GET   /api/health
    GET   /api/stats
    GET   /api/factors                (filters: status, category, origin, ic_gt, ic_lt, metric_name, limit)
    POST  /api/factors                body: {expression, name?, category?, origin?, ...}
    GET   /api/factors/{id}
    PUT   /api/factors/{id}/status    body: {status, reason?}
    GET   /api/factors/{id}/evaluations
    POST  /api/factors/{id}/evaluations  body: {eval_type, metric_name, metric_value, ...}
    GET   /api/factors/{id}/lineage
    GET   /api/origins                 origin distribution
    GET   /api/features                base-feature dependency table
    GET   /api/features/{name}         factors using a feature
    GET   /api/deployments
    POST  /api/deployments             body: {name, factor_ids, activate?, notes?}
    GET   /api/deployments/active
    PUT   /api/deployments/{id}/activate
    GET   /api/events                  filters: since, event_type, limit
    WS    /ws/events                   live event stream
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional

try:
    from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
except Exception as exc:  # noqa: BLE001
    raise SystemExit(
        "FastAPI is required for Factor Hub server. "
        "Install with: pip install fastapi uvicorn"
    ) from exc

from .client import Client
from . import db


# ============================================================
# App factory
# ============================================================

def create_app(db_path: Optional[str] = None) -> FastAPI:
    client = Client(db_path=db_path)
    client.init_db()

    app = FastAPI(title="Factor Hub", version="1.0.0",
                  description="Unified factor registry + evaluation + deployment service")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
    )
    app.state.client = client

    # --------------------------------------------------------
    # Pydantic request models
    # --------------------------------------------------------

    class ProposeBody(BaseModel):
        expression: str
        name: Optional[str] = None
        category: str = "misc"
        origin: str = "api"
        description: str = ""
        status: str = "candidate"
        created_by: str = "api"
        source_lib: str = ""
        metadata: Optional[Dict[str, Any]] = None

    class StatusBody(BaseModel):
        status: str
        reason: str = ""

    class EvaluationBody(BaseModel):
        eval_type: str
        metric_name: str
        metric_value: float
        period_start: str = ""
        period_end: str = ""
        n_samples: int = 0
        sign_agree: int = 0
        notes: str = ""
        metadata: Optional[Dict[str, Any]] = None

    class DeploymentBody(BaseModel):
        name: str
        factor_ids: List[int]
        activate: bool = False
        deployed_by: str = "api"
        notes: str = ""

    # --------------------------------------------------------
    # Basic endpoints
    # --------------------------------------------------------

    @app.get("/api/health")
    def health():
        return {"status": "ok", "db": str(client.path)}

    @app.get("/api/stats")
    def stats():
        return client.stats()

    # --------------------------------------------------------
    # Factors
    # --------------------------------------------------------

    @app.get("/api/factors")
    def list_factors(
        status: Optional[str] = "active",
        category: Optional[str] = None,
        origin: Optional[str] = None,
        ic_gt: Optional[float] = None,
        ic_lt: Optional[float] = None,
        metric_name: str = "oos_ic",
        limit: int = Query(100, ge=1, le=1000),
    ):
        return client.query(
            status=status, category=category, origin=origin,
            ic_gt=ic_gt, ic_lt=ic_lt, metric_name=metric_name, limit=limit,
        )

    @app.post("/api/factors")
    def create_factor(body: ProposeBody):
        fid = client.propose(
            expression=body.expression, name=body.name, category=body.category,
            origin=body.origin, description=body.description, status=body.status,
            created_by=body.created_by, source_lib=body.source_lib,
            metadata=body.metadata,
        )
        return {"id": fid}

    @app.get("/api/factors/{factor_id}")
    def get_factor(factor_id: int):
        f = client.get(factor_id)
        if not f:
            raise HTTPException(404, f"factor {factor_id} not found")
        return asdict(f)

    @app.put("/api/factors/{factor_id}/status")
    def set_status(factor_id: int, body: StatusBody):
        client.update_status(factor_id, body.status, reason=body.reason)
        return {"id": factor_id, "status": body.status}

    @app.get("/api/factors/{factor_id}/evaluations")
    def factor_evaluations(factor_id: int, metric_name: Optional[str] = None, limit: int = 100):
        return [asdict(e) for e in client.evaluations(factor_id, metric_name=metric_name, limit=limit)]

    @app.post("/api/factors/{factor_id}/evaluations")
    def add_factor_evaluation(factor_id: int, body: EvaluationBody):
        eid = client.add_evaluation(
            factor_id, body.eval_type, body.metric_name, body.metric_value,
            period_start=body.period_start, period_end=body.period_end,
            n_samples=body.n_samples, sign_agree=body.sign_agree,
            notes=body.notes, metadata=body.metadata,
        )
        return {"id": eid}

    @app.get("/api/factors/{factor_id}/lineage")
    def factor_lineage(factor_id: int):
        f = client.get(factor_id)
        if not f:
            raise HTTPException(404, f"factor {factor_id} not found")
        return {"factor_id": factor_id, "name": f.name, "features": client.lineage(factor_id)}

    # --------------------------------------------------------
    # Features / origins
    # --------------------------------------------------------

    @app.get("/api/origins")
    def origins():
        return client.origin_distribution()

    @app.get("/api/features")
    def features(top_n: int = 50):
        return client.feature_deps(top_n=top_n)

    @app.get("/api/features/{feature}")
    def feature_users(feature: str):
        return [asdict(f) for f in client.factors_by_feature(feature)]

    # --------------------------------------------------------
    # Deployments
    # --------------------------------------------------------

    @app.get("/api/deployments")
    def deployments(limit: int = 50):
        return [asdict(d) for d in client.deployments(limit=limit)]

    @app.post("/api/deployments")
    def create_deployment(body: DeploymentBody):
        did = client.deploy(
            body.name, body.factor_ids, activate=body.activate,
            deployed_by=body.deployed_by, notes=body.notes,
        )
        return {"id": did}

    @app.get("/api/deployments/active")
    def active_deployment(name: str = "production"):
        d = client.active_deployment(name=name)
        if not d:
            raise HTTPException(404, f"no active deployment for name={name}")
        return asdict(d)

    @app.put("/api/deployments/{deployment_id}/activate")
    def activate(deployment_id: int):
        client.activate(deployment_id)
        return {"id": deployment_id, "activated": True}

    # --------------------------------------------------------
    # Events
    # --------------------------------------------------------

    @app.get("/api/events")
    def events(since: int = 0, event_type: Optional[str] = None, limit: int = 100):
        return [asdict(e) for e in client.events(since_id=since, limit=limit, event_type=event_type)]

    @app.websocket("/ws/events")
    async def ws_events(websocket: WebSocket):
        await websocket.accept()
        try:
            event_type = websocket.query_params.get("event_type")
            with client.connect() as conn:
                row = conn.execute("SELECT MAX(id) AS mx FROM events").fetchone()
                since_id = int(row["mx"] or 0)
            while True:
                batch = client.events(since_id=since_id, limit=200, event_type=event_type)
                for e in batch:
                    since_id = max(since_id, int(e.id or 0))
                    await websocket.send_json(asdict(e))
                await asyncio.sleep(0.5)
        except WebSocketDisconnect:
            return

    return app


# ============================================================
# CLI runner
# ============================================================

def main() -> None:
    import argparse
    try:
        import uvicorn
    except Exception as exc:  # noqa: BLE001
        raise SystemExit("uvicorn required: pip install uvicorn") from exc

    p = argparse.ArgumentParser(prog="factor_hub.server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--db", default=None)
    p.add_argument("--reload", action="store_true")
    args = p.parse_args()

    if args.db:
        os.environ["FACTOR_HUB_DB"] = args.db

    if args.reload:
        os.environ.setdefault("FACTOR_HUB_DB", args.db or str(db.get_db_path()))
        uvicorn.run("agent_market.factor_hub.server:create_app",
                    host=args.host, port=args.port, reload=True, factory=True)
    else:
        app = create_app(db_path=args.db)
        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

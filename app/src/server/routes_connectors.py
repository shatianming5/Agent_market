from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, Field

from agent_market.connectors import ConnectorError

from .connector_service import ConnectorService
from .db import ConnectorRecord, ConnectorTestLog

router = APIRouter(prefix="/connectors", tags=["connectors"])


def get_service() -> ConnectorService:  # pragma: no cover - overridden in main
    raise RuntimeError("Connector service dependency not configured")


class ConnectorCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    type: str = Field(..., alias="connector_type")
    config: Dict[str, Any] = Field(default_factory=dict)
    credentials: Dict[str, Any] = Field(default_factory=dict)
    metadata: Optional[Dict[str, Any]] = None


class ConnectorUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=128)
    config: Optional[Dict[str, Any]] = None
    credentials: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class ConnectorResponse(BaseModel):
    id: str
    name: str
    type: str
    config: Dict[str, Any]
    status: Optional[str]
    last_tested_at: Optional[str]
    created_at: str
    updated_at: str
    metadata: Dict[str, Any]

    @classmethod
    def from_record(cls, record: ConnectorRecord) -> "ConnectorResponse":
        return cls(
            id=record.id,
            name=record.name,
            type=record.type,
            config=record.config,
            status=record.status,
            last_tested_at=record.last_tested_at,
            created_at=record.created_at,
            updated_at=record.updated_at,
            metadata=record.metadata,
        )


class ConnectorLogResponse(BaseModel):
    id: str
    connector_id: str
    status: str
    payload: Dict[str, Any]
    created_at: str


@router.get("/types", response_model=List[str])
def list_types(service: ConnectorService = Depends(get_service)) -> List[str]:
    return service.list_types()


@router.get("/", response_model=List[ConnectorResponse])
def list_connectors(service: ConnectorService = Depends(get_service)) -> List[ConnectorResponse]:
    return [ConnectorResponse.from_record(record) for record in service.list_connectors()]


@router.post("/", response_model=ConnectorResponse, status_code=201)
def create_connector(payload: ConnectorCreate, service: ConnectorService = Depends(get_service)) -> ConnectorResponse:
    try:
        record = service.create_connector(
            name=payload.name,
            connector_type=payload.type,
            config=payload.config,
            credentials=payload.credentials,
            metadata=payload.metadata,
        )
        return ConnectorResponse.from_record(record)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{connector_id}", response_model=ConnectorResponse)
def get_connector(connector_id: str, service: ConnectorService = Depends(get_service)) -> ConnectorResponse:
    record = service.get_connector(connector_id)
    if not record:
        raise HTTPException(status_code=404, detail="connector not found")
    return ConnectorResponse.from_record(record)


@router.patch("/{connector_id}", response_model=ConnectorResponse)
def update_connector(
    connector_id: str,
    payload: ConnectorUpdate,
    service: ConnectorService = Depends(get_service),
) -> ConnectorResponse:
    record = service.update_connector(
        connector_id,
        name=payload.name,
        config=payload.config,
        credentials=payload.credentials,
        metadata=payload.metadata,
    )
    if not record:
        raise HTTPException(status_code=404, detail="connector not found")
    return ConnectorResponse.from_record(record)


@router.delete("/{connector_id}", status_code=204)
def delete_connector(connector_id: str, service: ConnectorService = Depends(get_service)) -> Response:
    record = service.get_connector(connector_id)
    if not record:
        raise HTTPException(status_code=404, detail="connector not found")
    service.delete_connector(connector_id)
    return Response(status_code=204)


@router.post("/{connector_id}/test", response_model=Dict[str, Any])
def test_connector(connector_id: str, service: ConnectorService = Depends(get_service)) -> Dict[str, Any]:
    try:
        return service.test_connector(connector_id)
    except ConnectorError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{connector_id}/logs", response_model=List[ConnectorLogResponse])
def list_logs(connector_id: str, service: ConnectorService = Depends(get_service), limit: int = 20) -> List[ConnectorLogResponse]:
    logs = service.list_test_logs(connector_id, limit=limit)
    return [
        ConnectorLogResponse(
            id=log.id,
            connector_id=log.connector_id,
            status=log.status,
            payload=log.payload,
            created_at=log.created_at,
        )
        for log in logs
    ]


@router.post("/{connector_id}/run", response_model=Dict[str, Any])
def run_connector_action(connector_id: str, service: ConnectorService = Depends(get_service)) -> Dict[str, Any]:
    try:
        return service.run_action(connector_id)
    except ConnectorError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal

from fastapi import APIRouter, Depends, HTTPException, Body, Response
from pydantic import BaseModel, ConfigDict, Field

from .db import DB, TriggerRecord
from .triggers import TriggerManager

router = APIRouter(prefix="/triggers", tags=["triggers"])


def get_db() -> DB:  # pragma: no cover - overridden in main
    raise RuntimeError("DB dependency not configured")


def get_manager() -> TriggerManager:  # pragma: no cover - overridden in main
    raise RuntimeError("Trigger manager dependency not configured")


class TriggerConfig(BaseModel):
    model_config = ConfigDict(extra="allow")


class TriggerCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    type: Literal["webhook", "cron", "evm"]
    config: Dict[str, Any]
    enabled: bool = True


class TriggerUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=128)
    config: Optional[Dict[str, Any]] = None
    enabled: Optional[bool] = None


class TriggerResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    type: str
    config: Dict[str, Any]
    enabled: bool
    created_at: str
    updated_at: str
    last_fired_at: Optional[str] = None


def _to_response(record: TriggerRecord) -> TriggerResponse:
    return TriggerResponse.model_validate(record)


@router.get("/", response_model=List[TriggerResponse])
def list_triggers(db: DB = Depends(get_db)) -> List[TriggerResponse]:
    return [_to_response(record) for record in db.list_triggers()]


@router.post("/", response_model=TriggerResponse, status_code=201)
def create_trigger(
    payload: TriggerCreate,
    db: DB = Depends(get_db),
    manager: TriggerManager = Depends(get_manager),
) -> TriggerResponse:
    record = db.create_trigger(payload.name, payload.type, payload.config, payload.enabled)
    manager.refresh_single(record.id)
    return _to_response(record)


@router.get("/{trigger_id}", response_model=TriggerResponse)
def get_trigger(
    trigger_id: str,
    db: DB = Depends(get_db),
) -> TriggerResponse:
    record = db.get_trigger(trigger_id)
    if not record:
        raise HTTPException(status_code=404, detail="trigger not found")
    return _to_response(record)


@router.patch("/{trigger_id}", response_model=TriggerResponse)
def update_trigger(
    trigger_id: str,
    payload: TriggerUpdate,
    db: DB = Depends(get_db),
    manager: TriggerManager = Depends(get_manager),
) -> TriggerResponse:
    record = db.update_trigger(
        trigger_id,
        name=payload.name,
        config=payload.config,
        enabled=payload.enabled,
    )
    if not record:
        raise HTTPException(status_code=404, detail="trigger not found")
    manager.refresh_single(trigger_id)
    return _to_response(record)


@router.delete("/{trigger_id}", status_code=204, response_class=Response)
def delete_trigger(
    trigger_id: str,
    db: DB = Depends(get_db),
    manager: TriggerManager = Depends(get_manager),
) -> Response:
    if not db.get_trigger(trigger_id):
        raise HTTPException(status_code=404, detail="trigger not found")
    db.delete_trigger(trigger_id)
    manager.refresh_single(trigger_id)
    return Response(status_code=204)


@router.post("/{trigger_id}/fire", response_model=Dict[str, str])
def fire_trigger(
    trigger_id: str,
    payload: Dict[str, Any] = Body(default_factory=dict),
    manager: TriggerManager = Depends(get_manager),
) -> Dict[str, str]:
    try:
        job_id = manager.trigger_now(trigger_id, payload=payload)
    except KeyError:
        raise HTTPException(status_code=404, detail="trigger not found") from None
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"job_id": job_id}

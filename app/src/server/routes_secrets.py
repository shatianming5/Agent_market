from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Header, Response
from pydantic import BaseModel, Field

from .config import Settings
from .secrets import SecretsManager
from .db import SecretRecord

router = APIRouter(prefix="/secrets", tags=["secrets"])


def get_manager() -> SecretsManager:  # pragma: no cover - overridden
    raise RuntimeError("Secrets manager dependency not configured")


def get_settings() -> Settings:  # pragma: no cover - overridden
    raise RuntimeError("Settings dependency not configured")


class SecretCreate(BaseModel):
    scope: str = Field(..., min_length=1, max_length=64)
    name: str = Field(..., min_length=1, max_length=128)
    value: str = Field(..., min_length=1)


class SecretResponse(BaseModel):
    id: str
    scope: str
    name: str
    value: str
    created_at: str
    updated_at: str


def _serialize(record: SecretRecord) -> SecretResponse:
    return SecretResponse(
        id=record.id,
        scope=record.scope,
        name=record.name,
        value=record.value,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _require_token(manager: SecretsManager, token: Optional[str]) -> None:
    try:
        manager.validate_token(token)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


def _require_read(manager: SecretsManager, token: Optional[str]) -> None:
    try:
        manager.validate_read_token(token)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@router.get("/", response_model=List[SecretResponse])
def list_all(
    scope: Optional[str] = None,
    manager: SecretsManager = Depends(get_manager),
    token: Optional[str] = Header(default=None, alias="X-Secret-Token"),
) -> List[SecretResponse]:
    _require_read(manager, token)
    try:
        secrets = manager.list_secrets(scope)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    return [_serialize(s) for s in secrets]


@router.post("/", response_model=SecretResponse, status_code=201)
def create_secret(
    payload: SecretCreate,
    manager: SecretsManager = Depends(get_manager),
    token: Optional[str] = Header(default=None, alias="X-Secret-Token"),
) -> SecretResponse:
    _require_token(manager, token)
    try:
        record = manager.set_secret(payload.scope, payload.name, payload.value)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    return _serialize(record)


@router.get("/{scope}/{name}", response_model=SecretResponse)
def get_secret(
    scope: str,
    name: str,
    reveal: bool = False,
    manager: SecretsManager = Depends(get_manager),
    token: Optional[str] = Header(default=None, alias="X-Secret-Token"),
) -> SecretResponse:
    if reveal:
        _require_token(manager, token)
    else:
        _require_read(manager, token)
    try:
        record = manager.get_secret(scope, name, reveal=reveal)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    if not record:
        raise HTTPException(status_code=404, detail="secret not found")
    return _serialize(record)


@router.delete("/{scope}/{name}", status_code=204)
def delete_secret(
    scope: str,
    name: str,
    manager: SecretsManager = Depends(get_manager),
    token: Optional[str] = Header(default=None, alias="X-Secret-Token"),
) -> Response:
    _require_token(manager, token)
    try:
        manager.delete_secret(scope, name)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    return Response(status_code=204)

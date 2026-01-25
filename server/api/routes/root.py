from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/")
def root():
    return {"message": "Agent Market API", "docs": "/docs", "health": "/health"}


@router.get("/index")
def index():
    return {"message": "Agent Market API", "docs": "/docs", "health": "/health"}


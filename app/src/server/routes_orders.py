from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from .db import DB


router = APIRouter(prefix="/orders", tags=["orders"])


class OrderCreate(BaseModel):
    agent_id: str
    side: str = Field(..., pattern=r"^(buy|sell)$")
    qty: float = Field(..., gt=0)


class Order(BaseModel):
    id: str
    agent_id: str
    side: str
    qty: float
    status: str
    created_at: str


def get_db(db: DB = Depends()):  # provided via dependency override in main
    return db


@router.get("/", response_model=List[Order])
def list_orders(agent_id: Optional[str] = Query(default=None), db: DB = Depends(get_db)):
    return db.list_orders(agent_id)


@router.post("/", response_model=Order, status_code=201)
def create_order(payload: OrderCreate, db: DB = Depends(get_db)):
    # sanity: referenced agent must exist
    if not db.get_agent(payload.agent_id):
        raise HTTPException(status_code=404, detail="agent not found")
    return db.create_order(payload.agent_id, payload.side, payload.qty)


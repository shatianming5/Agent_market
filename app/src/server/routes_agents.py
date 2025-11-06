from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from .db import DB


router = APIRouter(prefix="/agents", tags=["agents"])


class AgentCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=64)


class Agent(BaseModel):
    id: str
    name: str
    created_at: str


def get_db(db: DB = Depends()):  # provided via dependency override in main
    return db


@router.get("/", response_model=List[Agent])
def list_agents(db: DB = Depends(get_db)):
    return db.list_agents()


@router.post("/", response_model=Agent, status_code=201)
def create_agent(payload: AgentCreate, db: DB = Depends(get_db)):
    return db.create_agent(payload.name)


@router.get("/{agent_id}", response_model=Agent)
def get_agent(agent_id: str, db: DB = Depends(get_db)):
    agent = db.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="agent not found")
    return agent


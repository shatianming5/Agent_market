from __future__ import annotations

from fastapi import APIRouter

from ..errors import not_found
from ...runtime import jobs

router = APIRouter()


@router.get("/jobs/{job_id}/status")
def job_status(job_id: str):
    res = jobs.status(job_id)
    if isinstance(res, dict) and res.get("error"):
        return not_found("JOB_NOT_FOUND", str(res.get("error")))
    return res


@router.get("/jobs/{job_id}/logs")
def job_logs(job_id: str, offset: int = 0, limit: int = 0):
    res = jobs.logs(job_id, offset, limit=limit)
    if isinstance(res, dict) and res.get("error"):
        return not_found("JOB_NOT_FOUND", str(res.get("error")))
    return res


@router.post("/jobs/{job_id}/cancel")
def job_cancel(job_id: str):
    res = jobs.cancel(job_id)
    if isinstance(res, dict) and res.get("error"):
        return not_found("JOB_NOT_FOUND", str(res.get("error")))
    return res

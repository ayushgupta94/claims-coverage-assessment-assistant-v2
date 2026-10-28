"""Health checks. `live` = the process is up (used by Render's health check and
cold-start polling); `ready` = it can also reach MongoDB."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live")
async def live() -> dict:
    return {"status": "ok"}


@router.get("/ready")
async def ready(request: Request) -> JSONResponse:
    mongo_ok = await request.app.state.mongo_database.ping()
    return JSONResponse(
        status_code=200 if mongo_ok else 503,
        content={"status": "ok" if mongo_ok else "degraded", "mongo_connected": mongo_ok},
    )

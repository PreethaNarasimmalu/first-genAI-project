"""Phase 6 — Meta endpoints: GET /health, GET /cuisines, GET /locations.

These endpoints expose dataset metadata and service liveness. The cuisine and
location lists are pre-loaded from the cleaned parquet file during application
startup (see src/api/main.py lifespan) and served from app.state for speed.
"""

from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter(tags=["meta"])


@router.get("/health")
def health() -> dict[str, str]:
    """Liveness probe — returns 200 whenever the service is running.

    Returns:
        JSON object with ``status`` and ``service`` keys.
    """
    return {"status": "ok", "service": "Restaurant Recommendation API"}


@router.get("/cuisines", response_model=list[str])
def cuisines(request: Request) -> list[str]:
    """Return all unique cuisine types present in the Zomato dataset.

    The list is sorted alphabetically and pre-loaded at startup.

    Returns:
        Sorted list of cuisine name strings (e.g. ``["Chinese", "Italian", ...]``).
    """
    return request.app.state.cuisines


@router.get("/locations", response_model=list[str])
def locations(request: Request) -> list[str]:
    """Return all unique restaurant locations present in the Zomato dataset.

    The list is sorted alphabetically and pre-loaded at startup.

    Returns:
        Sorted list of location/neighbourhood strings (e.g. ``["Indiranagar", ...]``).
    """
    return request.app.state.locations

"""Phase 6 — FastAPI middleware: CORS, request logging, and error handling.

Registers:
  - CORSMiddleware    — allows requests from the Vite dev server (:5173) and
                        other localhost ports during development.
  - LoggingMiddleware — logs method, path, status code, and latency per request.
  - Exception handlers for ValueError (→ 400) and EnvironmentError (→ 503).
"""

from __future__ import annotations

import logging
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("api.requests")

# Origins explicitly allowed — covers the Phase 7 Vite dev server.
_CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]


class _LoggingMiddleware(BaseHTTPMiddleware):
    """Log every HTTP request: method, path, status code, and elapsed time."""

    async def dispatch(self, request: Request, call_next):
        start = time.monotonic()
        response = await call_next(request)
        elapsed_ms = (time.monotonic() - start) * 1000
        logger.info(
            "%s %s → %d (%.1f ms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response


def setup_middleware(app: FastAPI) -> None:
    """Attach all middleware and exception handlers to *app*.

    Must be called before the app starts handling requests (typically inside
    the factory function that creates the FastAPI instance).

    Args:
        app: The FastAPI application instance to configure.
    """
    # CORS — add before other middleware so preflight requests are handled first.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # Per-request logging.
    app.add_middleware(_LoggingMiddleware)

    # -------------------------------------------------------------------------
    # Exception handlers
    # -------------------------------------------------------------------------

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        """Map unhandled ValueError to HTTP 400 Bad Request."""
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(EnvironmentError)
    async def env_error_handler(request: Request, exc: EnvironmentError) -> JSONResponse:
        """Map EnvironmentError (missing API keys, etc.) to HTTP 503."""
        logger.error("Environment configuration error: %s", exc)
        return JSONResponse(
            status_code=503,
            content={"detail": "Server configuration error — API key may be missing."},
        )

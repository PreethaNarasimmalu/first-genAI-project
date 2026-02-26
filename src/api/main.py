"""Phase 6 — FastAPI application entry point.

Creates the app instance, registers middleware, and includes all route modules.
A ``lifespan`` context manager pre-loads the ChromaDB collection and restaurant
metadata (cuisines, locations) once at startup so they are served from memory
on every request — no repeated disk I/O in the hot path.

Run with uvicorn (development):
    uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

Or via the installed entry-point (if configured):
    restaurant-api
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import ast

import numpy as np
import pandas as pd
from fastapi import FastAPI

from src.api.middleware import setup_middleware
from src.api.routes.meta import router as meta_router
from src.api.routes.recommend import router as recommend_router
from src.indexing.vector_store import get_client, get_collection

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PARQUET_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "clean" / "restaurants.parquet"
)


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown hooks
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load shared resources once at startup; release them on shutdown.

    Resources loaded:
      - ChromaDB persistent collection (Phase 2 artefact).
      - Sorted lists of unique cuisines and locations from the cleaned parquet
        (Phase 1 artefact) for the /cuisines and /locations meta-endpoints.
    """
    # --- ChromaDB collection -------------------------------------------------
    logger.info("Startup — loading ChromaDB collection …")
    client = get_client()
    collection = get_collection(client)
    app.state.collection = collection
    logger.info("ChromaDB ready: %d documents indexed.", collection.count())

    # --- Restaurant metadata from parquet ------------------------------------
    cuisines: list[str] = []
    locations: list[str] = []

    if _PARQUET_PATH.exists():
        logger.info("Loading restaurant metadata from %s …", _PARQUET_PATH)
        df = pd.read_parquet(_PARQUET_PATH, columns=["cuisines", "location"])

        # Flatten per-row cuisine values → sorted unique set.
        # The parquet stores cuisines as numpy arrays (dtype=object), Python
        # lists/tuples, or occasionally string-formatted lists depending on how
        # the column was written by Phase 1 preprocessing.
        all_cuisines: set[str] = set()
        for val in df["cuisines"].dropna():
            if isinstance(val, (list, tuple, np.ndarray)):
                all_cuisines.update(str(c).strip() for c in val if str(c).strip())
            elif isinstance(val, str):
                # Attempt to parse string-formatted list: "['A', 'B']"
                try:
                    parsed = ast.literal_eval(val)
                    if isinstance(parsed, (list, tuple)):
                        all_cuisines.update(c.strip() for c in parsed if str(c).strip())
                        continue
                except (ValueError, SyntaxError):
                    pass
                # Fall back: treat the string itself as a cuisine name
                tok = val.strip()
                if tok:
                    all_cuisines.add(tok)
        cuisines = sorted(all_cuisines)

        # Unique location strings
        locations = sorted(
            {str(loc).strip() for loc in df["location"].dropna() if str(loc).strip()}
        )

        logger.info(
            "Metadata ready: %d cuisines, %d locations.", len(cuisines), len(locations)
        )
    else:
        logger.warning(
            "Parquet not found at %s — /cuisines and /locations will return empty lists. "
            "Run Phase 1 ingestion to generate it.",
            _PARQUET_PATH,
        )

    app.state.cuisines = cuisines
    app.state.locations = locations

    yield  # --- application runs here ---

    logger.info("Shutdown complete.")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    """Construct and configure the FastAPI application.

    Separated from the module-level ``app`` assignment so tests can call
    ``create_app()`` independently and override state before making requests.

    Returns:
        Configured FastAPI instance with middleware and routes registered.
    """
    application = FastAPI(
        title="Restaurant Recommendation API",
        description=(
            "AI-powered restaurant recommendations backed by the Zomato dataset, "
            "ChromaDB semantic search, and Groq LLM."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    setup_middleware(application)
    application.include_router(meta_router)
    application.include_router(recommend_router)

    return application


# Module-level app instance — used by uvicorn and the test client.
app = create_app()

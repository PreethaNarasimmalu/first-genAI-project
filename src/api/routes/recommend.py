"""Phase 6 — POST /recommend endpoint.

Wires together all preceding phases:
  Phase 3 → preference validation (via Pydantic + UserPreference model)
  Phase 4 → retrieve and rank candidate restaurants from ChromaDB
  Phase 5 → build LLM prompts, call Groq, parse structured JSON response

The ChromaDB collection is injected from app.state (pre-loaded at startup)
so the embedding model and ChromaDB client are not re-initialised per request.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request

from src.engine.engine import recommend as run_engine
from src.llm.groq_client import call_llm
from src.llm.prompt_builder import build_prompts
from src.llm.response_parser import RecommendationResponse, parse_response
from src.preferences.models import UserPreference

logger = logging.getLogger(__name__)

router = APIRouter(tags=["recommend"])


@router.post("/recommend", response_model=RecommendationResponse)
def recommend(request: Request, body: UserPreference) -> RecommendationResponse:
    """Accept user restaurant preferences and return LLM-generated recommendations.

    The full pipeline:
      1. ``body`` is already validated by Pydantic / FastAPI before this function
         is called. A 422 is returned automatically for invalid field values.
      2. The recommendation engine (Phase 4) retrieves and ranks candidates from
         the ChromaDB collection loaded at startup.
      3. The LLM layer (Phase 5) builds prompts, calls Groq, and parses the
         structured JSON response into a ``RecommendationResponse``.

    Args:
        request: Starlette request — used to access ``app.state.collection``.
        body:    Validated ``UserPreference`` parsed from the JSON request body.

    Returns:
        ``RecommendationResponse`` with ``recommendations`` list and ``summary``.

    Raises:
        HTTP 404: No restaurants found matching the given preferences.
        HTTP 502: Groq returned a response that could not be parsed as valid JSON.
        HTTP 503: ``GROQ_API_KEY`` is missing, invalid, or Groq is unavailable.
    """
    collection = request.app.state.collection

    # ------------------------------------------------------------------
    # Phase 4: retrieval + ranking
    # ------------------------------------------------------------------
    logger.info(
        "Recommendation request: %s",
        body.model_dump(exclude_none=True),
    )

    candidates = run_engine(body, collection=collection)

    if not candidates:
        raise HTTPException(
            status_code=404,
            detail=(
                "No restaurants found matching your preferences. "
                "Try relaxing your filters (e.g. higher budget, broader location)."
            ),
        )

    logger.info("Engine returned %d candidates.", len(candidates))

    # ------------------------------------------------------------------
    # Phase 5: LLM recommendation generation
    # ------------------------------------------------------------------
    system_prompt, user_prompt = build_prompts(body, candidates)

    try:
        raw_llm_output = call_llm(system_prompt, user_prompt)
    except EnvironmentError as exc:
        logger.error("LLM call failed — configuration error: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("LLM call failed: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="LLM service is temporarily unavailable. Please try again shortly.",
        ) from exc

    try:
        result = parse_response(raw_llm_output)
    except ValueError as exc:
        logger.error("LLM response parsing failed: %s", exc)
        raise HTTPException(
            status_code=502,
            detail="LLM returned an unexpected response format. Please retry.",
        ) from exc

    logger.info("Returning %d recommendations.", len(result.recommendations))
    return result

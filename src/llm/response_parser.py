"""Phase 5 — LLM response parser.

Validates and deserializes the raw JSON string returned by Groq into typed
Pydantic models. Handles JSON wrapped inside markdown code fences gracefully.

Usage:
    from src.llm.response_parser import parse_response, RecommendationResponse
    result = parse_response(raw_llm_text)
    print(result.summary)
    for rec in result.recommendations:
        print(rec.rank, rec.name, rec.why)
"""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------

class RecommendationItem(BaseModel):
    """A single restaurant recommendation from the LLM."""

    rank: int = Field(..., ge=1)
    name: str
    location: str
    cuisine: str
    rating: float = Field(..., ge=0, le=5)
    approx_cost: int = Field(..., ge=0)
    why: str
    highlight: str


class RecommendationResponse(BaseModel):
    """Full structured response returned by Phase 5."""

    recommendations: list[RecommendationItem]
    summary: str


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_response(raw: str) -> RecommendationResponse:
    """Parse raw LLM output into a RecommendationResponse.

    Handles two common LLM output formats:
      1. Raw JSON object.
      2. JSON wrapped in ```json ... ``` or ``` ... ``` markdown fences.

    Args:
        raw: Raw string content from Groq completion.

    Returns:
        Validated RecommendationResponse.

    Raises:
        ValueError: If the text cannot be parsed as valid JSON or does not
                    match the expected schema.
    """
    text = raw.strip()

    # Strip markdown code fences if present (``` or ```json)
    fence_match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"LLM returned invalid JSON.\n"
            f"JSON error: {exc}\n"
            f"Raw response (first 500 chars):\n{raw[:500]}"
        ) from exc

    try:
        return RecommendationResponse(**data)
    except Exception as exc:
        raise ValueError(
            f"LLM JSON did not match expected schema.\n"
            f"Error: {exc}\n"
            f"Parsed data: {data}"
        ) from exc

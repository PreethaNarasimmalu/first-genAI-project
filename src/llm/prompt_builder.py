"""Phase 5 — Prompt builder.

Constructs the system and user prompts sent to Groq from a UserPreference
and a list of ranked RestaurantCandidate objects.

Usage:
    from src.llm.prompt_builder import build_prompts
    system_prompt, user_prompt = build_prompts(prefs, candidates)
"""

from __future__ import annotations

from src.engine.ranker import RestaurantCandidate
from src.preferences.models import UserPreference

# ---------------------------------------------------------------------------
# System prompt — static, defines the LLM's role and output contract
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are an expert restaurant concierge for Bangalore, India. "
    "You will be given a list of candidate restaurants retrieved from a real dataset. "
    "Your job is to rank and present ONLY those restaurants — do NOT invent, add, or "
    "substitute any restaurant not explicitly listed in the candidates. "
    "Every restaurant in your response must appear verbatim in the candidate list. "
    "Use only the fields provided (name, location, cuisine, rating, cost, dishes). "
    "If the user requested a specific cuisine and NONE of the candidates match that cuisine, "
    "return an empty recommendations list: "
    "{\"recommendations\": [], \"summary\": \"No matching restaurants found for your preferences.\"}. "
    "Never use placeholder values like 'None', 'null', or '0' for name, location, or cuisine. "
    "Always respond with a single valid JSON object — no prose before or "
    "after it, no markdown code fences, just the raw JSON."
)

# ---------------------------------------------------------------------------
# JSON schema shown inside the user prompt so the model knows the shape
# ---------------------------------------------------------------------------

_JSON_SCHEMA = """{
  "recommendations": [
    {
      "rank": 1,
      "name": "<restaurant name>",
      "location": "<neighbourhood>",
      "cuisine": "<cuisine types>",
      "rating": 4.2,
      "approx_cost": 600,
      "why": "<one sentence: why this matches the user preferences>",
      "highlight": "<a dish or feature worth trying>"
    }
  ],
  "summary": "<2-3 sentence conversational overview for the user>"
}"""


# ---------------------------------------------------------------------------
# Internal formatters
# ---------------------------------------------------------------------------

def _format_preferences(prefs: UserPreference) -> str:
    parts: list[str] = []
    if prefs.cuisine:
        parts.append(f"  Cuisine         : {', '.join(prefs.cuisine)}")
    if prefs.location:
        parts.append(f"  Location        : {prefs.location}")
    if prefs.max_price is not None:
        parts.append(f"  Budget (for two): \u20b9{prefs.max_price}")
    if prefs.min_rating is not None:
        parts.append(f"  Minimum rating  : {prefs.min_rating}\u2605")
    if prefs.meal_type:
        parts.append(f"  Meal type       : {prefs.meal_type}")
    if prefs.online_order is not None:
        parts.append(f"  Online order    : {'Yes' if prefs.online_order else 'No'}")
    if prefs.book_table is not None:
        parts.append(f"  Table booking   : {'Yes' if prefs.book_table else 'No'}")
    if prefs.free_text:
        parts.append(f"  Additional      : \"{prefs.free_text}\"")
    return "\n".join(parts) if parts else "  (no specific preferences provided)"


def _format_candidates(candidates: list[RestaurantCandidate]) -> str:
    lines: list[str] = []
    for i, c in enumerate(candidates, 1):
        lines.append(
            f"{i}. {c.name} | {c.location} | {c.cuisine_str} | "
            f"{c.rate}\u2605 | \u20b9{c.approx_cost} for two | "
            f"Votes: {c.votes} | "
            f"Online order: {'Yes' if c.online_order else 'No'} | "
            f"Table booking: {'Yes' if c.book_table else 'No'} | "
            f"Popular dishes: {c.dish_liked or 'N/A'}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_prompts(
    prefs: UserPreference,
    candidates: list[RestaurantCandidate],
) -> tuple[str, str]:
    """Build (system_prompt, user_prompt) for the Groq LLM call.

    Args:
        prefs:      Validated UserPreference from Phase 3.
        candidates: Ranked RestaurantCandidate list from Phase 4.

    Returns:
        A (system_prompt, user_prompt) tuple ready to pass to call_llm().
    """
    user_prompt = (
        f"User preferences:\n{_format_preferences(prefs)}\n\n"
        f"Candidate restaurants (from the real Zomato dataset — use ONLY these):\n"
        f"{_format_candidates(candidates)}\n\n"
        f"IMPORTANT: Recommend ONLY from the candidates listed above. "
        f"Do not add, invent, or substitute any other restaurant. "
        f"Include ALL {len(candidates)} candidates in your recommendations list — do not drop any.\n\n"
        f"Respond with a JSON object matching exactly this schema:\n{_JSON_SCHEMA}"
    )
    return _SYSTEM_PROMPT, user_prompt

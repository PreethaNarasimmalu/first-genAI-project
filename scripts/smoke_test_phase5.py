"""Phase 5 smoke test — Prompt Builder, Response Parser, Groq Client.

Offline tests (no API key needed):
  1.  build_prompts includes cuisine in user prompt
  2.  build_prompts includes location in user prompt
  3.  build_prompts includes budget in user prompt
  4.  build_prompts includes candidate name in user prompt
  5.  build_prompts includes candidate rating in user prompt
  6.  parse_response parses valid raw JSON
  7.  parse_response handles ```json ... ``` markdown fences
  8.  parse_response validates recommendation count
  9.  parse_response raises ValueError on bad JSON
  10. parse_response validates rating range (Pydantic)

Live test (requires GROQ_API_KEY in .env):
  11. Full end-to-end: prefs → Phase 4 candidates → prompts → Groq → parsed response
"""

from __future__ import annotations

import json
import os
import traceback
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def check(label: str, condition: bool) -> None:
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}")


# ---------------------------------------------------------------------------
# Minimal stubs so offline tests have no ChromaDB / embedding dependency
# ---------------------------------------------------------------------------

@dataclass
class _FakeCandidate:
    name: str
    location: str
    cuisine_str: str
    rest_type: str
    rate: float
    votes: int
    approx_cost: int
    online_order: bool
    book_table: bool
    dish_liked: str
    meal_type: str
    city: str
    score: float
    distance: float


_FAKE_CANDIDATES = [
    _FakeCandidate(
        name="Trattoria Roma",
        location="Indiranagar",
        cuisine_str="Italian, Pizza",
        rest_type="Casual Dining",
        rate=4.3,
        votes=1200,
        approx_cost=900,
        online_order=True,
        book_table=True,
        dish_liked="Margherita Pizza, Pasta Arrabiata",
        meal_type="Dine-out",
        city="Bangalore",
        score=0.82,
        distance=0.18,
    ),
    _FakeCandidate(
        name="Bella Napoli",
        location="Indiranagar",
        cuisine_str="Italian, Cafe",
        rest_type="Cafe",
        rate=4.0,
        votes=640,
        approx_cost=700,
        online_order=False,
        book_table=True,
        dish_liked="Tiramisu, Bruschetta",
        meal_type="Dine-out",
        city="Bangalore",
        score=0.74,
        distance=0.26,
    ),
]

_FAKE_PREFS_DICT = {
    "cuisine": ["Italian"],
    "location": "Indiranagar",
    "max_price": 1200,
    "min_rating": 3.5,
    "free_text": "romantic dinner with candles",
}


def _make_prefs():
    from src.preferences.parser import parse_preferences
    return parse_preferences(_FAKE_PREFS_DICT)


# ---------------------------------------------------------------------------
# Offline unit tests — prompt_builder
# ---------------------------------------------------------------------------

def test_prompt_cuisine() -> None:
    print("Test 1: build_prompts — cuisine in user prompt")
    from src.llm.prompt_builder import build_prompts
    prefs = _make_prefs()
    _, user = build_prompts(prefs, _FAKE_CANDIDATES)
    check("'Italian' appears in user prompt", "Italian" in user)


def test_prompt_location() -> None:
    print("\nTest 2: build_prompts — location in user prompt")
    from src.llm.prompt_builder import build_prompts
    prefs = _make_prefs()
    _, user = build_prompts(prefs, _FAKE_CANDIDATES)
    check("'Indiranagar' appears in user prompt", "Indiranagar" in user)


def test_prompt_budget() -> None:
    print("\nTest 3: build_prompts — budget in user prompt")
    from src.llm.prompt_builder import build_prompts
    prefs = _make_prefs()
    _, user = build_prompts(prefs, _FAKE_CANDIDATES)
    check("'1200' appears in user prompt", "1200" in user)


def test_prompt_candidate_name() -> None:
    print("\nTest 4: build_prompts — candidate name in user prompt")
    from src.llm.prompt_builder import build_prompts
    prefs = _make_prefs()
    _, user = build_prompts(prefs, _FAKE_CANDIDATES)
    check("'Trattoria Roma' in user prompt", "Trattoria Roma" in user)


def test_prompt_candidate_rating() -> None:
    print("\nTest 5: build_prompts — candidate rating in user prompt")
    from src.llm.prompt_builder import build_prompts
    prefs = _make_prefs()
    _, user = build_prompts(prefs, _FAKE_CANDIDATES)
    check("'4.3' in user prompt", "4.3" in user)


# ---------------------------------------------------------------------------
# Offline unit tests — response_parser
# ---------------------------------------------------------------------------

_VALID_JSON = {
    "recommendations": [
        {
            "rank": 1,
            "name": "Trattoria Roma",
            "location": "Indiranagar",
            "cuisine": "Italian",
            "rating": 4.3,
            "approx_cost": 900,
            "why": "Matches Italian cuisine preference in Indiranagar.",
            "highlight": "Margherita Pizza",
        }
    ],
    "summary": "Great choice for a romantic Italian dinner in Indiranagar.",
}


def test_parse_raw_json() -> None:
    print("\nTest 6: parse_response — raw JSON string")
    from src.llm.response_parser import parse_response
    result = parse_response(json.dumps(_VALID_JSON))
    check("got RecommendationResponse", result is not None)
    check("1 recommendation", len(result.recommendations) == 1)
    check("name matches", result.recommendations[0].name == "Trattoria Roma")
    check("summary present", len(result.summary) > 0)


def test_parse_markdown_fence() -> None:
    print("\nTest 7: parse_response — ```json ... ``` markdown fence")
    from src.llm.response_parser import parse_response
    fenced = f"```json\n{json.dumps(_VALID_JSON)}\n```"
    result = parse_response(fenced)
    check("parsed despite fence", result.recommendations[0].name == "Trattoria Roma")


def test_parse_recommendation_count() -> None:
    print("\nTest 8: parse_response — recommendation count preserved")
    from src.llm.response_parser import parse_response
    multi = {
        "recommendations": [
            {**_VALID_JSON["recommendations"][0], "rank": i}
            for i in range(1, 4)
        ],
        "summary": "Three great picks.",
    }
    result = parse_response(json.dumps(multi))
    check("3 recommendations", len(result.recommendations) == 3)


def test_parse_bad_json() -> None:
    print("\nTest 9: parse_response — raises ValueError on bad JSON")
    from src.llm.response_parser import parse_response
    raised = False
    try:
        parse_response("this is not JSON at all")
    except ValueError:
        raised = True
    check("ValueError raised on bad JSON", raised)


def test_parse_invalid_rating() -> None:
    print("\nTest 10: parse_response — Pydantic rejects rating > 5")
    from src.llm.response_parser import parse_response
    bad = {**_VALID_JSON}
    bad["recommendations"] = [{**_VALID_JSON["recommendations"][0], "rating": 9.9}]
    raised = False
    try:
        parse_response(json.dumps(bad))
    except (ValueError, Exception):
        raised = True
    check("error raised for rating=9.9", raised)


# ---------------------------------------------------------------------------
# Live end-to-end test (requires GROQ_API_KEY + cached HF model + ChromaDB)
# ---------------------------------------------------------------------------

def test_end_to_end() -> None:
    print("\nTest 11: Full end-to-end with Groq API")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        print("  Skipped — GROQ_API_KEY not found in .env")
        return

    try:
        from src.engine.engine import recommend
        from src.llm.groq_client import call_llm
        from src.llm.prompt_builder import build_prompts
        from src.llm.response_parser import parse_response
        from src.preferences.parser import parse_preferences

        prefs = parse_preferences({
            "cuisine": ["North Indian"],
            "location": "Koramangala",
            "max_price": 800,
            "min_rating": 3.5,
        })

        candidates = recommend(prefs, final_top_n=5)
        check(f"Phase 4 returned {len(candidates)} candidates", len(candidates) > 0)

        system_prompt, user_prompt = build_prompts(prefs, candidates)
        check("prompts built", len(system_prompt) > 0 and len(user_prompt) > 0)

        raw = call_llm(system_prompt, user_prompt)
        check("Groq returned text", len(raw.strip()) > 0)

        result = parse_response(raw)
        check("response parsed successfully", result is not None)
        check("has recommendations", len(result.recommendations) > 0)
        check("has summary", len(result.summary) > 0)

        print(f"\n  {'='*60}")
        print(f"  SUMMARY:\n  {result.summary}")
        print(f"\n  TOP RECOMMENDATIONS:")
        for rec in result.recommendations:
            print(f"\n  {rec.rank}. {rec.name} ({rec.location})")
            print(f"     Cuisine : {rec.cuisine}")
            print(f"     Rating  : {rec.rating}\u2605  |  Cost: \u20b9{rec.approx_cost}")
            print(f"     Why     : {rec.why}")
            print(f"     Try     : {rec.highlight}")
        print(f"  {'='*60}")

    except Exception as exc:
        if "ProxyError" in type(exc).__name__ or "ProxyError" in str(exc):
            print("  Skipped — no network access to HuggingFace (embedding model).")
            print("  Run on your local machine where the model is cached.")
        else:
            print("  End-to-end test FAILED:")
            traceback.print_exc()
            check("end-to-end passed", False)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Phase 5 smoke tests ===\n")
    test_prompt_cuisine()
    test_prompt_location()
    test_prompt_budget()
    test_prompt_candidate_name()
    test_prompt_candidate_rating()
    test_parse_raw_json()
    test_parse_markdown_fence()
    test_parse_recommendation_count()
    test_parse_bad_json()
    test_parse_invalid_rating()
    test_end_to_end()
    print("\n=== Done ===")

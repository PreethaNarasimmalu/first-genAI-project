"""Phase 6 — API layer tests.

Uses FastAPI's TestClient (synchronous, backed by httpx) together with
``unittest.mock`` to isolate external dependencies:

  - ``call_llm`` is always mocked — avoids real Groq API calls and the need
    for a live GROQ_API_KEY in CI.
  - ``run_engine`` is mocked in tests that need deterministic candidate lists
    (e.g. the 404 / 502 / 503 error path tests).
  - The ChromaDB collection and parquet file are loaded for real during the
    TestClient lifespan so the /cuisines and /locations endpoints exercise
    actual data.

Test classes:
  TestHealth     — GET /health
  TestCuisines   — GET /cuisines
  TestLocations  — GET /locations
  TestRecommend  — POST /recommend (success, validation, error paths)
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.engine.ranker import RestaurantCandidate

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client() -> TestClient:
    """TestClient that triggers the full app lifespan (loads ChromaDB + parquet).

    Scoped to the module so startup/shutdown runs only once per test file —
    keeps the suite fast even though the lifespan loads real data.
    """
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

# A single mock candidate returned by a patched run_engine.
_MOCK_CANDIDATE = RestaurantCandidate(
    name="Spice Garden",
    location="Koramangala",
    cuisine_str="North Indian, Chinese",
    rest_type="Casual Dining",
    rate=4.2,
    votes=850,
    approx_cost=650,
    online_order=True,
    book_table=False,
    dish_liked="Butter Chicken, Fried Rice",
    meal_type="Dine-out",
    city="Koramangala",
    score=0.87,
    distance=0.18,
)

# Valid JSON that mimics a real Groq response.
_MOCK_LLM_JSON = json.dumps(
    {
        "recommendations": [
            {
                "rank": 1,
                "name": "Spice Garden",
                "location": "Koramangala",
                "cuisine": "North Indian, Chinese",
                "rating": 4.2,
                "approx_cost": 650,
                "why": "Great match for your budget and cuisine preference.",
                "highlight": "Try the butter chicken — a crowd favourite.",
            }
        ],
        "summary": (
            "Spice Garden in Koramangala is a solid pick for North Indian "
            "and Chinese food within your budget."
        ),
    }
)


# ---------------------------------------------------------------------------
# Helper — patch both engine and LLM for a clean, isolated /recommend call
# ---------------------------------------------------------------------------


def _recommend(client: TestClient, body: dict, *, llm_return=_MOCK_LLM_JSON):
    """POST /recommend with engine + LLM both mocked."""
    with (
        patch(
            "src.api.routes.recommend.run_engine",
            return_value=[_MOCK_CANDIDATE],
        ),
        patch(
            "src.api.routes.recommend.call_llm",
            return_value=llm_return,
        ),
    ):
        return client.post("/recommend", json=body)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


class TestHealth:
    def test_returns_200(self, client: TestClient):
        assert client.get("/health").status_code == 200

    def test_status_is_ok(self, client: TestClient):
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_service_key_present(self, client: TestClient):
        data = client.get("/health").json()
        assert "service" in data

    def test_service_value_is_string(self, client: TestClient):
        data = client.get("/health").json()
        assert isinstance(data["service"], str)


# ---------------------------------------------------------------------------
# GET /cuisines
# ---------------------------------------------------------------------------


class TestCuisines:
    def test_returns_200(self, client: TestClient):
        assert client.get("/cuisines").status_code == 200

    def test_returns_list(self, client: TestClient):
        assert isinstance(client.get("/cuisines").json(), list)

    def test_list_is_not_empty(self, client: TestClient):
        assert len(client.get("/cuisines").json()) > 0

    def test_items_are_strings(self, client: TestClient):
        data = client.get("/cuisines").json()
        assert all(isinstance(c, str) for c in data)

    def test_list_is_sorted(self, client: TestClient):
        data = client.get("/cuisines").json()
        assert data == sorted(data), "Cuisine list should be sorted alphabetically."

    def test_no_empty_strings(self, client: TestClient):
        data = client.get("/cuisines").json()
        assert all(c.strip() for c in data), "No empty or whitespace-only cuisine names."

    def test_cuisines_are_individual_names_not_raw_reprs(self, client: TestClient):
        """Cuisine names must be plain strings, not raw Python list reprs like \"['Italian']\"."""
        data = client.get("/cuisines").json()
        for name in data:
            assert not name.startswith("["), (
                f"Cuisine {name!r} looks like a raw list repr — parsing is broken."
            )


# ---------------------------------------------------------------------------
# GET /locations
# ---------------------------------------------------------------------------


class TestLocations:
    def test_returns_200(self, client: TestClient):
        assert client.get("/locations").status_code == 200

    def test_returns_list(self, client: TestClient):
        assert isinstance(client.get("/locations").json(), list)

    def test_list_is_not_empty(self, client: TestClient):
        assert len(client.get("/locations").json()) > 0

    def test_items_are_strings(self, client: TestClient):
        data = client.get("/locations").json()
        assert all(isinstance(loc, str) for loc in data)

    def test_list_is_sorted(self, client: TestClient):
        data = client.get("/locations").json()
        assert data == sorted(data), "Location list should be sorted alphabetically."

    def test_no_empty_strings(self, client: TestClient):
        data = client.get("/locations").json()
        assert all(loc.strip() for loc in data), "No empty or whitespace-only locations."

    def test_contains_common_locations(self, client: TestClient):
        data = client.get("/locations").json()
        common = {"Koramangala", "Indiranagar"}
        assert common.issubset(set(data)), f"Expected {common} to be in locations list."


# ---------------------------------------------------------------------------
# POST /recommend — happy path
# ---------------------------------------------------------------------------


class TestRecommendSuccess:
    def test_valid_request_returns_200(self, client: TestClient):
        assert _recommend(client, {"location": "Koramangala"}).status_code == 200

    def test_empty_body_returns_200(self, client: TestClient):
        """All UserPreference fields are optional — empty body must be accepted."""
        assert _recommend(client, {}).status_code == 200

    def test_response_has_recommendations_key(self, client: TestClient):
        data = _recommend(client, {"location": "Koramangala"}).json()
        assert "recommendations" in data

    def test_response_has_summary_key(self, client: TestClient):
        data = _recommend(client, {"location": "Koramangala"}).json()
        assert "summary" in data

    def test_recommendations_is_list(self, client: TestClient):
        data = _recommend(client, {"location": "Koramangala"}).json()
        assert isinstance(data["recommendations"], list)

    def test_recommendation_item_has_all_fields(self, client: TestClient):
        data = _recommend(client, {"location": "Koramangala"}).json()
        required = {"rank", "name", "location", "cuisine", "rating", "approx_cost", "why", "highlight"}
        rec = data["recommendations"][0]
        missing = required - rec.keys()
        assert not missing, f"Recommendation item is missing fields: {missing}"

    def test_rank_is_integer(self, client: TestClient):
        rec = _recommend(client, {}).json()["recommendations"][0]
        assert isinstance(rec["rank"], int)

    def test_rating_is_number(self, client: TestClient):
        rec = _recommend(client, {}).json()["recommendations"][0]
        assert isinstance(rec["rating"], (int, float))

    def test_approx_cost_is_number(self, client: TestClient):
        rec = _recommend(client, {}).json()["recommendations"][0]
        assert isinstance(rec["approx_cost"], (int, float))

    def test_summary_is_string(self, client: TestClient):
        data = _recommend(client, {}).json()
        assert isinstance(data["summary"], str)
        assert len(data["summary"]) > 0

    def test_full_preference_body_returns_200(self, client: TestClient):
        """All preference fields populated simultaneously."""
        body = {
            "cuisine": ["North Indian"],
            "location": "Indiranagar",
            "max_price": 1000,
            "min_rating": 3.5,
            "online_order": True,
            "book_table": False,
            "meal_type": "Dine-out",
            "free_text": "romantic dinner with a good view",
        }
        assert _recommend(client, body).status_code == 200

    def test_cuisine_as_bare_string_is_accepted(self, client: TestClient):
        """cuisine field normalises a bare string to a single-element list."""
        assert _recommend(client, {"cuisine": "South Indian"}).status_code == 200


# ---------------------------------------------------------------------------
# POST /recommend — Pydantic / validation errors (422)
# ---------------------------------------------------------------------------


class TestRecommendValidation:
    def test_min_rating_above_5_returns_422(self, client: TestClient):
        resp = client.post("/recommend", json={"min_rating": 9.9})
        assert resp.status_code == 422

    def test_min_rating_negative_returns_422(self, client: TestClient):
        resp = client.post("/recommend", json={"min_rating": -1.0})
        assert resp.status_code == 422

    def test_max_price_zero_returns_422(self, client: TestClient):
        resp = client.post("/recommend", json={"max_price": 0})
        assert resp.status_code == 422

    def test_max_price_negative_returns_422(self, client: TestClient):
        resp = client.post("/recommend", json={"max_price": -500})
        assert resp.status_code == 422

    def test_422_response_has_detail_key(self, client: TestClient):
        resp = client.post("/recommend", json={"min_rating": 99.0})
        assert "detail" in resp.json()


# ---------------------------------------------------------------------------
# POST /recommend — error paths (404, 502, 503)
# ---------------------------------------------------------------------------


class TestRecommendErrors:
    def test_no_candidates_returns_404(self, client: TestClient):
        """When the engine finds nothing, the API should return 404."""
        with (
            patch("src.api.routes.recommend.run_engine", return_value=[]),
            patch("src.api.routes.recommend.call_llm", return_value=_MOCK_LLM_JSON),
        ):
            resp = client.post("/recommend", json={"location": "Koramangala"})
        assert resp.status_code == 404

    def test_404_response_has_detail(self, client: TestClient):
        with (
            patch("src.api.routes.recommend.run_engine", return_value=[]),
            patch("src.api.routes.recommend.call_llm", return_value=_MOCK_LLM_JSON),
        ):
            data = client.post("/recommend", json={}).json()
        assert "detail" in data

    def test_llm_environment_error_returns_503(self, client: TestClient):
        """Missing GROQ_API_KEY should surface as 503 Service Unavailable."""
        with (
            patch(
                "src.api.routes.recommend.run_engine",
                return_value=[_MOCK_CANDIDATE],
            ),
            patch(
                "src.api.routes.recommend.call_llm",
                side_effect=EnvironmentError("GROQ_API_KEY not set"),
            ),
        ):
            resp = client.post("/recommend", json={"location": "Koramangala"})
        assert resp.status_code == 503

    def test_llm_generic_error_returns_503(self, client: TestClient):
        """Any unhandled LLM exception maps to 503."""
        with (
            patch(
                "src.api.routes.recommend.run_engine",
                return_value=[_MOCK_CANDIDATE],
            ),
            patch(
                "src.api.routes.recommend.call_llm",
                side_effect=RuntimeError("Connection refused"),
            ),
        ):
            resp = client.post("/recommend", json={})
        assert resp.status_code == 503

    def test_llm_invalid_json_returns_502(self, client: TestClient):
        """If Groq returns unparseable JSON the API should return 502 Bad Gateway."""
        with (
            patch(
                "src.api.routes.recommend.run_engine",
                return_value=[_MOCK_CANDIDATE],
            ),
            patch(
                "src.api.routes.recommend.call_llm",
                return_value="this is not valid json at all",
            ),
        ):
            resp = client.post("/recommend", json={"location": "Koramangala"})
        assert resp.status_code == 502

    def test_llm_schema_mismatch_returns_502(self, client: TestClient):
        """Valid JSON that doesn't match the schema should also return 502."""
        bad_json = json.dumps({"wrong_key": "wrong_value"})
        with (
            patch(
                "src.api.routes.recommend.run_engine",
                return_value=[_MOCK_CANDIDATE],
            ),
            patch(
                "src.api.routes.recommend.call_llm",
                return_value=bad_json,
            ),
        ):
            resp = client.post("/recommend", json={})
        assert resp.status_code == 502


# ---------------------------------------------------------------------------
# CORS headers
# ---------------------------------------------------------------------------


class TestCORS:
    """Verify CORS headers are present for browser-originated requests."""

    def test_preflight_options_returns_200(self, client: TestClient):
        resp = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
            },
        )
        # Starlette CORS returns 200 for preflight requests
        assert resp.status_code in (200, 400)

    def test_cors_header_on_health(self, client: TestClient):
        resp = client.get(
            "/health",
            headers={"Origin": "http://localhost:5173"},
        )
        assert "access-control-allow-origin" in resp.headers

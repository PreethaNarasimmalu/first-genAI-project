"""Phase 4 — Ranker tests.

Tests deduplication, score computation, and top-N slicing in ranker.rank().
"""

from __future__ import annotations

from typing import Any

import pytest

from src.engine.ranker import RestaurantCandidate, _price_fit, _semantic_similarity, rank


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(
    name: str,
    location: str = "Koramangala",
    rate: float = 4.0,
    votes: int = 500,
    approx_cost: int = 600,
    distance: float = 0.2,
    cuisine_str: str = "North Indian",
) -> dict[str, Any]:
    return {
        "id": name.lower().replace(" ", "_"),
        "document": f"{name} restaurant",
        "metadata": {
            "name": name,
            "location": location,
            "cuisine_str": cuisine_str,
            "meal_type": "Dine-out",
            "rate": rate,
            "votes": votes,
            "approx_cost": approx_cost,
            "online_order": True,
            "book_table": False,
            "rest_type": "Casual Dining",
            "dish_liked": "Biryani",
            "city": location,
            "is_near_duplicate": False,
        },
        "distance": distance,
    }


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------

class TestEmptyInput:
    def test_empty_list_returns_empty(self):
        assert rank([]) == []

    def test_empty_list_with_max_price_returns_empty(self):
        assert rank([], max_price=800) == []


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

class TestDeduplication:
    def test_exact_duplicate_kept_once(self):
        raw = [
            _make_result("Place A", distance=0.3),
            _make_result("Place A", distance=0.3),
        ]
        assert len(rank(raw)) == 1

    def test_duplicate_with_lower_distance_wins(self):
        """When same name+location appears twice, the closer (lower distance) copy wins."""
        raw = [
            _make_result("Place A", distance=0.4),
            _make_result("Place A", distance=0.1),  # better match
        ]
        result = rank(raw)
        assert len(result) == 1
        assert result[0].distance == pytest.approx(0.1)

    def test_same_name_different_location_not_deduplicated(self):
        raw = [
            _make_result("Place A", location="Koramangala"),
            _make_result("Place A", location="Indiranagar"),
        ]
        assert len(rank(raw)) == 2

    def test_different_names_same_location_not_deduplicated(self):
        raw = [
            _make_result("Place A", location="Koramangala"),
            _make_result("Place B", location="Koramangala"),
        ]
        assert len(rank(raw)) == 2


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

class TestScoring:
    def test_higher_rating_gives_higher_score(self):
        """All else equal, the restaurant with a higher rating should rank first."""
        raw = [
            _make_result("Low Rated", rate=3.0, distance=0.2),
            _make_result("High Rated", rate=4.8, distance=0.2),
        ]
        results = rank(raw)
        assert results[0].name == "High Rated"

    def test_closer_distance_gives_higher_score(self):
        """All else equal, the more semantically similar restaurant should rank first."""
        raw = [
            _make_result("Far", rate=4.0, distance=0.8),
            _make_result("Near", rate=4.0, distance=0.1),
        ]
        results = rank(raw)
        assert results[0].name == "Near"

    def test_score_bounded_between_0_and_1(self):
        raw = [_make_result("Place", rate=5.0, votes=10000, distance=0.0)]
        result = rank(raw)[0]
        assert 0.0 <= result.score <= 1.0

    def test_score_is_rounded_to_4_dp(self):
        raw = [_make_result("Place")]
        result = rank(raw)[0]
        assert result.score == round(result.score, 4)

    def test_all_candidates_have_score(self):
        raw = [_make_result(f"Place {i}") for i in range(3)]
        for c in rank(raw):
            assert isinstance(c.score, float)


# ---------------------------------------------------------------------------
# Top-N slicing
# ---------------------------------------------------------------------------

class TestTopN:
    def test_returns_at_most_top_n(self):
        raw = [_make_result(f"Place {i}") for i in range(10)]
        assert len(rank(raw, top_n=5)) == 5

    def test_top_n_larger_than_pool_returns_all(self):
        raw = [_make_result(f"Place {i}") for i in range(3)]
        assert len(rank(raw, top_n=5)) == 3

    def test_results_sorted_by_score_descending(self):
        raw = [
            _make_result("Low Rated", rate=2.0, distance=0.5),
            _make_result("Mid Rated", rate=3.5, distance=0.3),
            _make_result("High Rated", rate=4.9, distance=0.1),
        ]
        results = rank(raw, top_n=3)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_n_one_returns_single_best(self):
        raw = [
            _make_result("Best", rate=4.9, distance=0.05),
            _make_result("Worse", rate=2.0, distance=0.9),
        ]
        result = rank(raw, top_n=1)
        assert len(result) == 1
        assert result[0].name == "Best"


# ---------------------------------------------------------------------------
# Price fit helper
# ---------------------------------------------------------------------------

class TestPriceFit:
    def test_no_max_price_returns_1(self):
        assert _price_fit(800, None) == 1.0

    def test_zero_cost_returns_1(self):
        assert _price_fit(0, 800) == 1.0

    def test_within_budget_returns_1(self):
        assert _price_fit(600, 800) == 1.0

    def test_exactly_at_budget_returns_1(self):
        assert _price_fit(800, 800) == 1.0

    def test_over_budget_decays(self):
        score = _price_fit(1600, 800)
        assert 0.0 < score < 1.0

    def test_over_budget_score_never_below_zero(self):
        assert _price_fit(999999, 1) >= 0.0


# ---------------------------------------------------------------------------
# Semantic similarity helper
# ---------------------------------------------------------------------------

class TestSemanticSimilarity:
    def test_distance_zero_is_similarity_one(self):
        assert _semantic_similarity(0.0) == pytest.approx(1.0)

    def test_distance_two_is_similarity_zero(self):
        assert _semantic_similarity(2.0) == pytest.approx(0.0)

    def test_distance_one_is_similarity_half(self):
        assert _semantic_similarity(1.0) == pytest.approx(0.5)

    def test_similarity_never_negative(self):
        assert _semantic_similarity(3.0) == 0.0  # clamped

    def test_output_in_range(self):
        for d in [0.0, 0.5, 1.0, 1.5, 2.0]:
            assert 0.0 <= _semantic_similarity(d) <= 1.0

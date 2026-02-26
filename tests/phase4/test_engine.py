"""Phase 4 — Engine filter tests.

Tests every post-filter applied by engine.recommend() against a mocked
retrieve() so we never touch the real ChromaDB or embedding model.

Each test verifies one filter in isolation, then combined cases, to make
sure adding one filter never silently corrupts another.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from src.engine.engine import recommend
from src.preferences.models import UserPreference


# ---------------------------------------------------------------------------
# Helpers — build fake ChromaDB result dicts
# ---------------------------------------------------------------------------

def _make_result(
    name: str,
    location: str,
    cuisine_str: str = "North Indian",
    meal_type: str = "Dine-out",
    rate: float = 4.0,
    votes: int = 500,
    approx_cost: int = 600,
    online_order: bool = True,
    book_table: bool = False,
    distance: float = 0.2,
) -> dict[str, Any]:
    return {
        "id": name.lower().replace(" ", "_"),
        "document": f"{name} is a restaurant in {location}.",
        "metadata": {
            "name": name,
            "location": location,
            "cuisine_str": cuisine_str,
            "meal_type": meal_type,
            "rate": rate,
            "votes": votes,
            "approx_cost": approx_cost,
            "online_order": online_order,
            "book_table": book_table,
            "rest_type": "Casual Dining",
            "dish_liked": "Biryani",
            "city": location,
            "is_near_duplicate": False,
        },
        "distance": distance,
    }


def _prefs(**kwargs) -> UserPreference:
    return UserPreference(**kwargs)


def _run(prefs: UserPreference, raw: list[dict]) -> list:
    """Run recommend() with retrieve() mocked to return `raw`."""
    with patch("src.engine.engine.retrieve", return_value=raw):
        return recommend(prefs)


def _names(candidates) -> list[str]:
    return [c.name for c in candidates]


# ---------------------------------------------------------------------------
# No filters — all retrieved results pass through to ranker
# ---------------------------------------------------------------------------

class TestNoFilters:
    def test_all_results_reach_ranker(self):
        raw = [
            _make_result("Restaurant A", "Koramangala"),
            _make_result("Restaurant B", "Indiranagar"),
            _make_result("Restaurant C", "Banaswadi"),
        ]
        candidates = _run(_prefs(), raw)
        assert len(candidates) == 3

    def test_unrated_restaurant_included_without_rating_filter(self):
        raw = [
            _make_result("Rated Place", "Koramangala", rate=4.2),
            _make_result("Unrated Place", "Koramangala", rate=0.0),
        ]
        candidates = _run(_prefs(), raw)
        assert "Unrated Place" in _names(candidates)

    def test_empty_retrieve_returns_empty(self):
        candidates = _run(_prefs(), [])
        assert candidates == []


# ---------------------------------------------------------------------------
# Location filter
# ---------------------------------------------------------------------------

class TestLocationFilter:
    def test_keeps_exact_location_match(self):
        raw = [_make_result("Good Place", "Koramangala")]
        candidates = _run(_prefs(location="Koramangala"), raw)
        assert "Good Place" in _names(candidates)

    def test_removes_non_matching_location(self):
        raw = [
            _make_result("In Koramangala", "Koramangala"),
            _make_result("In Indiranagar", "Indiranagar"),
        ]
        candidates = _run(_prefs(location="Koramangala"), raw)
        assert _names(candidates) == ["In Koramangala"]

    def test_sub_location_matches_parent(self):
        """'Koramangala 5th Block' should match filter 'Koramangala'."""
        raw = [_make_result("Block Place", "Koramangala 5th Block")]
        candidates = _run(_prefs(location="Koramangala"), raw)
        assert "Block Place" in _names(candidates)

    def test_location_match_is_case_insensitive(self):
        raw = [_make_result("Place", "KORAMANGALA")]
        candidates = _run(_prefs(location="koramangala"), raw)
        assert "Place" in _names(candidates)

    def test_no_location_match_returns_empty(self):
        """No fallback — strict filter means empty result when nothing matches."""
        raw = [
            _make_result("A", "Indiranagar"),
            _make_result("B", "Whitefield"),
        ]
        candidates = _run(_prefs(location="Banaswadi"), raw)
        assert candidates == []

    def test_no_fallback_to_unfiltered_when_location_empty(self):
        """The old (buggy) fallback would return all restaurants — this must not happen."""
        raw = [
            _make_result("Wrong Area 1", "Indiranagar"),
            _make_result("Wrong Area 2", "Whitefield"),
        ]
        candidates = _run(_prefs(location="Banaswadi"), raw)
        assert len(candidates) == 0  # strict — not 2


# ---------------------------------------------------------------------------
# Cuisine filter
# ---------------------------------------------------------------------------

class TestCuisineFilter:
    def test_keeps_matching_cuisine(self):
        raw = [_make_result("Italian Place", "Koramangala", cuisine_str="Italian, Continental")]
        candidates = _run(_prefs(cuisine=["Italian"]), raw)
        assert "Italian Place" in _names(candidates)

    def test_removes_non_matching_cuisine(self):
        raw = [
            _make_result("Italian Place", "Koramangala", cuisine_str="Italian"),
            _make_result("Chinese Place", "Koramangala", cuisine_str="Chinese"),
        ]
        candidates = _run(_prefs(cuisine=["Italian"]), raw)
        assert _names(candidates) == ["Italian Place"]

    def test_multiple_cuisines_any_match_is_enough(self):
        raw = [
            _make_result("North Indian Place", "Koramangala", cuisine_str="North Indian"),
            _make_result("South Indian Place", "Koramangala", cuisine_str="South Indian"),
            _make_result("Chinese Place", "Koramangala", cuisine_str="Chinese"),
        ]
        candidates = _run(_prefs(cuisine=["North Indian", "South Indian"]), raw)
        names = _names(candidates)
        assert "North Indian Place" in names
        assert "South Indian Place" in names
        assert "Chinese Place" not in names

    def test_cuisine_match_is_substring_case_insensitive(self):
        """'arabian' filter should match 'Arabian, Shawarma' cuisine_str."""
        raw = [_make_result("Arab Place", "Banaswadi", cuisine_str="Arabian, Shawarma")]
        candidates = _run(_prefs(cuisine=["Arabian"]), raw)
        assert "Arab Place" in _names(candidates)

    def test_no_cuisine_match_returns_empty(self):
        """No fallback — strict filter."""
        raw = [
            _make_result("Italian Place", "Koramangala", cuisine_str="Italian"),
            _make_result("Chinese Place", "Koramangala", cuisine_str="Chinese"),
        ]
        candidates = _run(_prefs(cuisine=["Arabian"]), raw)
        assert candidates == []

    def test_no_fallback_to_unfiltered_when_cuisine_empty(self):
        """The old (buggy) fallback would return all restaurants — must not happen."""
        raw = [
            _make_result("Italian Place", "Koramangala", cuisine_str="Italian"),
            _make_result("Chinese Place", "Koramangala", cuisine_str="Chinese"),
        ]
        candidates = _run(_prefs(cuisine=["Arabian"]), raw)
        assert len(candidates) == 0  # strict — not 2


# ---------------------------------------------------------------------------
# Rating filter
# ---------------------------------------------------------------------------

class TestRatingFilter:
    def test_keeps_restaurant_at_threshold(self):
        raw = [_make_result("Exactly 4", "Koramangala", rate=4.0)]
        candidates = _run(_prefs(min_rating=4.0), raw)
        assert "Exactly 4" in _names(candidates)

    def test_keeps_restaurant_above_threshold(self):
        raw = [_make_result("High Rated", "Koramangala", rate=4.5)]
        candidates = _run(_prefs(min_rating=4.0), raw)
        assert "High Rated" in _names(candidates)

    def test_removes_restaurant_below_threshold(self):
        raw = [
            _make_result("High Rated", "Koramangala", rate=4.5),
            _make_result("Low Rated", "Koramangala", rate=3.0),
        ]
        candidates = _run(_prefs(min_rating=4.0), raw)
        assert "Low Rated" not in _names(candidates)
        assert "High Rated" in _names(candidates)

    def test_unrated_restaurant_excluded_when_rating_filter_set(self):
        """rate=0.0 means unrated — must be excluded when min_rating is set."""
        raw = [
            _make_result("Rated Place", "Koramangala", rate=4.2),
            _make_result("Unrated Place", "Koramangala", rate=0.0),
        ]
        candidates = _run(_prefs(min_rating=3.5), raw)
        assert "Unrated Place" not in _names(candidates)
        assert "Rated Place" in _names(candidates)

    def test_unrated_restaurant_included_when_no_rating_filter(self):
        """Without min_rating, unrated places should still appear."""
        raw = [_make_result("Unrated Place", "Koramangala", rate=0.0)]
        candidates = _run(_prefs(), raw)
        assert "Unrated Place" in _names(candidates)

    def test_no_rating_filter_keeps_all_ratings(self):
        raw = [
            _make_result("Low", "Koramangala", rate=2.0),
            _make_result("Mid", "Koramangala", rate=3.5),
            _make_result("High", "Koramangala", rate=4.8),
        ]
        candidates = _run(_prefs(), raw)
        assert len(candidates) == 3


# ---------------------------------------------------------------------------
# The "Buff Buffet Buff" bug — meal_type filter + location + cuisine
# ---------------------------------------------------------------------------

class TestBuffetBugRegression:
    """Regression for the bug where adding a meal_type filter caused unrelated
    restaurants to appear because location/cuisine fallback silently dropped."""

    def _raw_pool(self):
        """A pool of 'Buffet' meal_type restaurants (as if ChromaDB returned them)."""
        return [
            _make_result(
                "Asia Kitchen",
                "Banaswadi",
                cuisine_str="Arabian, Chinese",
                meal_type="Buffet",
                rate=4.1,
            ),
            _make_result(
                "Buff Buffet Buff",
                "Indiranagar",           # NOT Banaswadi
                cuisine_str="Continental",  # NOT Arabian
                meal_type="Buffet",
                rate=3.8,
            ),
            _make_result(
                "The Grand Buffet",
                "Whitefield",            # NOT Banaswadi
                cuisine_str="South Indian",
                meal_type="Buffet",
                rate=4.0,
            ),
        ]

    def test_location_filter_excludes_non_banaswadi_buffet_restaurants(self):
        candidates = _run(_prefs(location="Banaswadi"), self._raw_pool())
        names = _names(candidates)
        assert "Buff Buffet Buff" not in names
        assert "The Grand Buffet" not in names
        assert "Asia Kitchen" in names

    def test_cuisine_filter_excludes_non_arabian_buffet_restaurants(self):
        candidates = _run(_prefs(cuisine=["Arabian"]), self._raw_pool())
        names = _names(candidates)
        assert "Buff Buffet Buff" not in names
        assert "The Grand Buffet" not in names
        assert "Asia Kitchen" in names

    def test_location_and_cuisine_combined_returns_only_exact_match(self):
        candidates = _run(
            _prefs(location="Banaswadi", cuisine=["Arabian"]),
            self._raw_pool(),
        )
        assert _names(candidates) == ["Asia Kitchen"]

    def test_no_match_after_location_cuisine_returns_empty_not_full_pool(self):
        """Old bug: fallback returned all 3 restaurants. Must return 0."""
        candidates = _run(
            _prefs(location="Banaswadi", cuisine=["Italian"]),  # no Italian in Banaswadi
            self._raw_pool(),
        )
        assert candidates == []


# ---------------------------------------------------------------------------
# Combined filters
# ---------------------------------------------------------------------------

class TestCombinedFilters:
    def _raw(self):
        return [
            _make_result("Perfect Match", "Koramangala", cuisine_str="Italian",
                         rate=4.5, approx_cost=700),
            _make_result("Wrong Location", "Indiranagar", cuisine_str="Italian",
                         rate=4.5, approx_cost=700),
            _make_result("Wrong Cuisine", "Koramangala", cuisine_str="Chinese",
                         rate=4.5, approx_cost=700),
            _make_result("Low Rating", "Koramangala", cuisine_str="Italian",
                         rate=2.5, approx_cost=700),
            _make_result("Unrated", "Koramangala", cuisine_str="Italian",
                         rate=0.0, approx_cost=700),
        ]

    def test_location_and_cuisine_and_rating_all_applied(self):
        candidates = _run(
            _prefs(location="Koramangala", cuisine=["Italian"], min_rating=4.0),
            self._raw(),
        )
        assert _names(candidates) == ["Perfect Match"]

    def test_unrated_excluded_when_rating_filter_and_location_set(self):
        candidates = _run(
            _prefs(location="Koramangala", min_rating=3.0),
            self._raw(),
        )
        names = _names(candidates)
        assert "Unrated" not in names

    def test_filters_applied_in_sequence_not_short_circuited(self):
        """All three filters must be active — not the first one that matches."""
        candidates = _run(
            _prefs(location="Koramangala", cuisine=["Italian"], min_rating=4.0),
            self._raw(),
        )
        # Only Perfect Match satisfies all three
        assert len(candidates) == 1
        assert candidates[0].name == "Perfect Match"

    def test_top_n_respected(self):
        raw = [
            _make_result(f"Place {i}", "Koramangala", cuisine_str="Italian",
                         rate=4.0, distance=0.1 * i)
            for i in range(10)
        ]
        candidates = _run(_prefs(location="Koramangala", cuisine=["Italian"]), raw)
        assert len(candidates) <= 5  # DEFAULT_FINAL_TOP_N = 5


# ---------------------------------------------------------------------------
# Meal type filter
# ---------------------------------------------------------------------------

class TestMealTypeFilter:
    """meal_type must be a POST-filter (not a ChromaDB hard filter).

    The key requirement: adding a meal_type filter must only NARROW results.
    It must never cause a restaurant to appear that was absent without the filter
    (the 'extra restaurant' bug caused by changing the ChromaDB retrieval pool).
    """

    def _raw(self):
        return [
            _make_result("Buffet Place", "Koramangala", cuisine_str="Chinese",
                         meal_type="Buffet", rate=4.5),
            _make_result("Dine-out Place", "Koramangala", cuisine_str="Chinese",
                         meal_type="Dine-out", rate=4.2),
            _make_result("Delivery Place", "Koramangala", cuisine_str="Chinese",
                         meal_type="Delivery", rate=3.8),
        ]

    def test_meal_type_filter_keeps_matching_meal_type(self):
        candidates = _run(_prefs(meal_type="Buffet"), self._raw())
        assert _names(candidates) == ["Buffet Place"]

    def test_meal_type_filter_removes_non_matching(self):
        candidates = _run(_prefs(meal_type="Buffet"), self._raw())
        names = _names(candidates)
        assert "Dine-out Place" not in names
        assert "Delivery Place" not in names

    def test_no_meal_type_filter_includes_all_types(self):
        """Without meal_type filter, all meal types should pass through."""
        candidates = _run(_prefs(), self._raw())
        assert len(candidates) == 3

    def test_meal_type_filter_is_case_insensitive(self):
        candidates = _run(_prefs(meal_type="buffet"), self._raw())
        assert "Buffet Place" in _names(candidates)

    def test_no_meal_type_match_returns_empty(self):
        candidates = _run(_prefs(meal_type="Desserts"), self._raw())
        assert candidates == []

    def test_adding_meal_type_can_only_narrow_not_expand(self):
        """The 'extra restaurant' bug: adding meal_type=Buffet must not return
        MORE restaurants than without the filter on the same pool."""
        without_filter = _run(_prefs(location="Koramangala", cuisine=["Chinese"]),
                              self._raw())
        with_filter = _run(_prefs(location="Koramangala", cuisine=["Chinese"],
                                  meal_type="Buffet"), self._raw())
        # With filter must be a strict subset of without filter
        assert len(with_filter) <= len(without_filter)
        filter_names = set(_names(with_filter))
        no_filter_names = set(_names(without_filter))
        assert filter_names.issubset(no_filter_names)

    def test_meal_type_combined_with_location_and_cuisine(self):
        raw = [
            _make_result("Match", "Koramangala", cuisine_str="Chinese", meal_type="Buffet"),
            _make_result("Wrong Type", "Koramangala", cuisine_str="Chinese", meal_type="Dine-out"),
            _make_result("Wrong Location", "Indiranagar", cuisine_str="Chinese", meal_type="Buffet"),
        ]
        candidates = _run(_prefs(location="Koramangala", cuisine=["Chinese"],
                                 meal_type="Buffet"), raw)
        assert _names(candidates) == ["Match"]

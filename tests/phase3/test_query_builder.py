"""Phase 3 — Query Builder tests.

Verifies that build_query produces the correct semantic query string and
ChromaDB where-clause filters for every combination of UserPreference fields.
"""

from __future__ import annotations

import pytest

from src.preferences.models import UserPreference
from src.preferences.query_builder import build_query


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prefs(**kwargs) -> UserPreference:
    return UserPreference(**kwargs)


# ---------------------------------------------------------------------------
# Semantic query construction
# ---------------------------------------------------------------------------

class TestSemanticQuery:
    def test_no_prefs_gives_fallback_query(self):
        query, _ = build_query(_prefs())
        assert query == "restaurant"

    def test_cuisine_appears_in_query(self):
        query, _ = build_query(_prefs(cuisine=["Italian"]))
        assert "italian" in query.lower()

    def test_multiple_cuisines_in_query(self):
        query, _ = build_query(_prefs(cuisine=["Italian", "Chinese"]))
        assert "italian" in query.lower()
        assert "chinese" in query.lower()

    def test_location_appears_in_query(self):
        query, _ = build_query(_prefs(location="Koramangala"))
        assert "koramangala" in query.lower()

    def test_meal_type_appears_in_query(self):
        query, _ = build_query(_prefs(meal_type="Buffet"))
        assert "buffet" in query.lower()

    def test_max_price_appears_in_query(self):
        query, _ = build_query(_prefs(max_price=800))
        assert "800" in query

    def test_min_rating_appears_in_query(self):
        query, _ = build_query(_prefs(min_rating=4.0))
        assert "4.0" in query

    def test_online_order_true_appears_in_query(self):
        query, _ = build_query(_prefs(online_order=True))
        assert "online" in query.lower()

    def test_book_table_true_appears_in_query(self):
        query, _ = build_query(_prefs(book_table=True))
        assert "table" in query.lower() or "booking" in query.lower()

    def test_free_text_appended_to_query(self):
        query, _ = build_query(_prefs(free_text="rooftop with live music"))
        assert "rooftop" in query.lower()

    def test_all_fields_produce_non_empty_query(self):
        query, _ = build_query(_prefs(
            cuisine=["North Indian"],
            location="Indiranagar",
            meal_type="Dine-out",
            max_price=1000,
            min_rating=3.5,
            online_order=True,
            book_table=True,
            free_text="anniversary dinner",
        ))
        assert len(query) > 0


# ---------------------------------------------------------------------------
# ChromaDB filter construction
# ---------------------------------------------------------------------------

class TestFilters:
    def test_no_prefs_gives_empty_filters(self):
        _, filters = build_query(_prefs())
        assert filters == {}

    # --- cuisine: semantic only, never in hard filters ---
    def test_cuisine_not_in_hard_filters(self):
        _, filters = build_query(_prefs(cuisine=["Italian"]))
        assert "cuisine" not in str(filters)

    # --- location: semantic only, never in hard filters ---
    def test_location_not_in_hard_filters(self):
        _, filters = build_query(_prefs(location="Koramangala"))
        assert "location" not in str(filters)

    # --- min_rating: ChromaDB hard filter (rate >= threshold) ---
    def test_min_rating_in_hard_filters(self):
        _, filters = build_query(_prefs(min_rating=4.0))
        filter_str = str(filters)
        assert "rate" in filter_str

    def test_min_rating_gte_value(self):
        _, filters = build_query(_prefs(min_rating=4.0))
        # Single condition — no $and wrapper
        assert filters == {"rate": {"$gte": 4.0}}

    # --- meal_type: ChromaDB hard filter (exact eq after capitalize) ---
    def test_meal_type_in_hard_filters(self):
        _, filters = build_query(_prefs(meal_type="Buffet"))
        assert "meal_type" in str(filters)

    def test_meal_type_only_produces_filter(self):
        _, filters = build_query(_prefs(meal_type="Dine-out"))
        assert filters == {"meal_type": {"$eq": "Dine-out"}}

    def test_meal_type_value_is_capitalized(self):
        """Lower-case input must be normalized to title-first casing for ChromaDB."""
        _, filters = build_query(_prefs(meal_type="buffet"))
        assert filters == {"meal_type": {"$eq": "Buffet"}}

    # --- max_price: two conditions (lte + gt 0) ---
    def test_max_price_produces_two_conditions(self):
        _, filters = build_query(_prefs(max_price=800))
        # Single condition wraps as $and when combined; just one price → still two conditions
        assert "$and" in filters
        conds = filters["$and"]
        keys = [list(c.keys())[0] for c in conds]
        assert keys.count("approx_cost") == 2

    def test_max_price_lte_value(self):
        _, filters = build_query(_prefs(max_price=800))
        lte_cond = next(c for c in filters["$and"] if c["approx_cost"].get("$lte"))
        assert lte_cond["approx_cost"]["$lte"] == 800

    def test_max_price_gt_zero_excludes_unknown_cost(self):
        _, filters = build_query(_prefs(max_price=800))
        gt_cond = next(c for c in filters["$and"] if "$gt" in c.get("approx_cost", {}))
        assert gt_cond["approx_cost"]["$gt"] == 0

    # --- online_order ---
    def test_online_order_true_in_hard_filters(self):
        _, filters = build_query(_prefs(online_order=True))
        assert filters == {"online_order": {"$eq": True}}

    def test_online_order_false_in_hard_filters(self):
        _, filters = build_query(_prefs(online_order=False))
        assert filters == {"online_order": {"$eq": False}}

    # --- book_table ---
    def test_book_table_true_in_hard_filters(self):
        _, filters = build_query(_prefs(book_table=True))
        assert filters == {"book_table": {"$eq": True}}

    # --- multiple hard-filter fields → $and ---
    def test_multiple_conditions_wrapped_in_and(self):
        _, filters = build_query(_prefs(online_order=True, book_table=True))
        assert "$and" in filters

    def test_and_contains_online_order_and_book_table(self):
        _, filters = build_query(_prefs(online_order=True, book_table=True))
        keys = [list(c.keys())[0] for c in filters["$and"]]
        assert "online_order" in keys
        assert "book_table" in keys

    # --- single hard-filter field: no $and wrapper ---
    def test_single_condition_not_wrapped_in_and(self):
        _, filters = build_query(_prefs(online_order=True))
        assert "$and" not in filters

    # --- meal_type + online_order → both appear in $and ---
    def test_meal_type_with_online_order_both_in_hard_filters(self):
        _, filters = build_query(_prefs(meal_type="Buffet", online_order=True))
        assert "$and" in filters
        keys = [list(c.keys())[0] for c in filters["$and"]]
        assert "meal_type" in keys
        assert "online_order" in keys

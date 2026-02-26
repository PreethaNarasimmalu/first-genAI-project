"""Phase 3 — Query Builder.

Translates a UserPreference into two artefacts consumed by Phase 4:

  1. A natural-language *semantic query* string for vector similarity search.
  2. A ChromaDB *where* clause dict for hard metadata filtering.

Usage:
    from src.preferences.query_builder import build_query

    query, filters = build_query(prefs)
"""

from __future__ import annotations

from typing import Any

from src.preferences.models import UserPreference


def build_query(prefs: UserPreference) -> tuple[str, dict[str, Any]]:
    """Build a semantic query string and ChromaDB metadata filter dict.

    Cuisine is intentionally kept only in the semantic query (not the
    hard filter) because cuisine_str is a free-form comma-separated field
    and substring matching is better handled by the embedding similarity
    than by an exact equality filter.

    Args:
        prefs: Validated UserPreference instance.

    Returns:
        Tuple of (semantic_query, filters) where:
            - semantic_query is a plain-English string ready for embedding.
            - filters is a ChromaDB ``where`` clause (empty dict = no filter).
    """
    # --- Semantic query -------------------------------------------------------
    parts: list[str] = []

    if prefs.cuisine:
        parts.append(f"{', '.join(prefs.cuisine)} cuisine")
    if prefs.location:
        parts.append(f"in {prefs.location}")
    if prefs.meal_type:
        parts.append(f"{prefs.meal_type} dining")
    if prefs.max_price is not None:
        parts.append(f"budget under ₹{prefs.max_price} for two")
    if prefs.min_rating is not None:
        parts.append(f"rated at least {prefs.min_rating} out of 5")
    if prefs.online_order is True:
        parts.append("online order available")
    if prefs.book_table is True:
        parts.append("table booking available")
    if prefs.free_text:
        parts.append(prefs.free_text)

    semantic_query = " ".join(parts) if parts else "restaurant"

    # --- ChromaDB metadata filter --------------------------------------------
    conditions: list[dict[str, Any]] = []

    # Location is intentionally excluded from hard filters because the dataset
    # has sub-location values like "Koramangala 5th Block" that won't match an
    # exact equality on "Koramangala". Substring post-filtering is applied in
    # engine.py instead, matching the same pattern used for cuisine.
    if prefs.max_price is not None:
        # approx_cost == 0 means "unknown" (stored as 0 fallback); exclude those
        conditions.append({"approx_cost": {"$lte": prefs.max_price}})
        conditions.append({"approx_cost": {"$gt": 0}})
    if prefs.min_rating is not None:
        conditions.append({"rate": {"$gte": prefs.min_rating}})
    if prefs.online_order is not None:
        conditions.append({"online_order": {"$eq": prefs.online_order}})
    if prefs.book_table is not None:
        conditions.append({"book_table": {"$eq": prefs.book_table}})
    if prefs.meal_type:
        conditions.append({"meal_type": {"$eq": prefs.meal_type}})

    if len(conditions) == 0:
        filters: dict[str, Any] = {}
    elif len(conditions) == 1:
        filters = conditions[0]
    else:
        filters = {"$and": conditions}

    return semantic_query, filters

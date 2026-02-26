"""Phase 4 — Recommendation Engine orchestrator.

Ties together Phase 3 (query building) and Phase 4 (retrieval + ranking)
into a single ``recommend()`` call that accepts a UserPreference and returns
a ranked list of RestaurantCandidate objects ready for the LLM in Phase 5.

Usage:
    from src.preferences.parser import parse_preferences
    from src.engine.engine import recommend

    prefs = parse_preferences({"cuisine": ["Italian"], "location": "Indiranagar"})
    candidates = recommend(prefs)
    for c in candidates:
        print(c.name, c.score)
"""

from __future__ import annotations

import chromadb

from src.engine.ranker import RestaurantCandidate, rank
from src.engine.retriever import DEFAULT_RETRIEVAL_TOP_K, retrieve
from src.preferences.models import UserPreference
from src.preferences.query_builder import build_query

DEFAULT_FINAL_TOP_N = 5


def recommend(
    prefs: UserPreference,
    collection: chromadb.Collection | None = None,
    retrieval_top_k: int = DEFAULT_RETRIEVAL_TOP_K,
    final_top_n: int = DEFAULT_FINAL_TOP_N,
) -> list[RestaurantCandidate]:
    """Run the full retrieval + ranking pipeline for a user preference.

    Steps:
        1. Build semantic query + ChromaDB filters from preferences (Phase 3).
        2. Embed query and retrieve top-K candidates from ChromaDB (Phase 4 retriever).
        3. Deduplicate, score, and rank candidates (Phase 4 ranker).

    Args:
        prefs:            Validated UserPreference from Phase 3.
        collection:       ChromaDB collection. If None, loads the default
                          persistent collection automatically.
        retrieval_top_k:  Number of raw results to fetch before ranking.
                          Higher values give more deduplication headroom.
        final_top_n:      Number of top candidates to return after ranking.

    Returns:
        List of up to ``final_top_n`` RestaurantCandidate objects, sorted
        by composite score descending.
    """
    semantic_query, filters = build_query(prefs)

    raw_results = retrieve(
        semantic_query,
        filters,
        collection=collection,
        top_k=retrieval_top_k,
    )

    # Apply cuisine post-filter on raw results (cuisine is a free-text
    # comma-separated field so substring matching is more reliable than
    # an exact ChromaDB equality filter).
    if prefs.cuisine:
        cuisine_lower = [c.lower() for c in prefs.cuisine]
        cuisine_matched = [
            r for r in raw_results
            if any(
                cu in r["metadata"].get("cuisine_str", "").lower()
                for cu in cuisine_lower
            )
        ]
        if cuisine_matched:
            raw_results = cuisine_matched

    return rank(raw_results, max_price=prefs.max_price, top_n=final_top_n)

"""Phase 4 — Ranker.

Takes raw ChromaDB results, deduplicates them, computes a composite score
for each unique candidate, and returns the top-N as RestaurantCandidate objects.

Composite score formula (weights sum to 1.0):
    score = 0.4 × semantic_similarity
          + 0.3 × normalised_rating
          + 0.2 × normalised_votes
          + 0.1 × price_fit_score

Usage:
    from src.engine.ranker import rank, RestaurantCandidate

    candidates = rank(raw_results, max_price=800, top_n=5)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass
class RestaurantCandidate:
    """A ranked restaurant candidate returned by the recommendation engine.

    Attributes:
        name:         Restaurant name.
        location:     Neighbourhood / area.
        cuisine_str:  Comma-separated cuisine types.
        rest_type:    Restaurant category (Casual Dining, Café…).
        rate:         Aggregate rating out of 5.
        votes:        Number of reviews.
        approx_cost:  Approximate cost for two people (INR).
        online_order: Whether online ordering is available.
        book_table:   Whether table booking is available.
        dish_liked:   Popular dishes mentioned by reviewers.
        meal_type:    Meal context (Dine-out, Delivery, Buffet…).
        city:         Broader city / area.
        score:        Composite recommendation score (higher = better).
        distance:     Raw ChromaDB cosine distance (lower = more similar).
    """

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


# ---------------------------------------------------------------------------
# Internal scoring helpers
# ---------------------------------------------------------------------------

def _semantic_similarity(distance: float) -> float:
    """Convert ChromaDB cosine distance to a [0, 1] similarity score.

    ChromaDB cosine distance = 1 − cosine_similarity, ranging [0, 2].
    We map it to [0, 1] where 1 means identical.
    """
    return max(0.0, 1.0 - distance / 2.0)


def _normalise_votes(votes_list: list[int]) -> list[float]:
    """Log-scale normalise a list of vote counts to [0, 1].

    Log scale prevents restaurants with extremely high vote counts from
    dominating the popularity dimension.
    """
    if not votes_list:
        return []
    max_log = math.log1p(max(votes_list)) or 1.0
    return [math.log1p(v) / max_log for v in votes_list]


def _price_fit(cost: int, max_price: int | None) -> float:
    """Score how well the cost fits the user's budget.

    Returns 1.0 if no budget constraint or within budget.
    Decays proportionally if over budget (never below 0).
    """
    if max_price is None or cost == 0:
        return 1.0
    if cost <= max_price:
        return 1.0
    return max(0.0, max_price / cost)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rank(
    raw_results: list[dict[str, Any]],
    max_price: int | None = None,
    top_n: int = 5,
) -> list[RestaurantCandidate]:
    """Deduplicate, score, and rank raw ChromaDB results.

    Deduplication keeps the copy of each (name, location) pair with the
    lowest ChromaDB distance (i.e. most semantically similar to the query).

    Args:
        raw_results: List of dicts from ``similarity_search``.
        max_price:   User's budget for two (INR), used in price_fit scoring.
        top_n:       Number of top candidates to return.

    Returns:
        List of up to ``top_n`` RestaurantCandidate objects, sorted by
        composite score descending.
    """
    # --- Deduplication -------------------------------------------------------
    seen: dict[tuple[str, str], dict[str, Any]] = {}
    for r in raw_results:
        m = r["metadata"]
        key = (m.get("name", "").strip().lower(), m.get("location", "").strip().lower())
        if key not in seen or r["distance"] < seen[key]["distance"]:
            seen[key] = r

    unique = list(seen.values())

    # --- Normalise votes across the candidate set ----------------------------
    votes_list = [int(r["metadata"].get("votes", 0)) for r in unique]
    norm_votes = _normalise_votes(votes_list)

    # --- Build scored candidates ---------------------------------------------
    candidates: list[RestaurantCandidate] = []
    for r, nv in zip(unique, norm_votes):
        m = r["metadata"]

        sem_sim = _semantic_similarity(r["distance"])
        norm_rating = float(m.get("rate", 0.0)) / 5.0
        price_score = _price_fit(int(m.get("approx_cost", 0)), max_price)

        score = (
            0.4 * sem_sim
            + 0.3 * norm_rating
            + 0.2 * nv
            + 0.1 * price_score
        )

        candidates.append(RestaurantCandidate(
            name=str(m.get("name", "")),
            location=str(m.get("location", "")),
            cuisine_str=str(m.get("cuisine_str", "")),
            rest_type=str(m.get("rest_type", "")),
            rate=float(m.get("rate", 0.0)),
            votes=int(m.get("votes", 0)),
            approx_cost=int(m.get("approx_cost", 0)),
            online_order=bool(m.get("online_order", False)),
            book_table=bool(m.get("book_table", False)),
            dish_liked=str(m.get("dish_liked", "")),
            meal_type=str(m.get("meal_type", "")),
            city=str(m.get("city", "")),
            score=round(score, 4),
            distance=round(r["distance"], 4),
        ))

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:top_n]

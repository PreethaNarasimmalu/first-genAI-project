"""Phase 4 smoke test — Retriever, Ranker, and Engine.

No API keys required.

Tests (offline — no network needed):
  1. Ranker deduplicates identical (name, location) pairs
  2. Ranker scores and sorts correctly
  3. price_fit penalises over-budget restaurants
  4. Ranker respects top_n limit
  5. Ranker handles empty input gracefully

Test (requires cached HuggingFace model + ChromaDB):
  6. End-to-end recommend() with a real preference
"""

from __future__ import annotations

import traceback

from src.engine.ranker import RestaurantCandidate, _price_fit, _semantic_similarity, rank


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def check(label: str, condition: bool) -> None:
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}")


def _make_result(
    name: str,
    location: str,
    rate: float = 4.0,
    votes: int = 500,
    cost: int = 400,
    distance: float = 0.2,
) -> dict:
    """Build a fake ChromaDB result dict for unit testing."""
    return {
        "id": f"{name}-{location}",
        "document": f"{name} in {location}",
        "distance": distance,
        "metadata": {
            "name": name,
            "location": location,
            "cuisine_str": "South Indian",
            "rest_type": "Casual Dining",
            "rate": rate,
            "votes": votes,
            "approx_cost": cost,
            "online_order": True,
            "book_table": False,
            "dish_liked": "Dosa",
            "meal_type": "Dine-out",
            "city": "Bangalore",
            "is_near_duplicate": False,
        },
    }


# ---------------------------------------------------------------------------
# Offline unit tests
# ---------------------------------------------------------------------------

def test_deduplication() -> None:
    print("Test 1: Deduplication — identical (name, location) keeps closest")
    raw = [
        _make_result("Cafe A", "Koramangala", distance=0.3),
        _make_result("Cafe A", "Koramangala", distance=0.1),  # closer — keep this
        _make_result("Cafe A", "Koramangala", distance=0.5),
        _make_result("Cafe B", "Indiranagar", distance=0.2),
    ]
    results = rank(raw, top_n=10)
    names = [c.name for c in results]
    check("only 2 unique restaurants returned", len(results) == 2)
    check("Cafe A present once", names.count("Cafe A") == 1)
    check("Cafe B present once", names.count("Cafe B") == 1)
    # The kept Cafe A should have distance=0.1
    cafe_a = next(c for c in results if c.name == "Cafe A")
    check("kept closest Cafe A (distance=0.1)", cafe_a.distance == 0.1)


def test_sorting() -> None:
    print("\nTest 2: Sorting — highest score first")
    raw = [
        _make_result("Low Rated", "Loc", rate=1.0, votes=10, cost=200, distance=0.8),
        _make_result("High Rated", "Loc", rate=4.8, votes=2000, cost=200, distance=0.1),
        _make_result("Mid Rated", "Loc2", rate=3.5, votes=300, cost=200, distance=0.4),
    ]
    results = rank(raw, top_n=5)
    check("3 unique results", len(results) == 3)
    check("High Rated is first", results[0].name == "High Rated")
    check("Low Rated is last", results[-1].name == "Low Rated")
    check("scores descending", all(
        results[i].score >= results[i + 1].score for i in range(len(results) - 1)
    ))


def test_price_fit() -> None:
    print("\nTest 3: price_fit scoring")
    check("within budget → 1.0", _price_fit(400, 500) == 1.0)
    check("at budget → 1.0", _price_fit(500, 500) == 1.0)
    check("no budget → 1.0", _price_fit(999, None) == 1.0)
    check("zero cost → 1.0", _price_fit(0, 500) == 1.0)
    fit_over = _price_fit(1000, 500)
    check("2× over budget → 0.5", abs(fit_over - 0.5) < 1e-9)


def test_top_n_limit() -> None:
    print("\nTest 4: top_n limit")
    raw = [_make_result(f"R{i}", f"L{i}") for i in range(20)]
    results = rank(raw, top_n=5)
    check("returns exactly 5", len(results) == 5)


def test_empty_input() -> None:
    print("\nTest 5: Empty input → empty list")
    results = rank([], top_n=5)
    check("empty list returned", results == [])


def test_semantic_similarity() -> None:
    print("\nTest 6: _semantic_similarity helper")
    check("distance=0 → similarity=1.0", _semantic_similarity(0.0) == 1.0)
    check("distance=2 → similarity=0.0", _semantic_similarity(2.0) == 0.0)
    check("distance=1 → similarity=0.5", abs(_semantic_similarity(1.0) - 0.5) < 1e-9)


# ---------------------------------------------------------------------------
# Live end-to-end test (requires cached model + ChromaDB)
# ---------------------------------------------------------------------------

def test_end_to_end() -> None:
    print("\nTest 7: End-to-end recommend() with live ChromaDB")
    try:
        from src.engine.engine import recommend
        from src.preferences.parser import parse_preferences

        prefs = parse_preferences({
            "cuisine": ["Italian"],
            "location": "Indiranagar",
            "max_price": 1200,
            "min_rating": 3.5,
        })

        candidates = recommend(prefs, final_top_n=5)

        check("got candidates", len(candidates) > 0)
        check("no duplicates", len({(c.name, c.location) for c in candidates}) == len(candidates))
        check("scores descending", all(
            candidates[i].score >= candidates[i + 1].score
            for i in range(len(candidates) - 1)
        ))

        print(f"\n  Top {len(candidates)} recommendations for Italian in Indiranagar (≤₹1200, ≥3.5★):")
        print(f"  {'#':<3} {'Name':<35} {'Location':<15} {'Rate':>5} {'Cost':>6} {'Score':>7}")
        print(f"  {'-'*3} {'-'*35} {'-'*15} {'-'*5} {'-'*6} {'-'*7}")
        for i, c in enumerate(candidates, 1):
            print(f"  {i:<3} {c.name:<35} {c.location:<15} {c.rate:>5.1f} {c.approx_cost:>6} {c.score:>7.4f}")

    except Exception as exc:
        if "ProxyError" in type(exc).__name__ or "ProxyError" in str(exc):
            print("  Skipped — no network access to HuggingFace in this environment.")
            print("  (Run on your local machine where the model is cached.)")
        else:
            print("  End-to-end test failed:")
            traceback.print_exc()
            check("end-to-end passed", False)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Phase 4 smoke tests ===\n")
    test_deduplication()
    test_sorting()
    test_price_fit()
    test_top_n_limit()
    test_empty_input()
    test_semantic_similarity()
    test_end_to_end()
    print("\n=== Done ===")

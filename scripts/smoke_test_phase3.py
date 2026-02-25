"""Phase 3 smoke test — preference parsing, validation, and query building.

No API keys required. Tests:
  1. Valid preference → correct semantic query and filter dict
  2. Single-string cuisine is normalised to a list
  3. Invalid min_rating raises ValueError
  4. Invalid max_price raises ValueError
  5. End-to-end: parse → build_query → ChromaDB similarity search
"""

import traceback

from src.preferences.parser import parse_preferences
from src.preferences.query_builder import build_query


def check(label: str, condition: bool) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")


def run_tests() -> None:
    print("=== Phase 3 smoke tests ===\n")

    # ------------------------------------------------------------------
    # Test 1: Full preference → semantic query contains expected tokens
    # ------------------------------------------------------------------
    print("Test 1: Full preference → semantic query")
    prefs = parse_preferences({
        "cuisine": ["South Indian"],
        "location": "Koramangala",
        "max_price": 500,
        "min_rating": 3.5,
        "online_order": True,
        "meal_type": "Delivery",
        "free_text": "good for a quick lunch",
    })
    query, filters = build_query(prefs)
    print(f"  Query   : {query}")
    print(f"  Filters : {filters}")
    check("query contains cuisine", "South Indian" in query)
    check("query contains location", "Koramangala" in query)
    check("query contains budget", "500" in query)
    check("query contains free_text", "quick lunch" in query)
    check("filter has $and", "$and" in filters)
    check("location filter present", any(
        "location" in c for c in filters.get("$and", [])
    ))
    check("approx_cost filter present", any(
        "approx_cost" in c for c in filters.get("$and", [])
    ))

    # ------------------------------------------------------------------
    # Test 2: Cuisine as bare string → normalised to list
    # ------------------------------------------------------------------
    print("\nTest 2: Bare-string cuisine normalisation")
    prefs2 = parse_preferences({"cuisine": "Italian"})
    check("cuisine is list", isinstance(prefs2.cuisine, list))
    check("cuisine value correct", prefs2.cuisine == ["Italian"])

    # ------------------------------------------------------------------
    # Test 3: No filters → empty filter dict, fallback query
    # ------------------------------------------------------------------
    print("\nTest 3: Empty preferences → no filters")
    prefs3 = parse_preferences({})
    query3, filters3 = build_query(prefs3)
    print(f"  Query   : {query3}")
    print(f"  Filters : {filters3}")
    check("query defaults to 'restaurant'", query3 == "restaurant")
    check("filters is empty dict", filters3 == {})

    # ------------------------------------------------------------------
    # Test 4: Invalid min_rating → ValueError
    # ------------------------------------------------------------------
    print("\nTest 4: Invalid min_rating (> 5) → ValueError")
    try:
        parse_preferences({"min_rating": 6.0})
        check("raised ValueError", False)
    except ValueError as e:
        check("raised ValueError", True)
        print(f"  Error   : {e}")

    # ------------------------------------------------------------------
    # Test 5: Invalid max_price → ValueError
    # ------------------------------------------------------------------
    print("\nTest 5: Invalid max_price (<= 0) → ValueError")
    try:
        parse_preferences({"max_price": -100})
        check("raised ValueError", False)
    except ValueError as e:
        check("raised ValueError", True)
        print(f"  Error   : {e}")

    # ------------------------------------------------------------------
    # Test 6: Single-condition preference → no $and wrapper
    # ------------------------------------------------------------------
    print("\nTest 6: Single filter condition → no $and wrapper")
    prefs6 = parse_preferences({"location": "Indiranagar"})
    _, filters6 = build_query(prefs6)
    print(f"  Filters : {filters6}")
    check("no $and wrapper", "$and" not in filters6)
    check("location key present", "location" in filters6)

    # ------------------------------------------------------------------
    # Test 7: End-to-end with ChromaDB (uses live index)
    # ------------------------------------------------------------------
    print("\nTest 7: End-to-end parse → query → ChromaDB search")
    try:
        from src.indexing.embedder import embed_documents
        from src.indexing.vector_store import get_client, get_collection, similarity_search

        prefs7 = parse_preferences({
            "cuisine": ["North Indian"],
            "max_price": 600,
            "min_rating": 3.5,
        })
        query7, filters7 = build_query(prefs7)
        print(f"  Query   : {query7}")
        print(f"  Filters : {filters7}")

        vec = embed_documents([query7])[0]
        client = get_client()
        col = get_collection(client)
        results = similarity_search(col, vec, filters=filters7, top_k=5)

        check("got results", len(results) > 0)
        for r in results:
            m = r["metadata"]
            print(f"    {m['name']} | {m['location']} | rate={m['rate']} | cost={m['approx_cost']}")
    except Exception as exc:
        if "ProxyError" in type(exc).__name__ or "ProxyError" in str(exc):
            print("  Skipped — no network access to HuggingFace in this environment.")
            print("  (Run on your local machine where the model is cached.)")
        else:
            print("  ChromaDB test failed:")
            traceback.print_exc()
            check("end-to-end passed", False)

    print("\n=== Done ===")


if __name__ == "__main__":
    run_tests()

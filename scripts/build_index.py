"""Build script — run once at deploy time (e.g. Render build command).

Downloads the Zomato dataset from Hugging Face, preprocesses it, and
builds the ChromaDB vector index.  Skips automatically if the index is
already populated so re-deploys don't re-embed everything.

Usage:
    python scripts/build_index.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make sure the project root is on sys.path when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocessing import preprocess
from src.data.ingestion import ingest
from src.indexing.indexer import index_dataframe
from src.indexing.vector_store import get_client, get_collection


def main() -> None:
    client = get_client()
    collection = get_collection(client)

    existing = collection.count()
    if existing > 0:
        print(f"Index already contains {existing} documents — skipping rebuild.")
        return

    print("=== Step 1: Download dataset from Hugging Face ===")
    raw_df = ingest()

    print("\n=== Step 2: Preprocess ===")
    clean_df = preprocess(raw_df)

    print("\n=== Step 3: Embed and index into ChromaDB ===")
    total = index_dataframe(clean_df)
    print(f"\nDone — {total} documents indexed.")


if __name__ == "__main__":
    main()

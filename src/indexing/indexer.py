"""Phase 2 — Indexer.

Orchestrates the full embed → upsert pipeline:
  1. Reads the cleaned parquet (or accepts a DataFrame directly)
  2. Converts rows to document strings via embedder.build_documents
  3. Batch-embeds via sentence-transformers
  4. Upserts into ChromaDB in configurable batches

Usage:
    python -m src.indexing.indexer
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

from src.indexing.embedder import build_documents, embed_documents
from src.indexing.vector_store import get_client, get_collection, upsert_documents

CLEAN_PARQUET = Path(__file__).resolve().parents[2] / "data" / "clean" / "restaurants.parquet"
UPSERT_BATCH = 500


def _row_to_metadata(row: pd.Series) -> dict[str, Any]:
    """Extract ChromaDB-compatible metadata from a restaurant row.

    ChromaDB metadata values must be str, int, float, or bool.
    Lists are serialised to comma-separated strings.

    Args:
        row: A pandas Series representing one restaurant record.

    Returns:
        Flat dict of filterable metadata fields.
    """
    cuisines = row.get("cuisines", [])
    if isinstance(cuisines, list):
        cuisine_str = ", ".join(cuisines)
    else:
        cuisine_str = str(cuisines) if cuisines else ""

    meta: dict[str, Any] = {
        "name": str(row.get("name") or ""),
        "location": str(row.get("location") or ""),
        "cuisine_str": cuisine_str,
        "rest_type": str(row.get("rest_type") or ""),
        "meal_type": str(row.get("meal_type") or ""),
        "city": str(row.get("city") or ""),
        "dish_liked": str(row.get("dish_liked") or ""),
        "online_order": bool(row.get("online_order", False)),
        "book_table": bool(row.get("book_table", False)),
        "is_near_duplicate": bool(row.get("is_near_duplicate", False)),
    }

    # Handle pandas nullable float (pd.NA / NaN)
    rate = row.get("rate")
    try:
        rate_val = float(rate)
        meta["rate"] = 0.0 if math.isnan(rate_val) else rate_val
    except (TypeError, ValueError):
        meta["rate"] = 0.0

    cost = row.get("approx_cost")
    try:
        cost_val = float(cost)
        meta["approx_cost"] = 0 if math.isnan(cost_val) else int(cost_val)
    except (TypeError, ValueError):
        meta["approx_cost"] = 0

    votes = row.get("votes")
    try:
        meta["votes"] = int(votes)
    except (TypeError, ValueError):
        meta["votes"] = 0

    return meta


def index_dataframe(
    df: pd.DataFrame,
    db_path: str | Path | None = None,
) -> int:
    """Embed and index all restaurants in a DataFrame.

    Args:
        df: Cleaned restaurant DataFrame from Phase 1.
        db_path: Path for ChromaDB persistent storage. Defaults to data/vectordb/.

    Returns:
        Total number of documents now in the ChromaDB collection.
    """
    print(f"Building document strings for {len(df)} restaurants...")
    documents = build_documents(df)
    ids = [str(i) for i in range(len(df))]

    print("Embedding documents (model downloads on first run)...")
    embeddings = embed_documents(documents)

    metadatas = [_row_to_metadata(row) for _, row in df.iterrows()]

    client = get_client(db_path)
    collection = get_collection(client)

    total = len(ids)
    for start in range(0, total, UPSERT_BATCH):
        end = min(start + UPSERT_BATCH, total)
        print(f"Upserting batch {start + 1}–{end} of {total}...")
        upsert_documents(
            collection,
            ids=ids[start:end],
            documents=documents[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
        )

    count = collection.count()
    print(f"Indexing complete — {count} documents stored in ChromaDB.")
    return count


def run_pipeline(
    parquet_path: str | Path = CLEAN_PARQUET,
    db_path: str | Path | None = None,
) -> int:
    """End-to-end Phase 2 pipeline: parquet → ChromaDB.

    Args:
        parquet_path: Path to the cleaned restaurants parquet from Phase 1.
        db_path: ChromaDB storage directory.

    Returns:
        Number of documents indexed.
    """
    print(f"Loading clean data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} rows.")
    return index_dataframe(df, db_path=db_path)


if __name__ == "__main__":
    run_pipeline()

"""Phase 2 — Document Embedder.

Converts Restaurant rows into natural-language document strings and
produces sentence-level embeddings using sentence-transformers.

Usage:
    from src.indexing.embedder import build_documents, embed_documents
"""

from __future__ import annotations

import pandas as pd

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 64


def build_document(row: pd.Series) -> str:
    """Convert a single restaurant row into a human-readable document string.

    Args:
        row: A pandas Series representing one restaurant record.

    Returns:
        A plain-English description suitable for semantic embedding.
    """
    name = row.get("name") or "Unknown"
    rest_type = row.get("rest_type") or "restaurant"
    location = row.get("location") or "unknown location"
    meal_type = row.get("meal_type") or ""

    cuisines = row.get("cuisines", [])
    if isinstance(cuisines, (list, tuple)):
        cuisine_str = ", ".join(cuisines) if cuisines else "various cuisines"
    else:
        # numpy arrays (from parquet) — convert to list first
        import numpy as np
        if isinstance(cuisines, np.ndarray):
            cuisine_str = ", ".join(cuisines.tolist()) if cuisines.size > 0 else "various cuisines"
        else:
            cuisine_str = str(cuisines) if cuisines else "various cuisines"

    rate = row.get("rate")
    votes = row.get("votes", 0)
    approx_cost = row.get("approx_cost")
    dish_liked = row.get("dish_liked") or "not specified"
    online_order = row.get("online_order", False)
    book_table = row.get("book_table", False)

    # Handle all NA variants: None, np.nan, pd.NA, pd.NaT
    rate_is_na = pd.isna(rate)
    cost_is_na = pd.isna(approx_cost)

    rate_str = f"{rate}/5" if not rate_is_na else "unrated"
    cost_str = f"₹{int(approx_cost)}" if not cost_is_na else "not specified"
    online_str = "Yes" if online_order else "No"
    table_str = "Yes" if book_table else "No"

    meal_str = f" Listed under: {meal_type}." if meal_type else ""

    return (
        f"{name} is a {rest_type} in {location} serving {cuisine_str}. "
        f"Rating: {rate_str} based on {votes} votes. "
        f"Approx cost for two: {cost_str}. "
        f"Popular dishes: {dish_liked}. "
        f"Online order: {online_str}. Table booking: {table_str}."
        f"{meal_str}"
    )


def build_documents(df: pd.DataFrame) -> list[str]:
    """Convert all rows in a DataFrame to document strings.

    Args:
        df: Cleaned restaurant DataFrame from Phase 1.

    Returns:
        List of document strings, one per restaurant row.
    """
    return [build_document(row) for _, row in df.iterrows()]


def embed_documents(
    documents: list[str],
    model_name: str = MODEL_NAME,
    batch_size: int = BATCH_SIZE,
) -> list[list[float]]:
    """Embed a list of document strings.

    Uses ChromaDB's built-in ONNX embedding function (all-MiniLM-L6-v2,
    cached locally) so no network download is required at runtime.

    Args:
        documents: List of text documents to embed.
        model_name: Ignored — kept for API compatibility.
        batch_size: Ignored — ChromaDB batches internally.

    Returns:
        List of embedding vectors (each a list of floats).
    """
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

    ef = DefaultEmbeddingFunction()
    return list(ef(documents))

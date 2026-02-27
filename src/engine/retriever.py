"""Phase 4 — Retriever.

Embeds a semantic query string and runs a similarity search against the
ChromaDB restaurant collection, optionally applying metadata filters built
by Phase 3's query_builder.

Usage:
    from src.engine.retriever import retrieve

    raw_results = retrieve("cosy Italian restaurant", filters, collection=col)
"""

from __future__ import annotations

from typing import Any

import chromadb

from src.indexing.embedder import embed_documents
from src.indexing.vector_store import get_client, get_collection, similarity_search

# Default top_k — overridden by engine.py to collection.count() so that
# all ChromaDB-filtered documents are retrieved before post-filtering.
DEFAULT_RETRIEVAL_TOP_K = 1000


def retrieve(
    semantic_query: str,
    filters: dict[str, Any],
    collection: chromadb.Collection | None = None,
    top_k: int = DEFAULT_RETRIEVAL_TOP_K,
) -> list[dict[str, Any]]:
    """Embed the query and search ChromaDB for candidate restaurants.

    Args:
        semantic_query: Natural-language query built by query_builder.
        filters:        ChromaDB ``where`` clause (empty dict = no filter).
        collection:     ChromaDB collection to search. If None, the default
                        persistent collection is loaded automatically.
        top_k:          Number of raw candidates to retrieve before ranking.

    Returns:
        List of raw ChromaDB result dicts, each with keys:
            ``id``, ``document``, ``metadata``, ``distance``.
    """
    if collection is None:
        client = get_client()
        collection = get_collection(client)

    query_vec = embed_documents([semantic_query])[0]

    # Pass filters only when non-empty (empty dict means no filter)
    chroma_filters = filters if filters else None

    return similarity_search(
        collection,
        query_vec,
        filters=chroma_filters,
        top_k=top_k,
    )

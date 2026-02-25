"""Phase 2 — Vector Store (ChromaDB).

Manages the ChromaDB collection for restaurant embeddings. Provides
helpers for upserting documents and running similarity search.

Usage:
    from src.indexing.vector_store import get_client, get_collection, similarity_search
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb

COLLECTION_NAME = "restaurants"
DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "vectordb"
DEFAULT_TOP_K = 10


def get_client(db_path: str | Path | None = None) -> chromadb.ClientAPI:
    """Return a persistent ChromaDB client.

    Args:
        db_path: Directory for persistent storage. Defaults to data/vectordb/.

    Returns:
        ChromaDB PersistentClient.
    """
    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(path))


def get_collection(
    client: chromadb.ClientAPI,
    name: str = COLLECTION_NAME,
) -> chromadb.Collection:
    """Get or create the restaurants ChromaDB collection.

    The collection uses cosine similarity so that query distances are
    in the range [0, 2] (0 = identical, 2 = opposite).

    Args:
        client: ChromaDB client (persistent or ephemeral).
        name: Collection name.

    Returns:
        ChromaDB Collection.
    """
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def upsert_documents(
    collection: chromadb.Collection,
    ids: list[str],
    documents: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict[str, Any]],
) -> None:
    """Upsert restaurant documents into the ChromaDB collection.

    Uses upsert so repeated runs are idempotent (same id → overwrite).

    Args:
        collection: Target ChromaDB collection.
        ids: Unique string ID per document (e.g. row index as string).
        documents: Raw text document strings.
        embeddings: Precomputed embedding vectors.
        metadatas: Metadata dict per document (used for filtering).
    """
    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )


def similarity_search(
    collection: chromadb.Collection,
    query_embedding: list[float],
    filters: dict[str, Any] | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict[str, Any]]:
    """Run a vector similarity search against the restaurant collection.

    Args:
        collection: ChromaDB collection to search.
        query_embedding: Embedding vector for the search query.
        filters: Optional ChromaDB ``where`` clause for metadata filtering.
                 Example: ``{"location": {"$eq": "Koramangala"}}``
        top_k: Maximum number of results to return.

    Returns:
        List of result dicts, each with keys:
            - ``id``:       Document ID string
            - ``document``: Original document text
            - ``metadata``: Metadata dict stored at upsert time
            - ``distance``: Cosine distance from query (lower = more similar)
    """
    count = collection.count()
    if count == 0:
        return []

    kwargs: dict[str, Any] = {
        "query_embeddings": [query_embedding],
        "n_results": min(top_k, count),
        "include": ["documents", "metadatas", "distances"],
    }
    if filters:
        kwargs["where"] = filters

    results = collection.query(**kwargs)

    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    return [
        {"id": doc_id, "document": doc, "metadata": meta, "distance": dist}
        for doc_id, doc, meta, dist in zip(ids, docs, metas, dists)
    ]

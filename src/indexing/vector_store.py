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
    """Fetch all documents matching *filters* and rank by cosine similarity.

    Uses ``collection.get()`` instead of ``collection.query()`` to avoid the
    SQLite "too many SQL variables" error that occurs when the collection is
    large (ChromaDB's query() builds a WHERE id IN (...) clause that exceeds
    SQLite's variable limit at ~51 k rows).

    All matching documents are fetched with their embeddings, cosine similarity
    is computed via numpy, and results are returned sorted by distance
    (ascending — lower = more similar), capped at ``top_k``.

    Args:
        collection: ChromaDB collection to search.
        query_embedding: Embedding vector for the search query.
        filters: Optional ChromaDB ``where`` clause for metadata filtering.
        top_k: Maximum number of results to return.

    Returns:
        List of result dicts, each with keys:
            - ``id``:       Document ID string
            - ``document``: Original document text
            - ``metadata``: Metadata dict stored at upsert time
            - ``distance``: Cosine distance from query (lower = more similar)
    """
    import numpy as np

    if collection.count() == 0:
        return []

    get_kwargs: dict[str, Any] = {
        "include": ["embeddings", "documents", "metadatas"],
    }
    if filters:
        get_kwargs["where"] = filters

    result = collection.get(**get_kwargs)

    ids: list[str] = result.get("ids") or []
    embeddings = result.get("embeddings")
    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []

    if not ids or embeddings is None or len(embeddings) == 0:
        return []

    # Compute cosine distances: distance = 1 − cosine_similarity
    q = np.array(query_embedding, dtype=np.float32)
    q_norm = q / (np.linalg.norm(q) + 1e-10)

    emb_matrix = np.array(embeddings, dtype=np.float32)
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    emb_matrix /= norms + 1e-10

    distances: np.ndarray = 1.0 - (emb_matrix @ q_norm)

    # Sort ascending (most similar first) and cap at top_k
    order = np.argsort(distances)[:top_k]

    return [
        {
            "id": ids[i],
            "document": documents[i],
            "metadata": metadatas[i],
            "distance": float(distances[i]),
        }
        for i in order
    ]

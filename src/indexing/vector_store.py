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


# ChromaDB's SQLite backend crashes with "too many SQL variables" when a
# single get() call tries to fetch thousands of rows at once (SQLite's
# SQLITE_MAX_VARIABLE_NUMBER limit). Paginating at this size stays safely
# below that limit on every SQLite version.
_GET_BATCH_SIZE = 5_000


def similarity_search(
    collection: chromadb.Collection,
    query_embedding: list[float],
    filters: dict[str, Any] | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict[str, Any]]:
    """Fetch all matching documents in batches, then rank by cosine similarity.

    Paginates ``collection.get()`` with ``limit``/``offset`` so each call
    stays below SQLite's variable limit, regardless of collection size.

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

    # Collect all matching documents across paginated batches
    all_ids: list[str] = []
    all_embeddings: list[list[float]] = []
    all_documents: list[str] = []
    all_metadatas: list[dict[str, Any]] = []

    offset = 0
    while True:
        get_kwargs: dict[str, Any] = {
            "limit": _GET_BATCH_SIZE,
            "offset": offset,
            "include": ["embeddings", "documents", "metadatas"],
        }
        if filters:
            get_kwargs["where"] = filters

        batch = collection.get(**get_kwargs)
        batch_ids: list[str] = batch.get("ids") or []
        if not batch_ids:
            break

        all_ids.extend(batch_ids)
        all_documents.extend(batch.get("documents") or [])
        all_metadatas.extend(batch.get("metadatas") or [])

        batch_embeddings = batch.get("embeddings")
        if batch_embeddings is not None:
            all_embeddings.extend(batch_embeddings)

        offset += len(batch_ids)
        if len(batch_ids) < _GET_BATCH_SIZE:
            break  # final (possibly partial) batch

    if not all_ids or not all_embeddings:
        return []

    # Compute cosine distances: distance = 1 − cosine_similarity
    q = np.array(query_embedding, dtype=np.float32)
    q_norm = q / (np.linalg.norm(q) + 1e-10)

    emb_matrix = np.array(all_embeddings, dtype=np.float32)
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    emb_matrix /= norms + 1e-10

    distances: np.ndarray = 1.0 - (emb_matrix @ q_norm)

    # Sort ascending (most similar first) and cap at top_k
    order = np.argsort(distances)[:top_k]

    return [
        {
            "id": all_ids[i],
            "document": all_documents[i],
            "metadata": all_metadatas[i],
            "distance": float(distances[i]),
        }
        for i in order
    ]

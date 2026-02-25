"""Tests for Phase 2 — vector_store.py."""

import uuid

import pytest
import chromadb

from src.indexing.vector_store import (
    COLLECTION_NAME,
    DEFAULT_TOP_K,
    get_collection,
    similarity_search,
    upsert_documents,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def client() -> chromadb.ClientAPI:
    """Ephemeral in-memory ChromaDB client — no disk writes."""
    return chromadb.EphemeralClient()


@pytest.fixture()
def collection(client: chromadb.ClientAPI) -> chromadb.Collection:
    # Use a unique name per test to guarantee isolation even if ChromaDB's
    # EphemeralClient shares state within the same process.
    unique_name = f"restaurants_{uuid.uuid4().hex[:8]}"
    return client.create_collection(
        name=unique_name,
        metadata={"hnsw:space": "cosine"},
    )


def _fake_embedding(dim: int = 384) -> list[float]:
    """Return a deterministic unit vector of given dimension."""
    import math
    val = 1.0 / math.sqrt(dim)
    return [val] * dim


def _seeded_embedding(seed: int, dim: int = 384) -> list[float]:
    """Return a deterministic but varied embedding based on seed."""
    import math
    return [math.sin(seed + i) for i in range(dim)]


# ---------------------------------------------------------------------------
# get_collection
# ---------------------------------------------------------------------------

class TestGetCollection:
    def test_returns_collection(self, client):
        col = get_collection(client)
        assert col is not None

    def test_collection_name_matches_constant(self, client):
        col = get_collection(client)
        assert col.name == COLLECTION_NAME

    def test_custom_name(self, client):
        col = get_collection(client, name="custom_col")
        assert col.name == "custom_col"

    def test_idempotent_get_or_create(self, client):
        col1 = get_collection(client)
        col2 = get_collection(client)
        assert col1.name == col2.name

    def test_uses_cosine_distance(self, client):
        col = get_collection(client)
        assert col.metadata.get("hnsw:space") == "cosine"


# ---------------------------------------------------------------------------
# upsert_documents
# ---------------------------------------------------------------------------

class TestUpsertDocuments:
    def test_upsert_increases_count(self, collection):
        upsert_documents(
            collection,
            ids=["0"],
            documents=["A test restaurant in Koramangala."],
            embeddings=[_fake_embedding()],
            metadatas=[{"name": "Test", "location": "Koramangala"}],
        )
        assert collection.count() == 1

    def test_upsert_multiple_documents(self, collection):
        n = 5
        upsert_documents(
            collection,
            ids=[str(i) for i in range(n)],
            documents=[f"Restaurant {i}" for i in range(n)],
            embeddings=[_fake_embedding() for _ in range(n)],
            metadatas=[{"name": f"R{i}"} for i in range(n)],
        )
        assert collection.count() == n

    def test_upsert_is_idempotent(self, collection):
        for _ in range(3):
            upsert_documents(
                collection,
                ids=["0"],
                documents=["Same doc"],
                embeddings=[_fake_embedding()],
                metadatas=[{"name": "Same"}],
            )
        assert collection.count() == 1

    def test_upsert_overwrites_metadata(self, collection):
        upsert_documents(
            collection,
            ids=["0"],
            documents=["Original"],
            embeddings=[_fake_embedding()],
            metadatas=[{"name": "Old Name"}],
        )
        upsert_documents(
            collection,
            ids=["0"],
            documents=["Updated"],
            embeddings=[_fake_embedding()],
            metadatas=[{"name": "New Name"}],
        )
        result = collection.get(ids=["0"], include=["metadatas"])
        assert result["metadatas"][0]["name"] == "New Name"


# ---------------------------------------------------------------------------
# similarity_search
# ---------------------------------------------------------------------------

class TestSimilaritySearch:
    def _populate(self, collection, n: int = 5) -> list[str]:
        ids = [str(i) for i in range(n)]
        upsert_documents(
            collection,
            ids=ids,
            documents=[f"Restaurant {i} doc" for i in range(n)],
            embeddings=[_seeded_embedding(i) for i in range(n)],
            metadatas=[
                {
                    "name": f"Restaurant {i}",
                    "location": "Koramangala" if i % 2 == 0 else "Indiranagar",
                    "rate": float(3.0 + i * 0.2),
                }
                for i in range(n)
            ],
        )
        return ids

    def test_returns_list(self, collection):
        self._populate(collection)
        results = similarity_search(collection, _seeded_embedding(0))
        assert isinstance(results, list)

    def test_result_has_required_keys(self, collection):
        self._populate(collection)
        results = similarity_search(collection, _seeded_embedding(0), top_k=1)
        assert len(results) == 1
        r = results[0]
        assert "id" in r
        assert "document" in r
        assert "metadata" in r
        assert "distance" in r

    def test_top_k_limits_results(self, collection):
        self._populate(collection, n=5)
        results = similarity_search(collection, _seeded_embedding(0), top_k=3)
        assert len(results) <= 3

    def test_top_k_one_returns_single(self, collection):
        self._populate(collection)
        results = similarity_search(collection, _seeded_embedding(0), top_k=1)
        assert len(results) == 1

    def test_distance_is_float(self, collection):
        self._populate(collection)
        results = similarity_search(collection, _seeded_embedding(0), top_k=1)
        assert isinstance(results[0]["distance"], float)

    def test_metadata_filter_applied(self, collection):
        self._populate(collection, n=5)
        results = similarity_search(
            collection,
            _seeded_embedding(0),
            filters={"location": {"$eq": "Koramangala"}},
            top_k=10,
        )
        for r in results:
            assert r["metadata"]["location"] == "Koramangala"

    def test_no_filter_returns_all_up_to_top_k(self, collection):
        self._populate(collection, n=5)
        results = similarity_search(collection, _seeded_embedding(0), top_k=10)
        assert len(results) == 5

    def test_empty_collection_returns_empty(self, collection):
        results = similarity_search(collection, _fake_embedding(), top_k=5)
        assert results == []

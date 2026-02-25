"""Tests for Phase 2 — indexer.py."""

import math
import uuid
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import chromadb

from src.indexing.indexer import _row_to_metadata, index_dataframe


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_row(**kwargs) -> pd.Series:
    defaults = {
        "name": "Jalsa",
        "location": "Banashankari",
        "cuisines": ["North Indian", "Mughlai"],
        "rest_type": "Casual Dining",
        "meal_type": "Dine-out",
        "city": "Bangalore",
        "dish_liked": "Biryani",
        "rate": 4.1,
        "votes": 775,
        "approx_cost": 800,
        "online_order": True,
        "book_table": True,
        "is_near_duplicate": False,
    }
    defaults.update(kwargs)
    return pd.Series(defaults)


def _make_df(n: int = 3, **kwargs) -> pd.DataFrame:
    return pd.DataFrame([_make_row(**kwargs).to_dict() for _ in range(n)])


def _fake_embeddings(n: int, dim: int = 384) -> list[list[float]]:
    return np.random.rand(n, dim).tolist()


# ---------------------------------------------------------------------------
# _row_to_metadata
# ---------------------------------------------------------------------------

class TestRowToMetadata:
    def test_returns_dict(self):
        meta = _row_to_metadata(_make_row())
        assert isinstance(meta, dict)

    def test_name_is_string(self):
        meta = _row_to_metadata(_make_row(name="Spice Elephant"))
        assert meta["name"] == "Spice Elephant"

    def test_location_is_string(self):
        meta = _row_to_metadata(_make_row(location="Indiranagar"))
        assert meta["location"] == "Indiranagar"

    def test_cuisines_list_serialised_to_string(self):
        meta = _row_to_metadata(_make_row(cuisines=["Italian", "Mexican"]))
        assert meta["cuisine_str"] == "Italian, Mexican"

    def test_single_cuisine(self):
        meta = _row_to_metadata(_make_row(cuisines=["Thai"]))
        assert meta["cuisine_str"] == "Thai"

    def test_empty_cuisines_gives_empty_string(self):
        meta = _row_to_metadata(_make_row(cuisines=[]))
        assert meta["cuisine_str"] == ""

    def test_rate_is_float(self):
        meta = _row_to_metadata(_make_row(rate=4.1))
        assert isinstance(meta["rate"], float)
        assert meta["rate"] == pytest.approx(4.1)

    def test_none_rate_becomes_zero(self):
        meta = _row_to_metadata(_make_row(rate=None))
        assert meta["rate"] == 0.0

    def test_nan_rate_becomes_zero(self):
        meta = _row_to_metadata(_make_row(rate=float("nan")))
        assert meta["rate"] == 0.0

    def test_approx_cost_is_int(self):
        meta = _row_to_metadata(_make_row(approx_cost=600))
        assert isinstance(meta["approx_cost"], int)
        assert meta["approx_cost"] == 600

    def test_none_cost_becomes_zero(self):
        meta = _row_to_metadata(_make_row(approx_cost=None))
        assert meta["approx_cost"] == 0

    def test_votes_is_int(self):
        meta = _row_to_metadata(_make_row(votes=1000))
        assert isinstance(meta["votes"], int)
        assert meta["votes"] == 1000

    def test_online_order_true(self):
        meta = _row_to_metadata(_make_row(online_order=True))
        assert meta["online_order"] is True

    def test_online_order_false(self):
        meta = _row_to_metadata(_make_row(online_order=False))
        assert meta["online_order"] is False

    def test_book_table_is_bool(self):
        meta = _row_to_metadata(_make_row(book_table=True))
        assert isinstance(meta["book_table"], bool)

    def test_is_near_duplicate_false(self):
        meta = _row_to_metadata(_make_row(is_near_duplicate=False))
        assert meta["is_near_duplicate"] is False

    def test_is_near_duplicate_true(self):
        meta = _row_to_metadata(_make_row(is_near_duplicate=True))
        assert meta["is_near_duplicate"] is True

    def test_all_values_are_chromadb_compatible_types(self):
        meta = _row_to_metadata(_make_row())
        allowed = (str, int, float, bool)
        for key, val in meta.items():
            assert isinstance(val, allowed), f"{key}: {type(val)} is not a ChromaDB-compatible type"


# ---------------------------------------------------------------------------
# index_dataframe
# ---------------------------------------------------------------------------

class TestIndexDataframe:
    @pytest.fixture()
    def ephemeral_client(self):
        return chromadb.EphemeralClient()

    @pytest.fixture()
    def fresh_collection(self, ephemeral_client):
        """A uniquely named, empty collection per test — guarantees isolation."""
        unique_name = f"restaurants_{uuid.uuid4().hex[:8]}"
        return ephemeral_client.create_collection(
            name=unique_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _patch_embed(self, n: int):
        """Patch embed_documents to return fake vectors without loading the model."""
        fake = _fake_embeddings(n)
        return patch("src.indexing.indexer.embed_documents", return_value=fake)

    def _patch_collection(self, collection):
        """Patch get_collection to return the pre-created fresh collection."""
        return patch("src.indexing.indexer.get_collection", return_value=collection)

    def test_returns_correct_count(self, fresh_collection):
        df = _make_df(n=4)
        with self._patch_embed(4), self._patch_collection(fresh_collection):
            count = index_dataframe(df)
        assert count == 4

    def test_documents_stored_in_collection(self, fresh_collection):
        df = _make_df(n=3)
        with self._patch_embed(3), self._patch_collection(fresh_collection):
            index_dataframe(df)
        assert fresh_collection.count() == 3

    def test_ids_are_string_row_indices(self, fresh_collection):
        df = _make_df(n=2)
        with self._patch_embed(2), self._patch_collection(fresh_collection):
            index_dataframe(df)
        result = fresh_collection.get(ids=["0", "1"])
        assert "0" in result["ids"]
        assert "1" in result["ids"]

    def test_metadata_stored_correctly(self, fresh_collection):
        df = _make_df(n=1, name="Spice Hub", location="Koramangala")
        with self._patch_embed(1), self._patch_collection(fresh_collection):
            index_dataframe(df)
        result = fresh_collection.get(ids=["0"], include=["metadatas"])
        meta = result["metadatas"][0]
        assert meta["name"] == "Spice Hub"
        assert meta["location"] == "Koramangala"

    def test_upsert_is_idempotent(self, fresh_collection):
        df = _make_df(n=3)
        with self._patch_embed(3), self._patch_collection(fresh_collection):
            index_dataframe(df)
        with self._patch_embed(3), self._patch_collection(fresh_collection):
            count = index_dataframe(df)
        assert count == 3

    def test_empty_dataframe_indexes_zero(self, fresh_collection):
        df = pd.DataFrame(columns=list(_make_row().index))
        with self._patch_embed(0), self._patch_collection(fresh_collection):
            count = index_dataframe(df)
        assert count == 0

"""Tests for Phase 2 — embedder.py."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.indexing.embedder import (
    MODEL_NAME,
    build_document,
    build_documents,
    embed_documents,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_row(**kwargs) -> pd.Series:
    defaults = {
        "name": "Test Restaurant",
        "rest_type": "Casual Dining",
        "location": "Koramangala",
        "cuisines": ["North Indian", "Chinese"],
        "rate": 4.2,
        "votes": 500,
        "approx_cost": 600,
        "dish_liked": "Butter Chicken",
        "online_order": True,
        "book_table": False,
    }
    defaults.update(kwargs)
    return pd.Series(defaults)


# ---------------------------------------------------------------------------
# build_document
# ---------------------------------------------------------------------------

class TestBuildDocument:
    def test_contains_restaurant_name(self):
        doc = build_document(_make_row(name="Spice Garden"))
        assert "Spice Garden" in doc

    def test_contains_location(self):
        doc = build_document(_make_row(location="Indiranagar"))
        assert "Indiranagar" in doc

    def test_contains_rest_type(self):
        doc = build_document(_make_row(rest_type="Café"))
        assert "Café" in doc

    def test_cuisines_list_joined(self):
        doc = build_document(_make_row(cuisines=["Italian", "Mexican"]))
        assert "Italian" in doc
        assert "Mexican" in doc

    def test_cuisine_string_passthrough(self):
        # If cuisines somehow arrives as a plain string
        doc = build_document(_make_row(cuisines="Thai"))
        assert "Thai" in doc

    def test_rate_formatted(self):
        doc = build_document(_make_row(rate=3.8))
        assert "3.8/5" in doc

    def test_none_rate_shows_unrated(self):
        doc = build_document(_make_row(rate=None))
        assert "unrated" in doc

    def test_nan_rate_shows_unrated(self):
        doc = build_document(_make_row(rate=float("nan")))
        assert "unrated" in doc

    def test_cost_formatted_with_rupee(self):
        doc = build_document(_make_row(approx_cost=800))
        assert "₹800" in doc

    def test_none_cost_shows_not_specified(self):
        doc = build_document(_make_row(approx_cost=None))
        assert "not specified" in doc

    def test_votes_included(self):
        doc = build_document(_make_row(votes=1234))
        assert "1234" in doc

    def test_dish_liked_included(self):
        doc = build_document(_make_row(dish_liked="Biryani"))
        assert "Biryani" in doc

    def test_none_dish_liked_shows_not_specified(self):
        doc = build_document(_make_row(dish_liked=None))
        assert "not specified" in doc

    def test_online_order_true(self):
        doc = build_document(_make_row(online_order=True))
        assert "Online order: Yes" in doc

    def test_online_order_false(self):
        doc = build_document(_make_row(online_order=False))
        assert "Online order: No" in doc

    def test_book_table_true(self):
        doc = build_document(_make_row(book_table=True))
        assert "Table booking: Yes" in doc

    def test_book_table_false(self):
        doc = build_document(_make_row(book_table=False))
        assert "Table booking: No" in doc

    def test_empty_cuisines_list(self):
        doc = build_document(_make_row(cuisines=[]))
        assert "various cuisines" in doc

    def test_missing_rest_type_defaults_to_restaurant(self):
        doc = build_document(_make_row(rest_type=None))
        assert "restaurant" in doc

    def test_returns_string(self):
        doc = build_document(_make_row())
        assert isinstance(doc, str)


# ---------------------------------------------------------------------------
# build_documents
# ---------------------------------------------------------------------------

class TestBuildDocuments:
    def test_returns_list_of_strings(self):
        df = pd.DataFrame([_make_row().to_dict(), _make_row(name="Other").to_dict()])
        docs = build_documents(df)
        assert isinstance(docs, list)
        assert all(isinstance(d, str) for d in docs)

    def test_length_matches_dataframe(self):
        df = pd.DataFrame([_make_row().to_dict() for _ in range(5)])
        docs = build_documents(df)
        assert len(docs) == 5

    def test_each_doc_contains_its_name(self):
        names = ["Alpha", "Beta", "Gamma"]
        rows = [_make_row(name=n).to_dict() for n in names]
        df = pd.DataFrame(rows)
        docs = build_documents(df)
        for name, doc in zip(names, docs):
            assert name in doc

    def test_empty_dataframe_returns_empty_list(self):
        df = pd.DataFrame(columns=["name", "location"])
        docs = build_documents(df)
        assert docs == []


# ---------------------------------------------------------------------------
# embed_documents
# ---------------------------------------------------------------------------

class TestEmbedDocuments:
    # SentenceTransformer is imported lazily inside the function body, so we
    # must patch the attribute on the sentence_transformers module itself.
    _PATCH_TARGET = "sentence_transformers.SentenceTransformer"

    def test_returns_list_of_float_lists(self):
        fake_embeddings = np.random.rand(3, 384).astype(np.float32)
        mock_model = MagicMock()
        mock_model.encode.return_value = fake_embeddings

        with patch(self._PATCH_TARGET, return_value=mock_model):
            result = embed_documents(["doc1", "doc2", "doc3"])

        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(vec, list) for vec in result)
        assert all(isinstance(v, float) for v in result[0])

    def test_uses_correct_model_name(self):
        fake_embeddings = np.random.rand(1, 384).astype(np.float32)
        mock_model = MagicMock()
        mock_model.encode.return_value = fake_embeddings

        with patch(self._PATCH_TARGET, return_value=mock_model) as mock_cls:
            embed_documents(["hello"])
            mock_cls.assert_called_once_with(MODEL_NAME)

    def test_custom_model_name_passed_through(self):
        fake_embeddings = np.random.rand(1, 384).astype(np.float32)
        mock_model = MagicMock()
        mock_model.encode.return_value = fake_embeddings

        with patch(self._PATCH_TARGET, return_value=mock_model) as mock_cls:
            embed_documents(["hello"], model_name="custom/model")
            mock_cls.assert_called_once_with("custom/model")

    def test_encode_called_with_documents(self):
        docs = ["doc a", "doc b"]
        fake_embeddings = np.random.rand(2, 384).astype(np.float32)
        mock_model = MagicMock()
        mock_model.encode.return_value = fake_embeddings

        with patch(self._PATCH_TARGET, return_value=mock_model):
            embed_documents(docs)
            call_args = mock_model.encode.call_args
            assert call_args[0][0] == docs

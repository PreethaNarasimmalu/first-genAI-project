"""Tests for Phase 1 — ingestion.py (streaming mode)."""

from unittest.mock import patch

import pandas as pd

from src.data.ingestion import download_dataset, ingest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_RECORDS = [
    {
        "name": "Restaurant A",
        "location": "Koramangala",
        "cuisines": "Italian",
        "approx_cost(for two people)": "600",
        "rate": "4.1/5",
        "votes": 100,
        "rest_type": "Casual Dining",
        "online_order": "Yes",
        "book_table": "No",
        "dish_liked": "Pizza",
        "listed_in(type)": "Dine-out",
        "listed_in(city)": "Bangalore",
    },
    {
        "name": "Restaurant B",
        "location": "Indiranagar",
        "cuisines": "Chinese",
        "approx_cost(for two people)": "800",
        "rate": "3.9/5",
        "votes": 200,
        "rest_type": "Café",
        "online_order": "No",
        "book_table": "Yes",
        "dish_liked": "Noodles",
        "listed_in(type)": "Delivery",
        "listed_in(city)": "Bangalore",
    },
]


# ---------------------------------------------------------------------------
# download_dataset
# ---------------------------------------------------------------------------

def test_download_dataset_returns_dataframe():
    mock_dataset = {"train": SAMPLE_RECORDS}

    with patch("src.data.ingestion.load_dataset", return_value=mock_dataset):
        result = download_dataset()

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2


def test_download_dataset_uses_first_split_if_no_train():
    mock_dataset = {"test": SAMPLE_RECORDS}

    with patch("src.data.ingestion.load_dataset", return_value=mock_dataset):
        result = download_dataset()

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2


def test_download_dataset_passes_token():
    mock_dataset = {"train": SAMPLE_RECORDS}

    with patch("src.data.ingestion.load_dataset", return_value=mock_dataset) as mock_load:
        download_dataset(hf_token="test-token")
        mock_load.assert_called_once_with(
            "ManikaSaini/zomato-restaurant-recommendation",
            token="test-token",
            streaming=True,
        )


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------

def test_ingest_returns_dataframe():
    mock_dataset = {"train": SAMPLE_RECORDS}

    with patch("src.data.ingestion.load_dataset", return_value=mock_dataset):
        result = ingest()

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2

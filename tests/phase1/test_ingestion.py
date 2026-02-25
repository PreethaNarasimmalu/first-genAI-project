"""Tests for Phase 1 — ingestion.py."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.ingestion import download_dataset, ingest, load_raw, save_raw


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": ["Restaurant A", "Restaurant B"],
            "location": ["Koramangala", "Indiranagar"],
            "cuisines": ["Italian", "Chinese"],
            "approx_cost(for two people)": ["600", "800"],
            "rate": ["4.1/5", "3.9/5"],
            "votes": [100, 200],
            "rest_type": ["Casual Dining", "Café"],
            "online_order": ["Yes", "No"],
            "book_table": ["No", "Yes"],
            "dish_liked": ["Pizza", "Noodles"],
            "listed_in(type)": ["Dine-out", "Delivery"],
            "listed_in(city)": ["Bangalore", "Bangalore"],
        }
    )


# ---------------------------------------------------------------------------
# download_dataset
# ---------------------------------------------------------------------------

def test_download_dataset_returns_dataframe(sample_df):
    mock_split = MagicMock()
    mock_split.__len__ = lambda self: 2
    mock_split.to_pandas.return_value = sample_df

    mock_dataset = {"train": mock_split}

    with patch("src.data.ingestion.load_dataset", return_value=mock_dataset):
        result = download_dataset()

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2


def test_download_dataset_uses_first_split_if_no_train(sample_df):
    mock_split = MagicMock()
    mock_split.__len__ = lambda self: 2
    mock_split.to_pandas.return_value = sample_df

    mock_dataset = {"test": mock_split}

    with patch("src.data.ingestion.load_dataset", return_value=mock_dataset):
        result = download_dataset()

    assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# save_raw / load_raw
# ---------------------------------------------------------------------------

def test_save_and_load_raw(tmp_path, sample_df):
    dest = tmp_path / "raw" / "restaurants_raw.parquet"
    save_raw(sample_df, path=dest)

    assert dest.exists()

    loaded = load_raw(path=dest)
    assert list(loaded.columns) == list(sample_df.columns)
    assert len(loaded) == len(sample_df)


def test_load_raw_raises_if_missing(tmp_path):
    missing = tmp_path / "nonexistent.parquet"
    with pytest.raises(FileNotFoundError):
        load_raw(path=missing)


# ---------------------------------------------------------------------------
# ingest (cache logic)
# ---------------------------------------------------------------------------

def test_ingest_uses_cache_when_file_exists(tmp_path, sample_df, monkeypatch):
    cached = tmp_path / "restaurants_raw.parquet"
    sample_df.to_parquet(cached, index=False)

    # Patch module-level RAW_PARQUET (controls the .exists() check in ingest)
    monkeypatch.setattr("src.data.ingestion.RAW_PARQUET", cached)

    # load_raw() uses its own default arg bound at def time, so patch it directly
    with patch("src.data.ingestion.load_raw", return_value=sample_df) as mock_load, \
         patch("src.data.ingestion.download_dataset") as mock_dl:
        result = ingest(force=False)
        mock_dl.assert_not_called()
        mock_load.assert_called_once()

    assert isinstance(result, pd.DataFrame)


def test_ingest_force_redownloads(tmp_path, sample_df, monkeypatch):
    cached = tmp_path / "restaurants_raw.parquet"
    sample_df.to_parquet(cached, index=False)

    monkeypatch.setattr("src.data.ingestion.RAW_PARQUET", cached)

    mock_split = MagicMock()
    mock_split.__len__ = lambda self: 2
    mock_split.to_pandas.return_value = sample_df
    mock_dataset = {"train": mock_split}

    with patch("src.data.ingestion.load_dataset", return_value=mock_dataset):
        result = ingest(force=True)

    assert isinstance(result, pd.DataFrame)


def test_ingest_downloads_when_no_cache(tmp_path, sample_df, monkeypatch):
    missing = tmp_path / "restaurants_raw.parquet"
    monkeypatch.setattr("src.data.ingestion.RAW_PARQUET", missing)

    mock_split = MagicMock()
    mock_split.__len__ = lambda self: 2
    mock_split.to_pandas.return_value = sample_df
    mock_dataset = {"train": mock_split}

    # save_raw() default arg is also bound at def time; patch it to avoid
    # writing to the real RAW_PARQUET location
    with patch("src.data.ingestion.load_dataset", return_value=mock_dataset), \
         patch("src.data.ingestion.save_raw", return_value=missing) as mock_save:
        result = ingest(force=False)
        mock_save.assert_called_once()

    assert isinstance(result, pd.DataFrame)

"""Phase 1 — Data Ingestion.

Downloads the Zomato restaurant dataset from Hugging Face using the
`datasets` library and saves the raw split as a Parquet file.

Usage:
    python -m src.data.ingestion
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from datasets import load_dataset

DATASET_ID = "ManikaSaini/zomato-restaurant-recommendation"
RAW_DIR = Path("data/raw")
RAW_PARQUET = RAW_DIR / "restaurants_raw.parquet"


def download_dataset(hf_token: str | None = None) -> pd.DataFrame:
    """Download the Zomato dataset from Hugging Face and return as a DataFrame.

    Args:
        hf_token: Optional Hugging Face token (needed if dataset is gated).
                  Falls back to the HF_TOKEN environment variable.

    Returns:
        Raw DataFrame with all original columns.
    """
    token = hf_token or os.getenv("HF_TOKEN")

    print(f"Downloading dataset: {DATASET_ID}")
    dataset = load_dataset(DATASET_ID, token=token)

    # The dataset typically ships with a single 'train' split
    split_name = "train" if "train" in dataset else list(dataset.keys())[0]
    print(f"Using split: '{split_name}'  ({len(dataset[split_name])} rows)")

    df: pd.DataFrame = dataset[split_name].to_pandas()
    return df


def save_raw(df: pd.DataFrame, path: Path = RAW_PARQUET) -> Path:
    """Persist the raw DataFrame to a Parquet file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"Raw data saved → {path}  ({len(df)} rows, {len(df.columns)} columns)")
    return path


def load_raw(path: Path = RAW_PARQUET) -> pd.DataFrame:
    """Load the raw Parquet file (skips download if already cached)."""
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {path}. Run ingestion first."
        )
    return pd.read_parquet(path)


def ingest(force: bool = False, hf_token: str | None = None) -> pd.DataFrame:
    """Download (if needed) and return the raw dataset.

    Args:
        force: Re-download even if the local cache already exists.
        hf_token: Optional Hugging Face token.

    Returns:
        Raw DataFrame.
    """
    if RAW_PARQUET.exists() and not force:
        print(f"Cache found — loading from {RAW_PARQUET}")
        return load_raw()

    df = download_dataset(hf_token=hf_token)
    save_raw(df)
    return df


if __name__ == "__main__":
    df = ingest()
    print("\nColumns:", list(df.columns))
    print(df.head(3).to_string())

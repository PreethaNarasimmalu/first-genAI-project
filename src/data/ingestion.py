"""Phase 1 — Data Ingestion (streaming mode).

Streams the Zomato restaurant dataset from Hugging Face using the
`datasets` library in streaming mode — no data is written to disk.

Usage:
    python -m src.data.ingestion
"""

from __future__ import annotations

import os

import pandas as pd
from datasets import load_dataset

DATASET_ID = "ManikaSaini/zomato-restaurant-recommendation"


def download_dataset(hf_token: str | None = None) -> pd.DataFrame:
    """Stream the Zomato dataset from Hugging Face and return as a DataFrame.

    Args:
        hf_token: Optional Hugging Face token (needed if dataset is gated).
                  Falls back to the HF_TOKEN environment variable.

    Returns:
        Raw DataFrame with all original columns.
    """
    token = hf_token or os.getenv("HF_TOKEN")

    print(f"Streaming dataset: {DATASET_ID}")
    dataset = load_dataset(DATASET_ID, token=token, streaming=True)

    split_name = "train" if "train" in dataset else list(dataset.keys())[0]
    df = pd.DataFrame(list(dataset[split_name]))
    print(f"Using split: '{split_name}'  ({len(df)} rows)")

    return df


def ingest(hf_token: str | None = None) -> pd.DataFrame:
    """Stream and return the raw dataset (no local caching).

    Args:
        hf_token: Optional Hugging Face token.

    Returns:
        Raw DataFrame.
    """
    return download_dataset(hf_token=hf_token)


if __name__ == "__main__":
    df = ingest()
    print("\nColumns:", list(df.columns))
    print(df.head(3).to_string())

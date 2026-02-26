"""Phase 1 — Data Preprocessing.

Cleans and normalizes the raw Zomato DataFrame produced by ingestion.py,
then persists the result as data/clean/restaurants.parquet.

Transformations applied:
  - Normalize `rate`        → float | None  (strips "/5", handles NEW/–)
  - Normalize `approx_cost` → int | None    (strips commas)
  - Normalize `cuisines`    → list[str]     (splits comma-separated string)
  - Normalize `online_order` / `book_table` → bool
  - Drop exact duplicate rows
  - Flag near-duplicates (same name + location) — kept but marked
  - Rename columns to snake_case

Usage:
    python -m src.data.preprocessing
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.data.ingestion import ingest

CLEAN_PARQUET = Path(__file__).resolve().parents[2] / "data" / "clean" / "restaurants.parquet"

# Raw → clean column rename map (only what needs renaming)
COLUMN_RENAMES: dict[str, str] = {
    "listed_in(type)": "meal_type",
    "listed_in(city)": "city",
}


# ---------------------------------------------------------------------------
# Individual normalizers
# ---------------------------------------------------------------------------


def _normalize_rate(series: pd.Series) -> pd.Series:
    """Convert rate strings like '4.1/5', 'NEW', '-' to float | NaN."""

    def _parse(val: object) -> float | None:
        if pd.isna(val):
            return None
        s = str(val).strip()
        if s in ("NEW", "-", "–", ""):
            return None
        s = s.split("/")[0].strip()
        try:
            return float(s)
        except ValueError:
            return None

    return series.map(_parse).astype("Float64")


def _normalize_cost(series: pd.Series) -> pd.Series:
    """Strip commas and cast approx_cost to Int64 (nullable int)."""

    def _parse(val: object) -> int | None:
        if pd.isna(val):
            return None
        s = re.sub(r"[,\s]", "", str(val))
        if not s:
            return None
        try:
            return int(float(s))
        except ValueError:
            return None

    return series.map(_parse).astype("Int64")


def _normalize_yes_no(series: pd.Series) -> pd.Series:
    """Convert 'Yes'/'No' strings to boolean."""
    return series.map(lambda v: str(v).strip().lower() == "yes").astype(bool)


def _normalize_cuisines(series: pd.Series) -> pd.Series:
    """Split comma-separated cuisine strings into lists."""

    def _parse(val: object) -> list[str]:
        if pd.isna(val) or str(val).strip() == "":
            return []
        return [c.strip() for c in str(val).split(",") if c.strip()]

    return series.map(_parse)


def _normalize_votes(series: pd.Series) -> pd.Series:
    """Coerce votes to int, defaulting to 0 on errors."""

    def _parse(val: object) -> int:
        try:
            return int(val)
        except (ValueError, TypeError):
            return 0

    return series.map(_parse).astype(int)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning steps and return the processed DataFrame.

    Args:
        df: Raw DataFrame from ingestion.

    Returns:
        Cleaned DataFrame ready for indexing.
    """
    df = df.copy()

    # 1. Rename awkward column names
    df.rename(columns=COLUMN_RENAMES, inplace=True)
    if "approx_cost(for two people)" in df.columns:
        df.rename(columns={"approx_cost(for two people)": "approx_cost"}, inplace=True)

    # 2. Drop exact duplicates BEFORE list-type conversions (lists are unhashable)
    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"Dropped {before - len(df)} exact duplicate rows.")

    # 3. Normalize typed fields
    if "rate" in df.columns:
        df["rate"] = _normalize_rate(df["rate"])

    if "approx_cost" in df.columns:
        df["approx_cost"] = _normalize_cost(df["approx_cost"])

    if "online_order" in df.columns:
        df["online_order"] = _normalize_yes_no(df["online_order"])

    if "book_table" in df.columns:
        df["book_table"] = _normalize_yes_no(df["book_table"])

    if "cuisines" in df.columns:
        df["cuisines"] = _normalize_cuisines(df["cuisines"])

    if "votes" in df.columns:
        df["votes"] = _normalize_votes(df["votes"])

    # 4. Flag near-duplicates (same name + location) — keep first occurrence
    if {"name", "location"}.issubset(df.columns):
        df["is_near_duplicate"] = df.duplicated(subset=["name", "location"], keep="first")
        near_dup_count = df["is_near_duplicate"].sum()
        print(f"Flagged {near_dup_count} near-duplicate rows (same name + location).")

    # 5. Reset index
    df.reset_index(drop=True, inplace=True)

    print(f"Preprocessing complete — {len(df)} rows retained.")
    return df


def run_pipeline() -> pd.DataFrame:
    """End-to-end Phase 1 pipeline: stream → preprocess (no disk writes)."""
    raw_df = ingest()
    return preprocess(raw_df)


if __name__ == "__main__":
    clean = run_pipeline()
    CLEAN_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(CLEAN_PARQUET, index=False)
    print(f"\nSaved {len(clean)} rows to {CLEAN_PARQUET}")
    print("\nSample output:")
    print(clean[["name", "location", "rate", "approx_cost", "cuisines"]].head(5).to_string())

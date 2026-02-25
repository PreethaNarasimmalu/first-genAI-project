"""Tests for Phase 1 — preprocessing.py."""

import pandas as pd
import pytest

from src.data.preprocessing import (
    _normalize_cost,
    _normalize_cuisines,
    _normalize_rate,
    _normalize_votes,
    _normalize_yes_no,
    preprocess,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _series(*values):
    return pd.Series(list(values))


# ---------------------------------------------------------------------------
# _normalize_rate
# ---------------------------------------------------------------------------

class TestNormalizeRate:
    def test_slash_five_format(self):
        result = _normalize_rate(_series("4.1/5"))
        assert result[0] == pytest.approx(4.1)

    def test_new_becomes_na(self):
        result = _normalize_rate(_series("NEW"))
        assert pd.isna(result[0])

    def test_dash_becomes_na(self):
        result = _normalize_rate(_series("-"))
        assert pd.isna(result[0])

    def test_em_dash_becomes_na(self):
        result = _normalize_rate(_series("–"))
        assert pd.isna(result[0])

    def test_plain_float_string(self):
        result = _normalize_rate(_series("3.8"))
        assert result[0] == pytest.approx(3.8)

    def test_none_becomes_na(self):
        result = _normalize_rate(_series(None))
        assert pd.isna(result[0])

    def test_multiple_values(self):
        result = _normalize_rate(_series("4.0/5", "NEW", "3.5/5", None))
        assert result[0] == pytest.approx(4.0)
        assert pd.isna(result[1])
        assert result[2] == pytest.approx(3.5)
        assert pd.isna(result[3])


# ---------------------------------------------------------------------------
# _normalize_cost
# ---------------------------------------------------------------------------

class TestNormalizeCost:
    def test_comma_stripped(self):
        result = _normalize_cost(_series("1,200"))
        assert result[0] == 1200

    def test_plain_int_string(self):
        result = _normalize_cost(_series("800"))
        assert result[0] == 800

    def test_none_becomes_na(self):
        result = _normalize_cost(_series(None))
        assert pd.isna(result[0])

    def test_empty_string_becomes_na(self):
        result = _normalize_cost(_series(""))
        assert pd.isna(result[0])

    def test_float_string_truncated_to_int(self):
        result = _normalize_cost(_series("600.0"))
        assert result[0] == 600


# ---------------------------------------------------------------------------
# _normalize_yes_no
# ---------------------------------------------------------------------------

class TestNormalizeYesNo:
    def test_yes_is_true(self):
        result = _normalize_yes_no(_series("Yes"))
        assert bool(result[0]) is True

    def test_no_is_false(self):
        result = _normalize_yes_no(_series("No"))
        assert bool(result[0]) is False

    def test_case_insensitive(self):
        result = _normalize_yes_no(_series("YES", "no", "No"))
        assert list(result) == [True, False, False]


# ---------------------------------------------------------------------------
# _normalize_cuisines
# ---------------------------------------------------------------------------

class TestNormalizeCuisines:
    def test_comma_separated(self):
        result = _normalize_cuisines(_series("North Indian, Chinese"))
        assert result[0] == ["North Indian", "Chinese"]

    def test_single_value(self):
        result = _normalize_cuisines(_series("Italian"))
        assert result[0] == ["Italian"]

    def test_none_gives_empty_list(self):
        result = _normalize_cuisines(_series(None))
        assert result[0] == []

    def test_empty_string_gives_empty_list(self):
        result = _normalize_cuisines(_series(""))
        assert result[0] == []

    def test_strips_whitespace(self):
        result = _normalize_cuisines(_series("  Thai , Sushi  "))
        assert result[0] == ["Thai", "Sushi"]


# ---------------------------------------------------------------------------
# _normalize_votes
# ---------------------------------------------------------------------------

class TestNormalizeVotes:
    def test_integer_string(self):
        result = _normalize_votes(_series("250"))
        assert result[0] == 250

    def test_none_becomes_zero(self):
        result = _normalize_votes(_series(None))
        assert result[0] == 0

    def test_invalid_string_becomes_zero(self):
        result = _normalize_votes(_series("N/A"))
        assert result[0] == 0


# ---------------------------------------------------------------------------
# preprocess (integration)
# ---------------------------------------------------------------------------

def _make_raw_df(n: int = 5) -> pd.DataFrame:
    """Build a minimal raw DataFrame that mimics the Zomato dataset schema."""
    costs = ["600", "1,200", "800", None, "400"]
    rates = ["4.1/5", "NEW", "3.8/5", "-", "4.5/5"]
    votes = [100, 200, 0, None, 50]
    orders = ["Yes", "No", "Yes", "Yes", "No"]
    books = ["No", "Yes", "No", "No", "Yes"]
    return pd.DataFrame(
        {
            "name": [f"Restaurant {i}" for i in range(n)],
            "location": ["Koramangala"] * n,
            "cuisines": ["North Indian, Chinese"] * n,
            "approx_cost(for two people)": (costs * ((n // 5) + 1))[:n],
            "rate": (rates * ((n // 5) + 1))[:n],
            "votes": (votes * ((n // 5) + 1))[:n],
            "rest_type": ["Casual Dining"] * n,
            "online_order": (orders * ((n // 5) + 1))[:n],
            "book_table": (books * ((n // 5) + 1))[:n],
            "dish_liked": ["Butter Chicken"] * n,
            "listed_in(type)": ["Dine-out"] * n,
            "listed_in(city)": ["Bangalore"] * n,
        }
    )


class TestPreprocess:
    def test_returns_dataframe(self):
        df = preprocess(_make_raw_df())
        assert isinstance(df, pd.DataFrame)

    def test_approx_cost_renamed(self):
        df = preprocess(_make_raw_df())
        assert "approx_cost" in df.columns
        assert "approx_cost(for two people)" not in df.columns

    def test_listed_in_type_renamed_to_meal_type(self):
        df = preprocess(_make_raw_df())
        assert "meal_type" in df.columns
        assert "listed_in(type)" not in df.columns

    def test_rate_is_float_or_na(self):
        df = preprocess(_make_raw_df())
        valid = df["rate"].dropna()
        assert valid.dtype.name in ("Float64", "float64")

    def test_cost_is_int_or_na(self):
        df = preprocess(_make_raw_df())
        valid = df["approx_cost"].dropna()
        assert valid.dtype.name in ("Int64", "int64", "int32")

    def test_cuisines_are_lists(self):
        df = preprocess(_make_raw_df())
        for val in df["cuisines"]:
            assert isinstance(val, list)

    def test_online_order_is_bool(self):
        df = preprocess(_make_raw_df())
        assert df["online_order"].dtype == bool

    def test_book_table_is_bool(self):
        df = preprocess(_make_raw_df())
        assert df["book_table"].dtype == bool

    def test_exact_duplicates_dropped(self):
        raw = _make_raw_df(3)
        # Append an exact duplicate of row 0
        raw = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
        df = preprocess(raw)
        assert len(df) == 3  # duplicate removed

    def test_near_duplicate_flag_column_added(self):
        df = preprocess(_make_raw_df())
        assert "is_near_duplicate" in df.columns

    def test_index_reset_after_dedup(self):
        df = preprocess(_make_raw_df())
        assert list(df.index) == list(range(len(df)))

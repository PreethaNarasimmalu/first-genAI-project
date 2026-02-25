"""Tests for Phase 1 — schema.py (Restaurant Pydantic model)."""

import pytest
from src.data.schema import Restaurant


# ---------------------------------------------------------------------------
# Happy-path construction
# ---------------------------------------------------------------------------

def test_basic_valid_restaurant():
    r = Restaurant(
        name="Trattoria",
        location="Indiranagar",
        cuisines=["Italian", "Continental"],
        approx_cost=900,
        rate=4.3,
        votes=250,
        online_order=True,
        book_table=False,
    )
    assert r.name == "Trattoria"
    assert r.rate == 4.3
    assert r.approx_cost == 900
    assert r.online_order is True
    assert r.book_table is False


# ---------------------------------------------------------------------------
# rate validator
# ---------------------------------------------------------------------------

def test_rate_parsed_from_slash_string():
    r = Restaurant(name="A", location="B", cuisines=[], rate="4.1/5")
    assert r.rate == pytest.approx(4.1)


def test_rate_new_becomes_none():
    r = Restaurant(name="A", location="B", cuisines=[], rate="NEW")
    assert r.rate is None


def test_rate_dash_becomes_none():
    r = Restaurant(name="A", location="B", cuisines=[], rate="-")
    assert r.rate is None


def test_rate_none_stays_none():
    r = Restaurant(name="A", location="B", cuisines=[], rate=None)
    assert r.rate is None


# ---------------------------------------------------------------------------
# approx_cost validator
# ---------------------------------------------------------------------------

def test_cost_strips_comma():
    r = Restaurant(name="A", location="B", cuisines=[], approx_cost="1,200")
    assert r.approx_cost == 1200


def test_cost_none_stays_none():
    r = Restaurant(name="A", location="B", cuisines=[], approx_cost=None)
    assert r.approx_cost is None


def test_cost_integer_passthrough():
    r = Restaurant(name="A", location="B", cuisines=[], approx_cost=500)
    assert r.approx_cost == 500


# ---------------------------------------------------------------------------
# cuisines validator
# ---------------------------------------------------------------------------

def test_cuisines_split_from_string():
    r = Restaurant(name="A", location="B", cuisines="North Indian, Chinese, Biryani")
    assert r.cuisines == ["North Indian", "Chinese", "Biryani"]


def test_cuisines_list_passthrough():
    r = Restaurant(name="A", location="B", cuisines=["Italian"])
    assert r.cuisines == ["Italian"]


def test_cuisines_empty_string_gives_empty_list():
    r = Restaurant(name="A", location="B", cuisines="")
    assert r.cuisines == []


def test_cuisines_none_gives_empty_list():
    r = Restaurant(name="A", location="B", cuisines=None)
    assert r.cuisines == []


# ---------------------------------------------------------------------------
# online_order / book_table validators
# ---------------------------------------------------------------------------

def test_yes_no_yes_is_true():
    r = Restaurant(name="A", location="B", cuisines=[], online_order="Yes")
    assert r.online_order is True


def test_yes_no_no_is_false():
    r = Restaurant(name="A", location="B", cuisines=[], book_table="No")
    assert r.book_table is False


def test_yes_no_bool_passthrough():
    r = Restaurant(name="A", location="B", cuisines=[], online_order=True)
    assert r.online_order is True


# ---------------------------------------------------------------------------
# votes validator
# ---------------------------------------------------------------------------

def test_votes_default_zero():
    r = Restaurant(name="A", location="B", cuisines=[])
    assert r.votes == 0


def test_votes_invalid_becomes_zero():
    r = Restaurant(name="A", location="B", cuisines=[], votes="unknown")
    assert r.votes == 0


def test_votes_integer_passthrough():
    r = Restaurant(name="A", location="B", cuisines=[], votes=1234)
    assert r.votes == 1234

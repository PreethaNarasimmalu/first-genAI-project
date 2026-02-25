"""Pydantic models for a validated Restaurant record."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class Restaurant(BaseModel):
    """A single cleaned restaurant record from the Zomato dataset."""

    name: str
    location: str
    cuisines: list[str]
    approx_cost: int | None = None
    rate: float | None = None
    votes: int = 0
    rest_type: str | None = None
    online_order: bool = False
    book_table: bool = False
    dish_liked: str | None = None
    meal_type: str | None = Field(None, alias="listed_in(type)")
    city: str | None = Field(None, alias="listed_in(city)")

    model_config = {"populate_by_name": True}

    @field_validator("rate", mode="before")
    @classmethod
    def parse_rate(cls, v: object) -> float | None:
        if v is None:
            return None
        s = str(v).strip()
        if s in ("NEW", "-", "–", ""):
            return None
        # Handle "4.1/5" format
        s = s.split("/")[0].strip()
        try:
            return float(s)
        except ValueError:
            return None

    @field_validator("approx_cost", mode="before")
    @classmethod
    def parse_cost(cls, v: object) -> int | None:
        if v is None:
            return None
        s = str(v).replace(",", "").strip()
        if s == "":
            return None
        try:
            return int(float(s))
        except ValueError:
            return None

    @field_validator("cuisines", mode="before")
    @classmethod
    def parse_cuisines(cls, v: object) -> list[str]:
        if isinstance(v, list):
            return [c.strip() for c in v if c.strip()]
        if v is None or str(v).strip() == "":
            return []
        return [c.strip() for c in str(v).split(",") if c.strip()]

    @field_validator("online_order", "book_table", mode="before")
    @classmethod
    def parse_yes_no(cls, v: object) -> bool:
        if isinstance(v, bool):
            return v
        return str(v).strip().lower() == "yes"

    @field_validator("votes", mode="before")
    @classmethod
    def parse_votes(cls, v: object) -> int:
        if v is None:
            return 0
        try:
            return int(v)
        except (ValueError, TypeError):
            return 0

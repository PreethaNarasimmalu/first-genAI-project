"""Phase 3 — UserPreference model.

Defines the structured input a user can provide to the recommendation engine.
All fields are optional so partial preferences are fully supported.
"""

from __future__ import annotations

from pydantic import BaseModel, field_validator


class UserPreference(BaseModel):
    """Structured representation of a user's restaurant preferences.

    Attributes:
        cuisine:      Desired cuisine types, e.g. ["Italian", "Chinese"].
        location:     Neighbourhood / area, e.g. "Koramangala".
        max_price:    Maximum budget for two people (INR).
        min_rating:   Minimum acceptable rating (0–5).
        online_order: Whether online ordering is required.
        book_table:   Whether table booking is required.
        meal_type:    Meal context: "Dine-out", "Delivery", "Buffet", etc.
        free_text:    Any additional free-form request from the user.
    """

    cuisine: list[str] | None = None
    location: str | None = None
    max_price: int | None = None
    min_rating: float | None = None
    online_order: bool | None = None
    book_table: bool | None = None
    meal_type: str | None = None
    free_text: str | None = None

    @field_validator("min_rating")
    @classmethod
    def validate_min_rating(cls, v: float | None) -> float | None:
        if v is not None and not (0.0 <= v <= 5.0):
            raise ValueError(f"min_rating must be between 0 and 5, got {v}")
        return v

    @field_validator("max_price")
    @classmethod
    def validate_max_price(cls, v: int | None) -> int | None:
        if v is not None and v <= 0:
            raise ValueError(f"max_price must be greater than 0, got {v}")
        return v

    @field_validator("cuisine", mode="before")
    @classmethod
    def normalise_cuisine(cls, v: list[str] | str | None) -> list[str] | None:
        """Accept a bare string as a single-element list for convenience."""
        if isinstance(v, str):
            return [v]
        return v

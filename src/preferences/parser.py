"""Phase 3 — Preference Parser.

Validates and normalises a raw preference dict (e.g. from an API request body
or a test harness) into a typed UserPreference model.

Usage:
    from src.preferences.parser import parse_preferences

    prefs = parse_preferences({"cuisine": ["Italian"], "max_price": 800})
"""

from __future__ import annotations

from pydantic import ValidationError

from src.preferences.models import UserPreference


def parse_preferences(raw: dict) -> UserPreference:
    """Validate and normalise raw preference payload.

    Args:
        raw: Dict of user preference fields (any subset of UserPreference).

    Returns:
        Validated UserPreference instance.

    Raises:
        ValueError: If any field fails validation (wraps Pydantic's error).
    """
    try:
        return UserPreference(**raw)
    except ValidationError as exc:
        # Flatten Pydantic errors into a single readable message
        messages = "; ".join(
            f"{'.'.join(str(l) for l in e['loc'])}: {e['msg']}"
            for e in exc.errors()
        )
        raise ValueError(f"Invalid preferences: {messages}") from exc

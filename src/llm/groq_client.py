"""Phase 5 — Groq SDK wrapper.

Loads GROQ_API_KEY from the environment (or .env file) and exposes a single
``call_llm()`` function used by the engine to get LLM recommendations.

Usage:
    from src.llm.groq_client import call_llm
    raw_text = call_llm(system_prompt, user_prompt)
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from groq import AuthenticationError, Groq, RateLimitError

load_dotenv()

_MODEL = "llama-3.3-70b-versatile"
_MAX_TOKENS = 1024
_TEMPERATURE = 0.3


def call_llm(system_prompt: str, user_prompt: str) -> str:
    """Send a chat completion request to Groq and return the raw text.

    Args:
        system_prompt: The system message (role definition + output contract).
        user_prompt:   The user message (preferences + candidates + schema).

    Returns:
        Raw string content from the first completion choice.

    Raises:
        EnvironmentError:      GROQ_API_KEY not found in environment.
        AuthenticationError:   Invalid API key.
        RateLimitError:        Groq rate limit exceeded.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not set. Add it to your .env file:\n"
            "  GROQ_API_KEY=gsk_..."
        )

    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=_MODEL,
            max_tokens=_MAX_TOKENS,
            temperature=_TEMPERATURE,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content

    except AuthenticationError as exc:
        raise AuthenticationError(
            "Groq authentication failed — check your GROQ_API_KEY."
        ) from exc
    except RateLimitError as exc:
        raise RateLimitError(
            "Groq rate limit reached — wait a moment and retry."
        ) from exc

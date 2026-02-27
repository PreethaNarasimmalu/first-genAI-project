"""Streamlit front-end for the Restaurant Recommendation API.

Run locally:
    # Terminal 1 – start the FastAPI backend
    uvicorn src.api.main:app --reload

    # Terminal 2 – start Streamlit
    streamlit run streamlit_app.py

Streamlit Community Cloud:
    Set the main file to streamlit_app.py and the backend URL via the
    BACKEND_URL secret (defaults to http://localhost:8000).
"""

from __future__ import annotations

import os

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(
    page_title="Bangalore Restaurant Finder",
    page_icon="🍽️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def fetch_options() -> tuple[list[str], list[str]]:
    """Fetch cuisine and location lists from the API (cached for the session)."""
    try:
        cuisines = requests.get(f"{BACKEND_URL}/cuisines", timeout=10).json()
        locations = requests.get(f"{BACKEND_URL}/locations", timeout=10).json()
        return sorted(cuisines), sorted(locations)
    except Exception:
        return [], []


def post_recommend(payload: dict) -> dict:
    resp = requests.post(f"{BACKEND_URL}/recommend", json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


def star_bar(rating: float) -> str:
    """Return a unicode star string for a given rating out of 5."""
    full = int(rating)
    half = 1 if (rating - full) >= 0.5 else 0
    empty = 5 - full - half
    return "★" * full + "½" * half + "☆" * empty


# ---------------------------------------------------------------------------
# Sidebar – Preferences
# ---------------------------------------------------------------------------
st.sidebar.title("🍽️ Your Preferences")

cuisines_all, locations_all = fetch_options()

selected_cuisines: list[str] = st.sidebar.multiselect(
    "Cuisine(s)",
    options=cuisines_all,
    placeholder="Any cuisine",
)

selected_location: str = st.sidebar.selectbox(
    "Neighbourhood",
    options=[""] + locations_all,
    format_func=lambda x: "Any location" if x == "" else x,
)

col1, col2 = st.sidebar.columns(2)
with col1:
    max_price: int | None = st.number_input(
        "Max budget (₹ for two)",
        min_value=0,
        max_value=10_000,
        value=0,
        step=100,
        help="Leave 0 for no limit",
    )
with col2:
    min_rating: float | None = st.number_input(
        "Min rating ★",
        min_value=0.0,
        max_value=5.0,
        value=0.0,
        step=0.5,
        help="Leave 0 for no minimum",
    )

meal_type: str = st.sidebar.selectbox(
    "Meal type",
    options=["", "Dine-out", "Delivery", "Buffet", "Cafes", "Desserts", "Pubs and bars"],
    format_func=lambda x: "Any" if x == "" else x,
)

online_order = st.sidebar.toggle("Online ordering available")
book_table = st.sidebar.toggle("Table booking available")

free_text: str = st.sidebar.text_area(
    "Anything else?",
    placeholder="e.g. rooftop seating, good for dates, vegetarian-friendly…",
    height=80,
)

search_clicked = st.sidebar.button("Find Restaurants 🔍", use_container_width=True, type="primary")

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("🍽️ Bangalore Restaurant Recommender")
st.caption("Powered by Zomato data · Groq LLaMA 3.3 · ChromaDB")

if not search_clicked:
    st.info("Set your preferences in the sidebar and click **Find Restaurants** to get started.")
    st.stop()

# Build request payload
payload: dict = {}
if selected_cuisines:
    payload["cuisine"] = selected_cuisines
if selected_location:
    payload["location"] = selected_location
if max_price and max_price > 0:
    payload["max_price"] = max_price
if min_rating and min_rating > 0:
    payload["min_rating"] = min_rating
if meal_type:
    payload["meal_type"] = meal_type
if online_order:
    payload["online_order"] = True
if book_table:
    payload["book_table"] = True
if free_text.strip():
    payload["free_text"] = free_text.strip()

# Call the API
with st.spinner("Finding the best restaurants for you…"):
    try:
        data = post_recommend(payload)
    except requests.exceptions.ConnectionError:
        st.error(
            f"Cannot reach the backend at `{BACKEND_URL}`. "
            "Make sure the FastAPI server is running."
        )
        st.stop()
    except requests.exceptions.HTTPError as exc:
        st.error(f"API error {exc.response.status_code}: {exc.response.text}")
        st.stop()
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")
        st.stop()

recommendations: list[dict] = data.get("recommendations", [])
summary: str = data.get("summary", "")

if not recommendations:
    st.warning("No restaurants found for your preferences. Try relaxing some filters.")
    st.stop()

# Summary banner
if summary:
    st.success(summary)

st.markdown(f"### Top {len(recommendations)} picks")

# Recommendation cards
for item in recommendations:
    rank = item.get("rank", "?")
    name = item.get("name", "Unknown")
    location = item.get("location", "")
    cuisine = item.get("cuisine", "")
    rating = float(item.get("rating", 0))
    cost = item.get("approx_cost", 0)
    why = item.get("why", "")
    highlight = item.get("highlight", "")

    with st.container(border=True):
        header_col, badge_col = st.columns([5, 1])
        with header_col:
            st.markdown(f"#### #{rank} &nbsp; {name}")
            st.caption(f"📍 {location}　　🍴 {cuisine}")
        with badge_col:
            st.markdown(
                f"<div style='text-align:right; font-size:1.5rem; color:#f5a623'>"
                f"{star_bar(rating)}</div>"
                f"<div style='text-align:right; color:#888'>{rating:.1f} / 5</div>",
                unsafe_allow_html=True,
            )

        info_col, why_col = st.columns([1, 2])
        with info_col:
            st.metric("Cost for two", f"₹{cost:,}")
        with why_col:
            if why:
                st.markdown(f"**Why this?** {why}")
            if highlight:
                st.markdown(f"**Must try:** 🌟 {highlight}")

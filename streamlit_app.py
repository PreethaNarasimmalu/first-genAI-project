"""Streamlit app — runs the full engine in-process (no separate FastAPI backend).

Streamlit Community Cloud:
  - Add GROQ_API_KEY in App Settings → Secrets
  - First load builds the ChromaDB index from HuggingFace (takes a few minutes)
  - All subsequent loads reuse the cached index instantly

Run locally:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

# ── Inject Streamlit secrets into os.environ so engine code can read them ──
for _key in ("GROQ_API_KEY",):
    if hasattr(st, "secrets") and _key in st.secrets:
        os.environ[_key] = st.secrets[_key]

# ── Ensure project root is importable ─────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ---------------------------------------------------------------------------
# Page config (must come before any other st.* call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Bangalore Restaurant Finder",
    page_icon="🍽️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# One-time resource: build / load the ChromaDB collection
# ---------------------------------------------------------------------------

@st.cache_resource(
    show_spinner="Loading restaurant index… (first run downloads & indexes ~51k restaurants — takes a few minutes)"
)
def _load_collection():
    """Build the vector index from HuggingFace if needed, then return the collection."""
    from scripts.build_index import main as build_index
    build_index()
    from src.indexing.vector_store import get_client, get_collection
    return get_collection(get_client())


@st.cache_data(show_spinner=False)
def _load_options(_collection) -> tuple[list[str], list[str]]:
    """Paginate collection metadata to extract unique cuisines and locations."""
    cuisines: set[str] = set()
    locations: set[str] = set()
    offset, batch = 0, 5_000
    while True:
        result = _collection.get(limit=batch, offset=offset, include=["metadatas"])
        ids = result.get("ids") or []
        if not ids:
            break
        for meta in result.get("metadatas") or []:
            loc = (meta.get("location") or "").strip()
            if loc:
                locations.add(loc)
            for c in (meta.get("cuisine_str") or "").split(","):
                c = c.strip()
                if c:
                    cuisines.add(c)
        offset += len(ids)
        if len(ids) < batch:
            break
    return sorted(cuisines), sorted(locations)


# Load (triggers index build on first run)
collection = _load_collection()
cuisines_all, locations_all = _load_options(collection)

# ---------------------------------------------------------------------------
# Engine imports (after sys.path is configured)
# ---------------------------------------------------------------------------
from src.engine.engine import recommend as run_engine
from src.llm.groq_client import call_llm
from src.llm.prompt_builder import build_prompts
from src.llm.response_parser import parse_response
from src.preferences.models import UserPreference

# ---------------------------------------------------------------------------
# Sidebar — Preferences
# ---------------------------------------------------------------------------
st.sidebar.title("🍽️ Your Preferences")

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
    max_price = int(st.number_input(
        "Max budget (₹ for two)",
        min_value=0, max_value=10_000, value=0, step=100,
        help="Leave 0 for no limit",
    ))
with col2:
    min_rating = float(st.number_input(
        "Min rating ★",
        min_value=0.0, max_value=5.0, value=0.0, step=0.5,
        help="Leave 0 for no minimum",
    ))

meal_type: str = st.sidebar.selectbox(
    "Meal type",
    options=["", "Dine-out", "Delivery", "Buffet", "Cafes", "Desserts", "Pubs and bars"],
    format_func=lambda x: "Any" if x == "" else x,
)

online_order = st.sidebar.toggle("Online ordering available")
book_table   = st.sidebar.toggle("Table booking available")

free_text: str = st.sidebar.text_area(
    "Anything else?",
    placeholder="e.g. rooftop seating, good for dates, vegetarian-friendly…",
    height=80,
)

search_clicked = st.sidebar.button(
    "Find Restaurants 🔍", use_container_width=True, type="primary"
)

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("🍽️ Bangalore Restaurant Recommender")
st.caption("Powered by Zomato data · Groq LLaMA 3.3 · ChromaDB")

if not search_clicked:
    st.info("Set your preferences in the sidebar and click **Find Restaurants** to get started.")
    st.stop()

# Build UserPreference object
prefs = UserPreference(
    cuisine=selected_cuisines or None,
    location=selected_location or None,
    max_price=max_price if max_price > 0 else None,
    min_rating=min_rating if min_rating > 0 else None,
    meal_type=meal_type or None,
    online_order=True if online_order else None,
    book_table=True if book_table else None,
    free_text=free_text.strip() or None,
)

with st.spinner("Finding the best restaurants for you…"):
    # Phase 4 — retrieve + rank
    try:
        candidates = run_engine(prefs, collection=collection)
    except Exception as exc:
        st.error(f"Retrieval error: {exc}")
        st.stop()

    if not candidates:
        st.warning("No restaurants found. Try relaxing your filters.")
        st.stop()

    # Phase 5 — LLM
    try:
        system_prompt, user_prompt = build_prompts(prefs, candidates)
        raw_output = call_llm(system_prompt, user_prompt)
        result = parse_response(raw_output)
    except EnvironmentError:
        st.error("GROQ_API_KEY is not set. Add it in App Settings → Secrets.")
        st.stop()
    except Exception as exc:
        st.error(f"LLM error: {exc}")
        st.stop()

# Strip placeholders and hallucinations (mirrors FastAPI route logic)
_PLACEHOLDERS = {"none", "null", "n/a", ""}
candidate_names = {c.name.strip().lower() for c in candidates}
result.recommendations = [
    r for r in result.recommendations
    if r.name.strip().lower() not in _PLACEHOLDERS
    and r.name.strip().lower() in candidate_names
]

if not result.recommendations:
    st.warning("No restaurants matched after filtering. Try different preferences.")
    st.stop()

# Summary banner
if result.summary:
    st.success(result.summary)

st.markdown(f"### Top {len(result.recommendations)} picks")


def star_bar(rating: float) -> str:
    full  = int(rating)
    half  = 1 if (rating - full) >= 0.5 else 0
    empty = 5 - full - half
    return "★" * full + "½" * half + "☆" * empty


for item in result.recommendations:
    with st.container(border=True):
        header_col, badge_col = st.columns([5, 1])
        with header_col:
            st.markdown(f"#### #{item.rank} &nbsp; {item.name}")
            st.caption(f"📍 {item.location}　　🍴 {item.cuisine}")
        with badge_col:
            st.markdown(
                f"<div style='text-align:right;font-size:1.5rem;color:#f5a623'>"
                f"{star_bar(float(item.rating))}</div>"
                f"<div style='text-align:right;color:#888'>{item.rating:.1f} / 5</div>",
                unsafe_allow_html=True,
            )

        info_col, why_col = st.columns([1, 2])
        with info_col:
            st.metric("Cost for two", f"₹{item.approx_cost:,}")
        with why_col:
            if item.why:
                st.markdown(f"**Why this?** {item.why}")
            if item.highlight:
                st.markdown(f"**Must try:** 🌟 {item.highlight}")

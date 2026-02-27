"""Streamlit app — visual design matches the React/Tailwind frontend exactly.

Layout: centered single column  (no sidebar)
  - Red hero banner  ("Zomato AI")
  - White card form  (Location, Cuisine, Budget, Rating pills, Meal type, Toggles, Free text)
  - HTML restaurant cards  (rank badge, gradient accent, cuisine tags, rating badge)

Streamlit Community Cloud:
  - Add GROQ_API_KEY in App Settings → Secrets (TOML: GROQ_API_KEY = "gsk_…")
  - First load builds ChromaDB index from HuggingFace (~51 k rows, a few minutes)
  - All subsequent loads reuse the cached index instantly

Run locally:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

# ── Page config (must be first st.* call) ────────────────────────────────────
st.set_page_config(
    page_title="Zomato AI",
    page_icon="🍽️",
    layout="centered",
)

# ── Inject Streamlit secrets → os.environ ────────────────────────────────────
for _key in ("GROQ_API_KEY",):
    if not os.environ.get(_key):
        try:
            _val = st.secrets.get(_key)
            if _val:
                os.environ[_key] = str(_val)
        except Exception:
            pass

if not os.environ.get("GROQ_API_KEY"):
    st.error(
        "**GROQ_API_KEY is missing.**\n\n"
        "Open your app's **Settings → Secrets** and add:\n"
        "```toml\nGROQ_API_KEY = \"gsk_your_key_here\"\n```\n"
        "Make sure the value is wrapped in **double quotes** (TOML format)."
    )
    st.stop()

# ── Project root on sys.path ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── Global CSS — matches React Tailwind theme exactly ────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ── Reset / base ── */
html, body, .stApp {
  background-color: #F8F8F8 !important;
  font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
  color: #1C1C1C;
}

/* ── Hide Streamlit chrome + kill every source of top whitespace ── */
[data-testid="stHeader"],
[data-testid="stDecoration"],
[data-testid="stToolbar"],
header, .stAppHeader,
#MainMenu, footer, .stDeployButton {
  display: none !important;
  height: 0 !important;
  min-height: 0 !important;
}

/* ── Centered container: zero top-padding so hero banner bleeds to the top ── */
.main .block-container {
  padding-top: 0 !important;
  padding-bottom: 3rem !important;
  max-width: 700px !important;
}

/* Belt-and-suspenders: every Streamlit wrapper above the hero */
.stApp > section > div,
.stApp > section,
.stMain > div,
.main > div:first-child {
  padding-top: 0 !important;
  margin-top: 0 !important;
}

/* ── Hero banner (full-width trick from centered layout) ── */
.zai-hero {
  width: 100vw;
  position: relative;
  left: 50%;
  transform: translateX(-50%);
  background-color: #E23744;
  padding: 3rem 1rem 3.5rem;
  text-align: center;
  margin-bottom: 2rem;
}
.zai-hero h1 {
  color: white;
  font-size: 3.2rem;
  font-weight: 700;
  line-height: 1.1;
  margin: 0;
  letter-spacing: -0.02em;
}
.zai-hero p {
  color: #fecdd3;
  font-size: 1.1rem;
  font-weight: 500;
  margin: 0.75rem 0 0;
}

/* ── Field section label ── */
.zai-label {
  font-size: 0.72rem;
  font-weight: 600;
  color: #1C1C1C;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  margin-bottom: 0.25rem;
  margin-top: 0.75rem;
}
.zai-label .req { color: #E23744; }

/* ── Style Streamlit native widgets ── */
/* Selectbox */
[data-testid="stSelectbox"] > div > div {
  border: 1px solid #E8E8E8 !important;
  border-radius: 0.5rem !important;
  font-size: 0.875rem !important;
  background: white !important;
}
[data-testid="stSelectbox"] > div > div:focus-within {
  border-color: #E23744 !important;
  box-shadow: 0 0 0 2px rgba(226,55,68,0.18) !important;
}

/* Number input — hide Streamlit ± buttons; show native spin arrows on hover only */
[data-testid="stNumberInputStepDown"],
[data-testid="stNumberInputStepUp"] {
  display: none !important;
}
[data-testid="stNumberInput"] input {
  border: 1px solid #E8E8E8 !important;
  border-radius: 0.5rem !important;
  font-size: 0.875rem !important;
  width: 100% !important;
}
[data-testid="stNumberInput"] input:focus {
  border-color: #E23744 !important;
  box-shadow: 0 0 0 2px rgba(226,55,68,0.18) !important;
  outline: none !important;
}
/* Hide native spin buttons by default */
[data-testid="stNumberInput"] input::-webkit-outer-spin-button,
[data-testid="stNumberInput"] input::-webkit-inner-spin-button {
  opacity: 0;
  cursor: pointer;
  transition: opacity 150ms;
}
/* Show native spin buttons (up/down arrows) on hover */
[data-testid="stNumberInput"]:hover input::-webkit-outer-spin-button,
[data-testid="stNumberInput"]:hover input::-webkit-inner-spin-button {
  opacity: 1;
}

/* Textarea */
[data-testid="stTextArea"] textarea {
  border: 1px solid #E8E8E8 !important;
  border-radius: 0.5rem !important;
  font-size: 0.875rem !important;
  resize: none !important;
}
[data-testid="stTextArea"] textarea:focus {
  border-color: #E23744 !important;
  box-shadow: 0 0 0 2px rgba(226,55,68,0.18) !important;
}

/* Radio (rating pills) — equal-width via flex-basis 0% */
[data-testid="stRadio"] > div {
  display: flex !important;
  flex-wrap: nowrap !important;
  gap: 0.4rem !important;
}
[data-testid="stRadio"] label {
  flex: 1 1 0% !important;   /* basis=0 → all grow equally regardless of text length */
  min-width: 0 !important;
  border: 1px solid #E8E8E8 !important;
  border-radius: 0.5rem !important;
  padding: 0.3rem 0.25rem !important;
  font-size: 0.75rem !important;
  font-weight: 600 !important;
  color: #696969 !important;
  cursor: pointer;
  text-align: center;
  transition: all 150ms;
  background: white !important;
  justify-content: center !important;
  white-space: nowrap !important;
  overflow: hidden !important;
}
[data-testid="stRadio"] label:has(input:checked) {
  background: #E23744 !important;
  color: white !important;
  border-color: #E23744 !important;
}
[data-testid="stRadio"] label > div:first-child { display: none !important; }

/* Toggle — label left, switch right */
[data-testid="stToggle"] > label {
  display: flex !important;
  flex-direction: row-reverse !important;
  justify-content: space-between !important;
  width: 100% !important;
  align-items: center !important;
}
[data-testid="stToggle"] p {
  font-size: 0.875rem !important;
  color: #1C1C1C !important;
}

/* Primary button — Zomato red */
[data-testid="stButton"] > button[kind="primary"] {
  background-color: #E23744 !important;
  border: none !important;
  border-radius: 0.75rem !important;
  font-weight: 600 !important;
  font-size: 0.9rem !important;
  padding: 0.6rem 1rem !important;
  color: white !important;
  box-shadow: 0 1px 4px rgba(0,0,0,0.12) !important;
  transition: background 150ms, box-shadow 150ms !important;
}
[data-testid="stButton"] > button[kind="primary"]:hover {
  background-color: #c62f3c !important;
  box-shadow: 0 3px 10px rgba(0,0,0,0.18) !important;
}

/* Secondary (Reset) button */
[data-testid="stButton"] > button[kind="secondary"] {
  border: 1px solid #E8E8E8 !important;
  border-radius: 0.75rem !important;
  color: #696969 !important;
  font-size: 0.875rem !important;
  background: white !important;
}
[data-testid="stButton"] > button[kind="secondary"]:hover {
  border-color: #E23744 !important;
  color: #E23744 !important;
}

/* ── Summary banner ── */
.zai-summary {
  background: linear-gradient(to right, #fff7ed, #fff1f2);
  border: 1px solid #fed7aa;
  border-radius: 0.75rem;
  padding: 1rem;
  margin-bottom: 1.25rem;
  display: flex;
  align-items: flex-start;
  gap: 0.75rem;
}
.zai-summary .icon { font-size: 1.4rem; flex-shrink: 0; line-height: 1.4; }
.zai-summary .title {
  font-size: 0.875rem;
  font-weight: 600;
  color: #1C1C1C;
  margin-bottom: 0.2rem;
}
.zai-summary .body {
  font-size: 0.875rem;
  color: #696969;
  line-height: 1.55;
}

/* ── Restaurant card ── */
.zai-card {
  background: white;
  border-radius: 1rem;
  box-shadow: 0 2px 12px rgba(0,0,0,0.08);
  border: 1px solid #E8E8E8;
  overflow: hidden;
  margin-bottom: 1rem;
  transition: box-shadow 150ms;
}
.zai-card:hover { box-shadow: 0 4px 20px rgba(0,0,0,0.13); }
.zai-card .accent { height: 6px; background: linear-gradient(to right, #E23744, #FC8019); }
.zai-card .body { padding: 1.1rem 1.25rem 1.25rem; position: relative; }

/* Rank badge */
.rank-badge {
  position: absolute; top: 0.75rem; left: 0.75rem;
  width: 1.65rem; height: 1.65rem;
  border-radius: 50%;
  background: #E23744; color: white;
  font-size: 0.7rem; font-weight: 700;
  display: flex; align-items: center; justify-content: center;
  box-shadow: 0 1px 4px rgba(0,0,0,0.18);
}

/* Header row */
.rest-header { display: flex; align-items: flex-start; justify-content: space-between; gap: 0.5rem; margin-left: 2rem; }
.rest-name { font-weight: 700; font-size: 1rem; line-height: 1.25; color: #1C1C1C; }
.rest-loc { font-size: 0.72rem; color: #696969; margin-top: 0.2rem; }

/* Rating badge */
.rating-badge {
  display: inline-flex; align-items: center; gap: 0.2rem;
  padding: 0.15rem 0.5rem;
  border-radius: 0.25rem;
  font-size: 0.72rem; font-weight: 700; white-space: nowrap;
  flex-shrink: 0;
}
.rgb-green  { background: #3D9B6D; color: white; }
.rgb-orange { background: #FC8019; color: white; }
.rgb-red    { background: #ef4444; color: white; }

/* Cuisine tags */
.cuisine-tags { display: flex; flex-wrap: wrap; gap: 0.35rem; margin-top: 0.6rem; }
.ctag {
  padding: 0.1rem 0.5rem;
  background: #fff1f2; color: #E23744;
  border: 1px solid #fecdd3;
  border-radius: 9999px;
  font-size: 0.7rem; font-weight: 500;
}

/* Cost */
.rest-cost { font-size: 0.72rem; color: #696969; margin-top: 0.6rem; }
.rest-cost strong { color: #1C1C1C; font-weight: 600; }

/* Details */
.details-sep { border: none; border-top: 1px solid #E8E8E8; margin: 0.9rem 0 0.7rem; }
.why-row { display: flex; gap: 0.5rem; align-items: flex-start; }
.detail-label { font-size: 0.7rem; font-weight: 600; color: #696969; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.2rem; }
.detail-text  { font-size: 0.85rem; color: #1C1C1C; line-height: 1.5; }
.highlight-row {
  display: flex; gap: 0.5rem; align-items: flex-start;
  background: #fff7ed; border-radius: 0.5rem; padding: 0.6rem; margin-top: 0.5rem;
}
.hl-label { font-size: 0.7rem; font-weight: 600; color: #FC8019; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.2rem; }

/* ── Empty state ── */
.zai-empty { text-align: center; padding: 4rem 1rem; }
.zai-empty .icon { font-size: 4rem; display: block; margin-bottom: 0.75rem; }
.zai-empty h2 { font-size: 1.2rem; font-weight: 700; color: #1C1C1C; margin-bottom: 0.5rem; }
.zai-empty p  { font-size: 0.875rem; color: #696969; max-width: 22rem; margin: 0 auto; line-height: 1.6; }

/* Remove Streamlit widget bottom-margin clutter */
[data-testid="stSelectbox"],
[data-testid="stNumberInput"],
[data-testid="stTextArea"],
[data-testid="stRadio"],
[data-testid="stToggle"] { margin-bottom: 0.1rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Load / build index (cached across sessions) ───────────────────────────────
@st.cache_resource(
    show_spinner="Loading restaurant index… (first run downloads & indexes ~51 k restaurants — takes a few minutes)"
)
def _load_collection():
    from scripts.build_index import main as build_index
    build_index()
    from src.indexing.vector_store import get_client, get_collection
    return get_collection(get_client())


@st.cache_data(show_spinner=False)
def _load_options(_collection) -> tuple[list[str], list[str]]:
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


collection = _load_collection()
cuisines_all, locations_all = _load_options(collection)

# ── Hero banner ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="zai-hero">
  <h1>Zomato AI</h1>
  <p>Tell us what you're craving. We'll find the perfect place.</p>
</div>
""", unsafe_allow_html=True)

# ── Preference form ───────────────────────────────────────────────────────────
col_loc, col_cui = st.columns(2)

with col_loc:
    st.markdown('<div class="zai-label">Location <span class="req">*</span></div>', unsafe_allow_html=True)
    selected_location = st.selectbox(
        "location", [""] + locations_all,
        format_func=lambda x: "Select a location" if x == "" else x,
        label_visibility="collapsed",
    )

with col_cui:
    st.markdown('<div class="zai-label">Cuisine <span class="req">*</span></div>', unsafe_allow_html=True)
    selected_cuisine = st.selectbox(
        "cuisine", [""] + cuisines_all,
        format_func=lambda x: "Select a cuisine" if x == "" else x,
        label_visibility="collapsed",
    )

st.markdown('<div class="zai-label">Max budget (₹ for two)</div>', unsafe_allow_html=True)
max_price = st.number_input(
    "max_price", min_value=1, max_value=10_000, value=None, step=100,
    placeholder="₹  e.g. 800 (optional)",
    label_visibility="collapsed",
)

st.markdown('<div class="zai-label">Minimum rating <span style="font-weight:400;text-transform:none;color:#9ca3af">(optional)</span></div>', unsafe_allow_html=True)
rating_choice = st.radio(
    "min_rating",
    ["★ 3+", "★ 3.5+", "★ 4+", "★ 4.5+"],
    horizontal=True,
    index=None,
    label_visibility="collapsed",
)
_rating_map = {"★ 3+": 3.0, "★ 3.5+": 3.5, "★ 4+": 4.0, "★ 4.5+": 4.5}
min_rating = _rating_map[rating_choice] if rating_choice else 0.0

st.markdown('<div class="zai-label">Meal type <span style="font-weight:400;text-transform:none;color:#9ca3af">(optional)</span></div>', unsafe_allow_html=True)
meal_type = st.selectbox(
    "meal_type",
    ["", "Dine-out", "Delivery", "Buffet", "Cafes", "Desserts", "Pubs and bars"],
    format_func=lambda x: "Any type" if x == "" else x,
    label_visibility="collapsed",
)

online_order = st.toggle("Online ordering available")
book_table   = st.toggle("Table booking available")

st.markdown('<div class="zai-label">Anything specific? <span style="font-weight:400;text-transform:none;color:#9ca3af">(optional)</span></div>', unsafe_allow_html=True)
free_text = st.text_area(
    "free_text",
    placeholder='e.g. "Romantic rooftop with cocktails"',
    height=80,
    label_visibility="collapsed",
)

btn_col, reset_col = st.columns([5, 1])
with btn_col:
    search_clicked = st.button("🔍  Find Restaurants", type="primary", use_container_width=True)
with reset_col:
    reset_clicked = st.button("Reset", use_container_width=True)

if reset_clicked:
    st.rerun()

# ── Empty state ───────────────────────────────────────────────────────────────
if not search_clicked:
    st.markdown("""
    <div class="zai-empty">
      <span class="icon">🍽️</span>
      <h2>Ready to explore?</h2>
      <p>Fill in your preferences above and hit <strong>Find Restaurants</strong>
         to get recommendations tailored to your taste.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Validation ────────────────────────────────────────────────────────────────
if not selected_location:
    st.error("Please select a location.")
    st.stop()
if not selected_cuisine:
    st.error("Please select a cuisine.")
    st.stop()

# ── Engine imports ────────────────────────────────────────────────────────────
from src.engine.engine import recommend as run_engine
from src.llm.groq_client import call_llm
from src.llm.prompt_builder import build_prompts
from src.llm.response_parser import parse_response
from src.preferences.models import UserPreference

prefs = UserPreference(
    cuisine=[selected_cuisine],
    location=selected_location or None,
    max_price=max_price if max_price > 0 else None,
    min_rating=min_rating if min_rating > 0.0 else None,
    meal_type=meal_type or None,
    online_order=True if online_order else None,
    book_table=True if book_table else None,
    free_text=free_text.strip() or None,
)

# ── Run engine ────────────────────────────────────────────────────────────────
with st.spinner("Finding the best restaurants for you…"):
    try:
        candidates = run_engine(prefs, collection=collection)
    except Exception as exc:
        st.error(f"Retrieval error: {exc}")
        st.stop()

    if not candidates:
        st.markdown("""
        <div class="zai-empty">
          <span class="icon">🔍</span>
          <h2>No restaurants found</h2>
          <p>Try relaxing your filters — broader location or higher budget.</p>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

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

# ── Filter hallucinations ─────────────────────────────────────────────────────
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

# ── Summary banner ────────────────────────────────────────────────────────────
count = len(result.recommendations)
if result.summary:
    st.markdown(f"""
    <div class="zai-summary">
      <span class="icon">✨</span>
      <div>
        <div class="title">Found {count} recommendation{"s" if count != 1 else ""} tailored to your taste</div>
        <div class="body">{result.summary}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Restaurant cards ──────────────────────────────────────────────────────────
for item in result.recommendations:
    rating = float(item.rating)
    rgb_class = "rgb-green" if rating >= 4.0 else ("rgb-orange" if rating >= 3.0 else "rgb-red")

    cuisine_tags = "".join(
        f'<span class="ctag">{c.strip()}</span>'
        for c in item.cuisine.split(",") if c.strip()
    )

    details_html = ""
    if item.why or item.highlight:
        details_html += '<hr class="details-sep">'
        if item.why:
            details_html += f"""
            <div class="why-row">
              <span style="font-size:1rem;flex-shrink:0">💡</span>
              <div>
                <div class="detail-label">Why this?</div>
                <div class="detail-text">{item.why}</div>
              </div>
            </div>"""
        if item.highlight:
            details_html += f"""
            <div class="highlight-row" style="margin-top:0.5rem">
              <span style="font-size:1rem;flex-shrink:0">⭐</span>
              <div>
                <div class="hl-label">Must Try</div>
                <div class="detail-text">{item.highlight}</div>
              </div>
            </div>"""

    st.markdown(f"""
    <article class="zai-card">
      <div class="accent"></div>
      <div class="body">
        <div class="rank-badge">#{item.rank}</div>
        <div class="rest-header">
          <div>
            <div class="rest-name">{item.name}</div>
            <div class="rest-loc">📍 {item.location}</div>
          </div>
          <span class="rating-badge {rgb_class}">★ {rating:.1f}</span>
        </div>
        <div class="cuisine-tags">{cuisine_tags}</div>
        <div class="rest-cost">💰 <strong>₹{item.approx_cost:,}</strong> for two</div>
        {details_html}
      </div>
    </article>
    """, unsafe_allow_html=True)

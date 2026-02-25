# AI Restaurant Recommendation Service — Architecture

## Overview

An end-to-end AI service that accepts user preferences (price, location, rating,
cuisine type) and returns personalized restaurant recommendations. The system
ingests the Zomato dataset from Hugging Face, processes and indexes it, then
uses an LLM to generate context-aware, natural-language recommendations.

**Dataset:** [ManikaSaini/zomato-restaurant-recommendation](https://huggingface.co/datasets/ManikaSaini/zomato-restaurant-recommendation)
**LLM:** Groq (llama-3.3-70b-versatile) via the Groq SDK
**Stack:** Python · FastAPI · Pandas · ChromaDB · Groq SDK · React · Vite

---

## Dataset Fields (Zomato)

| Field                  | Description                                              |
|------------------------|----------------------------------------------------------|
| `name`                 | Restaurant name                                          |
| `location`             | Neighborhood / area                                      |
| `cuisines`             | Comma-separated cuisine types                            |
| `approx_cost`          | Approximate cost for two people (INR)                    |
| `rate`                 | Aggregate rating (e.g. `4.1/5`)                          |
| `votes`                | Number of votes/reviews                                  |
| `rest_type`            | Restaurant category (Casual Dining, Café, Quick Bites…)  |
| `online_order`         | Whether online ordering is available (`Yes`/`No`)        |
| `book_table`           | Whether table booking is available (`Yes`/`No`)          |
| `dish_liked`           | Popular dishes mentioned by reviewers                    |
| `listed_in(type)`      | Meal type context (Delivery, Dine-out, Buffet…)          |
| `listed_in(city)`      | City / broader area                                      |

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     UI Layer  (React + Vite)                    │
│         Preference form · Results cards · Loading states        │
└────────────────────────┬────────────────────────────────────────┘
                         │  HTTP (fetch / axios)
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      API Layer  (FastAPI)                        │
│   POST /recommend  ·  GET /health  ·  GET /cuisines             │
└────────────────────────┬────────────────────────────────────────┘
                         │
            ┌────────────┴────────────┐
            ▼                         ▼
┌───────────────────┐     ┌───────────────────────────┐
│  Preference       │     │   Vector Store             │
│  Parser /         │     │   (ChromaDB)               │
│  Validator        │     │   Semantic restaurant      │
│                   │     │   embeddings               │
└─────────┬─────────┘     └───────────┬───────────────┘
          │                           │
          └──────────┬────────────────┘
                     │  Filtered + ranked candidates
                     ▼
          ┌──────────────────────┐
          │  Recommendation      │
          │  Engine              │
          │  (retrieval +        │
          │   re-ranking)        │
          └──────────┬───────────┘
                     │  Top-N candidates + user prefs
                     ▼
          ┌──────────────────────┐
          │  LLM Layer           │
          │  (Claude via         │
          │   Anthropic SDK)     │
          │                      │
          │  Prompt construction │
          │  + response parsing  │
          └──────────┬───────────┘
                     │  Natural-language recommendations
                     ▼
          ┌──────────────────────┐
          │  Response Formatter  │
          │  JSON + markdown     │
          └──────────────────────┘
```

---

## Project Phases

---

### Phase 1 — Data Ingestion & Preprocessing

**Goal:** Pull the raw Zomato dataset from Hugging Face and produce a clean,
analysis-ready dataframe that will be used in all subsequent phases.

**Components:**

```
src/
  data/
    ingestion.py     # Download dataset via `datasets` library
    preprocessing.py # Clean, normalize, and validate fields
    schema.py        # Pydantic models for a Restaurant record
```

**Key tasks:**

1. Load dataset with `datasets.load_dataset("ManikaSaini/zomato-restaurant-recommendation")`
2. Normalize `rate` — strip `/5`, convert to `float`, handle `NEW` / `–` as `None`
3. Normalize `approx_cost` — strip commas, cast to `int`
4. Expand `cuisines` into a list for multi-value filtering
5. Drop exact duplicates; flag near-duplicates (same name + location)
6. Persist cleaned data as `data/clean/restaurants.parquet`

**Output:** `restaurants.parquet` with typed, null-handled columns.

---

### Phase 2 — Vector Store & Indexing

**Goal:** Embed each restaurant as a semantic vector so that free-text preference
queries can retrieve relevant candidates beyond exact keyword matching.

**Components:**

```
src/
  indexing/
    embedder.py      # Build document strings & call embedding model
    vector_store.py  # ChromaDB collection management
    indexer.py       # Orchestrates embed → upsert pipeline
```

**Document string format (per restaurant):**

```
{name} is a {rest_type} in {location} serving {cuisines}.
Rating: {rate}/5 based on {votes} votes.
Approx cost for two: ₹{approx_cost}.
Popular dishes: {dish_liked}.
Online order: {online_order}. Table booking: {book_table}.
```

**Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` (local, free)
**Vector store:** ChromaDB (local persistent store at `data/vectordb/`)

**Key tasks:**

1. Convert each cleaned row to a document string
2. Batch-embed all documents
3. Store vectors + metadata (all filterable fields) in ChromaDB
4. Expose `similarity_search(query, filters, top_k)` helper

---

### Phase 3 — Preference Parsing & Filtering

**Goal:** Translate raw user inputs into structured filters and a semantic query
string that drives retrieval.

**Components:**

```
src/
  preferences/
    models.py        # UserPreference Pydantic model
    parser.py        # Validate & normalize incoming preference payload
    query_builder.py # Build ChromaDB metadata filters + query string
```

**UserPreference model:**

```python
class UserPreference(BaseModel):
    cuisine: list[str] | None       # ["Italian", "Chinese"]
    location: str | None            # "Koramangala"
    max_price: int | None           # 800  (per two people)
    min_rating: float | None        # 3.5
    online_order: bool | None       # True
    book_table: bool | None         # False
    meal_type: str | None           # "Dine-out" | "Delivery" | "Buffet"
    free_text: str | None           # "romantic rooftop with cocktails"
```

**Key tasks:**

1. Validate types and ranges (`min_rating` ∈ [0, 5], `max_price` > 0)
2. Build ChromaDB `where` clause from structured fields
3. Build semantic query string from structured fields + `free_text`
4. Return `(semantic_query: str, filters: dict)` pair

---

### Phase 4 — Recommendation Engine

**Goal:** Retrieve candidate restaurants using hybrid retrieval (semantic + metadata
filters) and re-rank them before sending to the LLM.

**Components:**

```
src/
  engine/
    retriever.py     # Semantic search against ChromaDB with metadata filters
    ranker.py        # Score and re-rank retrieved candidates
    engine.py        # Orchestrates retriever → ranker pipeline
```

**Retrieval strategy:**

1. Run `similarity_search(query, filters, top_k=20)` against ChromaDB
2. For each result compute a composite score:

   ```
   score = 0.4 × semantic_similarity
         + 0.3 × normalized_rating
         + 0.2 × normalized_votes (popularity)
         + 0.1 × price_fit_score   (1 if under budget, decays otherwise)
   ```

3. Return top 5 scored candidates with all metadata intact

**Output:** `list[RestaurantCandidate]` — top 5 restaurants with scores + metadata

---

### Phase 5 — LLM Integration

**Goal:** Use Groq to generate a natural-language, personalized recommendation
response from the top candidates and the original user preferences.

**Components:**

```
src/
  llm/
    prompt_builder.py  # Construct system + user prompt
    groq_client.py     # Groq SDK wrapper
    response_parser.py # Parse structured JSON from LLM output
```

**Prompt design:**

```
SYSTEM:
  You are an expert restaurant concierge. Given a list of candidate restaurants
  and user preferences, recommend the best options with clear reasoning.
  Always respond in valid JSON matching the schema provided.

USER:
  User preferences:
    Cuisine: {cuisine}
    Location: {location}
    Budget (for two): ₹{max_price}
    Minimum rating: {min_rating}
    Meal type: {meal_type}
    Additional request: "{free_text}"

  Top candidate restaurants:
  {formatted_candidates}

  Return JSON:
  {
    "recommendations": [
      {
        "rank": 1,
        "name": "...",
        "location": "...",
        "cuisine": "...",
        "rating": 4.2,
        "approx_cost": 600,
        "why": "One-sentence reason this matches the user's preferences",
        "highlight": "A dish or feature worth trying"
      }
    ],
    "summary": "A 2-3 sentence conversational summary for the user"
  }
```

**LLM call config:** `model=llama-3.3-70b-versatile`, `max_tokens=1024`, `temperature=0.3`

---

### Phase 6 — API Layer

**Goal:** Expose the full pipeline as a production-ready REST API.

**Components:**

```
src/
  api/
    main.py          # FastAPI app + lifespan (loads index on startup)
    routes/
      recommend.py   # POST /recommend
      meta.py        # GET /cuisines, GET /locations, GET /health
    middleware.py    # Request logging, error handling, CORS
```

**Endpoints:**

| Method | Path            | Description                                        |
|--------|-----------------|----------------------------------------------------|
| POST   | `/recommend`    | Accept preferences → return recommendations        |
| GET    | `/health`       | Liveness check                                     |
| GET    | `/cuisines`     | List all available cuisine types in the dataset    |
| GET    | `/locations`    | List all available locations in the dataset        |

**Request / Response example:**

```json
// POST /recommend
{
  "cuisine": ["Italian"],
  "location": "Indiranagar",
  "max_price": 1000,
  "min_rating": 4.0,
  "free_text": "good for a date night"
}

// 200 OK
{
  "recommendations": [
    {
      "rank": 1,
      "name": "Trattoria",
      "location": "Indiranagar",
      "cuisine": "Italian, Continental",
      "rating": 4.3,
      "approx_cost": 900,
      "why": "Cozy Italian spot with a romantic ambiance, well within budget.",
      "highlight": "Try the truffle pasta and the tiramisu."
    }
  ],
  "summary": "Based on your preferences, Trattoria is a great pick for a romantic Italian dinner in Indiranagar."
}
```

---

### Phase 7 — UI Layer

**Goal:** A clean, responsive web interface so users can submit preferences and
browse recommendations without touching the API directly.

**Tech:** React 18 · Vite · Tailwind CSS

**Components:**

```
ui/
  src/
    components/
      PreferenceForm.jsx   # Cuisine, location, budget, rating, free-text inputs
      RecommendationCard.jsx  # Single restaurant result card
      ResultsList.jsx      # Renders list of RecommendationCards
      LoadingSpinner.jsx   # Shown while API call is in flight
    pages/
      Home.jsx             # PreferenceForm + ResultsList wired together
    api/
      recommend.js         # fetch wrapper for POST /recommend
    App.jsx
    main.jsx
  index.html
  vite.config.js
  tailwind.config.js
  package.json
```

**Key features:**

1. **Preference form** — dropdowns for cuisine & meal type, text inputs for location, sliders for budget & min rating, free-text field
2. **Results view** — card per restaurant showing name, location, cuisine, rating, cost, "why" blurb, and highlight dish
3. **Summary banner** — displays the LLM-generated conversational summary above the cards
4. **Loading & error states** — spinner during API call, inline error message on failure
5. **CORS** — FastAPI middleware already configured to accept requests from the Vite dev server (`localhost:5173`)

**Dev setup:**
```
cd ui && npm install && npm run dev   # Vite dev server on :5173
# FastAPI backend running on :8000
```

---

## Directory Structure

```
first-genAI-project/
├── ARCHITECTURE.md
├── README.md                    # (Phase 6)
├── pyproject.toml               # Dependencies & project metadata
├── .env.example                 # GROQ_API_KEY, HF_TOKEN
│
├── data/
│   ├── raw/                     # Original Hugging Face download
│   ├── clean/
│   │   └── restaurants.parquet  # Phase 1 output
│   └── vectordb/                # ChromaDB persistent storage (Phase 2)
│
├── src/
│   ├── data/
│   │   ├── ingestion.py
│   │   ├── preprocessing.py
│   │   └── schema.py
│   ├── indexing/
│   │   ├── embedder.py
│   │   ├── vector_store.py
│   │   └── indexer.py
│   ├── preferences/
│   │   ├── models.py
│   │   ├── parser.py
│   │   └── query_builder.py
│   ├── engine/
│   │   ├── retriever.py
│   │   ├── ranker.py
│   │   └── engine.py
│   ├── llm/
│   │   ├── prompt_builder.py
│   │   ├── groq_client.py
│   │   └── response_parser.py
│   └── api/
│       ├── main.py
│       ├── middleware.py
│       └── routes/
│           ├── recommend.py
│           └── meta.py
│
├── ui/                          # Phase 7 — React frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── PreferenceForm.jsx
│   │   │   ├── RecommendationCard.jsx
│   │   │   ├── ResultsList.jsx
│   │   │   └── LoadingSpinner.jsx
│   │   ├── pages/
│   │   │   └── Home.jsx
│   │   ├── api/
│   │   │   └── recommend.js
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── index.html
│   ├── vite.config.js
│   ├── tailwind.config.js
│   └── package.json
│
└── tests/
    ├── test_preprocessing.py
    ├── test_retriever.py
    ├── test_llm.py
    └── test_api.py
```

---

## Data Flow (end-to-end)

```
User Request
     │
     ▼
[Phase 3] Parse & validate UserPreference
     │
     ├─── structured fields ──► ChromaDB metadata filters
     └─── free_text + fields ─► semantic query string
                                        │
                                        ▼
                             [Phase 4] ChromaDB similarity search
                             + metadata filter (top 20 candidates)
                                        │
                                        ▼
                             [Phase 4] Composite re-ranking
                             → top 5 RestaurantCandidate objects
                                        │
                                        ▼
                             [Phase 5] Prompt construction
                             (candidates + user prefs → Claude)
                                        │
                                        ▼
                             [Phase 5] Claude API call
                             → structured JSON response
                                        │
                                        ▼
                             [Phase 6] Format & return HTTP response
```

---

## Technology Choices & Rationale

| Technology                     | Role                        | Rationale                                          |
|--------------------------------|-----------------------------|----------------------------------------------------|
| `datasets` (Hugging Face)      | Data ingestion              | Official HF library, handles auth & streaming      |
| Pandas + Parquet               | Data processing & storage   | Fast columnar I/O, low overhead for ~10k rows      |
| `sentence-transformers`        | Local embeddings            | Free, no API key, good quality for semantic search |
| ChromaDB                       | Vector store                | Embedded, no extra infra, metadata filtering built-in |
| FastAPI                        | REST API                    | Async, automatic OpenAPI docs, Pydantic-native     |
| Groq SDK (llama-3.3-70b)       | LLM recommendations         | Very fast inference, free tier, OpenAI-compatible  |
| Pydantic v2                    | Data validation             | Type safety across all layers                      |
| React 18 + Vite                | Frontend UI                 | Fast dev server, component-based, easy API wiring  |
| Tailwind CSS                   | UI styling                  | Utility-first, no custom CSS needed                |

---

## Environment Variables

```
GROQ_API_KEY=...        # Required — Groq API access
HF_TOKEN=...            # Optional — needed if dataset becomes gated
```

---

## Phase Summary

| Phase | Name                        | Primary Output                          |
|-------|-----------------------------|-----------------------------------------|
| 1     | Data Ingestion & Preprocessing | `restaurants.parquet`                |
| 2     | Vector Store & Indexing     | ChromaDB collection + embeddings        |
| 3     | Preference Parsing          | `UserPreference` model + query builder  |
| 4     | Recommendation Engine       | Top-5 ranked `RestaurantCandidate` list |
| 5     | LLM Integration             | Natural-language recommendation JSON (Groq) |
| 6     | API Layer                   | FastAPI service, fully wired end-to-end |
| 7     | UI Layer                    | React + Vite frontend, talks to Phase 6 API |

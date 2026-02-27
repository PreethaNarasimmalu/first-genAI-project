# Bangalore Restaurant Recommender

An end-to-end AI-powered restaurant recommendation system built on the Zomato Bangalore dataset. Users describe what they want — cuisine, neighbourhood, budget, meal type — and the system returns ranked, LLM-narrated recommendations backed by semantic vector search.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [How the Architecture Was Designed](#2-how-the-architecture-was-designed)
3. [Phase-by-Phase Breakdown](#3-phase-by-phase-breakdown)
4. [Data Flow — End to End](#4-data-flow--end-to-end)
5. [How Data Is Stored](#5-how-data-is-stored)
6. [API Reference](#6-api-reference)
7. [Important Libraries and APIs Used](#7-important-libraries-and-apis-used)
8. [Deployment](#8-deployment)
9. [Running Locally](#9-running-locally)
10. [Future Improvements](#10-future-improvements)

---

## 1. Architecture Overview

```
User (Browser / Streamlit UI)
          │
          ▼
  ┌───────────────┐
  │  Streamlit UI │  streamlit_app.py
  │   or React UI │  ui/
  └──────┬────────┘
         │  HTTP POST /recommend
         ▼
  ┌───────────────────────────────────────┐
  │         FastAPI Backend               │
  │  POST /recommend   GET /cuisines      │  Phase 6
  │  GET /locations    GET /health        │
  └──────┬────────────────────────────────┘
         │
         ▼
  ┌───────────────────────────────────────┐
  │       Preference Parser               │  Phase 3
  │  UserPreference → semantic query      │
  │                 → ChromaDB filters    │
  └──────┬────────────────────────────────┘
         │
         ▼
  ┌───────────────────────────────────────┐
  │       Retriever (ChromaDB)            │  Phase 4a
  │  Embed query → paginated get()        │
  │  → cosine similarity (numpy)          │
  └──────┬────────────────────────────────┘
         │
         ▼
  ┌───────────────────────────────────────┐
  │       Ranker                          │  Phase 4b
  │  Dedup → composite score              │
  │  (semantic 40% + rating 30%           │
  │   + votes 20% + price fit 10%)        │
  └──────┬────────────────────────────────┘
         │
         ▼
  ┌───────────────────────────────────────┐
  │       LLM Layer (Groq)                │  Phase 5
  │  Build prompt → call LLaMA 3.3 70B   │
  │  → parse + validate JSON response     │
  └──────┬────────────────────────────────┘
         │
         ▼
  ┌───────────────────────────────────────┐
  │   Hallucination Guard + Response      │  Phase 6
  │   Validation (Pydantic)               │
  └───────────────────────────────────────┘
         │
         ▼
    RecommendationResponse (JSON)
```

---

## 2. How the Architecture Was Designed

### The Core Problem

Raw keyword search over 51,000 restaurant rows is brittle — a search for "romantic dinner with wine" won't match any metadata field directly. The goal was to understand *intent*, not just keywords.

### Why Vector Search + LLM (RAG Pattern)

The system follows a **Retrieval-Augmented Generation (RAG)** pattern:

1. **Retrieve** — Use semantic embeddings to find the restaurants most relevant to the user's intent from a real dataset.
2. **Augment** — Pass only the top candidates to the LLM, not the entire dataset.
3. **Generate** — Let the LLM write human-friendly explanations grounded in real data.

This approach avoids two failure modes:
- **Pure LLM**: The model would hallucinate restaurant names not in the dataset.
- **Pure keyword search**: Would miss semantic matches like "budget-friendly" → low `approx_cost`.

### Why ChromaDB

ChromaDB was chosen as the vector store because:
- It runs fully **embedded** (no separate server process needed), which simplifies deployment.
- It persists to disk automatically.
- It supports metadata filtering (`where` clauses), allowing hard constraints (budget, rating) to be applied at the database layer before similarity ranking.

### Why Groq + LLaMA 3.3 70B

- **Groq** provides extremely fast inference (tokens-per-second well above OpenAI/Anthropic hosted endpoints) — important for a responsive recommendation UX.
- **LLaMA 3.3 70B** is capable enough to follow a strict JSON output schema reliably, which is critical for the response parser.
- Temperature is set to **0.3** — low enough for consistent, factual output but not zero (which can cause repetitive phrasing).

### Why a Composite Ranker Before the LLM

The LLM sees only the top-N candidates. Without a pre-ranker, the LLM would receive semantically close but poorly rated or wildly over-budget restaurants. The composite ranker ensures the LLM always works with a high-quality shortlist.

### Hallucination Guard

The LLM is explicitly instructed to recommend only from the candidate list. A post-generation guard verifies every returned restaurant name exists in the candidate list (case-insensitive). Any fabricated name is silently dropped.

---

## 3. Phase-by-Phase Breakdown

### Phase 1 — Data Ingestion (`src/data/`)

- Reads the raw Zomato Bangalore CSV.
- Cleans and normalises fields:
  - Rating strings like `"4.1/5"` → `float 4.1`
  - `"Yes"` / `"No"` → `bool`
  - Comma-separated cuisine strings → `list[str]`
  - Cost strings → `int` (INR for two)
- Drops rows with missing critical fields (name, location).
- Outputs `data/clean/restaurants.parquet` — the canonical cleaned dataset used by all downstream phases.

**Key model:** `src/data/schema.py` — `Restaurant` Pydantic model.

---

### Phase 2 — Indexing & Vector Store (`src/indexing/`)

- **Embedder** (`embedder.py`): Converts each restaurant row into a rich natural-language document string. Example:
  ```
  Mainland China is a Casual Dining in Indiranagar serving North Indian, Chinese.
  Rating: 4.2/5 based on 1250 votes. Approx cost for two: ₹600.
  Popular dishes: Chilli Garlic Noodles. Online order: Yes. Table booking: Yes.
  Listed under: Dine-out.
  ```
  These strings are embedded using `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional vectors, ONNX runtime).

- **Indexer** (`indexer.py`): Batches documents and upserts them into ChromaDB along with their metadata. Upsert makes re-indexing idempotent — running it twice doesn't duplicate records.

- **Vector Store** (`vector_store.py`): Wrapper around the ChromaDB collection. The key function is `similarity_search()`, which paginates `collection.get()` in batches of 5,000 to avoid a SQLite variable-limit crash on large collections (see [known issue](#sqlite-variable-limit)).

**Stored per document in ChromaDB:**
| Field | Type | Description |
|---|---|---|
| `name` | str | Restaurant name |
| `location` | str | Neighbourhood |
| `cuisine_str` | str | Comma-separated cuisines |
| `rest_type` | str | Casual Dining, Café, etc. |
| `rate` | float | Star rating (0–5) |
| `votes` | int | Review count |
| `approx_cost` | int | INR for two |
| `online_order` | bool | Online ordering available |
| `book_table` | bool | Table booking available |
| `dish_liked` | str | Popular dishes |
| `meal_type` | str | Dine-out, Delivery, Buffet… |
| `city` | str | Broad city/zone |

---

### Phase 3 — Preference Parsing (`src/preferences/`)

Converts a `UserPreference` request into two things:

1. **Semantic query string** — a natural language sentence combining all soft preferences, embedded and compared against the vector store.
   ```
   Italian cuisine in Koramangala Dine-out dining budget under ₹800 for two rated at least 4.0 out of 5
   ```

2. **ChromaDB `where` filter** — hard constraints applied as metadata filters before similarity search:
   ```python
   {"$and": [
       {"approx_cost": {"$lte": 800}},
       {"rate":        {"$gte": 4.0}},
       {"meal_type":   {"$eq": "Dine-out"}},
       {"online_order":{"$eq": True}},
   ]}
   ```

**Design note:** Location and cuisine are intentionally kept as soft (post-retrieval) filters rather than ChromaDB `where` clauses. This is because the dataset has inconsistent location names (e.g. "Koramangala 5th Block" vs "Koramangala") and multi-value cuisine fields — substring matching after retrieval is more robust.

---

### Phase 4 — Retrieval & Ranking (`src/engine/`)

**Retriever (`retriever.py`):**
- Embeds the semantic query string.
- Calls `similarity_search()` with the ChromaDB `where` filter.
- Returns raw results: `[{id, document, metadata, distance}, ...]`

**Engine (`engine.py`):**
- Applies substring post-filters for location and cuisine (case-insensitive).
- Passes filtered results to the ranker.

**Ranker (`ranker.py`):**

Deduplicates by `(name, location)` pair — a restaurant appears multiple times in ChromaDB (once per meal type). The copy with the lowest cosine distance is kept; all meal types are merged.

Composite score formula:
```
score = 0.40 × semantic_similarity      # how well it matches the query
      + 0.30 × normalised_rating        # star rating / 5.0
      + 0.20 × normalised_votes         # log-scale popularity
      + 0.10 × price_fit_score          # 1.0 if within budget, decays if over
```

Returns the top-N `RestaurantCandidate` objects.

---

### Phase 5 — LLM Layer (`src/llm/`)

**Prompt builder (`prompt_builder.py`):**
- Constructs a **system prompt** instructing the LLM to act as a Bangalore restaurant concierge, grounded strictly in the candidate list.
- Constructs a **user prompt** with the formatted preferences and numbered candidate list.
- Includes the exact JSON schema the model must output.

**Groq client (`groq_client.py`):**
- Calls `llama-3.3-70b-versatile` via the Groq API.
- Settings: `max_tokens=1024`, `temperature=0.3`.

**Response parser (`response_parser.py`):**
- Strips any accidental markdown fences (` ```json ... ``` `).
- Parses and validates the JSON against the `RecommendationResponse` Pydantic schema.
- Raises a structured error on malformed output.

---

### Phase 6 — API & Hallucination Guard (`src/api/`)

**FastAPI app (`main.py`):**
- Uses a `lifespan` context manager to pre-load the ChromaDB collection and the cuisine/location lists at startup — so every request is served from memory with no cold-start disk I/O.
- CORS middleware enables cross-origin requests from the React/Streamlit frontends.

**Recommend endpoint (`routes/recommend.py`):**
- Validates the request body with Pydantic.
- Calls the engine pipeline.
- Strips any LLM response that:
  - Contains placeholder names (`"none"`, `"null"`, `"n/a"`, `""`)
  - References a restaurant name not found in the candidate list (hallucination)
- Returns a clean `RecommendationResponse`.

---

## 4. Data Flow — End to End

```
POST /recommend  {"cuisine": ["Italian"], "location": "Koramangala", "max_price": 800}
        │
        ▼
[Phase 3] Build query
        semantic_query = "Italian cuisine in Koramangala budget under ₹800"
        where_filter   = {"approx_cost": {"$lte": 800}}
        │
        ▼
[Phase 4a] Retrieve
        embed(semantic_query) → 384-dim vector
        ChromaDB paginated get() with where_filter → raw candidates
        cosine similarity computed via numpy
        │
        ▼
[Phase 4a→b] Post-filter
        substring match: location contains "koramangala" (case-insensitive)
        substring match: cuisine_str contains "italian" (case-insensitive)
        │
        ▼
[Phase 4b] Rank
        deduplicate by (name, location)
        composite score → top 5 RestaurantCandidate objects
        │
        ▼
[Phase 5] LLM
        build_prompts(prefs, candidates) → system_prompt + user_prompt
        Groq API → llama-3.3-70b-versatile → raw JSON string
        parse + validate → RecommendationResponse
        │
        ▼
[Phase 6] Guard
        strip hallucinated/placeholder names
        │
        ▼
200 OK  {"recommendations": [...], "summary": "..."}
```

---

## 5. How Data Is Stored

### Cleaned Dataset — Parquet

- **Path:** `data/clean/restaurants.parquet`
- **Format:** Apache Parquet (columnar, compressed)
- **Used by:** Phase 1 (written), Phase 2 (read for indexing), Phase 6 startup (cuisine/location lists)
- **Contents:** ~51,000 cleaned restaurant rows with all fields from the Zomato CSV.

### Vector Database — ChromaDB

- **Path:** `data/vectordb/` (persistent SQLite + HNSW index files)
- **Collection:** `restaurants` with cosine similarity metric
- **Contents:** One ChromaDB document per restaurant-per-meal-type row
  - **Embedding:** 384-dim float vector (all-MiniLM-L6-v2)
  - **Document:** Natural language string (used for display/debugging)
  - **Metadata:** All structured fields (name, location, rate, cost, etc.) — used for `where` filtering

#### Known Issue: SQLite Variable Limit

ChromaDB's Rust backend uses SQLite internally. When `collection.get()` is called without a `limit`, it fetches all 51k rows in one query, generating a SQL `WHERE id IN (id1, ..., id51000)` clause that exceeds SQLite's `SQLITE_MAX_VARIABLE_NUMBER` limit.

**Fix:** `similarity_search()` paginates with `limit=5000` and `offset` to keep each SQL call safely within the limit. All batches are concatenated in memory before cosine similarity is computed.

---

## 6. API Reference

### `POST /recommend`

Returns personalised restaurant recommendations.

**Request body:**
```json
{
  "cuisine":      ["Italian"],
  "location":     "Koramangala",
  "max_price":    800,
  "min_rating":   4.0,
  "meal_type":    "Dine-out",
  "online_order": true,
  "book_table":   false,
  "free_text":    "rooftop seating, good for dates"
}
```

All fields are optional. `cuisine` accepts a list of strings or a single string.

**Response:**
```json
{
  "recommendations": [
    {
      "rank":        1,
      "name":        "Toscano",
      "location":    "Koramangala",
      "cuisine":     "Italian, Continental",
      "rating":      4.3,
      "approx_cost": 700,
      "why":         "Authentic Italian in your neighbourhood with a warm ambience.",
      "highlight":   "Wood-fired Pizza Margherita"
    }
  ],
  "summary": "We found 3 Italian spots in Koramangala within your budget..."
}
```

**Status codes:**
| Code | Meaning |
|---|---|
| 200 | Success |
| 422 | Invalid request body (Pydantic validation) |
| 500 | Engine/retrieval error |
| 502 | LLM returned unparseable JSON |
| 503 | Groq API key missing or service unavailable |

---

### `GET /cuisines`

Returns a sorted list of all unique cuisine types in the dataset.

**Response:** `["American", "Biryani", "Chinese", ...]`

---

### `GET /locations`

Returns a sorted list of all unique neighbourhoods in the dataset.

**Response:** `["Banashankari", "Indiranagar", "Koramangala", ...]`

---

### `GET /health`

Liveness check.

**Response:** `{"status": "ok", "service": "Restaurant Recommendation API"}`

---

## 7. Important Libraries and APIs Used

| Library / API | Role |
|---|---|
| **FastAPI** | REST API framework — async, Pydantic-native, auto-generates OpenAPI docs |
| **Uvicorn** | ASGI server for FastAPI |
| **ChromaDB** | Embedded vector database — stores embeddings + metadata, handles `where` filters |
| **sentence-transformers** (`all-MiniLM-L6-v2`) | Lightweight, fast embedding model (384 dims) — runs locally via ONNX |
| **Groq API** | Hosted LLM inference — `llama-3.3-70b-versatile` at high throughput |
| **Pydantic v2** | Request/response validation and schema enforcement throughout all phases |
| **pandas + pyarrow** | Reading/writing the cleaned Parquet dataset |
| **numpy** | Cosine similarity computation across embedding matrices |
| **Streamlit** | Python-native UI — calls the FastAPI backend over HTTP |
| **python-dotenv** | Loads `GROQ_API_KEY` from `.env` in local development |
| **httpx** | Async HTTP client used in tests |
| **pytest** | Test suite covering all 6 phases (263 tests) |

---

## 8. Deployment

### Architecture Decision

The system has two separately deployed components:

| Component | Where it runs |
|---|---|
| **FastAPI backend** | Needs a persistent server (Render, Railway, EC2, etc.) — it holds the ChromaDB files and the embedding model |
| **Streamlit UI** | Streamlit Community Cloud — purely calls the FastAPI backend over HTTP |

### Why Streamlit Instead of (or alongside) React

The React UI (`ui/`) requires Node.js build tooling and a static host. Streamlit provides an equivalent UI with pure Python — easier to deploy for a data/ML project and natively supported by Streamlit Community Cloud.

### Streamlit Community Cloud Deployment

1. Push code to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Fill in:
   - **Repository:** `PreethaNarasimmalu/first-genAI-project`
   - **Branch:** your branch (or `main`)
   - **Main file path:** `streamlit_app.py`
4. Under **Advanced settings → Secrets**, add:
   ```toml
   BACKEND_URL = "https://your-fastapi-backend-url"
   GROQ_API_KEY = "your-groq-key"
   ```
5. Click **Deploy**.

The `BACKEND_URL` secret tells the Streamlit app where your FastAPI server is running. If not set, it defaults to `http://localhost:8000`.

### Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `GROQ_API_KEY` | Yes | — | Groq API key for LLM inference |
| `BACKEND_URL` | Streamlit only | `http://localhost:8000` | URL of the FastAPI backend |

---

## 9. Running Locally

### Prerequisites

- Python 3.11+
- A Groq API key ([console.groq.com](https://console.groq.com))

### Setup

```bash
git clone https://github.com/PreethaNarasimmalu/first-genAI-project.git
cd first-genAI-project

pip install -e ".[dev]"

# Create a .env file
echo 'GROQ_API_KEY=your-key-here' > .env
```

### Run the full pipeline (first time only)

```bash
# Phase 1+2: ingest data and build the vector index
python -m scripts.build_index
```

### Start the backend

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

API docs available at `http://localhost:8000/docs`.

### Start the Streamlit UI

```bash
streamlit run streamlit_app.py
# Opens at http://localhost:8501
```

### Run tests

```bash
pytest tests/ -q
```

---

## 10. Future Improvements

### Search & Retrieval

- **User feedback loop** — collect thumbs up/down on recommendations to fine-tune the composite score weights over time.
- **Hybrid search** — combine BM25 (keyword) with vector similarity (semantic) using a weighted merge, improving precision for exact restaurant name lookups.
- **Re-ranking with a cross-encoder** — use a small cross-encoder model (e.g. `ms-marco-MiniLM`) to re-rank the top-K results after retrieval for higher precision.

### Data & Freshness

- **Live data integration** — replace the static Zomato dataset with a periodically updated feed (web scraping or a restaurant data API) so ratings and new venues stay current.
- **Multi-city support** — extend beyond Bangalore by adding a `city` filter to the preference model and partitioning the ChromaDB collection.

### LLM Layer

- **Structured output mode** — use Groq's native JSON mode or function-calling to eliminate the need for the custom response parser and hallucination guard.
- **Conversational memory** — maintain a session state so the user can refine preferences across turns ("show me cheaper options", "only vegetarian").
- **Streaming responses** — stream the LLM output token by token to the Streamlit UI for a faster perceived response time.

### Infrastructure

- **Caching** — add Redis or an in-memory LRU cache keyed on the preference hash to serve repeated identical queries without re-running the full pipeline.
- **Async engine** — make the retrieval and LLM calls fully async so the FastAPI server can handle concurrent requests without blocking.
- **Containerisation** — package the FastAPI backend as a Docker image (including the ChromaDB data volume) for reproducible cloud deployment.
- **Monitoring** — add request latency metrics (Prometheus/Grafana) and LLM cost tracking per query.

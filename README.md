# NOVA — Production Multimodal Chatbot

> Text · Image · Voice — powered by **Groq** · **LangGraph** · **FastAPI** · **Neon** · **Upstash**

---

## Architecture

```
Browser (HTML/JS)
        │
        ▼
   FastAPI (main.py)
        │
        ├── JWT Middleware ──────── validate every request (security.py)
        ├── Rate Limiter ───────── Upstash sliding window  (database/redis.py)
        ├── Cache Check ────────── Upstash SHA-256 match   (database/redis.py)
        │
        ▼
   API Routes (api/)
   ├── /auth        → register, login, Google OAuth, /me
   ├── /chat        → main chat endpoint
   ├── /sessions    → end session, list sessions
   └── /preferences → user preference profile
        │
        ▼
   LangGraph Pipeline (agent_llm/graph.py)
   │
   ├── audio_transcription  @traceable ─ Groq Whisper large-v3
   ├── image_processing     @traceable ─ Groq llama-4-scout vision
   ├── merge_inputs                   ─ combine all modalities
   ├── llm_node             @traceable ─ llama-3.3-70b + bind_tools
   ├── tool_executor        @traceable ─ weather│currency│tavily│stock│wikipedia
   │     └── loops back to llm until no more tool calls (ReAct)
   └── finalize                       ─ extract final text
        │
        ├── Neon PostgreSQL ─── persist users, sessions, messages, preferences
        └── Upstash Redis ───── session history cache, response cache, rate limit
```

---

## Tech Stack

| Layer                | Technology                                    |
|----------------------|-----------------------------------------------|
| Graph Orchestration  | LangGraph 0.2                                 |
| LLM Provider         | Groq (ChatGroq via LangChain)                 |
| Audio Transcription  | Groq Whisper large-v3                         |
| Image Understanding  | Groq llama-4-scout-17b                        |
| Chat LLM             | llama-3.3-70b-versatile                       |
| Tools                | Weather · Currency · Tavily · Stock · Wikipedia |
| Backend              | FastAPI + uvicorn (async)                     |
| Database             | Neon PostgreSQL (SQLAlchemy async + asyncpg)  |
| Caching + Rate Limit | Upstash Redis (sliding window Lua script)     |
| Auth                 | JWT (python-jose) + Google OAuth (authlib)    |
| Tracing              | LangSmith @traceable                          |
| Testing              | Pytest + pytest-asyncio                       |
| Eval Metrics         | BLEU · ROUGE-L · BERTScore · WER · CER · FaithfulnessScore |
| Containerization     | Docker 3-stage multi-stage build              |
| Orchestration        | Kubernetes (deployment + service + ingress)   |

---

## Project Structure

```
nova-chatbot/
│
├── backend/
│   ├── main.py                    ← FastAPI app — wires everything, zero logic
│   ├── config.py                  ← pydantic-settings, all env vars validated
│   ├── security.py                ← JWT create/decode + Google OAuth flow
│   ├── schemas.py                 ← all Pydantic request/response models
│   ├── exceptions.py              ← named exceptions + global handler
│   │
│   ├── api/
│   │   ├── __init__.py            ← single api_router, one include in main.py
│   │   ├── auth.py                ← /auth/register, /login, /google, /me
│   │   ├── chat.py                ← /chat (rate limit → cache → graph → persist)
│   │   ├── sessions.py            ← /sessions/end, /sessions
│   │   └── preferences.py        ← /preferences
│   │
│   ├── database/
│   │   ├── __init__.py            ← clean re-exports (routes import from here)
│   │   ├── postgres.py            ← SQLAlchemy models + async engine + get_db
│   │   └── redis.py               ← Upstash cache + sliding window rate limiter
│   │
│   └── agent_llm/
│       ├── __init__.py            ← exports chatbot_graph, end_session
│       ├── graph.py               ← full LangGraph pipeline with ReAct loop
│       ├── tools.py               ← all 5 tools with @traceable
│       └── session_analysis.py   ← session summary + preference LLM analysis
│
├── tests/
│   ├── conftest.py                ← shared fixtures (token, mock_user, mock_db)
│   ├── unit/
│   │   ├── test_auth.py           ← password hashing, JWT encode/decode
│   │   ├── test_cache.py          ← cache key, hit/miss, multimodal bypass
│   │   ├── test_rate_limit.py     ← tier configs, allow/block, sliding window
│   │   └── test_graph_routing.py  ← all 7 input combos, tool loop routing
│   └── integration/
│       ├── conftest.py            ← AsyncClient, mock_redis_ok, mock_graph
│       ├── test_chat_api.py       ← HTTP round-trips, 401/429/200, cache hit
│       ├── test_session_flow.py   ← end session → summary → preferences
│       └── test_tool_calls.py     ← tool registry, routing, executor edge cases
│
├── evals/
│   ├── conftest.py                ← judge_llm, score_response, PASS_THRESHOLD
│   ├── eval_halluc.py             ← FaithfulnessScore · SelfCheckGPT · FactCC
│   ├── eval_text.py               ← BLEU · ROUGE-L · BERTScore · Tool+Coherence
│   ├── eval_img.py                ← CLIPScore · Entity Recall · CIDEr · OCR
│   └── eval_audio.py              ← WER · CER · Intent Accuracy · Disfluency
│
├── k8s/
│   ├── deployment.yaml            ← pods, image, env refs, health probes
│   ├── service.yaml               ← ClusterIP port 8000
│   ├── ingress.yaml               ← domain routing + TLS (cert-manager)
│   ├── configmap.yaml             ← non-sensitive config
│   └── secret.yaml                ← template only, gitignored
│
├── frontend/
│   ├── index.html                 ← landing page
│   ├── login.html                 ← email/password + Google OAuth
│   ├── register.html              ← sign up form
│   ├── chat.html                  ← main chat UI (text + image + audio)
│   ├── history.html               ← past sessions list
│   ├── profile.html               ← user info + preferences
│   ├── css/
│   │   ├── base.css               ← variables, reset, shared components
│   │   ├── auth.css               ← login/register styles
│   │   ├── chat.css               ← chat UI styles
│   │   └── pages.css              ← history + profile styles
│   └── js/
│       ├── api.js                 ← central API client (all fetch calls)
│       ├── auth.js                ← JWT storage, decode, expiry, logout
│       ├── auth-guard.js          ← protects all non-auth pages
│       ├── components.js          ← shared navbar + toast
│       ├── chat.js                ← chat logic, audio, image, session end
│       ├── history.js             ← fetch + render sessions
│       └── profile.js             ← fetch + render user + preferences
│
├── Dockerfile                     ← 3-stage: deps-builder → frontend → runtime
├── docker-compose.yml             ← local dev with hot-reload
├── .dockerignore                  ← excludes tests, evals, .env from image
├── requirements.txt               ← all deps pinned
├── pytest.ini                     ← test paths, asyncio_mode, eval marker
├── .env.example                   ← all env vars documented with sources
├── .gitignore                     ← secrets, __pycache__, .env gitignored
└── README.md
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Fill in all values — see comments in .env.example for where to get each key
```

**Required API keys:**
- `GROQ_API_KEY` — [console.groq.com](https://console.groq.com) (free)
- `DATABASE_URL` — [neon.tech](https://neon.tech) (free tier)
- `UPSTASH_REDIS_REST_URL` + `TOKEN` — [upstash.com](https://upstash.com) (free tier)
- `JWT_SECRET_KEY` — generate with `python -c "import secrets; print(secrets.token_hex(32))"`
- `GOOGLE_CLIENT_ID` + `SECRET` — [console.cloud.google.com](https://console.cloud.google.com)
- `LANGCHAIN_API_KEY` — [smith.langchain.com](https://smith.langchain.com)
- `TAVILY_API_KEY` — [tavily.com](https://tavily.com) (free tier)
- `ALPHA_VANTAGE_API_KEY` — [alphavantage.co](https://alphavantage.co) (free tier)

### 3. Run locally

```bash
uvicorn backend.main:app --reload --port 8000
```

Open [http://localhost:8000](http://localhost:8000)

### 4. Run with Docker

```bash
docker build -t nova-chatbot .
docker run -p 8000:8000 --env-file .env nova-chatbot
```

Or with Docker Compose:

```bash
docker compose up --build
```

### 5. Deploy to Kubernetes

```bash
# Fill in k8s/secret.yaml values, then:
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml
```

---

## Running Tests

```bash
# Unit tests only — fast, no API keys needed
pytest tests/unit/ -v

# Integration tests — requires mocked external services
pytest tests/integration/ -v

# All unit + integration (CI pipeline)
pytest -m "not eval" -v

# Eval suite — requires real API keys, costs tokens
pytest evals/ -m eval -v -s
```

### Eval Metrics Reference

| File           | Metrics Used                                        |
|----------------|-----------------------------------------------------|
| eval_halluc.py | FaithfulnessScore · SelfCheckGPT · FactCC           |
| eval_text.py   | BLEU · ROUGE-L · BERTScore F1 · LLM-as-judge        |
| eval_img.py    | CLIPScore (entity recall approx) · CIDEr (ROUGE-L) · Entity Recall |
| eval_audio.py  | WER · CER (jiwer) · Intent Accuracy · Disfluency Robustness |

All evals pass at score ≥ 3/5 (LLM-judged) or metric-specific thresholds.

---

## API Reference

### Auth

| Method | Endpoint               | Auth | Description                    |
|--------|------------------------|------|--------------------------------|
| POST   | `/auth/register`       | —    | Email + password signup         |
| POST   | `/auth/login`          | —    | Login, receive JWT              |
| GET    | `/auth/google`         | —    | Start Google OAuth flow         |
| GET    | `/auth/google/callback`| —    | OAuth callback (auto-redirect)  |
| GET    | `/auth/me`             | JWT  | Current user info               |

### Chat

| Method | Endpoint          | Auth | Description                              |
|--------|-------------------|------|------------------------------------------|
| POST   | `/chat`           | JWT  | Send message — any modality combination  |
| POST   | `/sessions/end`   | JWT  | End session → summary + preferences      |
| GET    | `/sessions`       | JWT  | List past sessions                       |
| GET    | `/preferences`    | JWT  | Get analyzed user preferences            |

### Chat Request Body

```json
{
  "session_id":       "optional-uuid-to-continue-session",
  "text":             "What is the stock price of Apple?",
  "image_b64":        "base64-encoded-image-data",
  "image_media_type": "image/jpeg",
  "audio_b64":        "base64-encoded-audio-webm"
}
```

All fields except `session_id` are optional — send any combination.

### Chat Response

```json
{
  "session_id":        "uuid",
  "message_id":        "uuid",
  "response":          "AAPL is currently trading at $189.42...",
  "transcribed_text":  "What is the stock price of Apple?",
  "image_description": null,
  "tools_called":      ["stock_tool"],
  "cache_hit":         false,
  "latency_ms":        1243,
  "error":             null
}
```

---

## Input Combinations

| Input                  | Pipeline Path                                  |
|------------------------|------------------------------------------------|
| Text only              | merge → llm → (tools?) → finalize             |
| Image only             | image_processing → merge → llm → finalize      |
| Audio only             | audio_transcription → merge → llm → finalize   |
| Image + Text           | image_processing → merge → llm → finalize      |
| Audio + Text           | audio_transcription → merge → llm → finalize   |
| Audio + Image          | audio → image → merge → llm → finalize         |
| Audio + Image + Text   | audio → image → merge → llm → finalize         |

---

## Rate Limits

| Tier | Requests / Minute | Strategy          |
|------|-------------------|-------------------|
| Free | 20                | Sliding window    |
| Pro  | 40                | Sliding window    |

Implemented as an atomic Lua script on Upstash Redis.
Burst-safe: 20 requests in 2 seconds are still counted and blocked correctly.

---

## Extending

Ready-to-add next layers:

- **Streaming** — replace `graph.invoke` with `graph.astream` + SSE endpoint
- **RAG** — add `retrieval_node` before `merge_inputs` (Chroma / Pinecone)
- **LangGraph Studio** — graph is already compatible, just connect
- **WebSocket** — replace `/chat` HTTP with persistent WS for real-time feel
- **GitHub OAuth** — add alongside Google in `security.py` (same pattern)
- **Admin dashboard** — add `/admin` routes + tier management
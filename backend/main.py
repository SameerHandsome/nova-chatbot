"""
main.py — FastAPI application entry point.
Wires everything together. Contains zero business logic.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.config import get_settings
from backend.database import init_db
from backend.api import api_router
from backend.exceptions import NovaBaseException, nova_exception_handler

settings = get_settings()


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()   # create tables on startup
    yield


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "NOVA Multimodal Chatbot API",
    description = "Text · Image · Voice — powered by Groq + LangGraph",
    version     = "1.0.0",
    lifespan    = lifespan,
)

# ── Middleware ─────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Global Exception Handler ──────────────────────────────────────────────────

app.add_exception_handler(NovaBaseException, nova_exception_handler)

# ── API Routes (must be before static mount) ──────────────────────────────────

app.include_router(api_router)

# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status":        "ok",
        "groq_key_set":  bool(settings.groq_api_key),
        "langsmith_set": bool(settings.langchain_api_key),
    }

# ── Frontend HTML page routes ─────────────────────────────────────────────────
# Each page needs its own explicit route so /login.html works directly.
# The CSS/JS assets are served by the StaticFiles mount below.

@app.get("/")
async def serve_index():
    return FileResponse("frontend/index.html")

@app.get("/login.html")
async def serve_login():
    return FileResponse("frontend/login.html")

@app.get("/register.html")
async def serve_register():
    return FileResponse("frontend/register.html")

@app.get("/chat.html")
async def serve_chat():
    return FileResponse("frontend/chat.html")

@app.get("/history.html")
async def serve_history():
    return FileResponse("frontend/history.html")

@app.get("/profile.html")
async def serve_profile():
    return FileResponse("frontend/profile.html")

# ── Static Assets (CSS, JS) ───────────────────────────────────────────────────
# Mount the entire frontend folder at root so:
#   /css/base.css       → frontend/css/base.css
#   /js/api.js          → frontend/js/api.js
# Must come AFTER the explicit HTML routes above.

app.mount("/", StaticFiles(directory="frontend"), name="static")
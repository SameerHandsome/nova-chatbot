"""
api/chat.py — Main chat endpoint.

POST /chat — accepts any combination of text, image, audio.
Full pipeline: rate limit → cache check → LangGraph → persist → cache set → respond.
"""

import time
import uuid

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.database import (
    get_db, User, Session as DBSession, Message,
    check_rate_limit, get_cached_response, set_cached_response,
    get_session_history, set_session_history,
)
from backend.security import get_current_user
from backend.agent_llm.graph import chatbot_graph
from backend.schemas import ChatRequest, ChatResponse
from backend.exceptions import RateLimitExceeded, InvalidInput, GraphExecutionError

router = APIRouter()


@router.post("", response_model=ChatResponse)
async def chat(
    req:          ChatRequest,
    current_user: User         = Depends(get_current_user),
    db:           AsyncSession = Depends(get_db),
):
    # ── Validate input ─────────────────────────────────────────────────────────
    if not any([req.text, req.image_b64, req.audio_b64]):
        raise InvalidInput("At least one input required: text, image_b64, or audio_b64")

    has_image = bool(req.image_b64)
    has_audio = bool(req.audio_b64)
    start_ms  = time.time()

    # ── Rate limit ─────────────────────────────────────────────────────────────
    rl = await check_rate_limit(str(current_user.id), current_user.tier)
    if not rl["allowed"]:
        raise RateLimitExceeded(retry_after=rl["retry_after"], limit=rl["limit"])

    # ── Cache check (text-only) ────────────────────────────────────────────────
    cached = await get_cached_response(req.text, has_image, has_audio)
    if cached:
        return ChatResponse(
            session_id = req.session_id or str(uuid.uuid4()),
            message_id = str(uuid.uuid4()),
            response   = cached["response"],
            cache_hit  = True,
            latency_ms = 0,
        )

    # ── Session ────────────────────────────────────────────────────────────────
    session_id = req.session_id
    if session_id:
        r = await db.execute(
            select(DBSession).where(
                DBSession.id       == session_id,
                DBSession.user_id  == current_user.id,
                DBSession.is_active== True,
            )
        )
        if not r.scalar_one_or_none():
            session_id = None

    if not session_id:
        db_session = DBSession(user_id=current_user.id)
        db.add(db_session)
        await db.flush()
        session_id = str(db_session.id)

    # ── Chat history from Redis ────────────────────────────────────────────────
    history = await get_session_history(session_id)

    # ── LangGraph ─────────────────────────────────────────────────────────────
    # NOTE: key is "messages" (not "lc_messages") — must match ChatState exactly
    # so that ToolNode and tools_condition can find the message list.
    state = {
        "raw_text":             req.text,
        "raw_image_b64":        req.image_b64,
        "raw_image_media_type": req.image_media_type or "image/jpeg",
        "raw_audio_b64":        req.audio_b64,
        "transcribed_text":     None,
        "image_description":    None,
        "merged_input":         None,
        "messages":             [],        # ← "messages", NOT "lc_messages"
        "chat_history":         history,
        "final_response":       None,
        "tools_called":         [],
        "tool_results":         [],
        "error":                None,
    }

    try:
        result = chatbot_graph.invoke(state)
    except Exception as e:
        raise GraphExecutionError(f"Graph failed: {e}")

    response   = result.get("final_response", "Sorry, something went wrong.")
    latency_ms = int((time.time() - start_ms) * 1000)
    message_id = str(uuid.uuid4())

    # ── Persist to Neon ────────────────────────────────────────────────────────
    msg = Message(
        id                = message_id,
        session_id        = session_id,
        role              = "user",
        has_text          = bool(req.text),
        has_image         = has_image,
        has_audio         = has_audio,
        raw_text          = req.text,
        audio_transcript  = result.get("transcribed_text"),
        image_description = result.get("image_description"),
        merged_input      = result.get("merged_input"),
        response_text     = response,
        tools_called      = result.get("tools_called", []),
        tool_results      = result.get("tool_results", []),
        latency_ms        = latency_ms,
        cache_hit         = False,
    )
    db.add(msg)
    await db.commit()

    # ── Update Redis history ───────────────────────────────────────────────────
    user_summary = result.get("merged_input") or req.text or "(multimodal input)"
    history      = (history + [
        {"role": "user",      "content": user_summary},
        {"role": "assistant", "content": response},
    ])[-40:]   # cap at 20 turns
    await set_session_history(session_id, history)

    # ── Cache text-only responses ──────────────────────────────────────────────
    if not has_image and not has_audio:
        await set_cached_response(req.text, False, False, {"response": response})

    return ChatResponse(
        session_id        = session_id,
        message_id        = message_id,
        response          = response,
        transcribed_text  = result.get("transcribed_text"),
        image_description = result.get("image_description"),
        tools_called      = result.get("tools_called", []),
        cache_hit         = False,
        latency_ms        = latency_ms,
        error             = result.get("error"),
    )
"""
api/sessions.py — Session management routes.

POST /sessions/end                  — end session → generate summary + preferences
GET  /sessions                      — list user's past sessions
GET  /sessions/{session_id}/messages — get all messages for a session
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List
from pydantic import BaseModel
from datetime import datetime

from backend.database import get_db, User, Session as DBSession, Message, delete_session_history
from backend.security import get_current_user
from backend.agent_llm import end_session
from backend.schemas import SessionEndRequest, SessionSummaryResponse, SessionPublic
from backend.exceptions import SessionNotFound

router = APIRouter()


# ── Message schema for history view ──────────────────────────────────────────

class MessagePublic(BaseModel):
    id:                str
    role:              str
    raw_text:          str | None
    audio_transcript:  str | None
    image_description: str | None
    response_text:     str | None
    tools_called:      list | None
    has_text:          bool
    has_image:         bool
    has_audio:         bool
    created_at:        datetime

    class Config:
        from_attributes = True


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/end", response_model=SessionSummaryResponse)
async def session_end(
    req:          SessionEndRequest,
    current_user: User             = Depends(get_current_user),
    db:           AsyncSession     = Depends(get_db),
):
    result = await end_session(db, req.session_id, str(current_user.id))
    if "error" in result:
        raise SessionNotFound(result["error"])

    await delete_session_history(req.session_id)
    return SessionSummaryResponse(**result)


@router.get("", response_model=list[SessionPublic])
async def list_sessions(
    current_user: User         = Depends(get_current_user),
    db:           AsyncSession = Depends(get_db),
    limit:        int          = 20,
    offset:       int          = 0,
):
    r = await db.execute(
        select(DBSession)
        .where(DBSession.user_id == current_user.id)
        .order_by(DBSession.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    return r.scalars().all()


@router.get("/{session_id}/messages", response_model=List[MessagePublic])
async def get_session_messages(
    session_id:   str,
    current_user: User         = Depends(get_current_user),
    db:           AsyncSession = Depends(get_db),
):
    """Load all messages for a specific session — used by history page."""
    # Verify the session belongs to the current user
    r = await db.execute(
        select(DBSession).where(
            DBSession.id      == session_id,
            DBSession.user_id == current_user.id,
        )
    )
    session = r.scalar_one_or_none()
    if not session:
        raise SessionNotFound(f"Session {session_id} not found")

    # Load all messages ordered by time
    r = await db.execute(
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(Message.created_at.asc())
    )
    messages = r.scalars().all()

    # Convert UUID to string for JSON serialization
    result = []
    for m in messages:
        result.append(MessagePublic(
            id                = str(m.id),
            role              = m.role,
            raw_text          = m.raw_text,
            audio_transcript  = m.audio_transcript,
            image_description = m.image_description,
            response_text     = m.response_text,
            tools_called      = m.tools_called or [],
            has_text          = m.has_text,
            has_image         = m.has_image,
            has_audio         = m.has_audio,
            created_at        = m.created_at,
        ))
    return result
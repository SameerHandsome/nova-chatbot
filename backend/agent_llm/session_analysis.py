"""
agent_llm/session_analysis.py — Session-end LLM analysis.

Called when POST /session/end is hit:
  1. generate_session_summary() — 2-3 sentence summary of the conversation
  2. analyze_user_preferences() — extract structured preferences as JSON
  3. end_session()              — orchestrates both, persists to Postgres
"""

import json
from datetime import datetime

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.config import get_settings
from backend.database.postgres import Session, Message, UserPreference

settings = get_settings()


def _analysis_llm():
    return ChatGroq(
        model       = "llama-3.3-70b-versatile",
        temperature = 0.1,   # low temp for structured, consistent output
        api_key     = settings.groq_api_key,
    )


# ── Summary ───────────────────────────────────────────────────────────────────

@traceable(name="generate_session_summary")
async def generate_session_summary(messages: list) -> str:
    """Generate a 2-3 sentence summary of all messages in the session."""
    if not messages:
        return "Empty session."

    lines = []
    for m in messages:
        content = m.merged_input or m.raw_text or m.response_text or ""
        if content:
            lines.append(f"{m.role.upper()}: {content[:400]}")

    response = _analysis_llm().invoke([
        SystemMessage(content=(
            "Summarize this chat session in 2-3 concise sentences. "
            "Focus on what the user wanted and what was accomplished. "
            "Be factual and brief. No preamble."
        )),
        HumanMessage(content="\n".join(lines)),
    ])
    return response.content.strip()


# ── Preferences ───────────────────────────────────────────────────────────────

@traceable(name="analyze_user_preferences")
async def analyze_user_preferences(messages: list) -> dict:
    """
    Analyze user messages to extract structured preferences.
    Returns dict matching UserPreference columns.
    """
    if len(messages) < 2:
        return {}

    user_lines = [
        m.merged_input or m.raw_text or ""
        for m in messages
        if m.role == "user" and (m.merged_input or m.raw_text)
    ]
    uses_voice  = any(m.has_audio for m in messages if m.role == "user")
    uses_images = any(m.has_image for m in messages if m.role == "user")

    response = _analysis_llm().invoke([
        SystemMessage(content=(
            "Analyze these user messages and extract preferences. "
            "Respond ONLY with valid JSON — no markdown, no explanation. "
            "Keys must be exactly:\n"
            "  communication_style: 'formal' | 'casual' | 'technical'\n"
            "  topics_of_interest: array of up to 5 topic strings\n"
            "  preferred_response_length: 'short' | 'medium' | 'detailed'\n"
            "  language: ISO 639-1 code e.g. 'en', 'ur', 'fr'\n"
            "Base analysis only on evidence in the messages."
        )),
        HumanMessage(content="\n".join(user_lines[:20])),
    ])

    try:
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        prefs = json.loads(raw.strip())
    except Exception:
        prefs = {}

    prefs["uses_voice"]  = uses_voice
    prefs["uses_images"] = uses_images
    return prefs


# ── Orchestrator ──────────────────────────────────────────────────────────────

async def end_session(db: AsyncSession, session_id: str, user_id: str) -> dict:
    """
    Full session-end flow:
      1. Load session + messages
      2. Generate summary + analyze preferences
      3. Persist everything to Postgres
      4. Return summary + preferences
    """
    # Load session
    r = await db.execute(
        select(Session).where(
            Session.id       == session_id,
            Session.user_id  == user_id,
            Session.is_active == True,
        )
    )
    session = r.scalar_one_or_none()
    if not session:
        return {"error": "Session not found or already ended"}

    # Load all messages
    r = await db.execute(
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(Message.created_at)
    )
    messages = r.scalars().all()

    # Run analysis
    summary = await generate_session_summary(messages)
    prefs   = await analyze_user_preferences(messages)

    # Update session row
    session.is_active = False
    session.ended_at  = datetime.utcnow()
    session.summary   = summary
    session.title     = summary[:80] if summary else "Untitled session"

    # Upsert user preferences
    r         = await db.execute(select(UserPreference).where(UserPreference.user_id == user_id))
    user_pref = r.scalar_one_or_none()

    pref_fields = {
        k: v for k, v in prefs.items()
        if k in UserPreference.__table__.columns.keys() and v
    }

    if user_pref:
        for k, v in pref_fields.items():
            setattr(user_pref, k, v)
        user_pref.last_analyzed_at = datetime.utcnow()
        user_pref.raw_analysis     = json.dumps(prefs)
    else:
        user_pref = UserPreference(
            user_id          = user_id,
            last_analyzed_at = datetime.utcnow(),
            raw_analysis     = json.dumps(prefs),
            **pref_fields,
        )
        db.add(user_pref)

    await db.commit()
    return {"session_id": session_id, "summary": summary, "preferences": prefs}
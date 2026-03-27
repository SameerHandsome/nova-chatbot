"""
database/postgres.py — SQLAlchemy async models + engine + get_db dependency.

Tables:
  users             — registered accounts (email/password or Google OAuth)
  sessions          — chat sessions, one per conversation
  messages          — every message with all modality fields
  user_preferences  — LLM-analyzed preferences, one row per user
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean, Column, DateTime, Integer,
    String, Text, ForeignKey, JSON,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, relationship

from backend.config import get_settings

settings = get_settings()

# ── Engine ────────────────────────────────────────────────────────────────────

engine = create_async_engine(
    settings.database_url,
    echo         = settings.app_env == "development",
    pool_size    = 5,
    max_overflow = 10,
    pool_pre_ping= True,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_          = AsyncSession,
    expire_on_commit= False,
)


# ── Base ──────────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ── Models ────────────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id              = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email           = Column(String(255), unique=True, nullable=False, index=True)
    username        = Column(String(100), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=True)   # null for OAuth-only users
    github_id       = Column(String(255), nullable=True, unique=True)
    is_active       = Column(Boolean, default=True)
    tier            = Column(String(20), default="free")   # "free" | "pro"
    created_at      = Column(DateTime, default=datetime.utcnow)
    updated_at      = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    sessions    = relationship("Session",        back_populates="user", cascade="all, delete-orphan")
    preferences = relationship("UserPreference", back_populates="user", uselist=False)


class Session(Base):
    __tablename__ = "sessions"

    id         = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id    = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    title      = Column(String(255), nullable=True)   # auto-generated on session end
    summary    = Column(Text,        nullable=True)   # LLM-generated summary
    is_active  = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    ended_at   = Column(DateTime, nullable=True)

    user     = relationship("User",    back_populates="sessions")
    messages = relationship("Message", back_populates="session", cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = "messages"

    id         = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    role       = Column(String(20), nullable=False)   # "user" | "assistant"
    created_at = Column(DateTime,  default=datetime.utcnow)

    # Modality flags
    has_text  = Column(Boolean, default=False)
    has_image = Column(Boolean, default=False)
    has_audio = Column(Boolean, default=False)

    # Content
    raw_text          = Column(Text, nullable=True)
    audio_transcript  = Column(Text, nullable=True)
    image_description = Column(Text, nullable=True)
    merged_input      = Column(Text, nullable=True)   # combined prompt sent to LLM
    response_text     = Column(Text, nullable=True)

    # Tool metadata
    tools_called = Column(JSON, nullable=True)   # ["stock_tool", ...]
    tool_results = Column(JSON, nullable=True)   # [{"tool": ..., "result": ...}]

    # Perf metadata
    token_count = Column(Integer, nullable=True)
    cache_hit   = Column(Boolean, default=False)
    latency_ms  = Column(Integer, nullable=True)

    session = relationship("Session", back_populates="messages")


class UserPreference(Base):
    __tablename__ = "user_preferences"

    id                        = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id                   = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), unique=True)
    communication_style       = Column(String(50),  nullable=True)   # "formal"|"casual"|"technical"
    topics_of_interest        = Column(JSON,         nullable=True)   # ["finance", "tech", ...]
    preferred_response_length = Column(String(20),  nullable=True)   # "short"|"medium"|"detailed"
    language                  = Column(String(10),  default="en")
    uses_voice                = Column(Boolean,     default=False)
    uses_images               = Column(Boolean,     default=False)
    last_analyzed_at          = Column(DateTime,    nullable=True)
    raw_analysis              = Column(Text,        nullable=True)   # full LLM JSON for debugging

    user = relationship("User", back_populates="preferences")


# ── Helpers ───────────────────────────────────────────────────────────────────

async def init_db():
    """Create all tables. Called on FastAPI startup lifespan."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    """FastAPI dependency — yields an async DB session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
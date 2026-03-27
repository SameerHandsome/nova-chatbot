"""
schemas.py — All Pydantic request/response models.
Imported by api/ routes and agent_llm/ nodes.
Single source of truth for data shapes.
"""

from typing import Optional, List
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, EmailStr, field_validator


# ── Auth ──────────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email:    EmailStr
    username: str
    password: str

    @field_validator("password")
    @classmethod
    def password_strength(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v

    @field_validator("username")
    @classmethod
    def username_valid(cls, v):
        v = v.strip()
        if len(v) < 3:
            raise ValueError("Username must be at least 3 characters")
        return v


class LoginRequest(BaseModel):
    email:    EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    user_id:      str
    email:        str
    username:     str
    tier:         str


class UserPublic(BaseModel):
    id:         UUID
    email:      str
    username:   str
    tier:       str
    created_at: datetime

    class Config:
        from_attributes = True


# ── Chat ──────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id:       Optional[str] = None
    text:             Optional[str] = None
    image_b64:        Optional[str] = None
    image_media_type: Optional[str] = None
    audio_b64:        Optional[str] = None

    @field_validator("image_media_type")
    @classmethod
    def valid_media_type(cls, v):
        allowed = {"image/jpeg", "image/png", "image/webp", "image/gif", None}
        if v not in allowed:
            raise ValueError(f"image_media_type must be one of {allowed}")
        return v


class ChatResponse(BaseModel):
    session_id:        str
    message_id:        str
    response:          str
    transcribed_text:  Optional[str] = None
    image_description: Optional[str] = None
    tools_called:      List[str]     = []
    cache_hit:         bool          = False
    latency_ms:        int           = 0
    error:             Optional[str] = None


# ── Session ───────────────────────────────────────────────────────────────────

class SessionEndRequest(BaseModel):
    session_id: str


class SessionSummaryResponse(BaseModel):
    session_id:  str
    summary:     str
    preferences: Optional[dict] = None


class SessionPublic(BaseModel):
    id:         UUID
    title:      Optional[str]
    summary:    Optional[str]
    is_active:  bool
    created_at: datetime
    ended_at:   Optional[datetime]

    class Config:
        from_attributes = True


# ── Preferences ───────────────────────────────────────────────────────────────

class UserPreferencePublic(BaseModel):
    communication_style:       Optional[str]
    topics_of_interest:        Optional[List[str]]
    preferred_response_length: Optional[str]
    language:                  str
    uses_voice:                bool
    uses_images:               bool
    last_analyzed_at:          Optional[datetime]

    class Config:
        from_attributes = True
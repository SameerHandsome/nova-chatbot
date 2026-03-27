"""
api/preferences.py — User preferences route.

GET /preferences — returns LLM-analyzed preferences for the current user.
Populated after the first session end.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.database import get_db, User, UserPreference
from backend.security import get_current_user
from backend.schemas import UserPreferencePublic
from backend.exceptions import PreferencesNotFound

router = APIRouter()


@router.get("", response_model=UserPreferencePublic)
async def get_preferences(
    current_user: User         = Depends(get_current_user),
    db:           AsyncSession = Depends(get_db),
):
    r    = await db.execute(
        select(UserPreference).where(UserPreference.user_id == current_user.id)
    )
    pref = r.scalar_one_or_none()
    if not pref:
        raise PreferencesNotFound()
    return pref
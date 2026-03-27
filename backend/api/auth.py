"""
api/auth.py — Authentication routes.

POST /auth/register        — email + password signup
POST /auth/login           — email + password login
GET  /auth/github          — start GitHub OAuth flow
GET  /auth/github/callback — GitHub OAuth callback
GET  /auth/me              — current user info
"""

from fastapi import APIRouter, Depends
from fastapi.responses import RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.database import get_db, User
from backend.security import (
    hash_password, verify_password, create_access_token,
    get_current_user, build_github_auth_url,
    exchange_github_code, get_or_create_github_user,
)
from backend.schemas import (
    RegisterRequest, LoginRequest, TokenResponse, UserPublic,
)
from backend.exceptions import (
    UserAlreadyExists, InvalidCredentials,
)
from backend.config import get_settings

settings = get_settings()
router   = APIRouter()


@router.post("/register", response_model=TokenResponse, status_code=201)
async def register(req: RegisterRequest, db: AsyncSession = Depends(get_db)):
    # Check uniqueness
    r = await db.execute(select(User).where(User.email == req.email))
    if r.scalar_one_or_none():
        raise UserAlreadyExists("Email already registered")

    r = await db.execute(select(User).where(User.username == req.username))
    if r.scalar_one_or_none():
        raise UserAlreadyExists("Username already taken")

    user = User(
        email           = req.email,
        username        = req.username,
        hashed_password = hash_password(req.password),
        tier            = "free",
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    return TokenResponse(
        access_token = create_access_token(str(user.id), user.email, user.tier),
        user_id      = str(user.id),
        email        = user.email,
        username     = user.username,
        tier         = user.tier,
    )


@router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest, db: AsyncSession = Depends(get_db)):
    r    = await db.execute(select(User).where(User.email == req.email))
    user = r.scalar_one_or_none()

    if not user or not user.hashed_password or not verify_password(req.password, user.hashed_password):
        raise InvalidCredentials()

    return TokenResponse(
        access_token = create_access_token(str(user.id), user.email, user.tier),
        user_id      = str(user.id),
        email        = user.email,
        username     = user.username,
        tier         = user.tier,
    )


@router.get("/github")
async def github_login():
    """Redirect browser to GitHub authorization page."""
    return RedirectResponse(build_github_auth_url())


@router.get("/github/callback")
async def github_callback(code: str, db: AsyncSession = Depends(get_db)):
    """GitHub sends the user back here with a one-time code."""
    try:
        info = await exchange_github_code(code)
    except Exception as e:
        raise InvalidCredentials(f"GitHub OAuth failed: {e}")

    user  = await get_or_create_github_user(db, info)
    token = create_access_token(str(user.id), user.email, user.tier)
    return RedirectResponse(f"{settings.frontend_url}/chat.html?token={token}")


@router.get("/me", response_model=UserPublic)
async def get_me(current_user: User = Depends(get_current_user)):
    return current_user
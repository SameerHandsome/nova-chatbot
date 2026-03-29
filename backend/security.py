"""
security.py — JWT creation/verification + GitHub OAuth flow.

GitHub OAuth flow:
  1. Browser → GET /auth/github → redirect to github.com/login/oauth/authorize
  2. User approves → GitHub → GET /auth/github/callback?code=xxx
  3. Exchange code → GitHub access token → GitHub user info
  4. Find or create user in DB → return JWT

Important GitHub quirk handled:
  GitHub users can set their email to private. In that case /user returns
  email=null. We make a second call to /user/emails to get their verified
  primary email. This is a known GitHub OAuth issue Google does not have.
"""

import uuid
from datetime import datetime, timezone, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer
import httpx
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.config import get_settings
from backend.database import get_db, User
from backend.exceptions import TokenInvalid, UserNotFound

settings = get_settings()

# ── Password ──────────────────────────────────────────────────────────────────

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(plain: str) -> str:
    return pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


# ── JWT ───────────────────────────────────────────────────────────────────────

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def create_access_token(user_id: str, email: str, tier: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expire_minutes)
    payload = {
        "sub":   user_id,
        "email": email,
        "tier":  tier,
        "exp":   expire,
        "jti":   str(uuid.uuid4()),
    }
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )
    except JWTError:
        raise TokenInvalid()


async def get_current_user(
    token: str          = Depends(oauth2_scheme),
    db:    AsyncSession = Depends(get_db),
) -> User:
    """FastAPI dependency — resolves JWT Bearer token to a live User row."""
    payload = decode_token(token)
    user_id = payload.get("sub")
    if not user_id:
        raise TokenInvalid()

    result = await db.execute(select(User).where(User.id == user_id))
    user   = result.scalar_one_or_none()

    if not user or not user.is_active:
        raise UserNotFound()

    return user


# ── GitHub OAuth ──────────────────────────────────────────────────────────────

GITHUB_AUTH_URL   = "https://github.com/login/oauth/authorize"
GITHUB_TOKEN_URL  = "https://github.com/login/oauth/access_token"
GITHUB_USER_URL   = "https://api.github.com/user"
GITHUB_EMAILS_URL = "https://api.github.com/user/emails"
GITHUB_SCOPES     = "read:user user:email"


def build_github_auth_url() -> str:
    """Build the GitHub authorization URL the browser redirects to."""
    params = (
        f"?client_id={settings.github_client_id}"
        f"&redirect_uri={settings.github_redirect_uri}"
        f"&scope={GITHUB_SCOPES}"
    )
    return GITHUB_AUTH_URL + params


async def exchange_github_code(code: str) -> dict:
    """
    Exchange callback code for access token, then fetch user profile + email.
    Handles private email: if /user returns email=null we call /user/emails.
    """
    async with httpx.AsyncClient(timeout=15) as client:
        # Step 1 — Exchange code for access token
        token_resp = await client.post(
            GITHUB_TOKEN_URL,
            headers={"Accept": "application/json"},
            data={
                "client_id":     settings.github_client_id,
                "client_secret": settings.github_client_secret,
                "code":          code,
                "redirect_uri":  settings.github_redirect_uri,
            },
        )
        token_data   = token_resp.json()
        access_token = token_data.get("access_token")

        if not access_token:
            raise ValueError(
                f"GitHub did not return an access token: "
                f"{token_data.get('error_description', token_data)}"
            )

        auth_headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept":        "application/vnd.github+json",
        }

        # Step 2 — Get user profile
        user_resp = await client.get(GITHUB_USER_URL, headers=auth_headers)
        user_info = user_resp.json()

        # Step 3 — Handle private email (GitHub quirk)
        if not user_info.get("email"):
            emails_resp   = await client.get(GITHUB_EMAILS_URL, headers=auth_headers)
            emails        = emails_resp.json()
            primary_email = next(
                (e["email"] for e in emails if e.get("primary") and e.get("verified")),
                None,
            )
            if not primary_email:
                primary_email = next(
                    (e["email"] for e in emails if e.get("verified")),
                    None,
                )
            user_info["email"] = primary_email

    return user_info


async def get_or_create_github_user(db: AsyncSession, info: dict) -> User:
    """
    Find existing user by github_id or email, or create new one.

    GitHub user info keys used:
      info["id"]    → numeric GitHub user ID (stored as string)
      info["login"] → GitHub username e.g. "johndoe"
      info["email"] → email (resolved above even if originally private)
    """
    github_id = str(info["id"])
    email     = info.get("email")
    login     = info.get("login", "")

    if not email:
        raise ValueError(
            "Could not retrieve email from GitHub. "
            "Please add a verified email to your GitHub account."
        )

    # Try by github_id (returning user)
    r    = await db.execute(select(User).where(User.github_id == github_id))
    user = r.scalar_one_or_none()
    if user:
        return user

    # Try by email — link accounts (user registered with email/password before)
    r    = await db.execute(select(User).where(User.email == email))
    user = r.scalar_one_or_none()
    if user:
        user.github_id = github_id
        await db.commit()
        return user

    # New user — create account
    username = f"{login}_{str(uuid.uuid4())[:6]}" if login else str(uuid.uuid4())[:12]
    user = User(
        email           = email,
        username        = username,
        github_id       = github_id,
        hashed_password = None,
        tier            = "free",
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user
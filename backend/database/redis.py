"""
database/redis.py — Upstash Redis client.

Two systems:
  1. Exact-match response cache  (SHA-256 of text input → cached response)
  2. Sliding window rate limiter (per user, per tier — atomic Lua script)
  3. Session history cache       (active chat history per session)
"""

import hashlib
import json
import time
from typing import Optional

from upstash_redis.asyncio import Redis

from backend.config import get_settings

settings = get_settings()

# ── Client ────────────────────────────────────────────────────────────────────

redis = Redis(
    url   = settings.upstash_redis_url,
    token = settings.upstash_redis_token,
)

# ── Constants ─────────────────────────────────────────────────────────────────

RATE_LIMITS     = {"free": 20, "pro": 40}
WINDOW_SECONDS  = 60
CACHE_TTL       = 3600   # 1 hour
SESSION_TTL     = 7200   # 2 hours inactivity

# ── Lua: Sliding Window Rate Limiter ─────────────────────────────────────────
# Atomic script — removes expired timestamps, counts window, adds if allowed.
# Returns [allowed (0|1), current_count, limit]

_SLIDING_WINDOW_LUA = """
local key    = KEYS[1]
local now    = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local limit  = tonumber(ARGV[3])
local req_id = ARGV[4]

redis.call('ZREMRANGEBYSCORE', key, 0, now - window * 1000)
local count = redis.call('ZCARD', key)

if count < limit then
    redis.call('ZADD', key, now, req_id)
    redis.call('EXPIRE', key, window + 1)
    return {1, count + 1, limit}
else
    return {0, count, limit}
end
"""


# ── Rate Limiting ─────────────────────────────────────────────────────────────

async def check_rate_limit(user_id: str, tier: str) -> dict:
    """
    Sliding window rate limit check.
    Returns: { allowed: bool, current: int, limit: int, retry_after: int }
    """
    limit  = RATE_LIMITS.get(tier, RATE_LIMITS["free"])
    now_ms = int(time.time() * 1000)
    key    = f"ratelimit:{user_id}"
    req_id = f"{now_ms}-{user_id}"

    result = await redis.eval(
        _SLIDING_WINDOW_LUA,
        keys = [key],
        args = [str(now_ms), str(WINDOW_SECONDS), str(limit), req_id],
    )

    allowed, current, lim = result
    return {
        "allowed":     bool(allowed),
        "current":     int(current),
        "limit":       int(lim),
        "retry_after": WINDOW_SECONDS if not allowed else 0,
    }


# ── Response Cache ────────────────────────────────────────────────────────────

def _cache_key(text: Optional[str], has_image: bool, has_audio: bool) -> str:
    raw    = f"{(text or '').strip().lower()}|img:{has_image}|audio:{has_audio}"
    digest = hashlib.sha256(raw.encode()).hexdigest()
    return f"cache:response:{digest}"


async def get_cached_response(
    text:      Optional[str],
    has_image: bool,
    has_audio: bool,
) -> Optional[dict]:
    """Return cached response if it exists. Multimodal inputs are never cached."""
    if has_image or has_audio:
        return None
    data = await redis.get(_cache_key(text, has_image, has_audio))
    return json.loads(data) if data else None


async def set_cached_response(
    text:      Optional[str],
    has_image: bool,
    has_audio: bool,
    response:  dict,
) -> None:
    """Cache a text-only response with TTL."""
    if has_image or has_audio:
        return
    await redis.setex(_cache_key(text, False, False), CACHE_TTL, json.dumps(response))


# ── Session History Cache ──────────────────────────────────────────────────────

async def get_session_history(session_id: str) -> list:
    """Retrieve active session chat history from Redis."""
    data = await redis.get(f"session:history:{session_id}")
    return json.loads(data) if data else []


async def set_session_history(session_id: str, history: list) -> None:
    """Save session chat history to Redis with inactivity TTL."""
    await redis.setex(f"session:history:{session_id}", SESSION_TTL, json.dumps(history))


async def delete_session_history(session_id: str) -> None:
    """Remove session history when session ends."""
    await redis.delete(f"session:history:{session_id}")
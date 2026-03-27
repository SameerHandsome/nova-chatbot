"""
tests/unit/test_rate_limit.py — Sliding window rate limiter unit tests.
Tests tier configs, allow/block logic, retry_after values.
"""

import pytest
from unittest.mock import AsyncMock, patch

from backend.database.redis import check_rate_limit, RATE_LIMITS, WINDOW_SECONDS


class TestRateLimitConfig:
    def test_free_tier_limit(self):
        assert RATE_LIMITS["free"] == 20

    def test_pro_tier_limit(self):
        assert RATE_LIMITS["pro"] == 40

    def test_window_is_60_seconds(self):
        assert WINDOW_SECONDS == 60

    def test_pro_limit_greater_than_free(self):
        assert RATE_LIMITS["pro"] > RATE_LIMITS["free"]

    def test_unknown_tier_falls_back_to_free(self):
        limit = RATE_LIMITS.get("enterprise", RATE_LIMITS["free"])
        assert limit == RATE_LIMITS["free"]


class TestCheckRateLimit:
    @pytest.mark.asyncio
    async def test_allowed_returns_correct_shape(self):
        with patch("backend.database.redis.redis") as mock_redis:
            mock_redis.eval = AsyncMock(return_value=[1, 5, 20])
            result = await check_rate_limit("user-123", "free")

        assert result["allowed"]     is True
        assert result["current"]     == 5
        assert result["limit"]       == 20
        assert result["retry_after"] == 0

    @pytest.mark.asyncio
    async def test_blocked_returns_retry_after(self):
        with patch("backend.database.redis.redis") as mock_redis:
            mock_redis.eval = AsyncMock(return_value=[0, 20, 20])
            result = await check_rate_limit("user-123", "free")

        assert result["allowed"]     is False
        assert result["current"]     == 20
        assert result["retry_after"] == WINDOW_SECONDS

    @pytest.mark.asyncio
    async def test_pro_tier_uses_higher_limit(self):
        with patch("backend.database.redis.redis") as mock_redis:
            mock_redis.eval = AsyncMock(return_value=[1, 1, 40])
            result = await check_rate_limit("user-pro", "pro")

        assert result["limit"] == 40

    @pytest.mark.asyncio
    async def test_lua_script_called_with_correct_keys(self):
        with patch("backend.database.redis.redis") as mock_redis:
            mock_redis.eval = AsyncMock(return_value=[1, 1, 20])
            await check_rate_limit("user-abc", "free")

        call_kwargs = mock_redis.eval.call_args
        keys = call_kwargs.kwargs.get("keys") or call_kwargs[1].get("keys", [])
        assert any("ratelimit:user-abc" in str(k) for k in keys)

    @pytest.mark.asyncio
    async def test_first_request_always_allowed(self):
        """Simulate a brand new user — first request must be allowed."""
        with patch("backend.database.redis.redis") as mock_redis:
            mock_redis.eval = AsyncMock(return_value=[1, 1, 20])
            result = await check_rate_limit("brand-new-user", "free")

        assert result["allowed"] is True
        assert result["current"] == 1

    @pytest.mark.asyncio
    async def test_at_limit_boundary_blocked(self):
        """Exactly at limit — next request must be blocked."""
        with patch("backend.database.redis.redis") as mock_redis:
            mock_redis.eval = AsyncMock(return_value=[0, 20, 20])
            result = await check_rate_limit("user-at-limit", "free")

        assert result["allowed"] is False
"""
tests/unit/test_cache.py — Redis cache unit tests.
Cache key generation, hit/miss, multimodal bypass.
"""

import json
import pytest
from unittest.mock import AsyncMock, patch

from backend.database.redis import (
    _cache_key,
    get_cached_response,
    set_cached_response,
)


class TestCacheKeyGeneration:
    def test_same_text_same_key(self):
        """Case-insensitive and whitespace-normalized."""
        k1 = _cache_key("Hello World", False, False)
        k2 = _cache_key("hello world", False, False)
        assert k1 == k2

    def test_different_text_different_key(self):
        k1 = _cache_key("What is AI?", False, False)
        k2 = _cache_key("What is ML?", False, False)
        assert k1 != k2

    def test_key_starts_with_prefix(self):
        k = _cache_key("hello", False, False)
        assert k.startswith("cache:response:")

    def test_modality_flags_affect_key(self):
        k1 = _cache_key("hello", False, False)
        k2 = _cache_key("hello", True,  False)
        k3 = _cache_key("hello", False, True)
        assert k1 != k2
        assert k1 != k3
        assert k2 != k3

    def test_none_text_handled(self):
        k = _cache_key(None, False, False)
        assert k.startswith("cache:response:")


class TestGetCachedResponse:
    @pytest.mark.asyncio
    async def test_cache_miss_returns_none(self):
        with patch("backend.database.redis.redis") as mock_redis:
            mock_redis.get = AsyncMock(return_value=None)
            result = await get_cached_response("hello", False, False)
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_hit_returns_dict(self):
        payload = json.dumps({"response": "Hello back!"})
        with patch("backend.database.redis.redis") as mock_redis:
            mock_redis.get = AsyncMock(return_value=payload)
            result = await get_cached_response("hello", False, False)
        assert result == {"response": "Hello back!"}

    @pytest.mark.asyncio
    async def test_image_input_bypasses_cache(self):
        """Multimodal inputs must never be served from cache."""
        with patch("backend.database.redis.redis") as mock_redis:
            mock_redis.get = AsyncMock(return_value=json.dumps({"response": "cached"}))
            result = await get_cached_response("text", True, False)
        assert result is None
        mock_redis.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_audio_input_bypasses_cache(self):
        with patch("backend.database.redis.redis") as mock_redis:
            mock_redis.get = AsyncMock(return_value=json.dumps({"response": "cached"}))
            result = await get_cached_response("text", False, True)
        assert result is None
        mock_redis.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_both_modalities_bypass_cache(self):
        with patch("backend.database.redis.redis") as mock_redis:
            mock_redis.get = AsyncMock(return_value=json.dumps({"response": "cached"}))
            result = await get_cached_response("text", True, True)
        assert result is None
        mock_redis.get.assert_not_called()


class TestSetCachedResponse:
    @pytest.mark.asyncio
    async def test_text_only_gets_cached(self):
        with patch("backend.database.redis.redis") as mock_redis:
            mock_redis.setex = AsyncMock(return_value=True)
            await set_cached_response("hello", False, False, {"response": "Hi"})
        mock_redis.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_image_input_not_cached(self):
        with patch("backend.database.redis.redis") as mock_redis:
            mock_redis.setex = AsyncMock()
            await set_cached_response("text", True, False, {"response": "Hi"})
        mock_redis.setex.assert_not_called()

    @pytest.mark.asyncio
    async def test_audio_input_not_cached(self):
        with patch("backend.database.redis.redis") as mock_redis:
            mock_redis.setex = AsyncMock()
            await set_cached_response("text", False, True, {"response": "Hi"})
        mock_redis.setex.assert_not_called()

    @pytest.mark.asyncio
    async def test_cached_value_is_valid_json(self):
        stored = None
        async def capture_setex(key, ttl, value):
            nonlocal stored
            stored = value
            return True

        with patch("backend.database.redis.redis") as mock_redis:
            mock_redis.setex = capture_setex
            await set_cached_response("test", False, False, {"response": "answer"})

        parsed = json.loads(stored)
        assert parsed["response"] == "answer"
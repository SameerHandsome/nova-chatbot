"""
tests/integration/test_chat_api.py — Chat endpoint integration tests.
Full HTTP round-trips with mocked DB, Redis, and LangGraph.
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

from backend.security import create_access_token
from backend.database.postgres import User, Session as DBSession


def _auth_header(tier="free"):
    uid   = str(uuid.uuid4())
    token = create_access_token(uid, f"{tier}@test.com", tier)
    return {"Authorization": f"Bearer {token}"}, uid


class TestChatEndpointAuth:
    @pytest.mark.asyncio
    async def test_unauthenticated_request_returns_401(self, async_client):
        res = await async_client.post("/chat", json={"text": "hello"})
        assert res.status_code == 401

    @pytest.mark.asyncio
    async def test_invalid_token_returns_401(self, async_client):
        res = await async_client.post(
            "/chat",
            json={"text": "hello"},
            headers={"Authorization": "Bearer not.a.real.token"},
        )
        assert res.status_code == 401


class TestChatEndpointValidation:
    @pytest.mark.asyncio
    async def test_empty_payload_returns_400(self, async_client, mock_redis_ok):
        headers, uid = _auth_header()

        mock_user    = MagicMock(spec=User)
        mock_user.id = uuid.UUID(uid)
        mock_user.tier = "free"

        with patch("backend.api.chat.get_current_user", return_value=mock_user), \
             patch("backend.api.chat.check_rate_limit", return_value={"allowed": True, "current": 1, "limit": 20, "retry_after": 0}):
            res = await async_client.post("/chat", json={}, headers=headers)

        assert res.status_code == 400
        assert "input" in res.json()["error"].lower()


class TestChatEndpointRateLimit:
    @pytest.mark.asyncio
    async def test_rate_limited_user_gets_429(self, async_client):
        headers, uid = _auth_header()

        mock_user      = MagicMock(spec=User)
        mock_user.id   = uuid.UUID(uid)
        mock_user.tier = "free"

        with patch("backend.api.chat.get_current_user", return_value=mock_user), \
             patch("backend.api.chat.check_rate_limit", return_value={
                 "allowed": False, "current": 20, "limit": 20, "retry_after": 60
             }):
            res = await async_client.post("/chat", json={"text": "hello"}, headers=headers)

        assert res.status_code == 429
        assert "Retry-After" in res.headers

    @pytest.mark.asyncio
    async def test_pro_user_has_higher_limit(self, async_client):
        """Pro users get 40 req/min vs 20 for free."""
        from backend.database.redis import RATE_LIMITS
        assert RATE_LIMITS["pro"] == 40
        assert RATE_LIMITS["free"] == 20


class TestChatEndpointSuccess:
    @pytest.mark.asyncio
    async def test_text_message_returns_200(self, async_client, mock_redis_ok, mock_graph):
        headers, uid = _auth_header()

        mock_user      = MagicMock(spec=User)
        mock_user.id   = uuid.UUID(uid)
        mock_user.tier = "free"

        mock_session    = MagicMock(spec=DBSession)
        mock_session.id = uuid.uuid4()

        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=MagicMock(scalar_one_or_none=lambda: None))
        mock_db.flush   = AsyncMock()
        mock_db.add     = MagicMock()
        mock_db.commit  = AsyncMock()

        with patch("backend.api.chat.get_current_user",   return_value=mock_user), \
             patch("backend.api.chat.check_rate_limit",   return_value={"allowed": True, "current": 1, "limit": 20, "retry_after": 0}), \
             patch("backend.api.chat.get_cached_response", return_value=None), \
             patch("backend.api.chat.get_session_history", return_value=[]), \
             patch("backend.api.chat.set_session_history", new_callable=lambda: AsyncMock), \
             patch("backend.api.chat.set_cached_response", new_callable=lambda: AsyncMock), \
             patch("backend.api.chat.get_db",             return_value=mock_db):

            res = await async_client.post(
                "/chat",
                json={"text": "Hello there"},
                headers=headers,
            )

        assert res.status_code == 200
        data = res.json()
        assert "response"   in data
        assert "session_id" in data
        assert "message_id" in data
        assert data["cache_hit"] is False

    @pytest.mark.asyncio
    async def test_cache_hit_returns_cached_response(self, async_client, mock_redis_ok):
        headers, uid = _auth_header()

        mock_user      = MagicMock(spec=User)
        mock_user.id   = uuid.UUID(uid)
        mock_user.tier = "free"

        with patch("backend.api.chat.get_current_user",    return_value=mock_user), \
             patch("backend.api.chat.check_rate_limit",    return_value={"allowed": True, "current": 1, "limit": 20, "retry_after": 0}), \
             patch("backend.api.chat.get_cached_response", return_value={"response": "Cached answer!"}):

            res = await async_client.post(
                "/chat",
                json={"text": "cached question"},
                headers=headers,
            )

        assert res.status_code == 200
        data = res.json()
        assert data["cache_hit"]     is True
        assert data["response"]      == "Cached answer!"
        assert data["latency_ms"]    == 0
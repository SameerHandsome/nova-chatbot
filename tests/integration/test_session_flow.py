"""
tests/integration/test_session_flow.py — Session lifecycle integration tests.
Tests the full flow: chat → end session → preferences generated.
"""

import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.security import create_access_token
from backend.database.postgres import User, Session as DBSession, UserPreference


def _auth_header(tier="free"):
    uid   = str(uuid.uuid4())
    token = create_access_token(uid, f"{tier}@test.com", tier)
    return {"Authorization": f"Bearer {token}"}, uid


class TestSessionEnd:
    @pytest.mark.asyncio
    async def test_end_session_unauthenticated_returns_401(self, async_client):
        res = await async_client.post("/sessions/end", json={"session_id": str(uuid.uuid4())})
        assert res.status_code == 401

    @pytest.mark.asyncio
    async def test_end_nonexistent_session_returns_404(self, async_client):
        headers, uid = _auth_header()

        mock_user      = MagicMock(spec=User)
        mock_user.id   = uuid.UUID(uid)
        mock_user.tier = "free"

        # Simulate end_session returning error
        with patch("backend.api.sessions.get_current_user", return_value=mock_user), \
             patch("backend.api.sessions.end_session",       return_value={"error": "Session not found"}), \
             patch("backend.api.sessions.delete_session_history", new_callable=AsyncMock):

            res = await async_client.post(
                "/sessions/end",
                json={"session_id": str(uuid.uuid4())},
                headers=headers,
            )

        assert res.status_code == 404

    @pytest.mark.asyncio
    async def test_end_session_returns_summary_and_preferences(self, async_client):
        headers, uid = _auth_header()

        mock_user      = MagicMock(spec=User)
        mock_user.id   = uuid.UUID(uid)
        mock_user.tier = "free"

        session_id = str(uuid.uuid4())
        mock_result = {
            "session_id":  session_id,
            "summary":     "User asked about AI and Python. Answered both questions.",
            "preferences": {
                "communication_style":       "casual",
                "topics_of_interest":        ["AI", "Python"],
                "preferred_response_length": "medium",
                "language":                  "en",
                "uses_voice":                False,
                "uses_images":               False,
            },
        }

        with patch("backend.api.sessions.get_current_user",       return_value=mock_user), \
             patch("backend.api.sessions.end_session",             return_value=mock_result), \
             patch("backend.api.sessions.delete_session_history",  new_callable=AsyncMock):

            res = await async_client.post(
                "/sessions/end",
                json={"session_id": session_id},
                headers=headers,
            )

        assert res.status_code == 200
        data = res.json()
        assert data["summary"]             == mock_result["summary"]
        assert data["session_id"]          == session_id
        assert data["preferences"]["language"] == "en"


class TestListSessions:
    @pytest.mark.asyncio
    async def test_list_sessions_returns_200(self, async_client):
        headers, uid = _auth_header()

        mock_user      = MagicMock(spec=User)
        mock_user.id   = uuid.UUID(uid)
        mock_user.tier = "free"

        mock_sessions = []   # empty list is valid

        mock_db     = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = mock_sessions
        mock_db.execute = AsyncMock(return_value=mock_result)

        with patch("backend.api.sessions.get_current_user", return_value=mock_user), \
             patch("backend.api.sessions.get_db",           return_value=mock_db):
            res = await async_client.get("/sessions", headers=headers)

        assert res.status_code == 200
        assert isinstance(res.json(), list)

    @pytest.mark.asyncio
    async def test_list_sessions_unauthenticated_returns_401(self, async_client):
        res = await async_client.get("/sessions")
        assert res.status_code == 401
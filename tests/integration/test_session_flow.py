import pytest
from unittest.mock import AsyncMock, patch
from backend.main import app
from backend.security import get_current_user

class TestSessionEnd:
    @pytest.mark.asyncio
    async def test_end_session_unauthenticated_returns_401(self, async_client):
        # 1. Temporarily remove the global auth mock so this actually fails
        del app.dependency_overrides[get_current_user]
        
        res = await async_client.post("/api/sessions/end", json={"session_id": "1234"})
        assert res.status_code == 401

    @pytest.mark.asyncio
    async def test_end_nonexistent_session_returns_404(self, async_client):
        # 2. Mock the actual end_session logic to return the error
        with patch("backend.api.sessions.end_session", return_value={"error": "Session not found"}), \
             patch("backend.api.sessions.delete_session_history", new_callable=AsyncMock):
            
            res = await async_client.post(
                "/api/sessions/end",
                json={"session_id": "12345678-1234-5678-1234-567812345678"},
            )
        assert res.status_code == 404

class TestListSessions:
    @pytest.mark.asyncio
    async def test_list_sessions_returns_200(self, async_client):
        # 3. Because of conftest.py, we don't need to mock anything here! 
        # The mock_db automatically returns the empty list for .scalars().all()
        res = await async_client.get("/api/sessions")
        assert res.status_code == 200
import pytest
from unittest.mock import AsyncMock, patch

class TestChatEndpoint:
    @pytest.mark.asyncio
    async def test_text_message_returns_200(self, async_client, mock_redis_ok):
        # 1. Update target to 'ainvoke' and explicitly make all awaited functions AsyncMocks
        with patch("backend.api.chat.chatbot_graph.ainvoke", new_callable=AsyncMock) as mock_ainvoke, \
             patch("backend.api.chat.check_rate_limit", new_callable=AsyncMock) as mock_rl, \
             patch("backend.api.chat.get_cached_response", new_callable=AsyncMock) as mock_cache, \
             patch("backend.api.chat.set_session_history", new_callable=AsyncMock):

            # Setup the mocked LLM response
            mock_ainvoke.return_value = {
                "final_response": "Mocked response",
                "tools_called": [],
                "merged_input": "hello"
            }
            
            # Setup rate limit and cache mock return values
            mock_rl.return_value = {"allowed": True, "current": 1, "limit": 20}
            mock_cache.return_value = None

            # Execute the request
            res = await async_client.post(
                "/api/chat",
                json={"text": "Hello there"}
            )

        # Assertions
        assert res.status_code == 200
        data = res.json()
        assert data["response"] == "Mocked response"
        assert data["cache_hit"] is False

    @pytest.mark.asyncio
    async def test_unauthenticated_request_returns_401(self, async_client):
        from backend.security import get_current_user
        from backend.main import app
        
        # Safely store the original override so we can put it back
        original_override = app.dependency_overrides.get(get_current_user)
        
        # Remove the mock just for this one check
        if get_current_user in app.dependency_overrides:
            del app.dependency_overrides[get_current_user]
        
        try:
            res = await async_client.post("/api/chat", json={"text": "hello"})
            assert res.status_code == 401
        finally:
            # RESTORE the override, otherwise all tests that run after this one will fail with 401s!
            if original_override:
                app.dependency_overrides[get_current_user] = original_override
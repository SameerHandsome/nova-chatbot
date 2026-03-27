"""
tests/integration/conftest.py — Integration-only fixtures.
Uses real AsyncClient pointed at the FastAPI app.
"""

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, patch

from backend.main import app


@pytest_asyncio.fixture
async def async_client():
    """Real HTTP client for integration tests."""
    async with AsyncClient(
        transport = ASGITransport(app=app),
        base_url  = "http://test",
    ) as client:
        yield client


@pytest.fixture
def mock_redis_ok():
    """Patch Redis so rate limit always allows and cache always misses."""
    with patch("backend.database.redis.redis") as r:
        r.eval   = AsyncMock(return_value=[1, 1, 20])   # allowed
        r.get    = AsyncMock(return_value=None)          # cache miss
        r.setex  = AsyncMock(return_value=True)
        r.delete = AsyncMock(return_value=True)
        yield r


@pytest.fixture
def mock_graph():
    """Patch the LangGraph pipeline to return a deterministic response."""
    with patch("backend.api.chat.chatbot_graph") as g:
        g.invoke = lambda state: {
            **state,
            "final_response":    "Mocked response from graph",
            "merged_input":      state.get("raw_text") or "input",
            "transcribed_text":  None,
            "image_description": None,
            "tools_called":      [],
            "tool_results":      [],
            "error":             None,
        }
        yield g
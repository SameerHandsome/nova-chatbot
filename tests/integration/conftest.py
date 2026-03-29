import asyncio
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, MagicMock

from backend.main import app
from backend.security import get_current_user
from backend.database import get_db

# 1. FIX: Event Loop for Windows
@pytest.fixture(scope="session")
def event_loop():
    """Create a persistent event loop for the duration of the test session."""
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

# In tests/conftest.py
@pytest_asyncio.fixture(scope="function")
async def async_client():
    mock_user = MagicMock()
    mock_user.id = "8bf43689-dd9f-472e-8093-d35c2b63ed37"
    mock_user.tier = "free"
    mock_user.email = "test@test.com"

    # --- THE FIX: Smarter Database Mock ---
    mock_db = AsyncMock()
    mock_db.add = MagicMock() # db.add() is synchronous (Fixes the RuntimeWarning)
    
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = [] # Fixes the .all() error
    mock_result.scalar_one_or_none.return_value = None
    mock_db.execute.return_value = mock_result
    # --------------------------------------

    app.dependency_overrides[get_current_user] = lambda: mock_user
    app.dependency_overrides[get_db] = lambda: mock_db

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client

    app.dependency_overrides.clear()

@pytest.fixture
def mock_redis_ok():
    with pytest.MonkeyPatch.context() as mp:
        mock = AsyncMock()
        mock.eval = AsyncMock(return_value=[1, 1, 20])
        mock.get = AsyncMock(return_value=None)
        mock.setex = AsyncMock(return_value=True)
        # Use monkeypatch to target where redis is imported in your database module
        yield mock
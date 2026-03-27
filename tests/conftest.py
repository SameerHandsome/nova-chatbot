"""
tests/conftest.py — Shared fixtures for ALL tests (unit + integration).
"""

import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock
from backend.security import create_access_token
from backend.database.postgres import User


@pytest.fixture
def fake_user_id():
    return str(uuid.uuid4())


@pytest.fixture
def free_user_token(fake_user_id):
    return create_access_token(fake_user_id, "free@test.com", "free")


@pytest.fixture
def pro_user_token(fake_user_id):
    return create_access_token(fake_user_id, "pro@test.com", "pro")


@pytest.fixture
def mock_user(fake_user_id):
    user          = MagicMock(spec=User)
    user.id       = uuid.UUID(fake_user_id)
    user.email    = "test@example.com"
    user.username = "testuser"
    user.tier     = "free"
    user.is_active= True
    return user


@pytest.fixture
def mock_db():
    db = AsyncMock()
    db.commit   = AsyncMock()
    db.rollback = AsyncMock()
    db.flush    = AsyncMock()
    db.add      = MagicMock()
    db.execute  = AsyncMock()
    db.close    = AsyncMock()
    return db
"""
database/__init__.py — Clean re-exports.
Routes import from backend.database, never from internal files directly.
"""

from backend.database.postgres import (
    get_db,
    init_db,
    User,
    Session,
    Message,
    UserPreference,
)

from backend.database.redis import (
    check_rate_limit,
    get_cached_response,
    set_cached_response,
    get_session_history,
    set_session_history,
    delete_session_history,
)

__all__ = [
    # Postgres
    "get_db",
    "init_db",
    "User",
    "Session",
    "Message",
    "UserPreference",
    # Redis
    "check_rate_limit",
    "get_cached_response",
    "set_cached_response",
    "get_session_history",
    "set_session_history",
    "delete_session_history",
]
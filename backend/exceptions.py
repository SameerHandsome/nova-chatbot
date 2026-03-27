"""
exceptions.py — Named exception classes and global FastAPI handler.
Routes raise these instead of raw HTTPException for consistency.
"""

from fastapi import Request
from fastapi.responses import JSONResponse


# ── Custom Exceptions ─────────────────────────────────────────────────────────

class NovaBaseException(Exception):
    status_code: int = 500
    detail:      str = "Internal server error"

    def __init__(self, detail: str = None):
        self.detail = detail or self.__class__.detail


class InvalidCredentials(NovaBaseException):
    status_code = 401
    detail      = "Invalid email or password"


class TokenExpired(NovaBaseException):
    status_code = 401
    detail      = "Token has expired"


class TokenInvalid(NovaBaseException):
    status_code = 401
    detail      = "Invalid token"


class UserNotFound(NovaBaseException):
    status_code = 404
    detail      = "User not found"


class UserAlreadyExists(NovaBaseException):
    status_code = 409
    detail      = "Email or username already registered"


class SessionNotFound(NovaBaseException):
    status_code = 404
    detail      = "Session not found or already ended"


class RateLimitExceeded(NovaBaseException):
    status_code  = 429
    detail       = "Rate limit exceeded"
    retry_after: int = 60

    def __init__(self, retry_after: int = 60, limit: int = 20):
        self.detail      = f"Rate limit exceeded ({limit} req/min). Retry in {retry_after}s."
        self.retry_after = retry_after


class InvalidInput(NovaBaseException):
    status_code = 400
    detail      = "Invalid input"


class GraphExecutionError(NovaBaseException):
    status_code = 500
    detail      = "LangGraph pipeline failed"


class PreferencesNotFound(NovaBaseException):
    status_code = 404
    detail      = "No preferences found yet. Complete and end a session first."


# ── Global Handler ────────────────────────────────────────────────────────────

async def nova_exception_handler(request: Request, exc: NovaBaseException):
    headers = {}
    if isinstance(exc, RateLimitExceeded):
        headers["Retry-After"] = str(exc.retry_after)

    return JSONResponse(
        status_code = exc.status_code,
        content     = {"error": exc.detail, "status_code": exc.status_code},
        headers     = headers,
    )
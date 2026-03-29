from fastapi import APIRouter
from backend.api.auth        import router as auth_router
from backend.api.chat        import router as chat_router
from backend.api.sessions    import router as sessions_router
from backend.api.preferences import router as preferences_router

# Industry Practice: All API routes now start with /api
api_router = APIRouter(prefix="/api")

api_router.include_router(auth_router,        prefix="/auth",        tags=["Auth"])
api_router.include_router(chat_router,        prefix="/chat",        tags=["Chat"])
api_router.include_router(sessions_router,    prefix="/sessions",    tags=["Sessions"])
api_router.include_router(preferences_router, prefix="/preferences", tags=["Preferences"])

__all__ = ["api_router"]
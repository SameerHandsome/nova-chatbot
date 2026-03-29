from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
class Settings(BaseSettings):
    # -- Groq --
    groq_api_key: str

    # -- Neon PostgreSQL --
    database_url: str

    # -- Upstash Redis (UPDATED TO MATCH YOUR .ENV) --
    upstash_redis_url: str 
    upstash_redis_token: str

    # -- JWT --
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 1440

    # -- GitHub OAuth --
    github_client_id: str
    github_client_secret: str
    github_redirect_uri: str = "http://localhost:8000/auth/github/callback"

    # -- LangSmith --
    langchain_tracing_v2: str = "true"
    langchain_api_key: str = ""
    langchain_project: str = "nova-multimodal-chatbot"

    # -- Tools --
    tavily_api_key: str
    alpha_vantage_api_key: str

    # -- App --
    app_env: str = "development"
    frontend_url: str = "http://localhost:8000"

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )


@lru_cache()
def get_settings() -> Settings:
    return Settings()
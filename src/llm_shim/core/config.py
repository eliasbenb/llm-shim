"""Configuration models and helpers."""

from functools import lru_cache

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

__all__ = ["Settings", "get_settings"]


class ServerSettings(BaseModel):
    """Runtime server configuration for uvicorn."""

    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    workers: int = 1
    log_level: str = "info"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    server: ServerSettings = Field(default_factory=ServerSettings)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton accessor for cached settings instance.

    Returns:
        Settings: Cached settings instance.
    """
    return Settings()

"""Configuration models and helpers."""

from functools import lru_cache
from typing import Any

import instructor
from instructor import Mode
from instructor.models import KnownModelName
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["InstructorSettings", "Settings", "get_settings"]


class InstructorSettings(BaseModel):
    """Configuration for a single Instructor client."""

    model: KnownModelName | str
    api_key: str | None = None
    base_url: str | None = None
    mode: Mode | None = None
    extra: dict[str, object] = Field(default_factory=dict)

    @property
    def provider_model_name(self) -> str:
        """Return provider-native model name without provider prefix."""
        model = str(self.model)
        if "/" in model:
            return model.split("/", maxsplit=1)[1]
        return model

    def instructor_kwargs(self) -> dict[str, Any]:
        """Return kwargs forwarded to instructor.from_provider."""
        kwargs: dict[str, Any] = dict(self.extra)
        if self.api_key is not None:
            kwargs["api_key"] = self.api_key
        if self.base_url is not None:
            kwargs["base_url"] = self.base_url
        if self.mode is not None:
            kwargs["mode"] = self.mode
        return kwargs

    def create_async_client(self) -> instructor.AsyncInstructor:
        """Create an async Instructor client."""
        return instructor.from_provider(
            str(self.model),
            async_client=True,
            **self.instructor_kwargs(),
        )

    @model_validator(mode="after")
    def validate_model_format(self) -> InstructorSettings:
        """Require provider/model format expected by instructor.from_provider."""
        if "/" not in str(self.model):
            raise ValueError(
                "LLM_SHIM_INSTRUCTOR__MODEL must use provider/model format"
            )
        return self


class ServerSettings(BaseModel):
    """Runtime server configuration for uvicorn."""

    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    workers: int = 1
    log_level: str = "info"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    default_provider: str = "default"
    providers: dict[str, InstructorSettings] = Field(default_factory=dict)
    server: ServerSettings = Field(default_factory=ServerSettings)

    model_config = SettingsConfigDict(
        case_sensitive=False,
        env_file=".env",
        env_prefix="LLM_SHIM_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    @model_validator(mode="after")
    def validate_providers(self) -> Settings:
        """Validate multi-provider Instructor configuration."""
        if not self.providers:
            raise ValueError(
                "LLM_SHIM_PROVIDERS must define at least one configured provider"
            )

        if self.default_provider not in self.providers:
            raise ValueError(
                "LLM_SHIM_DEFAULT_PROVIDER must be a key in LLM_SHIM_PROVIDERS"
            )

        return self

    def resolve_provider(
        self,
        requested_model: str | None,
    ) -> tuple[str, InstructorSettings]:
        """Resolve configured provider by alias or exact configured model id."""
        if requested_model is None:
            provider_name = self.default_provider
            return provider_name, self.providers[provider_name]

        if requested_model in self.providers:
            provider_name = requested_model
            return provider_name, self.providers[provider_name]

        for provider_name, provider in self.providers.items():
            if requested_model == str(provider.model):
                return provider_name, provider

        raise ValueError(
            "Requested model is not configured. Use a provider alias or exact "
            "configured model id."
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton accessor for cached settings instance.

    Returns:
        Settings: Cached settings instance.
    """
    return Settings()

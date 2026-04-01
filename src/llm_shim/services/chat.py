"""Service for OpenAI-compatible chat completions."""

import time
from functools import lru_cache
from uuid import uuid4

from llm_shim.api.models import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseMessage,
    ChatCompletionUsage,
    JsonSchemaModelFactory,
    ResponseFormatJsonSchema,
)
from llm_shim.core.config import Settings, get_settings


class ChatService:
    """Service for creating OpenAI-compatible chat completion responses."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize service dependencies."""
        self._settings = settings or get_settings()

    @staticmethod
    def _build_response(
        configured_model: str,
        content: str,
    ) -> ChatCompletionResponse:
        """Build an OpenAI-compatible chat completion response."""
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid4().hex}",
            created=int(time.time()),
            model=configured_model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionResponseMessage(content=content),
                    finish_reason="stop",
                ),
            ],
            usage=ChatCompletionUsage(),
        )

    async def create(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Create a provider chat completion normalized to OpenAI response format.

        Args:
            request (ChatCompletionRequest): OpenAI-compatible request.

        Returns:
            ChatCompletionResponse: OpenAI-compatible chat completion response.
        """
        _, provider = self._settings.resolve_provider(request.model)
        configured_model = str(provider.model)

        try:
            client = provider.create_async_client()
        except Exception as error:
            raise RuntimeError(
                f"Failed to initialize provider client: {error}"
            ) from error

        create_kwargs = request.instructor_create_kwargs()

        if isinstance(request.response_format, ResponseFormatJsonSchema):
            try:
                dynamic_model = JsonSchemaModelFactory.build_model(
                    request.response_format.json_schema,
                )
                structured_response = await client.create(
                    response_model=dynamic_model,
                    **create_kwargs,
                )
            except ValueError:
                raise
            except Exception as error:
                raise RuntimeError(
                    f"Provider chat completion failed: {error}"
                ) from error

            return self._build_response(
                configured_model=configured_model,
                content=structured_response.model_dump_json(),
            )

        try:
            response_text = await client.create(response_model=str, **create_kwargs)
        except Exception as error:
            raise RuntimeError(f"Provider chat completion failed: {error}") from error

        return self._build_response(
            configured_model=configured_model,
            content=response_text,
        )


@lru_cache(maxsize=1)
def get_chat_service() -> ChatService:
    """Singleton factory for ChatService.

    Returns:
        ChatService: Cached service instance.
    """
    return ChatService()

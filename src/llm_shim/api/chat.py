"""Chat completion API endpoint."""

import time
from uuid import uuid4

from fastapi import APIRouter, HTTPException

from llm_shim.api.models import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseMessage,
    ChatCompletionUsage,
    JsonSchemaModelFactory,
    ResponseFormatJsonSchema,
)
from llm_shim.core.config import get_settings

__all__ = ["router"]

router = APIRouter(tags=["chat"])


@router.post("/v1/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
) -> ChatCompletionResponse:
    """Create an OpenAI-compatible chat completion response.

    Args:
        request (ChatCompletionRequest): OpenAI chat completion request.

    Returns:
        ChatCompletionResponse: OpenAI chat completion response.
    """
    settings = get_settings()

    try:
        _, provider = settings.resolve_provider(request.model)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    configured_model = str(provider.model)

    try:
        client = provider.create_async_client()
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize provider client: {error}",
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
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except Exception as error:
            raise HTTPException(
                status_code=502,
                detail=f"Provider chat completion failed: {error}",
            ) from error

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid4().hex}",
            created=int(time.time()),
            model=configured_model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionResponseMessage(
                        content=structured_response.model_dump_json(),
                    ),
                    finish_reason="stop",
                ),
            ],
            usage=ChatCompletionUsage(),
        )

    try:
        response_text = await client.create(response_model=str, **create_kwargs)
    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=f"Provider chat completion failed: {error}",
        ) from error

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid4().hex}",
        created=int(time.time()),
        model=configured_model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatCompletionResponseMessage(content=response_text),
                finish_reason="stop",
            ),
        ],
        usage=ChatCompletionUsage(),
    )

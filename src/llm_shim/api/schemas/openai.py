"""Pydantic models for OpenAI-compatible request and response payloads."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class ChatMessage(BaseModel):
    """OpenAI-style chat message payload."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str | None = None
    tool_call_id: str | None = None


class ResponseFormatText(BaseModel):
    """OpenAI response format for plain text."""

    type: Literal["text"]


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model_config = ConfigDict(extra="allow")

    model: str | None = None
    messages: list[ChatMessage]
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    n: int = 1
    stream: Literal[False] = False
    response_format: ResponseFormatText | None = None
    user: str | None = None

    def chat_kwargs(self) -> dict[str, Any]:
        """Build kwargs forwarded to the pydantic-ai chat model."""
        kwargs: dict[str, Any] = {
            "messages": [
                message.model_dump(exclude_none=True) for message in self.messages
            ],
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.user is not None:
            kwargs["user"] = self.user
        return kwargs


class ChatCompletionUsage(BaseModel):
    """OpenAI-compatible token usage details."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponseMessage(BaseModel):
    """Assistant message in completion choices."""

    role: Literal["assistant"] = "assistant"
    content: str


class ChatCompletionChoice(BaseModel):
    """A chat completion choice entry."""

    index: int = 0
    message: ChatCompletionResponseMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage


class EmbeddingsRequest(BaseModel):
    """OpenAI-compatible embeddings request."""

    model_config = ConfigDict(extra="allow")

    model: str | None = None
    input: str | list[str]
    encoding_format: Literal["float"] | None = None
    dimensions: int | None = None
    user: str | None = None


class EmbeddingDatum(BaseModel):
    """OpenAI-compatible single embedding entry."""

    object: Literal["embedding"] = "embedding"
    embedding: list[float]
    index: int


class EmbeddingsUsage(BaseModel):
    """OpenAI-compatible embeddings usage section."""

    prompt_tokens: int = 0
    total_tokens: int = 0


class EmbeddingsResponse(BaseModel):
    """OpenAI-compatible embeddings response."""

    object: Literal["list"] = "list"
    data: list[EmbeddingDatum]
    model: str
    usage: EmbeddingsUsage


class ModelListItem(BaseModel):
    """OpenAI-compatible model card entry."""

    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str


class ModelListResponse(BaseModel):
    """OpenAI-compatible response payload for listing models."""

    object: Literal["list"] = "list"
    data: list[ModelListItem]

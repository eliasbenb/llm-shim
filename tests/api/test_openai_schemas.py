"""Tests for API models and schema handling."""

from typing import Any, cast

import pytest
from pydantic import ValidationError

from llm_shim.api.schemas.openai import (
    ChatCompletionRequest,
    ChatMessage,
    EmbeddingsRequest,
)


def test_chat_message_all_fields() -> None:
    """ChatMessage should accept all optional fields."""
    msg = ChatMessage(
        role="user",
        content="hello",
        name="test-user",
        tool_call_id="call-123",
    )
    assert msg.role == "user"
    assert msg.content == "hello"
    assert msg.name == "test-user"
    assert msg.tool_call_id == "call-123"


def test_chat_message_required_fields() -> None:
    """ChatMessage should require role and content."""
    msg = ChatMessage(role="assistant", content="response")
    assert msg.name is None
    assert msg.tool_call_id is None


def test_chat_message_invalid_role() -> None:
    """ChatMessage should validate role values."""
    with pytest.raises(ValidationError):
        ChatMessage(role=cast(Any, "invalid"), content="hello")


def test_chat_completion_request_minimal() -> None:
    """ChatCompletionRequest should work with minimal fields."""
    request = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="hello")]
    )
    assert request.model is None
    assert request.stream is False
    assert request.n == 1


def test_chat_completion_request_all_fields() -> None:
    """ChatCompletionRequest should accept all fields."""
    request = ChatCompletionRequest(
        model="gpt-4",
        messages=[ChatMessage(role="user", content="hello")],
        temperature=0.7,
        top_p=0.9,
        max_tokens=100,
        user="user-123",
    )
    assert request.model == "gpt-4"
    assert request.temperature == 0.7
    assert request.top_p == 0.9
    assert request.max_tokens == 100
    assert request.user == "user-123"


def test_chat_completion_request_extra_fields_allowed() -> None:
    """ChatCompletionRequest should allow extra fields."""
    request = ChatCompletionRequest.model_validate(
        {
            "messages": [ChatMessage(role="user", content="hello")],
            "frequency_penalty": 0.5,
        }
    )
    assert hasattr(request, "frequency_penalty")


def test_chat_completion_request_chat_kwargs() -> None:
    """chat_kwargs should format correctly."""
    request = ChatCompletionRequest(
        messages=[
            ChatMessage(role="system", content="You are helpful"),
            ChatMessage(role="user", content="hello"),
        ],
        temperature=0.5,
        top_p=0.8,
        max_tokens=50,
        user="user-1",
    )

    kwargs = request.chat_kwargs()

    assert kwargs["messages"] == [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "hello"},
    ]
    assert kwargs["temperature"] == 0.5
    assert kwargs["top_p"] == 0.8
    assert kwargs["max_tokens"] == 50
    assert kwargs["user"] == "user-1"


def test_chat_completion_request_chat_kwargs_excludes_none() -> None:
    """chat_kwargs should exclude None values."""
    request = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="hello")],
    )

    kwargs = request.chat_kwargs()

    assert "temperature" not in kwargs
    assert "top_p" not in kwargs
    assert "max_tokens" not in kwargs
    assert "user" not in kwargs


def test_embeddings_request_single_input() -> None:
    """EmbeddingsRequest should accept single string input."""
    request = EmbeddingsRequest(input="hello world")
    assert request.input == "hello world"


def test_embeddings_request_list_input() -> None:
    """EmbeddingsRequest should accept list input."""
    request = EmbeddingsRequest(input=["hello", "world"])
    assert request.input == ["hello", "world"]


def test_embeddings_request_all_fields() -> None:
    """EmbeddingsRequest should accept all fields."""
    request = EmbeddingsRequest(
        model="embedding-model",
        input=["text1", "text2"],
        encoding_format="float",
        dimensions=1536,
        user="user-1",
    )
    assert request.model == "embedding-model"
    assert request.dimensions == 1536
    assert request.encoding_format == "float"
    assert request.user == "user-1"


def test_chat_completion_usage_defaults() -> None:
    """ChatCompletionUsage should have zero defaults."""
    from llm_shim.api.schemas.openai import ChatCompletionUsage

    usage = ChatCompletionUsage()
    assert usage.prompt_tokens == 0
    assert usage.completion_tokens == 0
    assert usage.total_tokens == 0


def test_chat_completion_response() -> None:
    """ChatCompletionResponse should build correctly."""
    from llm_shim.api.schemas.openai import (
        ChatCompletionChoice,
        ChatCompletionResponse,
        ChatCompletionResponseMessage,
        ChatCompletionUsage,
    )

    response = ChatCompletionResponse(
        id="chatcmpl-123",
        created=1234567890,
        model="gpt-4",
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatCompletionResponseMessage(content="response text"),
                finish_reason="stop",
            )
        ],
        usage=ChatCompletionUsage(),
    )

    assert response.object == "chat.completion"
    assert response.id == "chatcmpl-123"
    assert response.model == "gpt-4"
    assert len(response.choices) == 1


def test_embeddings_response() -> None:
    """EmbeddingsResponse should build correctly."""
    from llm_shim.api.schemas.openai import (
        EmbeddingDatum,
        EmbeddingsResponse,
        EmbeddingsUsage,
    )

    response = EmbeddingsResponse(
        data=[
            EmbeddingDatum(index=0, embedding=[1.0, 2.0, 3.0]),
            EmbeddingDatum(index=1, embedding=[4.0, 5.0, 6.0]),
        ],
        model="embedding-model",
        usage=EmbeddingsUsage(),
    )

    assert response.object == "list"
    assert response.model == "embedding-model"
    assert len(response.data) == 2
    assert response.data[0].object == "embedding"

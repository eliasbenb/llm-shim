from typing import Any, cast

import pytest
from pydantic_ai.usage import RunUsage

from llm_shim.api.schemas.openai import (
    ChatCompletionRequest,
    ChatMessage,
)
from llm_shim.core.exceptions import BadRequestError, ProviderCallError
from llm_shim.services.chat import ChatService


class FakeProvider:
    def __init__(self) -> None:
        self.chat_models = ["gpt-4o-mini"]
        self.embedding_models = []
        self.env = {}
        self.chat_model_settings = {
            "temperature": 0.1,
            "max_tokens": 32,
            "timeout": 30,
        }


class FakeSettings:
    def resolve_chat_provider(
        self, requested_model: str | None
    ) -> tuple[str, str, FakeProvider]:
        assert requested_model == "openai:gpt-4o-mini"
        return "openai", "gpt-4o-mini", FakeProvider()


@pytest.mark.asyncio
async def test_chat_service_returns_text_response(monkeypatch: Any) -> None:
    service = ChatService(settings=cast(Any, FakeSettings()))

    async def fake_run_text(
        model_name: str,
        prompt: str,
        model_settings: dict[str, Any] | None,
    ) -> tuple[str, RunUsage]:
        assert model_name == "openai:gpt-4o-mini"
        assert "user: hello" in prompt
        assert model_settings is not None
        assert model_settings["temperature"] == 0.1
        assert model_settings["max_tokens"] == 32
        assert model_settings["timeout"] == 30
        return "plain completion", RunUsage(input_tokens=10, output_tokens=5)

    monkeypatch.setattr(service, "_run_text_model", fake_run_text)

    response = await service.create(
        ChatCompletionRequest(
            model="openai:gpt-4o-mini",
            messages=[ChatMessage(role="user", content="hello")],
        )
    )

    assert response.object == "chat.completion"
    assert response.model == "openai:gpt-4o-mini"
    assert response.choices[0].message.content == "plain completion"
    assert response.usage.prompt_tokens == 10
    assert response.usage.completion_tokens == 5
    assert response.usage.total_tokens == 15


@pytest.mark.asyncio
async def test_chat_service_wraps_provider_errors(monkeypatch: Any) -> None:
    service = ChatService(settings=cast(Any, FakeSettings()))

    async def failing_run_text(
        model_name: str,
        prompt: str,
        model_settings: dict[str, Any] | None,
    ) -> tuple[str, RunUsage]:
        del model_name
        del prompt
        del model_settings
        raise RuntimeError("rate limited")

    monkeypatch.setattr(service, "_run_text_model", failing_run_text)

    with pytest.raises(ProviderCallError, match="Provider chat completion failed"):
        await service.create(
            ChatCompletionRequest(
                model="openai:gpt-4o-mini",
                messages=[ChatMessage(role="user", content="hello")],
            )
        )


@pytest.mark.asyncio
async def test_request_chat_model_settings_override_provider_defaults(
    monkeypatch: Any,
) -> None:
    service = ChatService(settings=cast(Any, FakeSettings()))

    async def fake_run_text(
        model_name: str,
        prompt: str,
        model_settings: dict[str, Any] | None,
    ) -> tuple[str, RunUsage]:
        del model_name
        del prompt
        assert model_settings is not None
        assert model_settings["temperature"] == 0.8
        assert model_settings["max_tokens"] == 99
        assert model_settings["timeout"] == 30
        return "ok", RunUsage()

    monkeypatch.setattr(service, "_run_text_model", fake_run_text)

    await service.create(
        ChatCompletionRequest(
            model="openai:gpt-4o-mini",
            messages=[ChatMessage(role="user", content="hello")],
            temperature=0.8,
            max_tokens=99,
        )
    )


@pytest.mark.asyncio
async def test_chat_service_requires_chat_model() -> None:
    class MissingModelSettings:
        def resolve_chat_provider(
            self, requested_model: str | None
        ) -> tuple[str, str, Any]:
            del requested_model
            raise BadRequestError(
                "Request model is required and must use provider:model format"
            )

    service = ChatService(settings=cast(Any, MissingModelSettings()))

    with pytest.raises(BadRequestError, match="Request model is required"):
        await service.create(
            ChatCompletionRequest(messages=[ChatMessage(role="user", content="hello")])
        )

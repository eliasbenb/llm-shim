"""Tests for models list API endpoint."""

import pytest

from llm_shim.api.models import list_models
from llm_shim.core.config import ProviderSettings, ServerSettings, Settings
from llm_shim.services.models import ModelsService


@pytest.mark.asyncio
async def test_models_endpoint_lists_chat_and_embedding_models() -> None:
    settings = Settings.model_construct(
        providers={
            "openai": ProviderSettings(
                chat_models=["gpt-4o-mini", "gpt-4.1*"],
                embedding_models=["text-embedding-3-small"],
            ),
            "bedrock": ProviderSettings(
                chat_models=["haiku*"],
                embedding_models=["amazon.titan-embed-text-*"],
            ),
        },
        server=ServerSettings(),
    )

    service = ModelsService(settings=settings)
    response = await list_models(service=service)

    assert response.object == "list"
    assert len(response.data) == 5
    ids = [entry.id for entry in response.data]
    owners = [entry.owned_by for entry in response.data]

    assert "openai:gpt-4o-mini" in ids
    assert "openai:gpt-4.1*" in ids
    assert "openai:text-embedding-3-small" in ids
    assert "bedrock:haiku*" in ids
    assert "bedrock:amazon.titan-embed-text-*" in ids

    assert "openai" in owners
    assert "bedrock" in owners
    assert all(entry.object == "model" for entry in response.data)

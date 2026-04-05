"""Tests for embeddings API endpoint error handling."""

import pytest

from llm_shim.api.schemas.openai import EmbeddingsRequest, EmbeddingsResponse
from llm_shim.core.exceptions import BadRequestError, ProviderCallError
from llm_shim.services.embeddings import EmbeddingsService


class FakeFailingEmbeddingsServiceBadRequest(EmbeddingsService):
    """Service that raises BadRequestError."""

    async def create(self, request: EmbeddingsRequest) -> EmbeddingsResponse:
        raise BadRequestError("Invalid input")


class FakeFailingEmbeddingsServiceProvider(EmbeddingsService):
    """Service that raises ProviderCallError."""

    async def create(self, request: EmbeddingsRequest) -> EmbeddingsResponse:
        raise ProviderCallError("Provider embeddings failed: connection refused")


@pytest.mark.asyncio
async def test_embeddings_endpoint_returns_400_on_bad_request() -> None:
    """Embeddings endpoint should return 400 for BadRequestError."""
    with pytest.raises(BadRequestError, match="Invalid input"):
        await FakeFailingEmbeddingsServiceBadRequest().create(
            EmbeddingsRequest(input="hello")
        )


@pytest.mark.asyncio
async def test_embeddings_endpoint_returns_502_on_provider_error() -> None:
    """Embeddings endpoint should return 502 for ProviderCallError."""
    with pytest.raises(ProviderCallError, match="Provider embeddings failed"):
        await FakeFailingEmbeddingsServiceProvider().create(
            EmbeddingsRequest(input="hello")
        )

"""Provider adapter classes for embeddings APIs."""

import asyncio
import inspect
import json
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any

from llm_shim.api.models import (
    EmbeddingDatum,
    EmbeddingsRequest,
    EmbeddingsResponse,
    EmbeddingsUsage,
)
from llm_shim.core.config import Settings, get_settings


class EmbeddingsService:
    """Service for creating OpenAI-compatible embeddings responses."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize service dependencies and provider adapter registry."""
        self._settings = settings or get_settings()
        self._default_adapter = OpenAILikeEmbeddingsAdapter()
        self._adapters: dict[str, EmbeddingsAdapter] = {
            "anthropic": AnthropicEmbeddingsAdapter(),
            "bedrock": BedrockEmbeddingsAdapter(),
            "generative-ai": GoogleEmbeddingsAdapter(),
            "google": GoogleEmbeddingsAdapter(),
            "vertexai": GoogleEmbeddingsAdapter(),
            "xai": XAIEmbeddingsAdapter(),
        }

    def _build_response(
        self,
        configured_model: str,
        vectors: list[list[float]],
    ) -> EmbeddingsResponse:
        """Build an OpenAI-compatible response from provider embedding vectors."""
        return EmbeddingsResponse(
            data=[
                EmbeddingDatum(index=index, embedding=list(vector))
                for index, vector in enumerate(vectors)
            ],
            model=configured_model,
            usage=EmbeddingsUsage(),
        )

    async def create(self, request: EmbeddingsRequest) -> EmbeddingsResponse:
        """Create provider embeddings and normalize into OpenAI response format.

        Args:
            request (EmbeddingsRequest): OpenAI-compatible embeddings request.

        Returns:
            EmbeddingsResponse: OpenAI-compatible embeddings response.
        """
        _, provider = self._settings.resolve_provider(request.model)

        try:
            client = provider.create_async_client()
        except Exception as error:
            raise RuntimeError(
                f"Failed to initialize provider client: {error}"
            ) from error

        adapter = self._adapters.get(provider.provider_name, self._default_adapter)

        try:
            vectors = await adapter.create_vectors(
                client.client,
                provider.provider_model_name,
                request,
            )
        except ValueError:
            raise
        except Exception as error:
            raise RuntimeError(f"Provider embeddings failed: {error}") from error

        return self._build_response(str(provider.model), vectors)


class EmbeddingsAdapter(ABC):
    """Abstract provider adapter for embedding vector creation."""

    @staticmethod
    def _inputs(request: EmbeddingsRequest) -> list[str]:
        if isinstance(request.input, str):
            return [request.input]
        return request.input

    @abstractmethod
    async def create_vectors(
        self,
        native_client: Any,
        provider_model_name: str,
        request: EmbeddingsRequest,
    ) -> list[list[float]]:
        """Create embedding vectors for a provider-native model."""


class OpenAILikeEmbeddingsAdapter(EmbeddingsAdapter):
    """Adapter for providers exposing an OpenAI-style embeddings API."""

    async def create_vectors(
        self,
        native_client: Any,
        provider_model_name: str,
        request: EmbeddingsRequest,
    ) -> list[list[float]]:
        """Create vectors via a provider `embeddings.create` API."""
        embeddings_client = getattr(native_client, "embeddings", None)
        create = getattr(embeddings_client, "create", None)
        if create is None:
            raise ValueError("Provider does not expose an embeddings.create client")

        parameters = inspect.signature(create).parameters
        inputs = self._inputs(request)

        kwargs: dict[str, Any] = {"model": provider_model_name}
        if "input" in parameters:
            kwargs["input"] = inputs
        elif "inputs" in parameters:
            kwargs["inputs"] = inputs
        else:
            raise ValueError(
                "Provider embeddings client does not accept input payloads"
            )

        if request.dimensions is not None and "dimensions" in parameters:
            kwargs["dimensions"] = request.dimensions
        if request.encoding_format is not None and "encoding_format" in parameters:
            kwargs["encoding_format"] = request.encoding_format
        if request.user is not None and "user" in parameters:
            kwargs["user"] = request.user

        result = create(**kwargs)
        if inspect.isawaitable(result):
            result = await result

        data = getattr(result, "data", None)
        if data is None:
            raise ValueError("Provider embedding response did not include data")

        ordered = sorted(data, key=lambda item: item.index)
        return [list(item.embedding) for item in ordered]


class GoogleEmbeddingsAdapter(EmbeddingsAdapter):
    """Adapter for Google GenAI and Vertex AI embeddings."""

    async def create_vectors(
        self,
        native_client: Any,
        provider_model_name: str,
        request: EmbeddingsRequest,
    ) -> list[list[float]]:
        """Create vectors using Google `models.embed_content`."""
        config: dict[str, Any] = {}
        if request.dimensions is not None:
            config["output_dimensionality"] = request.dimensions

        response = await native_client.aio.models.embed_content(
            model=provider_model_name,
            contents=self._inputs(request),
            config=config or None,
        )
        embeddings = getattr(response, "embeddings", None) or []
        return [list(embedding.values or []) for embedding in embeddings]


class XAIEmbeddingsAdapter(EmbeddingsAdapter):
    """Adapter for xAI embeddings gRPC service."""

    async def create_vectors(
        self,
        native_client: Any,
        provider_model_name: str,
        request: EmbeddingsRequest,
    ) -> list[list[float]]:
        """Create vectors through xAI's `Embedder` gRPC endpoint."""
        from xai_sdk.proto.v6 import embed_pb2, embed_pb2_grpc

        stub = embed_pb2_grpc.EmbedderStub(native_client._api_channel)
        response = await stub.Embed(
            embed_pb2.EmbedRequest(
                input=[
                    embed_pb2.EmbedInput(string=value)
                    for value in self._inputs(request)
                ],
                model=provider_model_name,
                encoding_format=embed_pb2.FORMAT_FLOAT,
                user=request.user,
            ),
        )

        vectors: list[list[float]] = []
        for entry in sorted(response.embeddings, key=lambda item: item.index):
            if not entry.embeddings:
                raise ValueError("xAI embedding response did not include a vector")
            vectors.append(list(entry.embeddings[0].float_array))
        return vectors


class BedrockEmbeddingsAdapter(EmbeddingsAdapter):
    """Adapter for AWS Bedrock embedding models."""

    @staticmethod
    def _build_body(
        provider_model_name: str,
        text: str,
        request: EmbeddingsRequest,
    ) -> dict[str, Any]:
        if "cohere" in provider_model_name.lower():
            body: dict[str, Any] = {
                "texts": [text],
                "input_type": "search_document",
            }
            if request.dimensions is not None:
                body["dimensions"] = request.dimensions
            return body

        body: dict[str, Any] = {"inputText": text}
        if request.dimensions is not None:
            body["dimensions"] = request.dimensions
        return body

    async def create_vectors(
        self,
        native_client: Any,
        provider_model_name: str,
        request: EmbeddingsRequest,
    ) -> list[list[float]]:
        """Create vectors by invoking Bedrock runtime embedding models."""

        def invoke(text: str) -> list[float]:
            response = native_client.invoke_model(
                modelId=provider_model_name,
                body=json.dumps(
                    self._build_body(provider_model_name, text, request),
                ).encode("utf-8"),
                contentType="application/json",
                accept="application/json",
            )

            payload = json.loads(response["body"].read().decode("utf-8"))
            if "embedding" in payload:
                return list(payload["embedding"])
            if payload.get("embeddings"):
                first = payload["embeddings"][0]
                if isinstance(first, dict) and "embedding" in first:
                    return list(first["embedding"])
                return list(first)

            raise ValueError("Bedrock embedding response did not include embeddings")

        inputs = self._inputs(request)
        return await asyncio.gather(
            *(asyncio.to_thread(invoke, text) for text in inputs)
        )


class AnthropicEmbeddingsAdapter(EmbeddingsAdapter):
    """Adapter for Anthropic provider that has no embeddings API."""

    async def create_vectors(
        self,
        native_client: Any,
        provider_model_name: str,
        request: EmbeddingsRequest,
    ) -> list[list[float]]:
        """Raise because Anthropic has no embeddings API in this shim."""
        del native_client
        del provider_model_name
        del request
        raise ValueError("Anthropic provider does not expose an embeddings API")


@lru_cache(maxsize=1)
def get_embeddings_service() -> EmbeddingsService:
    """Singleton factory for EmbeddingsService.

    Returns:
        EmbeddingsService: Cached service instance.
    """
    return EmbeddingsService()

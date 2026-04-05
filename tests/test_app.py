"""Integration tests for main app and initialization."""

from typing import Any

from fastapi.testclient import TestClient

from llm_shim import app, create_app
from llm_shim.core.exceptions import (
    BadRequestError,
    ProviderCallError,
    ProviderConfigError,
    ShimError,
)


def _route_paths() -> list[str]:
    paths: list[str] = []
    for route in app.routes:
        path = getattr(route, "path", None)
        if isinstance(path, str):
            paths.append(path)
    return paths


def test_app_created() -> None:
    """Test that app is properly initialized."""
    assert app is not None
    assert app.title == "llm-shim"


def test_create_app_returns_fastapi_instance() -> None:
    """create_app should return a FastAPI instance."""
    from fastapi import FastAPI

    test_app = create_app()
    assert isinstance(test_app, FastAPI)
    assert test_app.title == "llm-shim"


def test_app_has_routers_configured(client: Any | None = None) -> None:
    """App should have chat and embeddings routers configured."""
    # Check that routes are registered
    routes = _route_paths()
    assert "/v1/chat/completions" in routes
    assert "/v1/embeddings" in routes
    assert "/v1/models" in routes


def test_health_endpoints() -> None:
    """App should have health check endpoints."""
    routes = _route_paths()
    assert "/livez" in routes
    assert "/healthz" in routes


def test_shim_error_status_codes() -> None:
    """ShimError subclasses should have correct status_code attributes."""
    assert ShimError.status_code == 500
    assert BadRequestError.status_code == 400
    assert ProviderConfigError.status_code == 500
    assert ProviderCallError.status_code == 502


def test_shim_error_inheritance() -> None:
    """All custom exceptions should inherit from ShimError."""
    assert issubclass(BadRequestError, ShimError)
    assert issubclass(ProviderConfigError, ShimError)
    assert issubclass(ProviderCallError, ShimError)


def test_health_endpoint_returns_ok() -> None:
    """Health endpoints should return ok status."""
    client = TestClient(app)
    for path in ("/healthz", "/livez"):
        response = client.get(path)
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


def test_request_id_header_assigned() -> None:
    """Responses should include x-request-id header."""
    client = TestClient(app)
    response = client.get("/healthz")
    assert "x-request-id" in response.headers


def test_request_id_header_echoed() -> None:
    """When client sends x-request-id, response should echo it back."""
    client = TestClient(app)
    response = client.get("/healthz", headers={"x-request-id": "test-123"})
    assert response.headers["x-request-id"] == "test-123"

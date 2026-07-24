"""Shared pytest fixtures for API tests."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

TEST_API_KEY = "test-secret-key"


@pytest.fixture(scope="session", autouse=True)
def initialize_artifacts() -> None:
    """Ensure pipeline artifacts exist before any tests run."""
    from shopper_segmentation.artifacts import ensure_artifacts

    ensure_artifacts()


@pytest.fixture(autouse=True)
def api_key_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set API_KEY for all tests."""
    monkeypatch.setenv("API_KEY", TEST_API_KEY)


@pytest.fixture
def client() -> TestClient:
    """Return a FastAPI test client with lifespan initialized."""
    from shopper_segmentation.api.app import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def auth_headers() -> dict[str, str]:
    """Return headers with a valid API key."""
    return {"X-API-Key": TEST_API_KEY}

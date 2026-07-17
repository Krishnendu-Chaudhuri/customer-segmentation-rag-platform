"""Shared pytest fixtures for API tests."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

TEST_API_KEY = "test-secret-key"


@pytest.fixture(autouse=True)
def api_key_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set API_KEY for all tests."""
    monkeypatch.setenv("API_KEY", TEST_API_KEY)


@pytest.fixture
def client() -> TestClient:
    """Return a FastAPI test client."""
    from shopper_segmentation.api.app import app

    return TestClient(app)


@pytest.fixture
def auth_headers() -> dict[str, str]:
    """Return headers with a valid API key."""
    return {"X-API-Key": TEST_API_KEY}

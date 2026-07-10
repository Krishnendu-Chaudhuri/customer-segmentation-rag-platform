"""Unit tests for FastAPI backend endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_list_segments() -> None:
    """GET /segments should return all segment summaries."""
    response = client.get("/segments")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 8
    assert data[0]["id"] == 0
    assert "name" in data[0]
    assert "size" in data[0]


def test_get_segment_detail() -> None:
    """GET /segments/{id} should return feature means."""
    response = client.get("/segments/2")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == 2
    assert "feature_means" in data
    assert "monetary" in data["feature_means"]


def test_get_segment_not_found() -> None:
    """Unknown segment ids should return 404."""
    response = client.get("/segments/999")
    assert response.status_code == 404


def test_get_segment_recommendations() -> None:
    """GET /segments/{id}/recommendations should return ranked products."""
    response = client.get("/segments/2/recommendations")
    assert response.status_code == 200
    data = response.json()
    assert data["segment_id"] == 2
    assert len(data["recommendations"]) > 0
    assert "lift" in data["recommendations"][0]


def test_chat_missing_api_key() -> None:
    """POST /chat should fail clearly when GROQ_API_KEY is missing."""
    response = client.post("/chat", json={"query": "Who are promo-sensitive shoppers?"})
    assert response.status_code == 503
    assert "GROQ_API_KEY" in response.json()["detail"]

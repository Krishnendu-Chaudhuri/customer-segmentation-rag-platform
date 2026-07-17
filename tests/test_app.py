"""Unit tests for FastAPI backend endpoints."""

from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient


def test_list_segments(client: TestClient, auth_headers: dict[str, str]) -> None:
    """GET /segments should return all segment summaries."""
    response = client.get("/segments", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 8
    assert data[0]["id"] == 0
    assert "name" in data[0]
    assert "size" in data[0]


def test_get_segment_detail(client: TestClient, auth_headers: dict[str, str]) -> None:
    """GET /segments/{id} should return feature means."""
    response = client.get("/segments/2", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == 2
    assert "feature_means" in data
    assert "monetary" in data["feature_means"]


def test_get_segment_not_found(client: TestClient, auth_headers: dict[str, str]) -> None:
    """Unknown segment ids should return 404."""
    response = client.get("/segments/999", headers=auth_headers)
    assert response.status_code == 404


def test_get_segment_recommendations(client: TestClient, auth_headers: dict[str, str]) -> None:
    """GET /segments/{id}/recommendations should return ranked products."""
    response = client.get("/segments/2/recommendations", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["segment_id"] == 2
    assert len(data["recommendations"]) > 0
    assert "lift" in data["recommendations"][0]


def test_missing_api_key_returns_401(client: TestClient) -> None:
    """Protected routes should reject requests without X-API-Key."""
    response = client.get("/segments")
    assert response.status_code == 401
    assert response.json()["detail"] == "Missing X-API-Key header"


def test_invalid_api_key_returns_401(client: TestClient) -> None:
    """Protected routes should reject requests with an invalid API key."""
    response = client.get("/segments", headers={"X-API-Key": "wrong-key"})
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid API key"


def test_health_without_api_key(client: TestClient) -> None:
    """GET /health should remain accessible without authentication."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_missing_groq_api_key(client: TestClient, auth_headers: dict[str, str]) -> None:
    """POST /chat should fail clearly when GROQ_API_KEY is missing."""
    with patch(
        "shopper_segmentation.api.app.get_groq_api_key",
        side_effect=RuntimeError("GROQ_API_KEY is not set."),
    ):
        response = client.post(
            "/chat",
            json={"query": "Who are promo-sensitive shoppers?"},
            headers=auth_headers,
        )
    assert response.status_code == 503
    assert "GROQ_API_KEY" in response.json()["detail"]


def test_chat_rate_limit(client: TestClient, auth_headers: dict[str, str]) -> None:
    """POST /chat should return 429 after exceeding the per-minute limit."""
    mock_result = {
        "query": "test",
        "answer": "mock answer",
        "retrieved_cards": [],
        "validation": {
            "validated": True,
            "numbers_found": [],
            "unsupported_numbers": [],
        },
    }

    with patch("shopper_segmentation.api.app.get_groq_api_key", return_value="gsk_test"):
        with patch("shopper_segmentation.api.app.answer_query", return_value=mock_result):
            statuses: list[int] = []
            for _ in range(12):
                response = client.post(
                    "/chat",
                    json={"query": "rate limit test"},
                    headers=auth_headers,
                )
                statuses.append(response.status_code)

    assert 200 in statuses
    assert 429 in statuses


def test_low_confidence_segments_flagged(client: TestClient, auth_headers: dict[str, str]) -> None:
    """Segments below the confidence threshold should expose low_confidence=true."""
    from shopper_segmentation.segmentation import is_low_confidence_segment

    assert is_low_confidence_segment(5) is True
    assert is_low_confidence_segment(100) is False

    response = client.get("/segments", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert all("low_confidence" in segment for segment in data)

    small_segments = [segment for segment in data if segment["size"] < 30]
    if small_segments:
        assert all(segment["low_confidence"] for segment in small_segments)

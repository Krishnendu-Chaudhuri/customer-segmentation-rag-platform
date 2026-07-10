"""HTTP client for the Shopper Segmentation FastAPI backend."""

from __future__ import annotations

import os
from typing import Any

import httpx

DEFAULT_API_BASE = "http://127.0.0.1:8000"


def get_api_base_url() -> str:
    """Return the configured API base URL.

    Returns:
        Base URL for FastAPI backend requests.
    """
    return os.getenv("API_BASE_URL", DEFAULT_API_BASE).rstrip("/")


def _request(method: str, path: str, **kwargs: Any) -> Any:
    """Perform an HTTP request against the backend API.

    Args:
        method: HTTP method name.
        path: API path beginning with /.
        **kwargs: Additional httpx request arguments.

    Returns:
        Parsed JSON response body.

    Raises:
        RuntimeError: If the API is unreachable or returns an error.
    """
    url = f"{get_api_base_url()}{path}"
    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
    except httpx.ConnectError as exc:
        raise RuntimeError(
            "Cannot connect to the API. Start the backend with: "
            "uvicorn app:app --host 127.0.0.1 --port 8000"
        ) from exc
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text
        try:
            detail = exc.response.json().get("detail", detail)
        except Exception:
            pass
        raise RuntimeError(f"API error ({exc.response.status_code}): {detail}") from exc


def get_segments() -> list[dict[str, Any]]:
    """Fetch all segment summaries."""
    return _request("GET", "/segments")


def get_segment(segment_id: int) -> dict[str, Any]:
    """Fetch detailed profile for one segment."""
    return _request("GET", f"/segments/{segment_id}")


def get_recommendations(segment_id: int) -> dict[str, Any]:
    """Fetch product recommendations for one segment."""
    return _request("GET", f"/segments/{segment_id}/recommendations")


def post_chat(query: str) -> dict[str, Any]:
    """Send an analyst question to the chat endpoint."""
    return _request("POST", "/chat", json={"query": query})


def check_health() -> bool:
    """Return True if the API health endpoint responds."""
    try:
        _request("GET", "/health")
        return True
    except RuntimeError:
        return False

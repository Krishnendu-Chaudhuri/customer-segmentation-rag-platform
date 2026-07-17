"""API key authentication for protected endpoints."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

load_dotenv()

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_expected_api_key() -> str:
    """Load the required API key from environment variables.

    Returns:
        Expected API key string.

    Raises:
        RuntimeError: If API_KEY is not configured.
    """
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise RuntimeError(
            "API_KEY is not set. Add it to a .env file in the project root."
        )
    return api_key


def verify_api_key(api_key: str | None = Security(API_KEY_HEADER)) -> str:
    """Validate the X-API-Key header against the configured API_KEY.

    Args:
        api_key: Value from the X-API-Key request header.

    Returns:
        The validated API key.

    Raises:
        HTTPException: If the header is missing or does not match.
    """
    try:
        expected = get_expected_api_key()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")
    if api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key

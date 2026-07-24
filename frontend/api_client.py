"""HTTP client for the Shopper Segmentation FastAPI backend."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import httpx

DEFAULT_API_BASE = "http://127.0.0.1:8000"
STREAM_TIMEOUT = httpx.Timeout(connect=30.0, read=300.0, write=300.0, pool=300.0)
STREAM_MAX_RETRIES = 2
STREAM_RETRY_BACKOFF_SECONDS = 1.0
RETRYABLE_ERROR_CODES = frozenset({"connection_lost", "stream_incomplete"})

logger = logging.getLogger(__name__)


class ChatStreamError(RuntimeError):
    """Structured error raised when chat streaming fails."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)

    def as_dict(self) -> dict[str, str]:
        """Return a JSON-serializable error payload."""
        return {"code": self.code, "message": self.message}


__all__ = [
    "ChatStreamError",
    "get_api_base_url",
    "post_chat",
    "post_chat_stream",
    "get_segments",
    "get_segment",
    "get_recommendations",
    "check_health",
]


def get_api_base_url() -> str:
    """Return the configured API base URL.

    Returns:
        Base URL for FastAPI backend requests.
    """
    return os.getenv("API_BASE_URL", DEFAULT_API_BASE).rstrip("/")


def _auth_headers() -> dict[str, str]:
    """Return API authentication headers when API_KEY is configured."""
    api_key = os.getenv("API_KEY")
    if api_key:
        return {"X-API-Key": api_key}
    return {}


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
    headers = kwargs.pop("headers", {})
    headers = {**_auth_headers(), **headers}
    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.request(method, url, headers=headers, **kwargs)
            response.raise_for_status()
            return response.json()
    except httpx.ConnectError as exc:
        raise RuntimeError(
            "Cannot connect to the API. Start the backend with: "
            "uvicorn shopper_segmentation.api.app:app --host 127.0.0.1 --port 8000"
        ) from exc
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text
        try:
            detail = exc.response.json().get("detail", detail)
        except Exception:
            pass
        raise RuntimeError(f"API error ({exc.response.status_code}): {detail}") from exc


def _consume_chat_sse(
    response: httpx.Response,
    on_token: Any | None = None,
) -> dict[str, Any]:
    """Consume SSE lines from a chat stream response.

    Args:
        response: Open httpx streaming response with successful status.
        on_token: Optional callback invoked with each streamed token string.

    Returns:
        Final chat payload from the terminal ``done`` SSE event.

    Raises:
        ChatStreamError: On connection, parse, backend, or incomplete stream errors.
    """
    final_event: dict[str, Any] | None = None
    resolved_thread_id: str | None = None

    try:
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            try:
                event = json.loads(line[6:])
            except json.JSONDecodeError as exc:
                raise ChatStreamError(
                    "parse_error",
                    f"Failed to parse streaming event: {exc}",
                ) from exc

            event_type = event.get("type")
            if event_type == "thread_id":
                resolved_thread_id = str(event["thread_id"])
            elif event_type == "token" and on_token is not None:
                on_token(str(event.get("content", "")))
            elif event_type == "error":
                error_payload = event.get("error") or {}
                code = str(error_payload.get("code", "backend_error"))
                message = str(error_payload.get("message", "Chat stream failed."))
                raise ChatStreamError(code, message)
            elif event_type == "done":
                final_event = event
    except ChatStreamError:
        raise
    except httpx.RemoteProtocolError as exc:
        raise ChatStreamError(
            "connection_lost",
            "The chat stream ended unexpectedly before completion.",
        ) from exc
    except httpx.ReadTimeout as exc:
        raise ChatStreamError(
            "read_timeout",
            "The chat stream timed out while waiting for a response.",
        ) from exc
    except httpx.ConnectError as exc:
        raise ChatStreamError(
            "connect_error",
            "Cannot connect to the API. Start the backend with: "
            "uvicorn shopper_segmentation.api.app:app --host 127.0.0.1 --port 8000",
        ) from exc
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text
        try:
            detail = exc.response.json().get("detail", detail)
        except Exception:
            pass
        raise ChatStreamError(
            "api_error",
            f"API error ({exc.response.status_code}): {detail}",
        ) from exc
    except Exception as exc:
        raise ChatStreamError(
            "stream_error",
            f"Unexpected chat stream error: {exc}",
        ) from exc

    if final_event is None:
        raise ChatStreamError(
            "stream_incomplete",
            "Chat stream ended without a final response event.",
        )

    if resolved_thread_id and "thread_id" not in final_event:
        final_event["thread_id"] = resolved_thread_id
    return final_event


def _is_retryable_stream_error(exc: ChatStreamError) -> bool:
    """Return True when a chat stream error should be retried."""
    return exc.code in RETRYABLE_ERROR_CODES


def _post_chat_stream_once(
    query: str,
    thread_id: str | None,
    on_token: Any | None,
) -> dict[str, Any]:
    """Perform a single chat streaming request attempt."""
    payload: dict[str, Any] = {"query": query}
    if thread_id:
        payload["thread_id"] = thread_id

    url = f"{get_api_base_url()}/chat"
    headers = _auth_headers()
    headers["Accept"] = "text/event-stream"

    with httpx.Client(timeout=STREAM_TIMEOUT) as client:
        with client.stream(
            "POST",
            url,
            params={"stream": "true"},
            json=payload,
            headers=headers,
        ) as response:
            logger.info(
                "Chat stream response status=%s url=%s",
                response.status_code,
                url,
            )
            response.raise_for_status()
            return _consume_chat_sse(response, on_token=on_token)


def post_chat_stream(
    query: str,
    thread_id: str | None = None,
    on_token: Any | None = None,
) -> dict[str, Any]:
    """Stream an analyst chat response via Server-Sent Events.

    Args:
        query: Natural language analyst question.
        thread_id: Optional session thread id for conversational memory.
        on_token: Optional callback invoked with each streamed token string.

    Returns:
        Final chat payload from the terminal ``done`` SSE event.

    Raises:
        ChatStreamError: On connection, timeout, backend, or incomplete stream errors.
    """
    last_error: ChatStreamError | None = None

    for attempt in range(STREAM_MAX_RETRIES + 1):
        try:
            logger.info(
                "Chat stream attempt=%s/%s thread_id=%s",
                attempt + 1,
                STREAM_MAX_RETRIES + 1,
                thread_id,
            )
            return _post_chat_stream_once(query, thread_id, on_token)
        except ChatStreamError as exc:
            last_error = exc
            if attempt >= STREAM_MAX_RETRIES or not _is_retryable_stream_error(exc):
                raise
            logger.warning(
                "Retrying chat stream attempt=%s/%s code=%s",
                attempt + 2,
                STREAM_MAX_RETRIES + 1,
                exc.code,
            )
            time.sleep(STREAM_RETRY_BACKOFF_SECONDS * (attempt + 1))
        except httpx.ConnectError as exc:
            last_error = ChatStreamError(
                "connect_error",
                "Cannot connect to the API. Start the backend with: "
                "uvicorn shopper_segmentation.api.app:app --host 127.0.0.1 --port 8000",
            )
            if attempt >= STREAM_MAX_RETRIES:
                raise last_error from exc
            logger.warning(
                "Retrying chat stream after connect error attempt=%s/%s",
                attempt + 2,
                STREAM_MAX_RETRIES + 1,
            )
            time.sleep(STREAM_RETRY_BACKOFF_SECONDS * (attempt + 1))
        except httpx.RemoteProtocolError as exc:
            last_error = ChatStreamError(
                "connection_lost",
                "The chat stream ended unexpectedly before completion.",
            )
            if attempt >= STREAM_MAX_RETRIES:
                raise last_error from exc
            logger.warning(
                "Retrying chat stream after protocol error attempt=%s/%s",
                attempt + 2,
                STREAM_MAX_RETRIES + 1,
            )
            time.sleep(STREAM_RETRY_BACKOFF_SECONDS * (attempt + 1))
        except httpx.ReadTimeout as exc:
            last_error = ChatStreamError(
                "read_timeout",
                "The chat stream timed out while waiting for a response.",
            )
            if attempt >= STREAM_MAX_RETRIES:
                raise last_error from exc
            logger.warning(
                "Retrying chat stream after read timeout attempt=%s/%s",
                attempt + 2,
                STREAM_MAX_RETRIES + 1,
            )
            time.sleep(STREAM_RETRY_BACKOFF_SECONDS * (attempt + 1))
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text
            try:
                detail = exc.response.json().get("detail", detail)
            except Exception:
                pass
            raise ChatStreamError(
                "api_error",
                f"API error ({exc.response.status_code}): {detail}",
            ) from exc

    if last_error is not None:
        raise last_error
    raise ChatStreamError("stream_error", "Chat stream failed without a response.")


def get_segments() -> list[dict[str, Any]]:
    """Fetch all segment summaries."""
    return _request("GET", "/segments")


def get_segment(segment_id: int) -> dict[str, Any]:
    """Fetch detailed profile for one segment."""
    return _request("GET", f"/segments/{segment_id}")


def get_recommendations(segment_id: int) -> dict[str, Any]:
    """Fetch product recommendations for one segment."""
    return _request("GET", f"/segments/{segment_id}/recommendations")


def post_chat(query: str, thread_id: str | None = None) -> dict[str, Any]:
    """Send an analyst question to the chat endpoint (non-streaming)."""
    payload: dict[str, Any] = {"query": query}
    if thread_id:
        payload["thread_id"] = thread_id
    return _request("POST", "/chat", params={"stream": "false"}, json=payload)


def check_health() -> bool:
    """Return True if the API health endpoint responds."""
    try:
        _request("GET", "/health")
        return True
    except RuntimeError:
        return False

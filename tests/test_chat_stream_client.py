"""Unit tests for resilient chat streaming client behavior."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from frontend.api_client import (
    STREAM_MAX_RETRIES,
    ChatStreamError,
    _consume_chat_sse,
    post_chat_stream,
)


def _mock_response(lines: list[str], status_code: int = 200) -> httpx.Response:
    """Build a mock httpx response whose iter_lines yields given SSE lines."""
    response = MagicMock(spec=httpx.Response)
    response.status_code = status_code
    response.iter_lines.return_value = iter(lines)
    response.raise_for_status = MagicMock()
    return response


def test_consume_chat_sse_returns_done_event() -> None:
    """Should return the terminal done payload from SSE lines."""
    response = _mock_response(
        [
            'data: {"type": "thread_id", "thread_id": "abc"}',
            'data: {"type": "token", "content": "Hello"}',
            'data: {"type": "done", "answer": "Hello", "thread_id": "abc"}',
        ]
    )

    tokens: list[str] = []
    result = _consume_chat_sse(response, on_token=tokens.append)

    assert result["answer"] == "Hello"
    assert result["thread_id"] == "abc"
    assert tokens == ["Hello"]


def test_consume_chat_sse_raises_on_backend_error_event() -> None:
    """Backend SSE error events should map to ChatStreamError."""
    response = _mock_response(
        [
            'data: {"type": "error", "error": {"code": "stream_failed", "message": "boom"}}',
        ]
    )

    with pytest.raises(ChatStreamError) as exc_info:
        _consume_chat_sse(response)

    assert exc_info.value.code == "stream_failed"
    assert exc_info.value.message == "boom"


def test_consume_chat_sse_raises_on_incomplete_stream() -> None:
    """Missing done event should raise stream_incomplete."""
    response = _mock_response(
        [
            'data: {"type": "thread_id", "thread_id": "abc"}',
        ]
    )

    with pytest.raises(ChatStreamError) as exc_info:
        _consume_chat_sse(response)

    assert exc_info.value.code == "stream_incomplete"


def test_consume_chat_sse_maps_remote_protocol_error() -> None:
    """RemoteProtocolError during iteration should map to connection_lost."""
    response = _mock_response([])
    response.iter_lines.side_effect = httpx.RemoteProtocolError(
        "peer closed connection without sending complete message body"
    )

    with pytest.raises(ChatStreamError) as exc_info:
        _consume_chat_sse(response)

    assert exc_info.value.code == "connection_lost"


def test_post_chat_stream_retries_transient_connection_lost() -> None:
    """Transient connection_lost errors should retry up to two times."""
    side_effects = [
        ChatStreamError("connection_lost", "lost"),
        ChatStreamError("connection_lost", "lost"),
        {"type": "done", "answer": "ok", "thread_id": "t1"},
    ]

    with patch(
        "frontend.api_client._post_chat_stream_once",
        side_effect=side_effects,
    ) as mock_once:
        with patch("frontend.api_client.time.sleep"):
            result = post_chat_stream("hello")

    assert result["answer"] == "ok"
    assert mock_once.call_count == 3


def test_post_chat_stream_does_not_retry_backend_error() -> None:
    """Explicit backend errors should not be retried."""
    with patch(
        "frontend.api_client._post_chat_stream_once",
        side_effect=ChatStreamError("stream_failed", "boom"),
    ) as mock_once:
        with pytest.raises(ChatStreamError) as exc_info:
            post_chat_stream("hello")

    assert exc_info.value.code == "stream_failed"
    assert mock_once.call_count == 1


def test_post_chat_stream_stops_after_max_retries() -> None:
    """Should fail after exhausting all retry attempts."""
    with patch(
        "frontend.api_client._post_chat_stream_once",
        side_effect=ChatStreamError("connection_lost", "lost"),
    ) as mock_once:
        with patch("frontend.api_client.time.sleep"):
            with pytest.raises(ChatStreamError) as exc_info:
                post_chat_stream("hello")

    assert exc_info.value.code == "connection_lost"
    assert mock_once.call_count == STREAM_MAX_RETRIES + 1

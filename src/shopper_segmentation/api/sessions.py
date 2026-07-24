"""Chat session thread id resolution for LangGraph checkpointer."""

from __future__ import annotations

import re
import uuid

from fastapi import HTTPException, Request

THREAD_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def resolve_thread_id(request: Request, body_thread_id: str | None) -> str:
    """Resolve a LangGraph thread id from the client or generate a new one.

    Args:
        request: Incoming HTTP request (reserved for future header-based ids).
        body_thread_id: Optional client-supplied thread id from the chat body.

    Returns:
        Valid thread id string for LangGraph configurable config.

    Raises:
        HTTPException: If the supplied thread id has an invalid format.
    """
    _ = request
    if body_thread_id:
        if not THREAD_ID_PATTERN.match(body_thread_id):
            raise HTTPException(
                status_code=400,
                detail="thread_id must be 1-128 characters of letters, digits, _ or -",
            )
        return body_thread_id
    return str(uuid.uuid4())

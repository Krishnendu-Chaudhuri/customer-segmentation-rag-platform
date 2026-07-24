"""LangGraph agent state definitions."""

from __future__ import annotations

from typing import TypedDict


class AgentState(TypedDict, total=False):
    """State passed between LangGraph agent nodes."""

    query: str
    route: str | None
    retrieved_cards: list[dict]
    answer: str | None
    validation: dict | None
    retry_count: int
    segment_id: int | None
    debug_raw_answer: str | None
    turn_history: list[dict[str, str]]

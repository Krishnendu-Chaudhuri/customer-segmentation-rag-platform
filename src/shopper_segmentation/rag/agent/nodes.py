"""LangGraph agent node implementations."""

from __future__ import annotations

import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage

from shopper_segmentation.rag.agent.prompts import (
    INSUFFICIENT_GROUNDED_DATA_MESSAGE,
    RETRY_PROMPT,
    ROUTER_PROMPT,
    SYSTEM_PROMPT,
)
from shopper_segmentation.rag.agent.state import AgentState
from shopper_segmentation.rag.rag_chain import (
    build_messages,
    validate_response_numbers,
)
from shopper_segmentation.rag.tools import (
    lookup_card_from_payload,
    recommendation_lookup,
    segment_lookup,
    semantic_search,
    uplift_lookup,
)

logger = logging.getLogger(__name__)

SEGMENT_ID_PATTERN = re.compile(r"segment\s+(\d+)", re.IGNORECASE)
LOOKUP_KEYWORDS = (
    "uplift",
    "incremental",
    "promo",
    "recommend",
    "product",
    "target",
    "profile",
    "metric",
    "size",
    "household",
)


def extract_segment_id(query: str) -> int | None:
    """Extract a segment id from an analyst query when present.

    Args:
        query: Natural language question.

    Returns:
        Parsed segment id or None.
    """
    match = SEGMENT_ID_PATTERN.search(query)
    if match:
        return int(match.group(1))
    return None


def _segment_name_for_id(segment_id: int) -> str:
    """Resolve a segment display name from structured lookup data."""
    profile = segment_lookup.invoke({"segment_id": segment_id})
    if isinstance(profile, dict) and "error" not in profile:
        return str(profile.get("name", f"Segment {segment_id}"))
    return f"Segment {segment_id}"


def _heuristic_route(query: str) -> str:
    """Classify route without an LLM when patterns are unambiguous."""
    lowered = query.lower()
    if extract_segment_id(query) is not None:
        return "lookup"
    if any(keyword in lowered for keyword in LOOKUP_KEYWORDS) and re.search(r"\b\d+\b", query):
        return "lookup"
    return "search"


def _needs_llm_router(query: str) -> bool:
    """Return True when lookup vs search cannot be decided heuristically."""
    lowered = query.lower()
    if extract_segment_id(query) is not None:
        return False
    if any(phrase in lowered for phrase in ("who are", "what are", "which segment")):
        return False
    has_lookup_signal = any(keyword in lowered for keyword in LOOKUP_KEYWORDS)
    has_digit = bool(re.search(r"\b\d+\b", query))
    return has_lookup_signal and not has_digit


def router_node(state: AgentState) -> dict[str, object]:
    """Classify the query as lookup or semantic search."""
    query = state["query"]
    route = _heuristic_route(query)

    if _needs_llm_router(query):
        try:
            from shopper_segmentation.rag import rag_chain

            chat_model = rag_chain.get_chat_model()
            response = chat_model.invoke(
                [
                    SystemMessage(content=ROUTER_PROMPT),
                    HumanMessage(content=query),
                ]
            )
            llm_route = rag_chain._message_content(response).strip().lower()
            if llm_route in {"lookup", "search"}:
                route = llm_route
        except RuntimeError:
            logger.debug("Groq unavailable for router; using heuristic route=%s", route)

    segment_id = extract_segment_id(query)
    if route == "lookup" and segment_id is None:
        number_match = re.search(r"\b(\d+)\b", query)
        if number_match:
            segment_id = int(number_match.group(1))

    return {"route": route, "segment_id": segment_id}


def retrieve_node(state: AgentState) -> dict[str, object]:
    """Retrieve structured or semantic context for the query."""
    query = state["query"]
    route = state.get("route", "search")

    if route == "search":
        cards = semantic_search.invoke({"query": query, "top_k": 3})
        return {"retrieved_cards": cards}

    segment_id = state.get("segment_id") or extract_segment_id(query)
    if segment_id is None:
        cards = semantic_search.invoke({"query": query, "top_k": 3})
        return {"retrieved_cards": cards}

    lowered = query.lower()
    segment_name = _segment_name_for_id(segment_id)
    cards: list[dict] = []

    if any(keyword in lowered for keyword in ("uplift", "incremental", "promo")):
        payload = uplift_lookup.invoke({"segment_id": segment_id})
        cards.append(
            lookup_card_from_payload(segment_id, segment_name, payload, "Uplift")
        )
    elif any(keyword in lowered for keyword in ("recommend", "product", "target")):
        payload = recommendation_lookup.invoke({"segment_id": segment_id})
        cards.append(
            lookup_card_from_payload(segment_id, segment_name, payload, "Recommendations")
        )
    else:
        profile = segment_lookup.invoke({"segment_id": segment_id})
        cards.append(
            lookup_card_from_payload(segment_id, segment_name, profile, "Profile")
        )

    return {"retrieved_cards": cards}


def generate_node(state: AgentState) -> dict[str, object]:
    """Generate an answer from retrieved context via ChatGroq."""
    query = state["query"]
    retrieved = state.get("retrieved_cards", [])
    retry_count = state.get("retry_count", 0)
    turn_history = state.get("turn_history", [])

    messages = build_messages(query, retrieved)
    if turn_history:
        prior = "\n\n".join(
            f"Q: {turn['query']}\nA: {turn['answer']}" for turn in turn_history[-3:]
        )
        prior_block = f"Prior conversation:\n{prior}\n\n"
        user_message = messages[1]
        messages[1] = HumanMessage(content=f"{prior_block}{user_message.content}")
    if retry_count > 0:
        messages[0] = SystemMessage(content=f"{SYSTEM_PROMPT}\n\n{RETRY_PROMPT}")

    from shopper_segmentation.rag import rag_chain

    chat_model = rag_chain.get_chat_model()
    response = chat_model.invoke(messages)
    answer = rag_chain._message_content(response)
    return {"answer": answer}


def guardrail_node(state: AgentState) -> dict[str, object]:
    """Validate numeric citations against retrieved context."""
    answer = state.get("answer") or ""
    retrieved = state.get("retrieved_cards", [])
    context = "\n\n---\n\n".join(str(card["content"]) for card in retrieved)
    validation = validate_response_numbers(answer, context)
    return {"validation": validation}


def retry_bump_node(state: AgentState) -> dict[str, object]:
    """Increment retry counter before a guardrail-driven regeneration."""
    return {"retry_count": state.get("retry_count", 0) + 1}


def respond_node(state: AgentState) -> dict[str, object]:
    """Finalize the agent response and apply refusal when guardrails fail."""
    history = list(state.get("turn_history", []))
    answer = state.get("answer")
    query = state.get("query")
    validation = state.get("validation") or {}
    retry_count = state.get("retry_count", 0)

    updates: dict[str, object] = {}
    if not validation.get("validated") and retry_count >= 1 and answer:
        updates["debug_raw_answer"] = answer
        updates["answer"] = INSUFFICIENT_GROUNDED_DATA_MESSAGE
        answer = INSUFFICIENT_GROUNDED_DATA_MESSAGE

    if answer and query:
        history.append({"query": query, "answer": answer})
    updates["turn_history"] = history
    return updates

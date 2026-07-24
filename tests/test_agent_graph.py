"""Tests for the LangGraph analyst agent."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

from shopper_segmentation.rag.agent.graph import get_compiled_graph, init_graph, shutdown_graph
from shopper_segmentation.rag.agent.nodes import router_node
from shopper_segmentation.rag.rag_chain import answer_query, clear_chat_cache


def _run_graph(input_state: dict, thread_id: str) -> dict:
    """Run the compiled graph asynchronously in tests."""

    async def _run() -> dict:
        await init_graph()
        try:
            return await get_compiled_graph().ainvoke(
                input_state,
                config={"configurable": {"thread_id": thread_id}},
            )
        finally:
            await shutdown_graph()

    return asyncio.run(_run())


def test_router_lookup_for_segment_uplift() -> None:
    """Segment-specific uplift questions should route to lookup."""
    state = router_node({"query": "what's segment 2's uplift"})
    assert state["route"] == "lookup"
    assert state["segment_id"] == 2


def test_router_search_for_open_ended_question() -> None:
    """Behavioral questions without segment ids should route to search."""
    state = router_node({"query": "who are promo-sensitive shoppers"})
    assert state["route"] == "search"


def test_agent_happy_path() -> None:
    """Graph should return answer, cards, and validation on success."""
    mock_cards = [
        {
            "segment_id": 2,
            "segment_name": "Promo-Sensitive",
            "content": "Segment 2 has 67 households.",
            "distance": 0.1,
        }
    ]
    mock_model = MagicMock()
    mock_model.invoke.return_value = AIMessage(
        content="Segment 2 has 67 households."
    )

    with patch(
        "shopper_segmentation.rag.vectorstore.retrieve_cards",
        return_value=mock_cards,
    ):
        with patch(
            "shopper_segmentation.rag.rag_chain.get_chat_model",
            return_value=mock_model,
        ):
            result = _run_graph(
                {"query": "who are promo-sensitive shoppers", "retry_count": 0},
                "test-thread",
            )

    assert result["answer"] == "Segment 2 has 67 households."
    assert result["retrieved_cards"] == mock_cards
    assert result["validation"]["validated"] is True
    assert mock_model.invoke.call_count == 1


def test_guardrail_retry_exactly_once() -> None:
    """Invalid first answer should trigger exactly one regeneration."""
    mock_cards = [
        {
            "segment_id": 2,
            "segment_name": "Promo-Sensitive",
            "content": "Segment 2 has 67 households.",
            "distance": 0.1,
        }
    ]
    mock_model = MagicMock()
    mock_model.invoke.side_effect = [
        AIMessage(content="Segment 2 has 99.9% uplift."),
        AIMessage(content="Segment 2 has 67 households."),
    ]

    with patch(
        "shopper_segmentation.rag.vectorstore.retrieve_cards",
        return_value=mock_cards,
    ):
        with patch(
            "shopper_segmentation.rag.rag_chain.get_chat_model",
            return_value=mock_model,
        ):
            result = _run_graph(
                {"query": "who are promo-sensitive shoppers", "retry_count": 0},
                "test-thread",
            )

    assert mock_model.invoke.call_count == 2
    assert result["retry_count"] == 1
    assert result["validation"]["validated"] is True


def test_guardrail_refusal_after_failed_retry() -> None:
    """Unsupported numbers after retry should return the refusal message."""
    mock_cards = [
        {
            "segment_id": 2,
            "segment_name": "Promo-Sensitive",
            "content": "Segment 2 has 67 households.",
            "distance": 0.1,
        }
    ]
    mock_model = MagicMock()
    mock_model.invoke.side_effect = [
        AIMessage(content="Segment 2 has 99.9% uplift."),
        AIMessage(content="Segment 2 still has 88.8% uplift."),
    ]

    with patch(
        "shopper_segmentation.rag.vectorstore.retrieve_cards",
        return_value=mock_cards,
    ):
        with patch(
            "shopper_segmentation.rag.rag_chain.get_chat_model",
            return_value=mock_model,
        ):
            from shopper_segmentation.rag.agent.prompts import (
                INSUFFICIENT_GROUNDED_DATA_MESSAGE,
            )

            result = _run_graph(
                {"query": "who are promo-sensitive shoppers", "retry_count": 0},
                "refusal-test",
            )

    assert result["answer"] == INSUFFICIENT_GROUNDED_DATA_MESSAGE
    assert result.get("debug_raw_answer")
    assert mock_model.invoke.call_count == 2


def test_get_chat_model_uses_fallback_chain() -> None:
    """Primary ChatGroq model should register an 8B fallback."""
    with patch("shopper_segmentation.rag.rag_chain.get_groq_api_key", return_value="gsk_test"):
        with patch("shopper_segmentation.rag.rag_chain.ChatGroq") as mock_chat:
            primary = MagicMock()
            fallback = MagicMock()
            primary.with_retry.return_value = primary
            fallback.with_retry.return_value = fallback
            primary.with_fallbacks.return_value = primary
            mock_chat.side_effect = [primary, fallback]

            from shopper_segmentation.rag.rag_chain import get_chat_model

            model = get_chat_model()
            assert model is primary
            primary.with_fallbacks.assert_called_once()


def test_answer_query_adapter_shape() -> None:
    """answer_query should preserve the legacy response contract."""
    clear_chat_cache()
    mock_cards = [
        {
            "segment_id": 1,
            "segment_name": "Test",
            "content": "Segment 1 size 10.",
            "distance": 0.0,
        }
    ]
    mock_model = MagicMock()
    mock_model.invoke.return_value = AIMessage(content="Segment 1 size 10.")

    with patch(
        "shopper_segmentation.rag.vectorstore.retrieve_cards",
        return_value=mock_cards,
    ):
        with patch(
            "shopper_segmentation.rag.rag_chain.get_chat_model",
            return_value=mock_model,
        ):
            result = answer_query("Segment 1 size?", client=mock_model)

    assert set(result.keys()) == {"query", "answer", "retrieved_cards", "validation"}
    assert result["query"] == "Segment 1 size?"
    clear_chat_cache()

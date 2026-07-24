"""Unit tests for RAG card building and response validation."""

from __future__ import annotations

from shopper_segmentation.rag.build_cards import build_all_cards
from shopper_segmentation.rag.rag_chain import (
    answer_query,
    clear_chat_cache,
    extract_numbers,
    number_in_context,
    validate_response_numbers,
)


def test_build_all_cards_count() -> None:
    """Should build one card per segment in profiles."""
    cards = build_all_cards()
    assert len(cards) == 8
    assert all("content" in card for card in cards)
    assert "Segment 0" in cards[0]["content"]


def test_build_segment_card_includes_sections() -> None:
    """Card markdown should include all required sections."""
    cards = build_all_cards()
    content = cards[2]["content"]
    assert "## Definition" in content
    assert "## Top Features" in content
    assert "## Top Recommended Products" in content
    assert "## Promo Performance" in content


def test_extract_numbers() -> None:
    """Should extract numeric tokens including percentages."""
    text = "Segment 2 has 67 households and +44.3% incremental spend."
    numbers = extract_numbers(text)
    assert "2" in numbers
    assert "67" in numbers
    assert "44.3" in numbers


def test_validate_response_numbers_flags_hallucination() -> None:
    """Unsupported numbers should be flagged during validation."""
    context = "Segment 2 has 67 households and 44.3% incremental spend."
    response = "Segment 2 has 67 households and 99.9% incremental spend."
    result = validate_response_numbers(response, context)
    assert result["validated"] is False
    assert "99.9" in result["unsupported_numbers"]


def test_number_in_context_allows_rounding() -> None:
    """Approximate numeric matches should count as supported."""
    context = "lift 10.66 and segment rate 0.149"
    assert number_in_context("10.66", context)
    assert number_in_context("0.149", context)


def test_answer_query_uses_cache_for_identical_questions() -> None:
    """Repeated identical queries should not call Groq more than once."""
    from unittest.mock import MagicMock, patch

    from langchain_core.messages import AIMessage

    clear_chat_cache()
    mock_model = MagicMock()
    mock_model.invoke.return_value = AIMessage(content="Cached answer without numbers.")

    with patch(
        "shopper_segmentation.rag.vectorstore.retrieve_cards",
        return_value=[],
    ):
        with patch("shopper_segmentation.rag.rag_chain.get_chat_model", return_value=mock_model):
            answer_query("Who are promo-sensitive shoppers?", client=mock_model)
            answer_query("Who are promo-sensitive shoppers?", client=mock_model)

    assert mock_model.invoke.call_count == 1
    clear_chat_cache()

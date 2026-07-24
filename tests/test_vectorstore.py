"""Unit tests for vector store retrieval."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from shopper_segmentation.rag import vectorstore


def test_score_to_distance_inverts_normalized_score() -> None:
    """Normalized similarity scores should map to lower-is-closer distances."""
    assert vectorstore._score_to_distance(0.9) == pytest.approx(0.1)
    assert vectorstore._score_to_distance(1.0) < vectorstore._score_to_distance(0.5)


def test_retrieve_cards_returns_expected_shape() -> None:
    """retrieve_cards should preserve the legacy response contract."""
    mock_store = MagicMock()
    mock_store.similarity_search_with_score.return_value = [
        (
            Document(
                page_content="Segment 2 card",
                metadata={"segment_id": 2, "segment_name": "Premium Grocery Loyalists"},
            ),
            0.8,
        )
    ]

    with patch("shopper_segmentation.rag.vectorstore.get_vectorstore", return_value=mock_store):
        cards = vectorstore.retrieve_cards("promo-sensitive shoppers", top_k=1)

    assert len(cards) == 1
    assert set(cards[0].keys()) == {"segment_id", "segment_name", "content", "distance"}
    assert cards[0]["segment_id"] == 2
    assert cards[0]["segment_name"] == "Premium Grocery Loyalists"
    assert cards[0]["content"] == "Segment 2 card"
    assert cards[0]["distance"] == pytest.approx(0.2)


def test_get_vectorstore_rebuilds_when_collection_empty() -> None:
    """An empty persisted collection should trigger a rebuild."""
    mock_store = MagicMock()
    mock_store._collection.count.return_value = 0

    with patch("shopper_segmentation.rag.vectorstore.Chroma", return_value=mock_store):
        with patch(
            "shopper_segmentation.rag.vectorstore.build_vectorstore",
            return_value=MagicMock(),
        ) as build_mock:
            vectorstore.get_vectorstore()

    build_mock.assert_called_once()

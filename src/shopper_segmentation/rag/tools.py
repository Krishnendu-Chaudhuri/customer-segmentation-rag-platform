"""LangChain tools for structured segment lookups and semantic search."""

from __future__ import annotations

import json

from langchain_core.tools import tool

from shopper_segmentation import artifacts
from shopper_segmentation.rag import vectorstore


def _find_segment_profile(segment_id: int) -> dict | None:
    """Return a segment profile dict by id."""
    profiles = artifacts.load_profiles()
    for segment in profiles["segments"]:
        if int(segment["id"]) == segment_id:
            return segment
    return None


def _find_segment_recommendations(segment_id: int) -> dict | None:
    """Return recommendation record for a segment id."""
    recommendations = artifacts.load_recommendations()
    for segment in recommendations["segments"]:
        if int(segment["segment_id"]) == segment_id:
            return segment
    return None


def _find_segment_uplift(segment_id: int) -> dict | None:
    """Return uplift record for a segment id."""
    uplift = artifacts.load_uplift()
    for segment in uplift["segments"]:
        if int(segment["segment_id"]) == segment_id:
            return segment
    return None


@tool
def segment_lookup(segment_id: int) -> dict:
    """Look up a segment profile by id.

    Use when the question asks about segment size, features, narrative, or definition
    for a specific segment id.
    """
    profile = _find_segment_profile(segment_id)
    if profile is None:
        return {"error": f"Segment {segment_id} not found"}
    return profile


@tool
def recommendation_lookup(segment_id: int) -> dict:
    """Look up ranked product recommendations for a segment id.

    Use when the question asks which products to target or recommended items for a
    specific segment.
    """
    record = _find_segment_recommendations(segment_id)
    if record is None:
        return {"error": f"Recommendations for segment {segment_id} not found"}
    return record


@tool
def uplift_lookup(segment_id: int) -> dict:
    """Look up campaign uplift metrics for a segment id.

    Use when the question asks about incremental spend, promo performance, or uplift
    for a specific segment.
    """
    record = _find_segment_uplift(segment_id)
    if record is None:
        return {"error": f"Uplift data for segment {segment_id} not found"}
    return record


@tool
def semantic_search(query: str, top_k: int = 3) -> list[dict]:
    """Search segment cards semantically for open-ended analyst questions.

    Use when the question does not specify a segment id and requires discovering
    which segments match a behavioral description.
    """
    return vectorstore.retrieve_cards(query, top_k=top_k)


def lookup_card_from_payload(
    segment_id: int,
    segment_name: str,
    payload: dict,
    label: str,
) -> dict[str, object]:
    """Wrap a structured lookup payload as a retrieved card.

    Args:
        segment_id: Segment identifier.
        segment_name: Human-readable segment name.
        payload: Structured lookup result.
        label: Card section label.

    Returns:
        Card dictionary compatible with the RAG chain contract.
    """
    return {
        "segment_id": segment_id,
        "segment_name": segment_name,
        "content": f"# {label}\n\n{json.dumps(payload, indent=2)}",
        "distance": 0.0,
    }

"""Evaluation dataset for the LangGraph analyst agent."""

from __future__ import annotations

from typing import TypedDict

from shopper_segmentation import artifacts


class EvalCase(TypedDict):
    """Single RAG evaluation case."""

    query: str
    expected_segment_ids: list[int]
    forbidden_numbers: list[str]


def _segment_profile(segment_id: int) -> dict:
    """Return a segment profile by id from loaded artifacts."""
    for segment in artifacts.load_profiles()["segments"]:
        if int(segment["id"]) == segment_id:
            return segment
    raise KeyError(f"Segment {segment_id} not found in profiles")


def _segment_uplift(segment_id: int) -> dict:
    """Return uplift metrics for a segment id."""
    for segment in artifacts.load_uplift()["segments"]:
        if int(segment["segment_id"]) == segment_id:
            return segment
    raise KeyError(f"Segment {segment_id} not found in uplift report")


def _segment_recommendations(segment_id: int) -> dict:
    """Return recommendations for a segment id."""
    for segment in artifacts.load_recommendations()["segments"]:
        if int(segment["segment_id"]) == segment_id:
            return segment
    raise KeyError(f"Segment {segment_id} not found in recommendations")


def build_eval_dataset() -> list[EvalCase]:
    """Build evaluation cases from real pipeline artifact data."""
    seg2 = _segment_profile(2)
    seg4 = _segment_profile(4)
    seg0 = _segment_profile(0)

    uplift3 = _segment_uplift(3)
    uplift7 = _segment_uplift(7)

    rec1 = _segment_recommendations(1)
    rec2 = _segment_recommendations(2)
    top_rec2 = rec2["recommendations"][0]

    return [
        {
            "query": f"How many households are in segment 0?",
            "expected_segment_ids": [0],
            "forbidden_numbers": ["99999"],
        },
        {
            "query": f"Describe segment 1 profile size and name.",
            "expected_segment_ids": [1],
            "forbidden_numbers": ["88888"],
        },
        {
            "query": "What's segment 2's uplift?",
            "expected_segment_ids": [2],
            "forbidden_numbers": ["77777"],
        },
        {
            "query": (
                f"What is segment 3 incremental spend percent uplift "
                f"({uplift3['incremental_spend_pct']})?"
            ),
            "expected_segment_ids": [3],
            "forbidden_numbers": ["66666"],
        },
        {
            "query": "What products should we target to segment 2 this week?",
            "expected_segment_ids": [2],
            "forbidden_numbers": ["55555"],
        },
        {
            "query": (
                f"What is the top recommended product id for segment 2 "
                f"({top_rec2['product_id']})?"
            ),
            "expected_segment_ids": [2],
            "forbidden_numbers": ["44444"],
        },
        {
            "query": "Who are promo-sensitive shoppers?",
            "expected_segment_ids": [1],
            "forbidden_numbers": ["33333"],
        },
        {
            "query": f"Which segment is named {seg2['name']}?",
            "expected_segment_ids": [2],
            "forbidden_numbers": ["22222"],
        },
        {
            "query": f"How large is segment 4 ({seg4['name']})?",
            "expected_segment_ids": [4],
            "forbidden_numbers": ["11111"],
        },
        {
            "query": f"What's segment 5 uplift percentage?",
            "expected_segment_ids": [5],
            "forbidden_numbers": ["98765"],
        },
        {
            "query": f"List recommendations for segment 6.",
            "expected_segment_ids": [6],
            "forbidden_numbers": ["87654"],
        },
        {
            "query": f"Tell me about {seg0['name']} segment size.",
            "expected_segment_ids": [0],
            "forbidden_numbers": ["76543"],
        },
        {
            "query": (
                f"Segment 7 uplift percent is {uplift7['incremental_spend_pct']} — confirm."
            ),
            "expected_segment_ids": [7],
            "forbidden_numbers": ["65432"],
        },
        {
            "query": (
                f"What lift does product {rec1['recommendations'][0]['product_id']} "
                f"have for segment 1?"
            ),
            "expected_segment_ids": [1],
            "forbidden_numbers": ["54321"],
        },
        {
            "query": "What is our company stock price today?",
            "expected_segment_ids": [],
            "forbidden_numbers": ["123456789"],
        },
    ]


EVAL_DATASET: list[EvalCase] = build_eval_dataset()

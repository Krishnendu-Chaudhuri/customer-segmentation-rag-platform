"""FastAPI backend for shopper segmentation and RAG chatbot."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from shopper_segmentation.etl import OUTPUT_DIR
from shopper_segmentation.rag.rag_chain import answer_query, get_groq_api_key

PROFILES_PATH = OUTPUT_DIR / "segment_profiles.json"
RECOMMENDATIONS_PATH = OUTPUT_DIR / "segment_recommendations.json"


class SegmentSummary(BaseModel):
    """Summary view of a shopper segment."""

    id: int
    name: str
    size: int
    narrative: str


class SegmentDetail(BaseModel):
    """Detailed segment profile."""

    id: int
    name: str
    size: int
    narrative: str
    feature_means: dict[str, float]


class ProductRecommendation(BaseModel):
    """Product recommendation for a segment."""

    product_id: int
    department: str | None = None
    brand: str | None = None
    commodity_desc: str | None = None
    lift: float
    segment_purchase_rate: float
    population_purchase_rate: float
    segment_buyers: int
    segment_size: int


class SegmentRecommendations(BaseModel):
    """Recommendations payload for one segment."""

    segment_id: int
    segment_name: str
    recommendations: list[ProductRecommendation]


class ChatRequest(BaseModel):
    """Chat request body."""

    query: str = Field(..., min_length=1, description="Natural language analyst question")


class RetrievedSegment(BaseModel):
    """Segment card retrieved for a chat query."""

    segment_id: int
    segment_name: str
    distance: float


class ValidationResult(BaseModel):
    """Number validation result for chat responses."""

    validated: bool
    numbers_found: list[str]
    unsupported_numbers: list[str]


class ChatResponse(BaseModel):
    """Chat response with answer and validation metadata."""

    query: str
    answer: str
    retrieved_segments: list[RetrievedSegment]
    validation: ValidationResult


@lru_cache
def _load_profiles() -> dict[str, Any]:
    """Load segment profiles from disk."""
    return _read_json(PROFILES_PATH)


@lru_cache
def _load_recommendations() -> dict[str, Any]:
    """Load segment recommendations from disk."""
    return _read_json(RECOMMENDATIONS_PATH)


def _read_json(path: Path) -> dict[str, Any]:
    """Read and parse a JSON file.

    Args:
        path: Path to JSON file.

    Returns:
        Parsed JSON object.

    Raises:
        HTTPException: If the file is missing.
    """
    if not path.exists():
        raise HTTPException(status_code=500, detail=f"Missing required data file: {path.name}")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _get_segment_profile(segment_id: int) -> dict[str, Any]:
    """Return a segment profile by id.

    Args:
        segment_id: Segment identifier.

    Returns:
        Segment profile dictionary.

    Raises:
        HTTPException: If segment is not found.
    """
    profiles = _load_profiles()["segments"]
    for segment in profiles:
        if int(segment["id"]) == segment_id:
            return segment
    raise HTTPException(status_code=404, detail=f"Segment {segment_id} not found")


def _get_segment_recommendations(segment_id: int) -> dict[str, Any]:
    """Return recommendations for a segment.

    Args:
        segment_id: Segment identifier.

    Returns:
        Recommendation record for the segment.

    Raises:
        HTTPException: If segment is not found.
    """
    for segment in _load_recommendations()["segments"]:
        if int(segment["segment_id"]) == segment_id:
            return segment
    raise HTTPException(status_code=404, detail=f"Segment {segment_id} not found")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm cached data on startup."""
    _load_profiles()
    _load_recommendations()
    yield


app = FastAPI(
    title="Shopper Segmentation & Personalization API",
    description=(
        "REST API for household segment profiles, product recommendations, "
        "and RAG-powered analyst chat."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/segments", response_model=list[SegmentSummary], tags=["segments"])
def list_segments() -> list[SegmentSummary]:
    """List all shopper segments with summary metadata."""
    return [
        SegmentSummary(
            id=int(segment["id"]),
            name=str(segment["name"]),
            size=int(segment["size"]),
            narrative=str(segment["narrative"]),
        )
        for segment in _load_profiles()["segments"]
    ]


@app.get("/segments/{segment_id}", response_model=SegmentDetail, tags=["segments"])
def get_segment(segment_id: int) -> SegmentDetail:
    """Get detailed profile for a single segment."""
    segment = _get_segment_profile(segment_id)
    return SegmentDetail(
        id=int(segment["id"]),
        name=str(segment["name"]),
        size=int(segment["size"]),
        narrative=str(segment["narrative"]),
        feature_means={k: float(v) for k, v in segment["feature_means"].items()},
    )


@app.get(
    "/segments/{segment_id}/recommendations",
    response_model=SegmentRecommendations,
    tags=["segments"],
)
def get_recommendations(segment_id: int) -> SegmentRecommendations:
    """Get top product recommendations for a segment."""
    segment = _get_segment_recommendations(segment_id)
    recommendations = [
        ProductRecommendation(
            product_id=int(rec["product_id"]),
            department=rec.get("department"),
            brand=rec.get("brand"),
            commodity_desc=rec.get("commodity_desc"),
            lift=float(rec["lift"]),
            segment_purchase_rate=float(rec["segment_purchase_rate"]),
            population_purchase_rate=float(rec["population_purchase_rate"]),
            segment_buyers=int(rec["segment_buyers"]),
            segment_size=int(rec["segment_size"]),
        )
        for rec in segment.get("recommendations", [])
    ]
    return SegmentRecommendations(
        segment_id=int(segment["segment_id"]),
        segment_name=str(segment["segment_name"]),
        recommendations=recommendations,
    )


@app.post("/chat", response_model=ChatResponse, tags=["chat"])
def chat(request: ChatRequest) -> ChatResponse:
    """Ask the analyst chatbot a question about segments and promotions."""
    try:
        get_groq_api_key()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        result = answer_query(request.query)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Chat request failed: {exc}") from exc

    retrieved = [
        RetrievedSegment(
            segment_id=int(card["segment_id"]),
            segment_name=str(card["segment_name"]),
            distance=float(card["distance"]),
        )
        for card in result["retrieved_cards"]
    ]
    validation_raw = result["validation"]
    validation = ValidationResult(
        validated=bool(validation_raw["validated"]),
        numbers_found=[str(n) for n in validation_raw["numbers_found"]],
        unsupported_numbers=[str(n) for n in validation_raw["unsupported_numbers"]],
    )
    return ChatResponse(
        query=str(result["query"]),
        answer=str(result["answer"]),
        retrieved_segments=retrieved,
        validation=validation,
    )


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}

"""FastAPI backend for shopper segmentation and RAG chatbot."""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from shopper_segmentation.api.auth import verify_api_key
from shopper_segmentation.etl import OUTPUT_DIR
from shopper_segmentation.rag.rag_chain import answer_query, get_groq_api_key
from shopper_segmentation.segmentation import MIN_SEGMENT_CONFIDENT_N, is_low_confidence_segment

load_dotenv()

PROFILES_PATH = OUTPUT_DIR / "segment_profiles.json"
RECOMMENDATIONS_PATH = OUTPUT_DIR / "segment_recommendations.json"
DEFAULT_ALLOWED_ORIGINS = "http://localhost:8504"


def get_rate_limit_key(request: Request) -> str:
    """Return rate-limit bucket key from API key or client IP.

    Args:
        request: Incoming HTTP request.

    Returns:
        Identifier used for rate limiting.
    """
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return api_key
    return get_remote_address(request)


def get_allowed_origins() -> list[str]:
    """Parse comma-separated CORS origins from environment.

    Returns:
        List of allowed origin URLs.
    """
    raw = os.getenv("ALLOWED_ORIGINS", DEFAULT_ALLOWED_ORIGINS)
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


limiter = Limiter(key_func=get_rate_limit_key)


class SegmentSummary(BaseModel):
    """Summary view of a shopper segment."""

    id: int
    name: str
    size: int
    narrative: str
    low_confidence: bool


class SegmentDetail(BaseModel):
    """Detailed segment profile."""

    id: int
    name: str
    size: int
    narrative: str
    feature_means: dict[str, float]
    low_confidence: bool


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


def _segment_low_confidence(segment: dict[str, Any]) -> bool:
    """Return whether a segment should be flagged as low confidence.

    Args:
        segment: Segment profile dictionary.

    Returns:
        True when the segment is below the configured household threshold.
    """
    if "low_confidence" in segment:
        return bool(segment["low_confidence"])
    return is_low_confidence_segment(int(segment["size"]), MIN_SEGMENT_CONFIDENT_N)


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
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)

protected = [Depends(verify_api_key)]


@app.get("/segments", response_model=list[SegmentSummary], tags=["segments"], dependencies=protected)
def list_segments() -> list[SegmentSummary]:
    """List all shopper segments with summary metadata."""
    return [
        SegmentSummary(
            id=int(segment["id"]),
            name=str(segment["name"]),
            size=int(segment["size"]),
            narrative=str(segment["narrative"]),
            low_confidence=_segment_low_confidence(segment),
        )
        for segment in _load_profiles()["segments"]
    ]


@app.get(
    "/segments/{segment_id}",
    response_model=SegmentDetail,
    tags=["segments"],
    dependencies=protected,
)
def get_segment(segment_id: int) -> SegmentDetail:
    """Get detailed profile for a single segment."""
    segment = _get_segment_profile(segment_id)
    return SegmentDetail(
        id=int(segment["id"]),
        name=str(segment["name"]),
        size=int(segment["size"]),
        narrative=str(segment["narrative"]),
        feature_means={k: float(v) for k, v in segment["feature_means"].items()},
        low_confidence=_segment_low_confidence(segment),
    )


@app.get(
    "/segments/{segment_id}/recommendations",
    response_model=SegmentRecommendations,
    tags=["segments"],
    dependencies=protected,
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


@app.post("/chat", response_model=ChatResponse, tags=["chat"], dependencies=protected)
@limiter.limit("10/minute")
async def chat(request: Request, body: ChatRequest) -> ChatResponse:
    """Ask the analyst chatbot a question about segments and promotions."""
    try:
        get_groq_api_key()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        result = await run_in_threadpool(answer_query, body.query)
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

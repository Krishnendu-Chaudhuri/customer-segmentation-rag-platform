"""RAG chain: retrieve segment cards and answer via Groq Llama 3 70B."""

from __future__ import annotations

import logging
import os
import re
import time
from collections.abc import Callable
from typing import Any

from cachetools import TTLCache
from dotenv import load_dotenv
from groq import Groq

from shopper_segmentation.logging_config import configure_logging
from shopper_segmentation.rag.embed_store import retrieve_cards

load_dotenv()

logger = logging.getLogger(__name__)

GROQ_MODEL = "llama3-70b-8192"
MAX_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 2.0
DEFAULT_CHAT_CACHE_TTL_SECONDS = 3600
CHAT_CACHE_TTL_SECONDS = int(
    os.getenv("CHAT_CACHE_TTL_SECONDS", str(DEFAULT_CHAT_CACHE_TTL_SECONDS))
)
_chat_cache: TTLCache[str, dict[str, object]] = TTLCache(
    maxsize=256,
    ttl=CHAT_CACHE_TTL_SECONDS,
)

SYSTEM_PROMPT = """You are a retail analytics assistant for a shopper segmentation project.
Answer ONLY using the segment context provided below.

Rules:
- Cite specific numbers from the context when making claims.
- Never invent statistics, percentages, product IDs, or segment sizes.
- If the context lacks information to answer, say you do not have enough data.
- Reference segment IDs and names exactly as shown in the context.
- Be concise and business-friendly.
"""

NUMBER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])-?\d+(?:\.\d+)?(?:%|\s*(?:percent|pct))?(?![A-Za-z0-9_])",
    re.IGNORECASE,
)


def get_groq_api_key() -> str:
    """Load the Groq API key from environment variables.

    Returns:
        Groq API key string.

    Raises:
        RuntimeError: If GROQ_API_KEY is missing.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to a .env file in the project root. "
            "Sign up at https://console.groq.com/ and create an API key."
        )
    return api_key


def get_groq_client() -> Groq:
    """Create a Groq client using the configured API key.

    Returns:
        Initialized Groq client.
    """
    return Groq(api_key=get_groq_api_key())


def extract_numbers(text: str) -> list[str]:
    """Extract numeric tokens from text for validation.

    Args:
        text: Input text.

    Returns:
        List of numeric strings found in the text.
    """
    matches = NUMBER_PATTERN.findall(text)
    normalized: list[str] = []
    for match in matches:
        token = match.strip().lower().replace("percent", "").replace("pct", "").strip()
        token = token.rstrip("%").strip()
        if token and token not in {"-"}:
            normalized.append(token)
    return normalized


def number_in_context(number: str, context: str) -> bool:
    """Check whether a numeric token appears in the retrieved context.

    Args:
        number: Numeric string extracted from model output.
        context: Retrieved context documents concatenated.

    Returns:
        True if the number appears in context directly or approximately.
    """
    if number in context:
        return True

    try:
        value = float(number)
    except ValueError:
        return False

    context_numbers = extract_numbers(context)
    for ctx_num in context_numbers:
        try:
            ctx_value = float(ctx_num)
        except ValueError:
            continue
        if abs(value - ctx_value) < max(0.01, abs(ctx_value) * 0.01):
            return True
    return False


def validate_response_numbers(response: str, context: str) -> dict[str, object]:
    """Validate that numbers in the model response appear in retrieved context.

    Args:
        response: Model-generated answer.
        context: Concatenated retrieved card content.

    Returns:
        Validation summary with unsupported numbers flagged.
    """
    numbers = extract_numbers(response)
    unsupported = [num for num in numbers if not number_in_context(num, context)]
    if unsupported:
        logger.warning("Unsupported numbers in response: %s", unsupported)

    return {
        "validated": len(unsupported) == 0,
        "numbers_found": numbers,
        "unsupported_numbers": unsupported,
    }


def build_messages(query: str, retrieved_cards: list[dict[str, object]]) -> list[dict[str, str]]:
    """Build chat messages with retrieved context injected.

    Args:
        query: User question.
        retrieved_cards: Retrieved segment cards.

    Returns:
        OpenAI-compatible message list for Groq chat completions.
    """
    context = "\n\n---\n\n".join(str(card["content"]) for card in retrieved_cards)
    user_prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer using only the context above."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def call_with_retry(
    func: Callable[[], Any],
    max_retries: int = MAX_RETRIES,
    initial_backoff: float = INITIAL_BACKOFF_SECONDS,
) -> Any:
    """Call a Groq API function with exponential backoff on rate limits.

    Args:
        func: Zero-argument callable performing the API request.
        max_retries: Maximum retry attempts.
        initial_backoff: Initial sleep duration in seconds.

    Returns:
        API response from the callable.

    Raises:
        Exception: Re-raises the last error after retries are exhausted.
    """
    delay = initial_backoff
    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            return func()
        except Exception as exc:
            last_error = exc
            message = str(exc).lower()
            if "rate limit" not in message and "429" not in message:
                raise
            if attempt == max_retries - 1:
                break
            logger.warning(
                "Groq rate limit hit (attempt %s/%s). Retrying in %.1fs.",
                attempt + 1,
                max_retries,
                delay,
            )
            time.sleep(delay)
            delay *= 2

    assert last_error is not None
    raise last_error


def _normalize_query(query: str) -> str:
    """Normalize a query string for cache lookup.

    Args:
        query: Raw user query.

    Returns:
        Lowercased, whitespace-trimmed cache key.
    """
    return query.strip().lower()


def clear_chat_cache() -> None:
    """Clear the in-memory chat response cache."""
    _chat_cache.clear()


def answer_query(
    query: str,
    top_k: int = 3,
    client: Groq | None = None,
) -> dict[str, object]:
    """Retrieve context and answer a user query via Groq.

    Args:
        query: Natural language question.
        top_k: Number of segment cards to retrieve.
        client: Optional preconfigured Groq client.

    Returns:
        Dictionary with answer, retrieved cards, and validation metadata.
    """
    cache_key = _normalize_query(query)
    if cache_key in _chat_cache:
        logger.info("Chat cache hit for query: %s", query)
        return _chat_cache[cache_key]

    retrieved = retrieve_cards(query, top_k=top_k)
    context = "\n\n---\n\n".join(str(card["content"]) for card in retrieved)
    messages = build_messages(query, retrieved)

    groq_client = client or get_groq_client()

    def _call() -> Any:
        return groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.1,
        )

    response = call_with_retry(_call)
    answer = response.choices[0].message.content or ""
    validation = validate_response_numbers(answer, context)

    result = {
        "query": query,
        "answer": answer,
        "retrieved_cards": retrieved,
        "validation": validation,
    }
    _chat_cache[cache_key] = result
    return result


def main() -> None:
    """Run demo queries against the RAG chain."""
    configure_logging()
    logger.info("Module 5: RAG — Explainability Chatbot")

    demo_queries = [
        "Who are our high-value promo-sensitive shoppers?",
        "What products should we target to segment 2 this week?",
    ]

    try:
        get_groq_api_key()
        has_key = True
    except RuntimeError as exc:
        has_key = False
        logger.warning("%s", exc)
        logger.info("Showing retrieval results only (no LLM call).")

    for query in demo_queries:
        logger.info("Query: %s", query)
        retrieved = retrieve_cards(query, top_k=3)
        logger.info(
            "Retrieved segments: %s",
            [c["segment_name"] for c in retrieved],
        )

        if not has_key:
            logger.info(
                "Top retrieved card preview:\n%s\n...",
                str(retrieved[0]["content"])[:600],
            )
            continue

        result = answer_query(query, client=get_groq_client())
        logger.info("%s", result["answer"])
        validation = result["validation"]
        if validation["unsupported_numbers"]:
            logger.warning("Validation flags: %s", validation["unsupported_numbers"])
        else:
            logger.info("Validation: all cited numbers found in context.")


if __name__ == "__main__":
    main()

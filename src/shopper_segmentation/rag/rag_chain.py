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
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from shopper_segmentation.logging_config import configure_logging
from shopper_segmentation.rag.agent.prompts import SYSTEM_PROMPT
from shopper_segmentation.rag.vectorstore import retrieve_cards

load_dotenv()

logger = logging.getLogger(__name__)

GROQ_MODEL = "llama3-70b-8192"
GROQ_FALLBACK_MODEL = "llama3-8b-8192"
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


def get_chat_model() -> ChatGroq:
    """Create a Groq chat model with retry handling and fallback.

    Returns:
        Configured ChatGroq instance with 70B primary and 8B fallback.
    """
    primary = ChatGroq(
        model=GROQ_MODEL,
        temperature=0.1,
        api_key=get_groq_api_key(),
    ).with_retry(stop_after_attempt=MAX_RETRIES)
    fallback = ChatGroq(
        model=GROQ_FALLBACK_MODEL,
        temperature=0.1,
        api_key=get_groq_api_key(),
    ).with_retry(stop_after_attempt=MAX_RETRIES)
    return primary.with_fallbacks([fallback])


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


def build_messages(query: str, retrieved_cards: list[dict[str, object]]) -> list[Any]:
    """Build chat messages with retrieved context injected.

    Args:
        query: User question.
        retrieved_cards: Retrieved segment cards.

    Returns:
        LangChain message list for the chat model.
    """
    context = "\n\n---\n\n".join(str(card["content"]) for card in retrieved_cards)
    user_prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer using only the context above."
    )
    return [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]


def call_with_retry(
    func: Callable[[], Any],
    max_retries: int = MAX_RETRIES,
    initial_backoff: float = INITIAL_BACKOFF_SECONDS,
) -> Any:
    """Call a function with exponential backoff on rate limits.

    Args:
        func: Zero-argument callable performing the request.
        max_retries: Maximum retry attempts.
        initial_backoff: Initial sleep duration in seconds.

    Returns:
        Result from the callable.

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


def _message_content(response: AIMessage) -> str:
    """Extract string content from a chat model response.

    Args:
        response: LangChain AIMessage response.

    Returns:
        Response text content.
    """
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [part.get("text", "") for part in content if isinstance(part, dict)]
        return "".join(parts)
    return str(content)


def answer_query(
    query: str,
    top_k: int = 3,
    client: Any | None = None,
    thread_id: str | None = None,
) -> dict[str, object]:
    """Retrieve context and answer a user query via the LangGraph agent.

    Args:
        query: Natural language question.
        top_k: Number of segment cards to retrieve (search route only).
        client: Optional preconfigured chat model (ChatGroq); used by generate node.
        thread_id: LangGraph session thread id for conversational memory.

    Returns:
        Dictionary with answer, retrieved cards, and validation metadata.
    """
    import asyncio

    async def _run() -> dict[str, object]:
        from shopper_segmentation.rag.agent.graph import (
            get_compiled_graph_if_ready,
            init_graph,
            shutdown_graph,
        )

        managed_lifecycle = get_compiled_graph_if_ready() is None
        if managed_lifecycle:
            await init_graph()
        try:
            return await answer_query_async(query, top_k, client, thread_id)
        finally:
            if managed_lifecycle:
                await shutdown_graph()

    return asyncio.run(_run())


async def answer_query_async(
    query: str,
    top_k: int = 3,
    client: Any | None = None,
    thread_id: str | None = None,
) -> dict[str, object]:
    """Async variant of answer_query using graph.ainvoke.

    Args:
        query: Natural language question.
        top_k: Number of segment cards to retrieve (search route only).
        client: Optional preconfigured chat model (ChatGroq); used by generate node.
        thread_id: LangGraph session thread id for conversational memory.

    Returns:
        Dictionary with answer, retrieved cards, and validation metadata.
    """
    import uuid

    from shopper_segmentation.rag.agent.graph import get_compiled_graph

    cache_key = _normalize_query(query)
    if cache_key in _chat_cache:
        logger.info("Chat cache hit for query: %s", query)
        return _chat_cache[cache_key]

    session_thread_id = thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_thread_id}}
    graph = get_compiled_graph()
    input_state = {"query": query, "retry_count": 0}

    if client is not None:
        from unittest.mock import patch

        with patch(
            "shopper_segmentation.rag.rag_chain.get_chat_model",
            return_value=client,
        ):
            final_state = await graph.ainvoke(input_state, config=config)
    else:
        final_state = await graph.ainvoke(input_state, config=config)

    result = {
        "query": query,
        "answer": final_state.get("answer", ""),
        "retrieved_cards": final_state.get("retrieved_cards", []),
        "validation": final_state.get("validation", {}),
    }
    _chat_cache[cache_key] = result
    return result


def main() -> None:
    """Run demo queries against the RAG chain."""
    import asyncio

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

    async def _run_demo() -> None:
        from shopper_segmentation.rag.agent.graph import init_graph, shutdown_graph

        await init_graph()
        try:
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

                result = await answer_query_async(query, client=get_chat_model())
                logger.info("%s", result["answer"])
                validation = result["validation"]
                if validation["unsupported_numbers"]:
                    logger.warning("Validation flags: %s", validation["unsupported_numbers"])
                else:
                    logger.info("Validation: all cited numbers found in context.")
        finally:
            await shutdown_graph()

    asyncio.run(_run_demo())


if __name__ == "__main__":
    main()

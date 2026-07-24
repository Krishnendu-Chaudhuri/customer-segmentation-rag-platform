"""Prompt templates for the LangGraph analyst agent."""

from __future__ import annotations

SYSTEM_PROMPT = """You are a retail analytics assistant for a shopper segmentation project.
Answer ONLY using the segment context provided below.

Rules:
- Cite specific numbers from the context when making claims.
- Never invent statistics, percentages, product IDs, or segment sizes.
- If the context lacks information to answer, say you do not have enough data.
- Reference segment IDs and names exactly as shown in the context.
- Be concise and business-friendly.
"""

ROUTER_PROMPT = """Classify the analyst question as either "lookup" or "search".

Use "lookup" when the question references a specific segment id or asks for a direct
segment metric, recommendation list, or uplift value (e.g. "segment 2 uplift").

Use "search" for open-ended questions about shopper types or promo sensitivity that
require semantic retrieval across segment cards.

Respond with only one word: lookup or search.
"""

RETRY_PROMPT = (
    "Your previous answer cited numbers not found in context. "
    "Answer again using ONLY numbers present in context, or say the data isn't available."
)

INSUFFICIENT_GROUNDED_DATA_MESSAGE = (
    "I do not have enough grounded data in the retrieved segment context to answer "
    "that question with validated numbers."
)

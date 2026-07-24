"""Run the LangGraph RAG evaluation suite."""

from __future__ import annotations

import asyncio
import sys
import uuid

from dotenv import load_dotenv
from tabulate import tabulate

from shopper_segmentation.artifacts import ensure_artifacts
from shopper_segmentation.logging_config import configure_logging
from shopper_segmentation.rag.agent.graph import get_compiled_graph, init_graph, shutdown_graph
from shopper_segmentation.rag.evals.eval_dataset import EVAL_DATASET, EvalCase
from shopper_segmentation.rag.rag_chain import extract_numbers

load_dotenv()


def _segment_overlap(retrieved_cards: list[dict], expected_ids: list[int]) -> bool:
    """Return True when retrieved cards overlap expected segment ids."""
    if not expected_ids:
        return True
    retrieved_ids = {int(card["segment_id"]) for card in retrieved_cards}
    return bool(retrieved_ids.intersection(expected_ids))


def _forbidden_present(answer: str, forbidden_numbers: list[str]) -> list[str]:
    """Return forbidden numbers found in the model answer."""
    answer_numbers = set(extract_numbers(answer))
    return [number for number in forbidden_numbers if number in answer_numbers]


async def evaluate_case(case: EvalCase, thread_id: str) -> dict[str, object]:
    """Run one evaluation case through the compiled graph."""
    graph = get_compiled_graph()
    state = await graph.ainvoke(
        {"query": case["query"], "retry_count": 0},
        config={"configurable": {"thread_id": thread_id}},
    )
    validation = state.get("validation") or {}
    answer = str(state.get("answer", ""))
    retrieved_cards = state.get("retrieved_cards") or []

    overlap_ok = _segment_overlap(retrieved_cards, case["expected_segment_ids"])
    forbidden_hits = _forbidden_present(answer, case["forbidden_numbers"])
    validated = bool(validation.get("validated"))

    passed = validated and overlap_ok and not forbidden_hits
    return {
        "query": case["query"],
        "passed": passed,
        "validated": validated,
        "overlap_ok": overlap_ok,
        "forbidden_hits": forbidden_hits,
        "answer_preview": answer[:120],
    }


async def main_async() -> int:
    """Execute all evaluation cases and print a summary table."""
    configure_logging()
    ensure_artifacts()
    await init_graph()

    rows: list[list[object]] = []
    failures = 0

    try:
        for index, case in enumerate(EVAL_DATASET):
            thread_id = f"eval-{index}-{uuid.uuid4().hex[:8]}"
            result = await evaluate_case(case, thread_id)
            if not result["passed"]:
                failures += 1
            rows.append(
                [
                    "PASS" if result["passed"] else "FAIL",
                    case["query"][:60],
                    result["validated"],
                    result["overlap_ok"],
                    result["forbidden_hits"],
                ]
            )
    finally:
        await shutdown_graph()

    print(tabulate(
        rows,
        headers=["Status", "Query", "Validated", "Segment overlap", "Forbidden hits"],
        tablefmt="github",
    ))
    print(f"\n{len(EVAL_DATASET) - failures}/{len(EVAL_DATASET)} cases passed.")
    return 1 if failures else 0


def main() -> int:
    """Run the async evaluation suite."""
    return asyncio.run(main_async())


if __name__ == "__main__":
    sys.exit(main())

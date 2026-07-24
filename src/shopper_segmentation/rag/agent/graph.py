"""LangGraph analyst agent graph definition."""

from __future__ import annotations

import os
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, StateGraph

from shopper_segmentation.rag.agent.nodes import (
    generate_node,
    guardrail_node,
    respond_node,
    retrieve_node,
    retry_bump_node,
    router_node,
)
from shopper_segmentation.rag.agent.state import AgentState

DEFAULT_CHECKPOINT_DB_PATH = "output/checkpoints.sqlite"


def get_checkpoint_db_path() -> str:
    """Return the SQLite checkpoint database path from environment."""
    return os.getenv("CHECKPOINT_DB_PATH", DEFAULT_CHECKPOINT_DB_PATH)


def _guardrail_route(state: AgentState) -> str:
    """Route to a single retry or final response after guardrail validation."""
    validation = state.get("validation") or {}
    retry_count = state.get("retry_count", 0)
    if not validation.get("validated") and retry_count < 1:
        return "retry"
    return "respond"


def build_graph() -> StateGraph:
    """Construct the analyst agent state graph."""
    graph = StateGraph(AgentState)
    graph.add_node("router", router_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("guardrail", guardrail_node)
    graph.add_node("retry_bump", retry_bump_node)
    graph.add_node("respond", respond_node)

    graph.set_entry_point("router")
    graph.add_edge("router", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "guardrail")
    graph.add_conditional_edges(
        "guardrail",
        _guardrail_route,
        {"retry": "retry_bump", "respond": "respond"},
    )
    graph.add_edge("retry_bump", "generate")
    graph.add_edge("respond", END)
    return graph


_checkpointer: AsyncSqliteSaver | None = None
_checkpointer_cm: AbstractAsyncContextManager[AsyncSqliteSaver] | None = None
_compiled_graph = None


async def init_graph() -> None:
    """Open AsyncSqliteSaver and compile the agent graph."""
    global _checkpointer, _checkpointer_cm, _compiled_graph
    if _compiled_graph is not None:
        return

    db_path = Path(get_checkpoint_db_path())
    db_path.parent.mkdir(parents=True, exist_ok=True)
    _checkpointer_cm = AsyncSqliteSaver.from_conn_string(str(db_path))
    assert _checkpointer_cm is not None
    _checkpointer = await _checkpointer_cm.__aenter__()
    await _checkpointer.setup()
    _compiled_graph = build_graph().compile(checkpointer=_checkpointer)


async def shutdown_graph() -> None:
    """Close the async checkpointer and reset compiled graph state."""
    global _checkpointer, _checkpointer_cm, _compiled_graph
    if _checkpointer_cm is not None:
        await _checkpointer_cm.__aexit__(None, None, None)
    _checkpointer_cm = None
    _checkpointer = None
    _compiled_graph = None


def get_compiled_graph_if_ready() -> Any | None:
    """Return the compiled graph when initialized, else None."""
    return _compiled_graph


def get_compiled_graph() -> Any:
    """Return the compiled LangGraph agent with session checkpointer."""
    if _compiled_graph is None:
        raise RuntimeError("LangGraph not initialized; call init_graph() first.")
    return _compiled_graph

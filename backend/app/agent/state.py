"""Agent state definitions for RepoPilot's LangGraph workflow."""

from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    """Minimal state representation for repository investigation."""

    question: str
    top_k: int
    retrieved_chunks: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    tools_used: list[str]
    source_inspections: list[dict[str, Any]]
    graph_investigations: list[dict[str, Any]]
    needs_more_investigation: bool
    needs_graph_investigation: bool
    target_inspection: dict[str, Any] | None
    investigation_status: str
    answer: str
    iteration: int

"""Agent package for RepoPilot."""

from __future__ import annotations

from functools import lru_cache

from app.agent.graph import create_agent_graph
from app.agent.service import AgentService
from app.agent.state import AgentState
from app.agent.tools import DependencyGraphTool, SemanticSearchTool, SourceInspectionTool
from app.rag import get_default_llm_client, get_retriever


@lru_cache
def get_agent_service() -> AgentService:
    """Return the application-wide AgentService singleton."""
    from app.ingestion.repository import get_dependency_graph

    graph_tool = DependencyGraphTool(graph=get_dependency_graph())
    return AgentService(
        retriever=get_retriever(),
        inspection_tool=SourceInspectionTool(),
        graph_tool=graph_tool,
        llm_client=get_default_llm_client(),
    )


__all__ = [
    "AgentService",
    "AgentState",
    "DependencyGraphTool",
    "SemanticSearchTool",
    "SourceInspectionTool",
    "create_agent_graph",
    "get_agent_service",
]

"""Agent package for RepoPilot."""

from __future__ import annotations

from functools import lru_cache

from app.agent.graph import create_agent_graph
from app.agent.service import AgentService
from app.agent.state import AgentState
from app.agent.tools import SemanticSearchTool, SourceInspectionTool
from app.rag import get_default_llm_client, get_retriever


@lru_cache
def get_agent_service() -> AgentService:
    """Return the application-wide AgentService singleton."""
    return AgentService(
        retriever=get_retriever(),
        inspection_tool=SourceInspectionTool(),
        llm_client=get_default_llm_client(),
    )


__all__ = [
    "AgentService",
    "AgentState",
    "SemanticSearchTool",
    "SourceInspectionTool",
    "create_agent_graph",
    "get_agent_service",
]

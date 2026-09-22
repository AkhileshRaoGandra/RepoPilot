"""Agent service orchestrating LangGraph execution for RepoPilot."""

from __future__ import annotations

from typing import Any

from app.agent.graph import create_agent_graph
from app.agent.tools import DependencyGraphTool, SemanticSearchTool, SourceInspectionTool
from app.rag.llm import LLMClient, get_default_llm_client
from app.rag.retriever import Retriever


class AgentService:
    """Service executing LangGraph stateful repository investigations."""

    def __init__(
        self,
        retriever: Retriever,
        inspection_tool: SourceInspectionTool | None = None,
        graph_tool: DependencyGraphTool | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.retriever = retriever
        self.inspection_tool = inspection_tool or SourceInspectionTool()
        self.graph_tool = graph_tool or DependencyGraphTool()
        self.search_tool = SemanticSearchTool(retriever=retriever)
        self.llm_client = llm_client or get_default_llm_client()
        self.graph = create_agent_graph(
            search_tool=self.search_tool,
            inspection_tool=self.inspection_tool,
            llm_client=self.llm_client,
            graph_tool=self.graph_tool,
        )

    def run_investigation(self, query: str, top_k: int = 5) -> dict[str, Any]:
        """Execute the LangGraph workflow and return the investigation outcome."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        initial_state = {
            "question": query.strip(),
            "top_k": top_k,
        }
        result = self.graph.invoke(initial_state)

        return {
            "question": result.get("question", query.strip()),
            "answer": result.get("answer", ""),
            "evidence": result.get("evidence", []),
            "tools_used": result.get("tools_used", []),
            "investigation_status": result.get("investigation_status", "completed"),
            "retrieved_chunks": result.get("retrieved_chunks", []),
            "source_inspections": result.get("source_inspections", []),
            "graph_investigations": result.get("graph_investigations", []),
        }

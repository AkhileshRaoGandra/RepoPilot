"""LangGraph workflow for RepoPilot's agentic repository investigation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langgraph.graph import END, START, StateGraph

from app.agent.state import AgentState
from app.rag.context import build_rag_context, extract_evidence
from app.rag.service import DEFAULT_SYSTEM_PROMPT, RAGService

if TYPE_CHECKING:
    from app.agent.tools import SemanticSearchTool, SourceInspectionTool
    from app.rag.llm import LLMClient


def route_evaluation(state: AgentState) -> str:
    """Conditional edge router: decide whether additional source inspection is needed."""
    if state.get("needs_more_investigation", False) and state.get("iteration", 0) == 0:
        return "source_inspection"
    return "generate_answer"


def create_agent_graph(
    search_tool: SemanticSearchTool,
    inspection_tool: SourceInspectionTool,
    llm_client: LLMClient,
) -> Any:
    """Construct and compile the LangGraph StateGraph for agentic investigation."""

    def decide_initial_action(state: AgentState) -> dict[str, Any]:
        """Node 1: Validate input question and initialize investigation state."""
        question = state.get("question", "").strip()
        if not question:
            raise ValueError("Question must be a non-empty string.")

        return {
            "question": question,
            "top_k": state.get("top_k", 5),
            "tools_used": [],
            "retrieved_chunks": [],
            "evidence": [],
            "source_inspections": [],
            "iteration": 0,
            "needs_more_investigation": False,
            "target_inspection": None,
            "investigation_status": "searching",
        }

    def repository_search(state: AgentState) -> dict[str, Any]:
        """Node 2: Execute semantic repository retrieval using Tool 1."""
        question = state["question"]
        top_k = state.get("top_k", 5)

        chunks = search_tool.search(query=question, top_k=top_k)
        evidence = extract_evidence(chunks)

        tools_used = [*state.get("tools_used", []), "semantic_search"]
        return {
            "retrieved_chunks": chunks,
            "evidence": evidence,
            "tools_used": tools_used,
            "investigation_status": "evaluating",
        }

    def evaluate_context(state: AgentState) -> dict[str, Any]:
        """Node 3: Determine if retrieved chunks suffice or if source inspection is warranted."""
        chunks = state.get("retrieved_chunks", [])
        if not chunks:
            # Nothing in vector store; mark as insufficient context
            return {
                "needs_more_investigation": False,
                "target_inspection": None,
                "investigation_status": "insufficient",
            }

        question_lower = state["question"].lower()
        needs_more = False
        target: dict[str, Any] | None = None

        # Check for clues indicating deeper source inspection is desirable:
        # 1. The question explicitly asks for inspection / surrounding lines / full context.
        # 2. A top retrieved chunk was split across parts (contains 'part' metadata).
        inspect_keywords = {"inspect", "full", "surrounding", "lines", "detail", "complete", "around"}
        has_inspect_intent = any(kw in question_lower for kw in inspect_keywords)
        is_partial_chunk = bool(chunks[0].get("metadata", {}).get("part"))

        if (has_inspect_intent or is_partial_chunk) and state.get("iteration", 0) == 0:
            top_metadata = chunks[0].get("metadata", {})
            file_path = top_metadata.get("file")
            start_line = top_metadata.get("start_line")
            end_line = top_metadata.get("end_line")

            if file_path:
                needs_more = True
                # Slightly broaden window if inspecting context lines
                target_start = max(1, start_line - 5) if start_line is not None else 1
                target_end = end_line + 5 if end_line is not None else None
                target = {
                    "file": file_path,
                    "start_line": target_start,
                    "end_line": target_end,
                }

        return {
            "needs_more_investigation": needs_more,
            "target_inspection": target,
            "investigation_status": "inspecting" if needs_more else "answering",
        }

    def source_inspection(state: AgentState) -> dict[str, Any]:
        """Node 4: Execute direct source code inspection using Tool 2."""
        target = state.get("target_inspection")
        source_inspections = list(state.get("source_inspections", []))
        evidence = list(state.get("evidence", []))

        if target and target.get("file"):
            result = inspection_tool.inspect(
                file_path=target["file"],
                start_line=target.get("start_line"),
                end_line=target.get("end_line"),
            )
            source_inspections.append(result)

            if result.get("found") and result.get("lines"):
                citation = f"{result['file']}:{result['lines']}"
                if not any(e.get("citation") == citation for e in evidence):
                    evidence.append(
                        {
                            "file": result["file"],
                            "symbol": "inspected_range",
                            "start_line": result.get("start_line"),
                            "end_line": result.get("end_line"),
                            "lines": result.get("lines"),
                            "citation": citation,
                            "score": 1.0,
                        }
                    )

        tools_used = [*state.get("tools_used", []), "source_inspection"]
        return {
            "source_inspections": source_inspections,
            "evidence": evidence,
            "tools_used": tools_used,
            "iteration": state.get("iteration", 0) + 1,
            "investigation_status": "answering",
        }

    def generate_answer(state: AgentState) -> dict[str, Any]:
        """Node 5: Formulate the final grounded answer with repository evidence."""
        if state.get("investigation_status") == "insufficient" or not state.get("retrieved_chunks"):
            return {
                "answer": "The provided repository context is insufficient to answer this question.",
                "evidence": [],
                "investigation_status": "insufficient",
            }

        # Combine semantic chunks with any inspected source blocks
        context = build_rag_context(state.get("retrieved_chunks", []))
        inspections = state.get("source_inspections", [])

        if inspections:
            inspection_blocks = []
            for item in inspections:
                if item.get("found") and item.get("content"):
                    inspection_blocks.append(
                        f"Inspected File: {item['file']}\n"
                        f"Lines: {item.get('lines', 'Unknown')}\n\n"
                        f"{item['content'].strip()}"
                    )
            if inspection_blocks:
                context = f"{context}\n\n---\n\n" + "\n\n---\n\n".join(inspection_blocks)

        user_prompt = RAGService.build_user_prompt(state["question"], context)
        answer = llm_client.generate(DEFAULT_SYSTEM_PROMPT, user_prompt)

        return {
            "answer": answer,
            "investigation_status": "completed",
        }

    # Build the StateGraph
    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("decide_initial_action", decide_initial_action)
    graph_builder.add_node("repository_search", repository_search)
    graph_builder.add_node("evaluate_context", evaluate_context)
    graph_builder.add_node("source_inspection", source_inspection)
    graph_builder.add_node("generate_answer", generate_answer)

    graph_builder.add_edge(START, "decide_initial_action")
    graph_builder.add_edge("decide_initial_action", "repository_search")
    graph_builder.add_edge("repository_search", "evaluate_context")
    graph_builder.add_conditional_edges(
        "evaluate_context",
        route_evaluation,
        {
            "source_inspection": "source_inspection",
            "generate_answer": "generate_answer",
        },
    )
    graph_builder.add_edge("source_inspection", "generate_answer")
    graph_builder.add_edge("generate_answer", END)

    return graph_builder.compile()

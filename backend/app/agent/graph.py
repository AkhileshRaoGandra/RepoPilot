"""LangGraph workflow for RepoPilot's agentic repository investigation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langgraph.graph import END, START, StateGraph

from app.agent.state import AgentState
from app.rag.context import build_rag_context, extract_evidence
from app.rag.service import DEFAULT_SYSTEM_PROMPT, RAGService

if TYPE_CHECKING:
    from app.agent.tools import DependencyGraphTool, SemanticSearchTool, SourceInspectionTool
    from app.rag.llm import LLMClient


# Keywords that suggest the user is asking about code relationships / flow.
_GRAPH_KEYWORDS = frozenset({
    "depends", "dependency", "dependencies", "dependents",
    "imports", "imported",
    "uses", "used by", "calls", "called",
    "connected", "related", "relationship",
    "flow", "reach", "chain", "path",
    "how does", "where does",
})


def _wants_graph_investigation(question: str) -> bool:
    """Heuristic: does the question suggest relationship / dependency investigation?"""
    q = question.lower()
    return any(kw in q for kw in _GRAPH_KEYWORDS)


def route_evaluation(state: AgentState) -> str:
    """Conditional edge router after context evaluation."""
    if state.get("needs_graph_investigation", False):
        return "graph_investigation"
    if state.get("needs_more_investigation", False) and state.get("iteration", 0) == 0:
        return "source_inspection"
    return "generate_answer"


def route_after_graph(state: AgentState) -> str:
    """After graph investigation, decide whether source inspection is also warranted."""
    if state.get("needs_more_investigation", False) and state.get("iteration", 0) == 0:
        return "source_inspection"
    return "generate_answer"


def create_agent_graph(
    search_tool: SemanticSearchTool,
    inspection_tool: SourceInspectionTool,
    llm_client: LLMClient,
    graph_tool: DependencyGraphTool | None = None,
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
            "graph_investigations": [],
            "iteration": 0,
            "needs_more_investigation": False,
            "needs_graph_investigation": False,
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
        """Node 3: Determine if additional investigation is warranted."""
        chunks = state.get("retrieved_chunks", [])
        if not chunks:
            return {
                "needs_more_investigation": False,
                "needs_graph_investigation": False,
                "target_inspection": None,
                "investigation_status": "insufficient",
            }

        question_lower = state["question"].lower()
        needs_more = False
        needs_graph = False
        target: dict[str, Any] | None = None

        # Check for graph investigation intent
        if graph_tool is not None and graph_tool.available and _wants_graph_investigation(state["question"]):
            needs_graph = True

        # Check for source inspection intent
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
                target_start = max(1, start_line - 5) if start_line is not None else 1
                target_end = end_line + 5 if end_line is not None else None
                target = {
                    "file": file_path,
                    "start_line": target_start,
                    "end_line": target_end,
                }

        status = "answering"
        if needs_graph:
            status = "investigating_graph"
        elif needs_more:
            status = "inspecting"

        return {
            "needs_more_investigation": needs_more,
            "needs_graph_investigation": needs_graph,
            "target_inspection": target,
            "investigation_status": status,
        }

    def graph_investigation(state: AgentState) -> dict[str, Any]:
        """Node 3b: Explore repository relationships using Tool 3 (dependency graph)."""
        if graph_tool is None or not graph_tool.available:
            return {
                "graph_investigations": [],
                "needs_graph_investigation": False,
                "investigation_status": "answering",
            }

        chunks = state.get("retrieved_chunks", [])
        graph_results: list[dict[str, Any]] = []

        # Collect unique files from top search results
        seen_files: set[str] = set()
        target_files: list[str] = []
        for chunk in chunks[:5]:
            fp = chunk.get("metadata", {}).get("file")
            if fp and fp not in seen_files:
                seen_files.add(fp)
                target_files.append(fp)

        # For each top file, get dependencies and dependents
        for fp in target_files[:3]:  # Limit to top 3 files
            deps_result = graph_tool.find_dependencies(fp)
            if deps_result.get("dependencies") or deps_result.get("symbols"):
                graph_results.append(deps_result)

            dependents_result = graph_tool.find_dependents(fp)
            if dependents_result.get("dependents"):
                graph_results.append(dependents_result)

        # Also try symbol-based search using key terms from the question
        question = state.get("question", "")
        # Extract potential symbol names (capitalized words or specific terms)
        words = question.split()
        for word in words:
            cleaned = word.strip("?.,!\"'()[]{}:")
            if cleaned and (cleaned[0].isupper() or len(cleaned) > 3):
                related = graph_tool.find_related(cleaned)
                if related.get("related"):
                    graph_results.append(related)

        tools_used = [*state.get("tools_used", []), "graph_investigation"]
        evidence = list(state.get("evidence", []))

        # Add graph-discovered files as evidence leads
        for result in graph_results:
            for dep in result.get("dependencies", []):
                citation = f"{dep}:graph"
                if not any(e.get("citation") == citation for e in evidence):
                    evidence.append({
                        "file": dep,
                        "symbol": "dependency",
                        "start_line": None,
                        "end_line": None,
                        "lines": "graph",
                        "citation": citation,
                        "score": None,
                    })

        return {
            "graph_investigations": graph_results,
            "tools_used": tools_used,
            "evidence": evidence,
            "needs_graph_investigation": False,
            "investigation_status": "inspecting" if state.get("needs_more_investigation") else "answering",
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

        # Add graph investigation results to context
        graph_investigations = state.get("graph_investigations", [])
        if graph_investigations:
            graph_blocks: list[str] = []
            for result in graph_investigations:
                if "dependencies" in result and result["dependencies"]:
                    graph_blocks.append(
                        f"File: {result['file']}\n"
                        f"Imports: {', '.join(result['dependencies'])}"
                    )
                if "dependents" in result and result["dependents"]:
                    graph_blocks.append(
                        f"File: {result['file']}\n"
                        f"Imported by: {', '.join(result['dependents'])}"
                    )
                if "related" in result and result["related"]:
                    related_strs = [
                        f"{r['name']} ({r['type']}) in {r.get('file', '?')}"
                        for r in result["related"]
                    ]
                    graph_blocks.append(
                        f"Related to '{result.get('query', '?')}':\n" +
                        "\n".join(f"  - {s}" for s in related_strs)
                    )
            if graph_blocks:
                context = (
                    f"{context}\n\n---\n\n"
                    "Repository Dependency Graph Information:\n\n"
                    + "\n\n".join(graph_blocks)
                )

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
    graph_builder.add_node("graph_investigation", graph_investigation)
    graph_builder.add_node("source_inspection", source_inspection)
    graph_builder.add_node("generate_answer", generate_answer)

    graph_builder.add_edge(START, "decide_initial_action")
    graph_builder.add_edge("decide_initial_action", "repository_search")
    graph_builder.add_edge("repository_search", "evaluate_context")
    graph_builder.add_conditional_edges(
        "evaluate_context",
        route_evaluation,
        {
            "graph_investigation": "graph_investigation",
            "source_inspection": "source_inspection",
            "generate_answer": "generate_answer",
        },
    )
    graph_builder.add_conditional_edges(
        "graph_investigation",
        route_after_graph,
        {
            "source_inspection": "source_inspection",
            "generate_answer": "generate_answer",
        },
    )
    graph_builder.add_edge("source_inspection", "generate_answer")
    graph_builder.add_edge("generate_answer", END)

    return graph_builder.compile()

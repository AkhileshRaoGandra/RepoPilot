"""Orchestration service for RAG retrieval, context building, and grounded answer generation."""

from __future__ import annotations

from typing import Any

from app.rag.context import build_rag_context, extract_evidence
from app.rag.llm import LLMClient, get_default_llm_client
from app.rag.retriever import Retriever


DEFAULT_SYSTEM_PROMPT = (
    "You are a repository analysis assistant for RepoPilot.\n"
    "Your task is to answer questions about the codebase strictly using the provided repository context.\n\n"
    "Guidelines:\n"
    "1. Rely only on the code, documentation, and metadata provided in the context.\n"
    "2. Do not invent, assume, or hallucinate repository details, files, classes, or functions that are not in the context.\n"
    "3. If the provided context is empty, missing, or insufficient to answer the question, clearly state:\n"
    '   "The provided repository context is insufficient to answer this question."\n'
    "4. Reference specific files and line numbers (e.g. path/to/file.py:10-25 or README.md:1-30) when referencing code or documentation.\n"
    "5. If asked about a feature, module, or integration that is absent from the repository context (e.g. payment gateway when no payments exist), clearly explain that the feature is not implemented or present in the repository.\n"
    "6. When answering high-level questions (such as architecture, design, concepts, or installation), synthesize BOTH the documentation (e.g. README.md) AND the corresponding source code chunks provided in the context to explain how the concepts map to the actual codebase.\n"
)


class RAGService:
    """Coordinate semantic retrieval, context compilation, and grounded LLM generation."""

    def __init__(
        self,
        retriever: Retriever,
        llm_client: LLMClient | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        self.retriever = retriever
        self.llm_client = llm_client or get_default_llm_client()
        self.system_prompt = system_prompt

    @staticmethod
    def build_user_prompt(question: str, context: str) -> str:
        """Format the user prompt with clear separation between context and question."""
        context_str = context.strip() if context.strip() else "No relevant repository context found."
        return (
            f"Repository Context:\n"
            f"-------------------\n"
            f"{context_str}\n\n"
            f"-------------------\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )

    def answer_question(self, query: str, top_k: int = 5) -> dict[str, Any]:
        """Retrieve top-K chunks and generate a grounded answer with evidence citations."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        cleaned_query = query.strip()
        chunks = list(self.retriever.retrieve(query=cleaned_query, top_k=top_k))

        # Check for architectural / conceptual queries to balance README and code context
        concept_keywords = {
            "architecture", "architectural", "design", "structure", "overview",
            "install", "installation", "setup", "run", "concept", "concepts",
            "what does", "how does it work", "components", "stack", "readme",
        }
        question_lower = cleaned_query.lower()
        is_concept_query = any(kw in question_lower for kw in concept_keywords)

        def _is_doc(c: Mapping[str, Any]) -> bool:
            meta = c.get("metadata", {})
            f = str(meta.get("file", "")).lower()
            return meta.get("type") == "documentation" or "readme" in f or f.endswith((".md", ".rst", ".txt"))

        def _is_code(c: Mapping[str, Any]) -> bool:
            return not _is_doc(c)

        if is_concept_query and chunks:
            seen = {
                f"{c.get('metadata', {}).get('file')}:{c.get('metadata', {}).get('start_line')}"
                for c in chunks
            }
            if not any(_is_doc(c) for c in chunks):
                try:
                    for dc in self.retriever.retrieve(query=f"README architecture overview installation {cleaned_query}", top_k=2):
                        if _is_doc(dc):
                            cit = f"{dc.get('metadata', {}).get('file')}:{dc.get('metadata', {}).get('start_line')}"
                            if cit not in seen:
                                seen.add(cit)
                                chunks.append(dc)
                except Exception:
                    pass
            if not any(_is_code(c) for c in chunks):
                try:
                    for cc in self.retriever.retrieve(query=f"main entrypoint implementation {cleaned_query}", top_k=2):
                        if _is_code(cc):
                            cit = f"{cc.get('metadata', {}).get('file')}:{cc.get('metadata', {}).get('start_line')}"
                            if cit not in seen:
                                seen.add(cit)
                                chunks.append(cc)
                except Exception:
                    pass

        evidence = extract_evidence(chunks)

        if not chunks:
            return {
                "question": cleaned_query,
                "answer": "The provided repository context is insufficient to answer this question.",
                "evidence": [],
                "retrieved_chunks": [],
            }

        context = build_rag_context(chunks)
        user_prompt = self.build_user_prompt(cleaned_query, context)
        answer = self.llm_client.generate(self.system_prompt, user_prompt)

        return {
            "question": cleaned_query,
            "answer": answer,
            "evidence": evidence,
            "retrieved_chunks": chunks,
        }

"""Orchestration service for RAG retrieval, context building, and grounded answer generation."""

from __future__ import annotations

from typing import Any

from app.rag.context import build_rag_context, extract_evidence, is_documentation_chunk
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
    "6. Ground your technical explanation primarily in the provided source code chunks. Use the README chunks for architecture and high-level context, but always explain how it is implemented in the actual source code.\n"
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
        """Retrieve separated code and README chunks and generate a grounded answer."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        cleaned_query = query.strip()
        if (
            hasattr(self.retriever, "retrieve_separated")
            and getattr(self.retriever, "embedding_service", None) is not None
        ):
            sep = self.retriever.retrieve_separated(query=cleaned_query, top_k=top_k)
            code_chunks = sep.get("code_chunks", [])
            readme_chunks = sep.get("readme_chunks", [])
            chunks = sep.get("all_chunks", [])
        else:
            chunks = list(self.retriever.retrieve(query=cleaned_query, top_k=top_k))
            code_chunks = [c for c in chunks if not is_documentation_chunk(c)]
            readme_chunks = [c for c in chunks if is_documentation_chunk(c)]

        evidence = extract_evidence(chunks)
        code_evidence = [e for e in evidence if e.get("category") == "code"]
        readme_evidence = [e for e in evidence if e.get("category") == "readme"]

        if not chunks:
            return {
                "question": cleaned_query,
                "answer": "The provided repository context is insufficient to answer this question.",
                "evidence": [],
                "code_evidence": [],
                "readme_evidence": [],
                "code_chunks": [],
                "readme_chunks": [],
                "retrieved_chunks": [],
            }

        context = build_rag_context(chunks)
        user_prompt = self.build_user_prompt(cleaned_query, context)
        answer = self.llm_client.generate(self.system_prompt, user_prompt)

        return {
            "question": cleaned_query,
            "answer": answer,
            "evidence": evidence,
            "code_evidence": code_evidence,
            "readme_evidence": readme_evidence,
            "code_chunks": code_chunks,
            "readme_chunks": readme_chunks,
            "retrieved_chunks": chunks,
        }

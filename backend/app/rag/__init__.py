"""RAG package for RepoPilot."""

from __future__ import annotations

from functools import lru_cache

from app.embeddings import EmbeddingService
from app.ingestion.repository import embedding_service, get_vector_store
from app.rag.context import (
    build_rag_context,
    extract_evidence,
    format_citation,
    format_lines_range,
    format_symbol_name,
)
from app.rag.llm import (
    FakeLLMClient,
    HttpLLMClient,
    LLMClient,
    TemplateGroundedClient,
    get_default_llm_client,
)
from app.rag.retriever import Retriever
from app.rag.service import DEFAULT_SYSTEM_PROMPT, RAGService
from app.vectorstore import QdrantStore


@lru_cache
def get_retriever() -> Retriever:
    """Return the application-wide retriever instance."""
    return Retriever(embedding_service=embedding_service, vector_store=get_vector_store())


@lru_cache
def get_rag_service() -> RAGService:
    """Return the application-wide RAG service instance."""
    return RAGService(retriever=get_retriever(), llm_client=get_default_llm_client())


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "FakeLLMClient",
    "HttpLLMClient",
    "LLMClient",
    "RAGService",
    "Retriever",
    "TemplateGroundedClient",
    "build_rag_context",
    "extract_evidence",
    "format_citation",
    "format_lines_range",
    "format_symbol_name",
    "get_default_llm_client",
    "get_rag_service",
    "get_retriever",
]

"""Semantic retrieval over Qdrant using BGE-M3 query embeddings."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.rag.context import is_documentation_chunk

if TYPE_CHECKING:
    from app.embeddings import EmbeddingService
    from app.vectorstore import QdrantStore


class Retriever:
    """Retrieve top-K relevant repository chunks for natural-language queries."""

    def __init__(self, embedding_service: EmbeddingService, vector_store: QdrantStore) -> None:
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def retrieve_separated(
        self,
        query: str,
        top_k: int = 5,
        code_k: int = 3,
        doc_k: int = 2,
    ) -> dict[str, list[dict[str, Any]]]:
        """Separately retrieve code implementation chunks and README documentation chunks.

        This guarantees that natural-language documentation chunks do not crowd out
        technical source code chunks in the embedding vector space.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        if self.embedding_service is None:
            return {
                "code_chunks": [],
                "readme_chunks": [],
                "all_chunks": [],
            }

        query_vector = self.embedding_service.embed_query(query.strip())
        candidate_pool = self.vector_store.search(
            query_vector=query_vector,
            limit=max(top_k * 4, 30),
        )

        code_candidates: list[dict[str, Any]] = []
        readme_candidates: list[dict[str, Any]] = []

        for chunk in candidate_pool:
            if is_documentation_chunk(chunk):
                readme_candidates.append(chunk)
            else:
                code_candidates.append(chunk)

        # Target proportions: prioritize code implementation, cap README to at most doc_k
        selected_code = code_candidates[: max(code_k, top_k - doc_k)]
        selected_readme = readme_candidates[:doc_k]

        # If one category is empty (e.g. no README or doc-only repo), backfill from other
        if not selected_code and readme_candidates:
            selected_readme = readme_candidates[:top_k]
        elif not selected_readme and code_candidates:
            selected_code = code_candidates[:top_k]

        all_chunks = [*selected_code, *selected_readme]
        return {
            "code_chunks": selected_code,
            "readme_chunks": selected_readme,
            "all_chunks": all_chunks,
        }

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        query_filter: Any | None = None,
    ) -> list[dict[str, Any]]:
        """Embed the query and return a balanced set of code and documentation chunks."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        if query_filter is not None:
            query_vector = self.embedding_service.embed_query(query.strip())
            return self.vector_store.search(
                query_vector=query_vector,
                limit=top_k,
                query_filter=query_filter,
            )

        separated = self.retrieve_separated(query=query, top_k=top_k)
        return separated["all_chunks"]

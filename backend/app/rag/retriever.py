"""Semantic retrieval over Qdrant using BGE-M3 query embeddings."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.embeddings import EmbeddingService
    from app.vectorstore import QdrantStore


class Retriever:
    """Retrieve top-K relevant repository chunks for natural-language queries."""

    def __init__(self, embedding_service: EmbeddingService, vector_store: QdrantStore) -> None:
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        query_filter: Any | None = None,
    ) -> list[dict[str, Any]]:
        """Embed the query with BGE-M3 and return the nearest chunks from Qdrant."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        query_vector = self.embedding_service.embed_query(query.strip())
        return self.vector_store.search(
            query_vector=query_vector,
            limit=top_k,
            query_filter=query_filter,
        )

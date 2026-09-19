"""Generate BGE-M3 embeddings for code-aware chunks."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from typing import Any


MODEL_NAME = "BAAI/bge-m3"


class EmbeddingService:
    """Lazily load BGE-M3 and embed Day 2 chunks in batches.

    A service instance caches its model, so callers should create one instance per
    application process rather than creating one per chunk or request.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        batch_size: int = 16,
        model_factory: Callable[[str], Any] | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.model_name = model_name
        self.batch_size = batch_size
        self._model_factory = model_factory or self._default_model_factory
        self._model: Any | None = None

    @staticmethod
    def _default_model_factory(model_name: str) -> Any:
        """Import lazily so code paths that do not embed need no model startup."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "sentence-transformers is required for embeddings. "
                "Install backend/requirements.txt first."
            ) from error
        return SentenceTransformer(model_name)

    @property
    def model(self) -> Any:
        """Return the cached embedding model, loading it only on first use."""
        if self._model is None:
            self._model = self._model_factory(self.model_name)
        return self._model

    @staticmethod
    def build_embedding_input(content: str, metadata: Mapping[str, Any]) -> str:
        """Add repository context without modifying the original source content."""
        context = [
            f"Language: {metadata.get('language', 'Unknown')}",
            f"File: {metadata.get('file', 'Unknown')}",
            f"Symbol type: {metadata.get('type', 'Unknown')}",
            f"Class: {metadata.get('class_name', 'None')}",
            f"Symbol name: {metadata.get('name', 'Unknown')}",
        ]
        return "\n".join([*context, "Source code:", content])

    @staticmethod
    def _validate_chunk(chunk: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
        if not isinstance(chunk, Mapping):
            raise ValueError("Each chunk must be a mapping with content and metadata.")
        content = chunk.get("content")
        metadata = chunk.get("metadata")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Each chunk must have non-empty string content.")
        if not isinstance(metadata, Mapping):
            raise ValueError("Each chunk must have a metadata mapping.")
        return content, metadata

    def embed_chunks(self, chunks: Sequence[Mapping[str, Any]]) -> list[dict[str, object]]:
        """Return chunks with normalized dense embeddings and preserved metadata.

        An empty collection needs no model call and returns an empty collection.
        Invalid individual chunks raise ``ValueError`` before embedding begins.
        """
        if not chunks:
            return []

        validated = [self._validate_chunk(chunk) for chunk in chunks]
        inputs = [self.build_embedding_input(content, metadata) for content, metadata in validated]
        vectors = self.model.encode(
            inputs,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        if len(vectors) != len(validated):
            raise RuntimeError("Embedding model returned a different number of vectors than chunks.")

        embedded_chunks: list[dict[str, object]] = []
        for (content, metadata), vector in zip(validated, vectors, strict=True):
            embedded_chunks.append(
                {
                    "content": content,
                    "metadata": deepcopy(dict(metadata)),
                    "embedding": vector.tolist(),
                }
            )
        return embedded_chunks

"""Local Qdrant storage for Day 3 embedded repository chunks."""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models


DEFAULT_COLLECTION_NAME = "repository_chunks"


class QdrantStore:
    """Store embedded chunks and retrieve their nearest semantic matches.

    The store deliberately accepts vectors only. BGE-M3 embedding remains the
    responsibility of :class:`app.embeddings.EmbeddingService`.
    """

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        path: str | Path = ":memory:",
        client: QdrantClient | None = None,
    ) -> None:
        self.collection_name = collection_name
        if client is not None:
            self.client = client
        else:
            if path != ":memory:":
                Path(path).mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(path))

    def _collection_vector_size(self) -> int | None:
        """Read the existing collection's dimension, if it has been created."""
        if not self.client.collection_exists(self.collection_name):
            return None
        vectors = self.client.get_collection(self.collection_name).config.params.vectors
        size = getattr(vectors, "size", None)
        if size is None:
            raise RuntimeError("The repository collection does not use one unnamed vector.")
        return int(size)

    def initialize_collection(self, vector_size: int) -> None:
        """Create the collection once, or validate its already-established size."""
        if vector_size < 1:
            raise ValueError("Vector size must be at least 1.")

        existing_size = self._collection_vector_size()
        if existing_size is None:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
            )
        elif existing_size != vector_size:
            raise ValueError(
                f"Collection expects {existing_size}-dimensional vectors, "
                f"but received {vector_size}-dimensional vectors."
            )

    @staticmethod
    def _validate_embedded_chunk(chunk: Mapping[str, Any]) -> tuple[str, Mapping[str, Any], list[float]]:
        if not isinstance(chunk, Mapping):
            raise ValueError("Each embedded chunk must be a mapping.")
        content = chunk.get("content")
        metadata = chunk.get("metadata")
        embedding = chunk.get("embedding")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Each embedded chunk needs non-empty string content.")
        if not isinstance(metadata, Mapping):
            raise ValueError("Each embedded chunk needs metadata.")
        if not isinstance(embedding, Sequence) or isinstance(embedding, (str, bytes)) or not embedding:
            raise ValueError("Each embedded chunk needs a non-empty embedding vector.")
        try:
            vector = [float(value) for value in embedding]
        except (TypeError, ValueError) as error:
            raise ValueError("Embedding vectors must contain numeric values.") from error
        return content, metadata, vector

    @staticmethod
    def _point_id(content: str, metadata: Mapping[str, Any]) -> str:
        """Create a stable ID, making repeated ingestion of a chunk an upsert."""
        identity = json.dumps(
            {"content": content, "metadata": dict(metadata)},
            sort_keys=True,
            default=str,
        )
        return str(uuid.uuid5(uuid.NAMESPACE_URL, identity))

    def upsert_chunks(self, chunks: Sequence[Mapping[str, Any]]) -> list[str]:
        """Persist Day 3 output and return the point IDs written to Qdrant."""
        if not chunks:
            return []

        validated = [self._validate_embedded_chunk(chunk) for chunk in chunks]
        vector_size = len(validated[0][2])
        self.initialize_collection(vector_size)

        points: list[models.PointStruct] = []
        for content, metadata, vector in validated:
            if len(vector) != vector_size:
                raise ValueError("All embeddings in one upsert must have the same dimension.")
            points.append(
                models.PointStruct(
                    id=self._point_id(content, metadata),
                    vector=vector,
                    payload={"content": content, "metadata": deepcopy(dict(metadata))},
                )
            )

        self.client.upsert(collection_name=self.collection_name, points=points, wait=True)
        return [str(point.id) for point in points]

    def search(
        self,
        query_vector: Sequence[float],
        limit: int = 5,
        query_filter: models.Filter | None = None,
    ) -> list[dict[str, object]]:
        """Return top matching code chunks for a precomputed query vector.

        ``query_filter`` is intentionally exposed for future metadata filters,
        such as language or file path, without coupling filtering to embeddings.
        """
        if limit < 1:
            raise ValueError("Search limit must be at least 1.")
        if not self.client.collection_exists(self.collection_name):
            return []

        vector = [float(value) for value in query_vector]
        expected_size = self._collection_vector_size()
        if len(vector) != expected_size:
            raise ValueError(f"Query vector must have dimension {expected_size}.")

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return [
            {
                "id": str(point.id),
                "score": point.score,
                "content": point.payload["content"],
                "metadata": point.payload["metadata"],
            }
            for point in response.points
        ]

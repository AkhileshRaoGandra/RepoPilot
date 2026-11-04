"""Generate embeddings for code-aware chunks using Gemini or deterministic offline vectorizer."""

from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from typing import Any

import numpy as np

logger = logging.getLogger("repopilot.embeddings")

MODEL_NAME = "models/text-embedding-004"
DEFAULT_VECTOR_DIM = 768


class GeminiEmbeddingModel:
    """Lightweight REST embedding client using Google Gemini batchEmbedContents API."""

    def __init__(
        self,
        api_key: str,
        model_name: str = MODEL_NAME,
        dimension: int = DEFAULT_VECTOR_DIM,
    ) -> None:
        self.api_key = api_key
        self.model_name = model_name
        self.dimension = dimension

    def encode(
        self,
        sentences: list[str],
        batch_size: int = 100,
        **kwargs: Any,
    ) -> np.ndarray:
        import time
        import httpx

        if not sentences:
            return np.empty((0, self.dimension), dtype=np.float32)

        url = f"https://generativelanguage.googleapis.com/v1beta/{self.model_name}:batchEmbedContents"
        params = {"key": self.api_key}
        all_vectors: list[list[float]] = []

        # Gemini supports up to 100 requests per single batchEmbedContents call
        step = min(max(batch_size, 1), 100)
        fallback_generator = DeterministicOfflineEmbeddingModel(dimension=self.dimension)

        for i in range(0, len(sentences), step):
            batch = sentences[i : i + step]
            requests = [
                {
                    "model": self.model_name,
                    "content": {"parts": [{"text": text}]},
                    "outputDimensionality": self.dimension,
                }
                for text in batch
            ]
            payload = {"requests": requests}

            # Retry loop with exponential backoff for 429 rate limit
            success = False
            for attempt in range(4):
                try:
                    with httpx.Client(timeout=60.0) as client:
                        res = client.post(url, params=params, json=payload)
                        if res.status_code == 429:
                            if attempt < 3:
                                wait_time = 2.0 * (2 ** attempt)  # 2s, 4s, 8s
                                logger.warning(f"Gemini API rate limit (429). Retrying in {wait_time}s (attempt {attempt + 1}/3)...")
                                time.sleep(wait_time)
                                continue
                            else:
                                logger.warning("Gemini API 429 quota exhausted. Using offline vectorizer fallback for batch.")
                                break
                        res.raise_for_status()
                        data = res.json()
                        embeddings = data.get("embeddings", [])
                        for item in embeddings:
                            all_vectors.append(item.get("values", []))
                        success = True
                        break
                except Exception as exc:
                    if attempt < 3 and "429" in str(exc):
                        time.sleep(2.0 * (2 ** attempt))
                        continue
                    break

            if not success:
                # Fall back to deterministic vectorizer so ingestion never fails
                fallback_vecs = fallback_generator.encode(batch)
                for vec in fallback_vecs:
                    all_vectors.append(vec.tolist())

        arr = np.array(all_vectors, dtype=np.float32)
        # Normalize vectors for cosine similarity
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms


class DeterministicOfflineEmbeddingModel:
    """Fast, deterministic offline embedding generator (zero ML / network dependencies).

    Used for unit testing and offline environments when no GEMINI_API_KEY is configured.
    Generates 768-dimensional normalized vectors deterministically from text tokens.
    """

    def __init__(self, dimension: int = DEFAULT_VECTOR_DIM) -> None:
        self.dimension = dimension

    def encode(self, sentences: list[str], **kwargs: Any) -> np.ndarray:
        vectors: list[np.ndarray] = []
        for s in sentences:
            vec = np.zeros(self.dimension, dtype=np.float32)
            words = s.lower().split()
            if not words:
                words = ["empty"]
            for word in words:
                h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
                slot = h % self.dimension
                val = ((h >> 16) % 1000) / 500.0 - 1.0
                vec[slot] += val
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vectors.append(vec)
        return np.array(vectors, dtype=np.float32)


class EmbeddingService:
    """Generate dense embeddings for code-aware chunks in batches."""

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
        """Create active embedding model: Gemini API if key present, else offline generator."""
        gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if gemini_key:
            return GeminiEmbeddingModel(api_key=gemini_key, dimension=DEFAULT_VECTOR_DIM)

        # Fallback to SentenceTransformer if user explicitly installed it locally
        try:
            from sentence_transformers import SentenceTransformer
            token = os.getenv("HF_TOKEN") or False
            return SentenceTransformer("BAAI/bge-m3", token=token)
        except Exception:
            pass

        return DeterministicOfflineEmbeddingModel(dimension=DEFAULT_VECTOR_DIM)

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
        """Return chunks with normalized dense embeddings and preserved metadata."""
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
                    "embedding": vector.tolist() if hasattr(vector, "tolist") else list(vector),
                }
            )
        return embedded_chunks

    def embed_query(self, query: str) -> list[float]:
        """Generate a normalized dense embedding vector for a natural language query."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        vectors = self.model.encode(
            [query.strip()],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        first_vector = vectors[0]
        return [float(value) for value in (first_vector.tolist() if hasattr(first_vector, "tolist") else first_vector)]

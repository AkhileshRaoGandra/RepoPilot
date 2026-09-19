from __future__ import annotations

import numpy as np
import pytest

from app.embeddings import EmbeddingService


class FakeEmbeddingModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def encode(self, sentences: list[str], **kwargs: object) -> np.ndarray:
        self.calls.append({"sentences": sentences, **kwargs})
        return np.array([[float(index), 1.0, 2.0] for index, _ in enumerate(sentences)])


@pytest.fixture
def fake_model() -> FakeEmbeddingModel:
    return FakeEmbeddingModel()


@pytest.fixture
def service(fake_model: FakeEmbeddingModel) -> EmbeddingService:
    return EmbeddingService(model_factory=lambda _: fake_model, batch_size=2)


@pytest.fixture
def chunk() -> dict[str, object]:
    return {
        "content": "def login(token: str):\n    return token\n",
        "metadata": {
            "file": "app/auth.py",
            "language": "Python",
            "type": "method",
            "class_name": "AuthService",
            "name": "login",
            "start_line": 5,
            "end_line": 6,
        },
    }


def test_embedding_generation_includes_code_context(service: EmbeddingService, fake_model: FakeEmbeddingModel, chunk: dict[str, object]) -> None:
    result = service.embed_chunks([chunk])

    assert result[0]["embedding"] == [0.0, 1.0, 2.0]
    model_input = fake_model.calls[0]["sentences"][0]
    assert "Language: Python" in model_input
    assert "Class: AuthService" in model_input
    assert "def login(token: str):" in model_input


def test_embedding_shape(service: EmbeddingService, chunk: dict[str, object]) -> None:
    result = service.embed_chunks([chunk])

    assert len(result) == 1
    assert len(result[0]["embedding"]) == 3


def test_multiple_chunks_are_embedded_together(service: EmbeddingService, fake_model: FakeEmbeddingModel, chunk: dict[str, object]) -> None:
    second_chunk = {**chunk, "content": "def logout():\n    return None\n"}
    result = service.embed_chunks([chunk, second_chunk])

    assert len(result) == 2
    assert len(fake_model.calls) == 1
    assert fake_model.calls[0]["batch_size"] == 2


def test_empty_chunks_do_not_load_model(service: EmbeddingService, fake_model: FakeEmbeddingModel) -> None:
    assert service.embed_chunks([]) == []
    assert fake_model.calls == []


def test_invalid_chunk_is_rejected(service: EmbeddingService) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        service.embed_chunks([{"content": "   ", "metadata": {}}])


def test_metadata_is_preserved_without_sharing_mutable_state(service: EmbeddingService, chunk: dict[str, object]) -> None:
    result = service.embed_chunks([chunk])

    assert result[0]["content"] == chunk["content"]
    assert result[0]["metadata"] == chunk["metadata"]
    assert result[0]["metadata"] is not chunk["metadata"]

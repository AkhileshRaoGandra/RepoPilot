from __future__ import annotations

import pytest
from qdrant_client import QdrantClient

from app.vectorstore import QdrantStore


AUTH_CHUNK = {
    "content": "def login(token):\n    return validate_token(token)\n",
    "metadata": {
        "file": "auth/service.py",
        "language": "Python",
        "type": "method",
        "class_name": "AuthService",
        "name": "login",
        "start_line": 25,
        "end_line": 26,
    },
    "embedding": [1.0, 0.0, 0.0],
}
DATABASE_CHUNK = {
    "content": "def connect_database():\n    return Database.connect()\n",
    "metadata": {"file": "database/connection.py", "language": "Python", "type": "function", "name": "connect_database", "start_line": 1, "end_line": 2},
    "embedding": [0.0, 1.0, 0.0],
}
LOGGING_CHUNK = {
    "content": "def log_request():\n    logger.info('request')\n",
    "metadata": {"file": "logging/events.py", "language": "Python", "type": "function", "name": "log_request", "start_line": 1, "end_line": 2},
    "embedding": [0.0, 0.0, 1.0],
}


@pytest.fixture
def store() -> QdrantStore:
    return QdrantStore(collection_name="test_repository_chunks", client=QdrantClient(path=":memory:"))


def test_upsert_creates_collection_with_embedding_dimension(store: QdrantStore) -> None:
    store.upsert_chunks([AUTH_CHUNK])

    info = store.client.get_collection(store.collection_name)
    assert info.config.params.vectors.size == 3


def test_upsert_preserves_content_and_metadata(store: QdrantStore) -> None:
    store.upsert_chunks([AUTH_CHUNK])
    result = store.search([1.0, 0.0, 0.0])[0]

    assert result["content"] == AUTH_CHUNK["content"]
    assert result["metadata"] == AUTH_CHUNK["metadata"]


def test_search_returns_nearest_chunks_and_honors_top_k(store: QdrantStore) -> None:
    store.upsert_chunks([AUTH_CHUNK, DATABASE_CHUNK, LOGGING_CHUNK])
    results = store.search([0.95, 0.05, 0.0], limit=2)

    assert len(results) == 2
    assert results[0]["metadata"]["name"] == "login"
    assert results[0]["score"] > results[1]["score"]


def test_empty_collection_returns_no_matches(store: QdrantStore) -> None:
    assert store.search([1.0, 0.0, 0.0]) == []


def test_vector_dimension_is_validated(store: QdrantStore) -> None:
    store.upsert_chunks([AUTH_CHUNK])

    with pytest.raises(ValueError, match="3-dimensional"):
        store.upsert_chunks([{**DATABASE_CHUNK, "embedding": [0.0, 1.0]}])
    with pytest.raises(ValueError, match="dimension 3"):
        store.search([1.0, 0.0])


def test_end_to_end_deterministic_retrieval(store: QdrantStore) -> None:
    store.upsert_chunks([AUTH_CHUNK, DATABASE_CHUNK, LOGGING_CHUNK])

    results = store.search([0.99, 0.01, 0.0], limit=1)
    assert len(results) == 1
    assert results[0]["content"] == AUTH_CHUNK["content"]
    assert results[0]["metadata"]["file"] == "auth/service.py"

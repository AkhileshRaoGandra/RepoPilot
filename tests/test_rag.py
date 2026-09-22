"""Unit and pipeline tests for Day 5 RAG + Grounded Answers."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from app.embeddings import EmbeddingService
from app.main import app
from app.rag import (
    DEFAULT_SYSTEM_PROMPT,
    FakeLLMClient,
    GeminiLLMClient,
    RAGService,
    Retriever,
    TemplateGroundedClient,
    build_rag_context,
    extract_evidence,
    format_citation,
    format_lines_range,
    format_symbol_name,
    get_default_llm_client,
)
from app.vectorstore import QdrantStore


class FakeEmbeddingModel:
    """Deterministic embedding model producing 3D unit vectors for testing."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def encode(self, sentences: list[str], **kwargs: object) -> np.ndarray:
        self.calls.append({"sentences": sentences, **kwargs})
        # If the query contains 'auth' or 'login', return vector close to [1, 0, 0]
        vectors = []
        for s in sentences:
            text = s.lower()
            if "auth" in text or "login" in text:
                vectors.append([0.95, 0.05, 0.0])
            elif "database" in text or "connect" in text:
                vectors.append([0.0, 0.95, 0.05])
            else:
                vectors.append([0.05, 0.05, 0.9])
        return np.array(vectors)


AUTH_CHUNK = {
    "content": "def login(token: str):\n    return validate_token(token)\n",
    "metadata": {
        "file": "auth/service.py",
        "language": "Python",
        "type": "method",
        "class_name": "AuthService",
        "name": "login",
        "start_line": 25,
        "end_line": 48,
    },
    "embedding": [1.0, 0.0, 0.0],
}

TOKEN_CHUNK = {
    "content": "def generate_token(user_id: str):\n    return jwt_encode(user_id)\n",
    "metadata": {
        "file": "auth/token.py",
        "language": "Python",
        "type": "function",
        "name": "generate_token",
        "start_line": 10,
        "end_line": 30,
    },
    "embedding": [0.9, 0.1, 0.0],
}

DB_CHUNK = {
    "content": "def connect_db():\n    return Database()\n",
    "metadata": {
        "file": "db/conn.py",
        "language": "Python",
        "type": "function",
        "name": "connect_db",
        "start_line": 1,
        "end_line": 15,
    },
    "embedding": [0.0, 1.0, 0.0],
}


@pytest.fixture
def fake_model() -> FakeEmbeddingModel:
    return FakeEmbeddingModel()


@pytest.fixture
def embedding_service(fake_model: FakeEmbeddingModel) -> EmbeddingService:
    return EmbeddingService(model_factory=lambda _: fake_model)


@pytest.fixture
def vector_store() -> QdrantStore:
    store = QdrantStore(collection_name="test_rag_collection", client=QdrantClient(path=":memory:"))
    store.upsert_chunks([AUTH_CHUNK, TOKEN_CHUNK, DB_CHUNK])
    return store


@pytest.fixture
def empty_store() -> QdrantStore:
    return QdrantStore(collection_name="test_empty_rag", client=QdrantClient(path=":memory:"))


@pytest.fixture
def retriever(embedding_service: EmbeddingService, vector_store: QdrantStore) -> Retriever:
    return Retriever(embedding_service=embedding_service, vector_store=vector_store)


@pytest.fixture
def fake_llm() -> FakeLLMClient:
    return FakeLLMClient(
        responses=(
            "Authentication is handled by AuthService.login() in auth/service.py:25-48. "
            "Tokens are generated via generate_token() in auth/token.py:10-30."
        )
    )


# 1. Query Embedding Tests
def test_query_embedding_uses_same_model(embedding_service: EmbeddingService, fake_model: FakeEmbeddingModel) -> None:
    vector = embedding_service.embed_query("How does login work?")
    assert len(vector) == 3
    assert len(fake_model.calls) == 1
    assert fake_model.calls[0]["sentences"] == ["How does login work?"]


def test_query_embedding_rejects_empty_query(embedding_service: EmbeddingService) -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        embedding_service.embed_query("")
    with pytest.raises(ValueError, match="non-empty string"):
        embedding_service.embed_query("   ")


# 2. Retrieval Tests
def test_retriever_returns_relevant_chunks(retriever: Retriever) -> None:
    results = retriever.retrieve("How does authentication login work?", top_k=2)
    assert len(results) == 2
    assert results[0]["metadata"]["name"] == "login"
    assert results[0]["metadata"]["file"] == "auth/service.py"
    assert results[0]["score"] >= results[1]["score"]


def test_retriever_honors_top_k(retriever: Retriever) -> None:
    results = retriever.retrieve("login", top_k=1)
    assert len(results) == 1
    assert results[0]["metadata"]["name"] == "login"


def test_retriever_validates_inputs(retriever: Retriever) -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        retriever.retrieve("")
    with pytest.raises(ValueError, match="at least 1"):
        retriever.retrieve("auth", top_k=0)


# 3. Context Construction Tests
def test_context_construction_formats_metadata_and_content() -> None:
    chunks = [AUTH_CHUNK, TOKEN_CHUNK]
    context = build_rag_context(chunks)

    assert "File: auth/service.py" in context
    assert "Symbol: AuthService.login()" in context
    assert "Lines: 25-48" in context
    assert "def login(token: str):" in context

    assert "File: auth/token.py" in context
    assert "Symbol: generate_token()" in context
    assert "Lines: 10-30" in context
    assert "def generate_token(user_id: str):" in context


def test_context_construction_empty_chunks() -> None:
    assert build_rag_context([]) == ""


# 4. LLM Prompt Construction Tests
def test_user_prompt_construction() -> None:
    user_prompt = RAGService.build_user_prompt("Where is login?", "File: auth/service.py\ncode...")
    assert "Repository Context:" in user_prompt
    assert "File: auth/service.py" in user_prompt
    assert "Question: Where is login?" in user_prompt


def test_system_prompt_enforces_grounding() -> None:
    assert "strictly using the provided repository context" in DEFAULT_SYSTEM_PROMPT
    assert "Do not invent" in DEFAULT_SYSTEM_PROMPT
    assert "The provided repository context is insufficient" in DEFAULT_SYSTEM_PROMPT


# 5. Empty Retrieval Tests
def test_empty_retrieval_returns_no_chunks(embedding_service: EmbeddingService, empty_store: QdrantStore) -> None:
    retriever = Retriever(embedding_service=embedding_service, vector_store=empty_store)
    results = retriever.retrieve("any query")
    assert results == []


def test_empty_retrieval_returns_grounded_fallback_without_calling_llm(
    embedding_service: EmbeddingService, empty_store: QdrantStore, fake_llm: FakeLLMClient
) -> None:
    retriever = Retriever(embedding_service=embedding_service, vector_store=empty_store)
    rag_service = RAGService(retriever=retriever, llm_client=fake_llm)

    result = rag_service.answer_question("Where is auth?")
    assert result["answer"] == "The provided repository context is insufficient to answer this question."
    assert result["evidence"] == []
    assert result["retrieved_chunks"] == []
    assert len(fake_llm.calls) == 0  # LLM was not called unnecessarily


# 6. Missing/Insufficient Context Tests
def test_template_client_reports_insufficient_context() -> None:
    client = TemplateGroundedClient()
    response = client.generate(DEFAULT_SYSTEM_PROMPT, "Repository Context:\nNo relevant repository context found.\n")
    assert "insufficient" in response.lower()


def test_get_default_llm_client_selects_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    client = get_default_llm_client()
    assert isinstance(client, GeminiLLMClient)
    assert client.api_key == "test-gemini-key"
    assert client.model == "gemini-flash-lite-latest"


def test_gemini_llm_client_generate(monkeypatch: pytest.MonkeyPatch) -> None:
    class MockResponse:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": "Gemini generated answer based on context."}]
                        }
                    }
                ]
            }

    class MockHttpClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def __enter__(self) -> MockHttpClient:
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def post(self, url: str, **kwargs: object) -> MockResponse:
            assert "generativelanguage.googleapis.com" in url
            return MockResponse()

    import httpx
    monkeypatch.setattr(httpx, "Client", MockHttpClient)

    client = GeminiLLMClient(api_key="test-key", model="gemini-1.5-flash")
    answer = client.generate("system prompt", "user prompt")
    assert answer == "Gemini generated answer based on context."


# 7. Source Metadata Preservation & Evidence Formatting Tests
def test_extract_evidence_preserves_metadata() -> None:
    chunks = [
        {
            "content": AUTH_CHUNK["content"],
            "metadata": AUTH_CHUNK["metadata"],
            "score": 0.98,
        }
    ]
    evidence = extract_evidence(chunks)
    assert len(evidence) == 1
    item = evidence[0]
    assert item["file"] == "auth/service.py"
    assert item["symbol"] == "AuthService.login()"
    assert item["lines"] == "25-48"
    assert item["citation"] == "auth/service.py:25-48"
    assert item["score"] == 0.98


def test_format_helpers() -> None:
    assert format_symbol_name({"name": "bar", "type": "function"}) == "bar()"
    assert format_symbol_name({"name": "bar", "type": "method", "class_name": "Foo"}) == "Foo.bar()"
    assert format_lines_range({"start_line": 5, "end_line": 10}) == "5-10"
    assert format_lines_range({"start_line": 5, "end_line": 5}) == "5"
    assert format_citation({"file": "a.py", "start_line": 1, "end_line": 10}) == "a.py:1-10"


# 8. End-to-End RAG Pipeline Tests
def test_end_to_end_rag_service(retriever: Retriever, fake_llm: FakeLLMClient) -> None:
    rag_service = RAGService(retriever=retriever, llm_client=fake_llm)
    result = rag_service.answer_question("How does authentication work?", top_k=2)

    assert result["question"] == "How does authentication work?"
    assert "AuthService.login()" in result["answer"]
    assert len(result["evidence"]) == 2
    assert result["evidence"][0]["citation"] == "auth/service.py:25-48"
    assert result["evidence"][1]["citation"] == "auth/token.py:10-30"
    assert len(fake_llm.calls) == 1
    assert "strictly using the provided repository context" in fake_llm.calls[0]["system_prompt"]
    assert "auth/service.py" in fake_llm.calls[0]["user_prompt"]


# 9. FastAPI Integration Tests
def test_api_query_and_retrieve_endpoints(monkeypatch: pytest.MonkeyPatch, retriever: Retriever, fake_llm: FakeLLMClient) -> None:
    client = TestClient(app)
    rag_service = RAGService(retriever=retriever, llm_client=fake_llm)

    monkeypatch.setattr("app.main.get_retriever", lambda: retriever)
    monkeypatch.setattr("app.main.get_rag_service", lambda: rag_service)

    # Test /retrieve endpoint
    retrieve_resp = client.post("/retrieve", json={"query": "login", "top_k": 2})
    assert retrieve_resp.status_code == 200
    retrieve_data = retrieve_resp.json()
    assert retrieve_data["query"] == "login"
    assert len(retrieve_data["chunks"]) == 2
    assert retrieve_data["chunks"][0]["metadata"]["file"] == "auth/service.py"

    # Test /query endpoint
    query_resp = client.post("/query", json={"query": "How does authentication work?", "top_k": 2})
    assert query_resp.status_code == 200
    query_data = query_resp.json()
    assert query_data["question"] == "How does authentication work?"
    assert "AuthService.login()" in query_data["answer"]
    assert len(query_data["evidence"]) == 2
    assert query_data["evidence"][0]["citation"] == "auth/service.py:25-48"

    # Test validation
    invalid_resp = client.post("/query", json={"query": "", "top_k": 2})
    assert invalid_resp.status_code == 422

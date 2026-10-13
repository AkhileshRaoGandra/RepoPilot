"""Integration tests verifying frontend API contracts, CORS headers, and response schemas."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.agent import AgentService, DependencyGraphTool, SourceInspectionTool
from app.graph import build_graph
from app.main import app
from app.rag import RAGService
from app.rag.llm import FakeLLMClient
from app.rag.retriever import Retriever
from app.vectorstore import QdrantStore
from qdrant_client import QdrantClient


SAMPLE_CHUNK = {
    "content": "class AuthService:\n    def login(self, token):\n        return token\n",
    "metadata": {
        "file": "auth/service.py",
        "language": "Python",
        "type": "class",
        "name": "AuthService",
        "start_line": 1,
        "end_line": 5,
    },
    "embedding": [0.95, 0.05, 0.0],
}


@pytest.fixture
def api_client() -> TestClient:
    return TestClient(app)


def test_cors_headers_for_frontend_origin(api_client: TestClient) -> None:
    """Verify that requests from Vite frontend origins (localhost:5173, 127.0.0.1:5173) receive CORS headers."""
    response = api_client.options(
        "/query",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_cors_headers_127_0_0_1(api_client: TestClient) -> None:
    """Verify CORS for 127.0.0.1:5173."""
    response = api_client.options(
        "/agent/query",
        headers={
            "Origin": "http://127.0.0.1:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://127.0.0.1:5173"


def test_rag_query_contract(monkeypatch: pytest.MonkeyPatch, api_client: TestClient) -> None:
    """Verify the POST /query contract matches frontend expectations."""
    store = QdrantStore(collection_name="test_fe_contract", client=QdrantClient(path=":memory:"))
    store.upsert_chunks([SAMPLE_CHUNK])
    fake_retriever = Retriever(embedding_service=None, vector_store=store)  # type: ignore[arg-type]
    monkeypatch.setattr(fake_retriever, "retrieve", lambda query, top_k=5: [SAMPLE_CHUNK])

    fake_rag = RAGService(retriever=fake_retriever, llm_client=FakeLLMClient(responses="Auth answer."))
    monkeypatch.setattr("app.main.get_rag_service", lambda: fake_rag)

    resp = api_client.post(
        "/query",
        json={"query": "How does auth work?", "top_k": 3},
        headers={"Origin": "http://localhost:5173"},
    )
    assert resp.status_code == 200
    data = resp.json()

    # Required top-level fields
    assert "question" in data
    assert "answer" in data
    assert "evidence" in data
    assert "retrieved_chunks" in data
    assert data["question"] == "How does auth work?"
    assert data["answer"] == "Auth answer."

    # Evidence item contract
    assert len(data["evidence"]) >= 1
    ev = data["evidence"][0]
    assert "file" in ev
    assert "symbol" in ev
    assert "start_line" in ev
    assert "end_line" in ev
    assert "lines" in ev
    assert "citation" in ev
    assert ev["file"] == "auth/service.py"
    assert ev["lines"] == "1-5"


def test_agent_query_contract(monkeypatch: pytest.MonkeyPatch, api_client: TestClient) -> None:
    """Verify the POST /agent/query contract matches frontend expectations."""
    store = QdrantStore(collection_name="test_fe_agent_contract", client=QdrantClient(path=":memory:"))
    store.upsert_chunks([SAMPLE_CHUNK])
    fake_retriever = Retriever(embedding_service=None, vector_store=store)  # type: ignore[arg-type]
    monkeypatch.setattr(fake_retriever, "retrieve", lambda query, top_k=5: [SAMPLE_CHUNK])

    graph = build_graph([], {})
    fake_agent = AgentService(
        retriever=fake_retriever,
        inspection_tool=SourceInspectionTool(),
        graph_tool=DependencyGraphTool(graph=graph),
        llm_client=FakeLLMClient(responses="Agent answer."),
    )
    monkeypatch.setattr("app.main.get_agent_service", lambda: fake_agent)

    resp = api_client.post(
        "/agent/query",
        json={"query": "How does auth work?", "top_k": 5},
        headers={"Origin": "http://localhost:5173"},
    )
    assert resp.status_code == 200
    data = resp.json()

    # Required top-level fields for Agent Mode
    assert "question" in data
    assert "answer" in data
    assert "evidence" in data
    assert "tools_used" in data
    assert "investigation_status" in data
    assert "retrieved_chunks" in data
    assert "source_inspections" in data
    assert "graph_investigations" in data
    assert isinstance(data["tools_used"], list)
    assert isinstance(data["source_inspections"], list)
    assert isinstance(data["graph_investigations"], list)


def test_retrieve_chunks_contract(monkeypatch: pytest.MonkeyPatch, api_client: TestClient) -> None:
    """Verify the POST /retrieve contract matches frontend expectations."""
    store = QdrantStore(collection_name="test_fe_retrieve", client=QdrantClient(path=":memory:"))
    store.upsert_chunks([SAMPLE_CHUNK])
    fake_retriever = Retriever(embedding_service=None, vector_store=store)  # type: ignore[arg-type]
    monkeypatch.setattr(fake_retriever, "retrieve", lambda query, top_k=5: [SAMPLE_CHUNK])
    monkeypatch.setattr("app.main.get_retriever", lambda: fake_retriever)

    resp = api_client.post(
        "/retrieve",
        json={"query": "token", "top_k": 2},
        headers={"Origin": "http://localhost:5173"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["query"] == "token"
    assert "chunks" in data
    assert len(data["chunks"]) == 1
    assert data["chunks"][0]["metadata"]["file"] == "auth/service.py"


def test_query_validation_error_format(api_client: TestClient) -> None:
    """Verify that validation errors return 422 with a structured detail message."""
    resp = api_client.post(
        "/query",
        json={"query": "", "top_k": 5},
        headers={"Origin": "http://localhost:5173"},
    )
    assert resp.status_code == 422
    data = resp.json()
    assert "detail" in data


def test_repository_analyze_validation_error_format(api_client: TestClient) -> None:
    """Verify that invalid GitHub URLs return 422."""
    resp = api_client.post(
        "/repository/analyze",
        json={"github_url": "not-a-valid-url"},
        headers={"Origin": "http://localhost:5173"},
    )
    assert resp.status_code == 422
    data = resp.json()
    assert "detail" in data

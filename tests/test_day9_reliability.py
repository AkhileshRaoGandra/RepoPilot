"""Deterministic unit and integration tests for Day 9: Reliability, Safety, and Answer Quality."""

from __future__ import annotations

from pathlib import Path
import pytest
from fastapi.testclient import TestClient


from app.agent import AgentService, DependencyGraphTool, SourceInspectionTool
from app.code.chunker import create_chunks
from app.code.parser import parse_source
from app.graph import DependencyGraph, build_graph

from app.ingestion.repository import (
    get_active_repository,
    is_repository_indexed,
)
import app.ingestion.repository as repo_module
from app.main import app
from app.rag import RAGService
from app.rag.llm import FakeLLMClient
from app.rag.retriever import Retriever
from app.vectorstore import QdrantStore
from qdrant_client import QdrantClient


@pytest.fixture
def api_client() -> TestClient:
    return TestClient(app)


def test_unindexed_repository_rejection_query(api_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify POST /query returns HTTP 400 when no repository has been indexed."""
    monkeypatch.setattr(repo_module, "_active_repository", None)
    response = api_client.post("/query", json={"query": "What does this repo do?"})
    assert response.status_code == 400
    assert "No repository has been indexed yet" in response.json()["detail"]


def test_unindexed_repository_rejection_retrieve(api_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify POST /retrieve returns HTTP 400 when no repository has been indexed."""
    monkeypatch.setattr(repo_module, "_active_repository", None)
    response = api_client.post("/retrieve", json={"query": "find authentication"})
    assert response.status_code == 400
    assert "No repository has been indexed yet" in response.json()["detail"]


def test_unindexed_repository_rejection_agent_query(api_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify POST /agent/query returns HTTP 400 when no repository has been indexed."""
    monkeypatch.setattr(repo_module, "_active_repository", None)
    response = api_client.post("/agent/query", json={"query": "Trace controller to repository"})
    assert response.status_code == 400
    assert "No repository has been indexed yet" in response.json()["detail"]


def test_repository_status_endpoint_unindexed(api_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify GET /repository/status returns correct schema when unindexed."""
    monkeypatch.setattr(repo_module, "_active_repository", None)
    response = api_client.get("/repository/status")
    assert response.status_code == 200
    data = response.json()
    assert data["indexed"] is False
    assert data["repository"] is None
    assert "chunks_stored" in data
    assert "graph_available" in data


def test_repository_status_endpoint_indexed(api_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify GET /repository/status returns correct schema when indexed."""
    monkeypatch.setattr(repo_module, "_active_repository", "test-repo")
    # Mock store to report chunks > 0
    from unittest.mock import MagicMock
    mock_store = MagicMock()
    mock_store.count.return_value = 10
    monkeypatch.setattr("app.main.get_vector_store", lambda: mock_store)
    response = api_client.get("/repository/status")
    assert response.status_code == 200
    data = response.json()
    assert data["indexed"] is True
    assert data["repository"] == "test-repo"
    assert data["chunks_stored"] == 10


def test_module_fallback_chunking_for_files_without_symbols() -> None:
    """Files with constants or module-level code (no classes/functions) must produce module chunks."""
    script_code = 'API_URL = "https://api.example.com/v1"\nTIMEOUT = 30\nDEBUG = True\n'
    parsed = parse_source(script_code, "python")
    assert len(parsed["classes"]) == 0
    assert len(parsed["functions"]) == 0

    chunks = create_chunks(script_code, "config/settings.py", "python", parsed)
    assert len(chunks) == 1
    assert chunks[0]["metadata"]["type"] == "module"
    assert chunks[0]["metadata"]["name"] == "settings"
    assert chunks[0]["metadata"]["file"] == "config/settings.py"
    assert "API_URL" in chunks[0]["content"]


def test_vector_store_isolation_and_reset() -> None:
    """Vector store reset_collection cleans up previous points to prevent cross-repo contamination."""
    in_memory_client = QdrantClient(":memory:")
    store = QdrantStore(client=in_memory_client, collection_name="test_isolation")

    chunk_repo1 = {
        "content": "class RepoOneAuth: pass",
        "metadata": {"file": "repo1/auth.py", "language": "Python", "type": "class", "name": "RepoOneAuth", "start_line": 1, "end_line": 1},
        "embedding": [1.0, 0.0, 0.0],
    }
    store.upsert_chunks([chunk_repo1])
    assert store.count() == 1

    # Reset collection as done on new repo ingestion
    store.reset_collection()
    assert store.count() == 0
    assert store.is_empty() is True

    chunk_repo2 = {
        "content": "class RepoTwoBilling: pass",
        "metadata": {"file": "repo2/billing.py", "language": "Python", "type": "class", "name": "RepoTwoBilling", "start_line": 1, "end_line": 1},
        "embedding": [0.0, 1.0, 0.0],
    }
    store.upsert_chunks([chunk_repo2])
    assert store.count() == 1

    matches = store.search([1.0, 0.0, 0.0], limit=5)
    assert len(matches) == 1
    assert matches[0]["metadata"]["name"] == "RepoTwoBilling"
    assert "RepoOneAuth" not in [m["metadata"]["name"] for m in matches]


def test_rag_unanswerable_question_returns_insufficient_context() -> None:
    """RAG pipeline must not fabricate answers when retrieval produces low relevance / missing features."""
    in_memory_client = QdrantClient(":memory:")
    store = QdrantStore(client=in_memory_client, collection_name="test_rag_unanswerable")
    store.upsert_chunks([{
        "content": "def escape(s): return str(s).replace('&', '&amp;')\n",
        "metadata": {"file": "escape.py", "language": "Python", "type": "function", "name": "escape", "start_line": 1, "end_line": 2},
        "embedding": [1.0, 0.0, 0.0],
    }])

    retriever = Retriever(vector_store=store, embedding_service=type("MockEmbed", (), {"embed_query": lambda self, t: [0.0, 1.0, 0.0], "embed_text": lambda self, t: [0.0, 1.0, 0.0]})())
    # Fake LLM simulates grounded refusal when evidence lacks the requested subject
    llm = FakeLLMClient(responses="The provided repository context does not contain any information about a cryptocurrency payment gateway. Based on the available source files, this feature is not implemented.")
    rag = RAGService(retriever=retriever, llm_client=llm)

    result = rag.answer_question("Where is the cryptocurrency payment gateway implemented?", top_k=3)
    answer = result["answer"].lower()
    assert "does not contain" in answer or "not implemented" in answer
    assert "cryptocurrency" in answer


def test_agent_unanswerable_question_grounding() -> None:
    """Agent returns insufficient context when semantic search and graph cannot find non-existent feature."""
    in_memory_client = QdrantClient(":memory:")
    store = QdrantStore(client=in_memory_client, collection_name="test_agent_unanswerable")
    store.upsert_chunks([{
        "content": "def escape(s): return str(s)\n",
        "metadata": {"file": "escape.py", "language": "Python", "type": "function", "name": "escape", "start_line": 1, "end_line": 1},
        "embedding": [1.0, 0.0, 0.0],
    }])
    retriever = Retriever(vector_store=store, embedding_service=type("MockEmbed", (), {"embed_query": lambda self, t: [0.0, 0.0, 1.0], "embed_text": lambda self, t: [0.0, 0.0, 1.0]})())
    graph = DependencyGraph()
    graph_tool = DependencyGraphTool(graph)
    source_tool = SourceInspectionTool(base_dir=Path("."))

    llm = FakeLLMClient(responses="Insufficient context. The repository does not contain any Stripe billing webhook listener.")

    agent = AgentService(
        retriever=retriever,
        llm_client=llm,
        graph_tool=graph_tool,
        inspection_tool=source_tool,
    )
    result = agent.run_investigation("Where is the stripe billing webhook listener configured?")
    assert "insufficient" in result["answer"].lower() or "does not contain" in result["answer"].lower()

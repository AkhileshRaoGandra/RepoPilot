"""Unit and pipeline tests for Day 6 LangGraph Agentic Orchestration."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from app.agent import (
    AgentService,
    AgentState,
    SemanticSearchTool,
    SourceInspectionTool,
    create_agent_graph,
)
from app.embeddings import EmbeddingService
from app.main import app
from app.rag.llm import FakeLLMClient
from app.rag.retriever import Retriever
from app.vectorstore import QdrantStore


class FakeEmbeddingModel:
    """Deterministic embedding model producing 3D unit vectors for testing."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def encode(self, sentences: list[str], **kwargs: object) -> np.ndarray:
        self.calls.append({"sentences": sentences, **kwargs})
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
        "end_line": 26,
    },
    "embedding": [1.0, 0.0, 0.0],
}

PARTIAL_CHUNK = {
    "content": "def large_handler():\n    step1()\n",
    "metadata": {
        "file": "handlers/big.py",
        "language": "Python",
        "type": "function",
        "name": "large_handler",
        "start_line": 1,
        "end_line": 10,
        "part": 1,
        "parts": 3,
    },
    "embedding": [0.05, 0.05, 0.9],
}


@pytest.fixture
def fake_model() -> FakeEmbeddingModel:
    return FakeEmbeddingModel()


@pytest.fixture
def embedding_service(fake_model: FakeEmbeddingModel) -> EmbeddingService:
    return EmbeddingService(model_factory=lambda _: fake_model)


@pytest.fixture
def vector_store() -> QdrantStore:
    store = QdrantStore(collection_name="test_agent_collection", client=QdrantClient(path=":memory:"))
    store.upsert_chunks([AUTH_CHUNK, PARTIAL_CHUNK])
    return store


@pytest.fixture
def empty_store() -> QdrantStore:
    return QdrantStore(collection_name="test_agent_empty", client=QdrantClient(path=":memory:"))


@pytest.fixture
def sample_repo_dir(tmp_path: Path) -> Path:
    """Create a temporary repository directory with files to test source inspection."""
    repo = tmp_path / "test-owner--test-repo"
    repo.mkdir(parents=True, exist_ok=True)

    auth_file = repo / "auth" / "service.py"
    auth_file.parent.mkdir(parents=True, exist_ok=True)
    auth_lines = [f"# Line {i}\n" for i in range(1, 25)]
    auth_lines.append("def login(token: str):\n    return validate_token(token)\n")
    auth_lines.extend([f"# Line {i}\n" for i in range(27, 51)])
    auth_file.write_text("".join(auth_lines), encoding="utf-8")

    handler_file = repo / "handlers" / "big.py"
    handler_file.parent.mkdir(parents=True, exist_ok=True)
    handler_file.write_text("".join(f"line_{i} = {i}\n" for i in range(1, 30)), encoding="utf-8")

    return tmp_path


@pytest.fixture
def inspection_tool(sample_repo_dir: Path) -> SourceInspectionTool:
    return SourceInspectionTool(base_dir=sample_repo_dir)


@pytest.fixture
def fake_llm() -> FakeLLMClient:
    return FakeLLMClient(
        responses="AuthService.login() validates the user token in auth/service.py:25-26."
    )


# 1. Tool 1: Semantic Search Tool Tests
def test_semantic_search_tool(embedding_service: EmbeddingService, vector_store: QdrantStore) -> None:
    retriever = Retriever(embedding_service=embedding_service, vector_store=vector_store)
    tool = SemanticSearchTool(retriever=retriever)

    results = tool.search("login auth", top_k=1)
    assert len(results) == 1
    assert results[0]["metadata"]["name"] == "login"
    assert results[0]["metadata"]["file"] == "auth/service.py"


# 2. Tool 2: Source Inspection Tool Tests
def test_source_inspection_tool_reads_lines(inspection_tool: SourceInspectionTool) -> None:
    result = inspection_tool.inspect("auth/service.py", start_line=24, end_line=26)
    assert result["found"] is True
    assert result["lines"] == "24-26"
    assert "def login(token: str):" in result["content"]


def test_source_inspection_tool_handles_missing_and_traversal(inspection_tool: SourceInspectionTool) -> None:
    # Non-existent file
    missing = inspection_tool.inspect("nonexistent/file.py", 1, 10)
    assert missing["found"] is False
    assert "not found" in missing["error"]

    # Path traversal safety
    traversal = inspection_tool.inspect("../../secret.txt", 1, 10)
    assert traversal["found"] is False


# 3. LangGraph Workflow: Standard Question (Sufficient Context, Single Search Tool)
def test_agent_graph_sufficient_context(
    embedding_service: EmbeddingService,
    vector_store: QdrantStore,
    inspection_tool: SourceInspectionTool,
    fake_llm: FakeLLMClient,
) -> None:
    retriever = Retriever(embedding_service=embedding_service, vector_store=vector_store)
    agent = AgentService(
        retriever=retriever,
        inspection_tool=inspection_tool,
        llm_client=fake_llm,
    )

    result = agent.run_investigation("How does login authentication work?", top_k=1)
    assert result["question"] == "How does login authentication work?"
    assert "AuthService.login()" in result["answer"]
    # For a straightforward question, semantic search alone is sufficient
    assert result["tools_used"] == ["semantic_search"]
    assert len(result["evidence"]) >= 1
    assert result["investigation_status"] == "completed"


# 4. LangGraph Workflow: Deeper Investigation (Triggers Source Inspection)
def test_agent_graph_triggers_source_inspection_on_intent(
    embedding_service: EmbeddingService,
    vector_store: QdrantStore,
    inspection_tool: SourceInspectionTool,
    fake_llm: FakeLLMClient,
) -> None:
    retriever = Retriever(embedding_service=embedding_service, vector_store=vector_store)
    agent = AgentService(
        retriever=retriever,
        inspection_tool=inspection_tool,
        llm_client=fake_llm,
    )

    # Asking with intent keyword "inspect surrounding lines"
    result = agent.run_investigation("Inspect surrounding lines around auth login", top_k=1)
    assert "semantic_search" in result["tools_used"]
    assert "source_inspection" in result["tools_used"]
    assert len(result["source_inspections"]) == 1
    assert result["source_inspections"][0]["found"] is True
    assert result["investigation_status"] == "completed"


def test_agent_graph_triggers_source_inspection_on_partial_chunk(
    embedding_service: EmbeddingService,
    vector_store: QdrantStore,
    inspection_tool: SourceInspectionTool,
    fake_llm: FakeLLMClient,
) -> None:
    retriever = Retriever(embedding_service=embedding_service, vector_store=vector_store)
    agent = AgentService(
        retriever=retriever,
        inspection_tool=inspection_tool,
        llm_client=fake_llm,
    )

    # Query matching PARTIAL_CHUNK which has 'part': 1
    result = agent.run_investigation("large_handler function details", top_k=1)
    assert "semantic_search" in result["tools_used"]
    assert "source_inspection" in result["tools_used"]


# 5. Empty / Insufficient Context Handling
def test_agent_graph_handles_empty_collection(
    embedding_service: EmbeddingService,
    empty_store: QdrantStore,
    inspection_tool: SourceInspectionTool,
    fake_llm: FakeLLMClient,
) -> None:
    retriever = Retriever(embedding_service=embedding_service, vector_store=empty_store)
    agent = AgentService(
        retriever=retriever,
        inspection_tool=inspection_tool,
        llm_client=fake_llm,
    )

    result = agent.run_investigation("Where is authentication?", top_k=1)
    assert result["answer"] == "The provided repository context is insufficient to answer this question."
    assert result["evidence"] == []
    assert result["investigation_status"] == "insufficient"
    assert len(fake_llm.calls) == 0  # LLM call skipped for speed and token savings


# 6. Input Validation
def test_agent_service_validates_inputs(
    embedding_service: EmbeddingService,
    vector_store: QdrantStore,
    inspection_tool: SourceInspectionTool,
) -> None:
    retriever = Retriever(embedding_service=embedding_service, vector_store=vector_store)
    agent = AgentService(retriever=retriever, inspection_tool=inspection_tool)

    with pytest.raises(ValueError, match="non-empty string"):
        agent.run_investigation("   ")
    with pytest.raises(ValueError, match="at least 1"):
        agent.run_investigation("auth", top_k=0)


# 7. FastAPI /agent/query Endpoint Integration
def test_api_agent_query_endpoint(
    monkeypatch: pytest.MonkeyPatch,
    embedding_service: EmbeddingService,
    vector_store: QdrantStore,
    inspection_tool: SourceInspectionTool,
    fake_llm: FakeLLMClient,
) -> None:
    client = TestClient(app)
    retriever = Retriever(embedding_service=embedding_service, vector_store=vector_store)
    agent_service = AgentService(
        retriever=retriever,
        inspection_tool=inspection_tool,
        llm_client=fake_llm,
    )

    monkeypatch.setattr("app.main.get_agent_service", lambda: agent_service)

    response = client.post("/agent/query", json={"query": "How does authentication work?", "top_k": 2})
    assert response.status_code == 200
    data = response.json()
    assert data["question"] == "How does authentication work?"
    assert "AuthService.login()" in data["answer"]
    assert "semantic_search" in data["tools_used"]
    assert len(data["evidence"]) >= 1

    # Validation test
    err_resp = client.post("/agent/query", json={"query": "", "top_k": 2})
    assert err_resp.status_code == 422

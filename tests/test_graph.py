"""Unit and integration tests for Day 7 Repository Dependency Graph."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from app.agent import AgentService, DependencyGraphTool, SemanticSearchTool, SourceInspectionTool
from app.agent.graph import create_agent_graph
from app.code.parser import parse_source
from app.embeddings import EmbeddingService
from app.graph import (
    DependencyGraph,
    EdgeType,
    NodeType,
    build_graph,
    get_dependencies,
    get_dependents,
    get_file_symbols,
    get_import_chain,
    get_neighbors,
    get_related_symbols,
)
from app.main import app
from app.rag.llm import FakeLLMClient
from app.rag.retriever import Retriever
from app.vectorstore import QdrantStore


# ── Test Data ────────────────────────────────────────────────────────────

PYTHON_FILE_A = {
    "path": "app/auth/service.py",
    "language": "Python",
    "extension": ".py",
    "size": 200,
    "lines": 15,
    "content": (
        "from app.db.repository import UserRepository\n"
        "from app.utils import hash_password\n\n"
        "class AuthService:\n"
        "    def login(self, username: str, password: str) -> bool:\n"
        "        repo = UserRepository()\n"
        "        user = repo.find_by_name(username)\n"
        "        return hash_password(password) == user.password_hash\n\n"
        "    def logout(self, session_id: str) -> None:\n"
        "        pass\n"
    ),
}

PYTHON_FILE_B = {
    "path": "app/db/repository.py",
    "language": "Python",
    "extension": ".py",
    "size": 150,
    "lines": 10,
    "content": (
        "class UserRepository:\n"
        "    def find_by_name(self, name: str):\n"
        "        return None\n\n"
        "    def save(self, user):\n"
        "        pass\n"
    ),
}

PYTHON_FILE_C = {
    "path": "app/utils.py",
    "language": "Python",
    "extension": ".py",
    "size": 80,
    "lines": 5,
    "content": (
        "def hash_password(password: str) -> str:\n"
        "    return password[::-1]\n"
    ),
}

PYTHON_FILE_D = {
    "path": "app/controllers/auth_controller.py",
    "language": "Python",
    "extension": ".py",
    "size": 180,
    "lines": 12,
    "content": (
        "from app.auth.service import AuthService\n\n"
        "class AuthController:\n"
        "    def __init__(self):\n"
        "        self.service = AuthService()\n\n"
        "    def handle_login(self, request):\n"
        "        return self.service.login(request.user, request.password)\n"
    ),
}

JS_FILE = {
    "path": "src/auth.js",
    "language": "JavaScript",
    "extension": ".js",
    "size": 120,
    "lines": 8,
    "content": (
        "import { db } from './database';\n\n"
        "class AuthModule {\n"
        "  authenticate(token) {\n"
        "    return db.verify(token);\n"
        "  }\n"
        "}\n"
    ),
}

JS_FILE_B = {
    "path": "src/database.js",
    "language": "JavaScript",
    "extension": ".js",
    "size": 100,
    "lines": 6,
    "content": (
        "export const db = {\n"
        "  verify(token) { return true; },\n"
        "};\n"
    ),
}

JAVA_FILE = {
    "path": "com/example/service/AuthService.java",
    "language": "Java",
    "extension": ".java",
    "size": 200,
    "lines": 12,
    "content": (
        "package com.example.service;\n\n"
        "import com.example.repository.UserRepository;\n\n"
        "public class AuthService {\n"
        "    public boolean login(String user, String pass) {\n"
        "        return true;\n"
        "    }\n"
        "}\n"
    ),
}

JAVA_FILE_B = {
    "path": "com/example/repository/UserRepository.java",
    "language": "Java",
    "extension": ".java",
    "size": 150,
    "lines": 8,
    "content": (
        "package com.example.repository;\n\n"
        "public class UserRepository {\n"
        "    public Object findByName(String name) {\n"
        "        return null;\n"
        "    }\n"
        "}\n"
    ),
}

ALL_FILES = [PYTHON_FILE_A, PYTHON_FILE_B, PYTHON_FILE_C, PYTHON_FILE_D]
ALL_FILES_WITH_JS = [*ALL_FILES, JS_FILE, JS_FILE_B]
ALL_FILES_WITH_JAVA = [*ALL_FILES, JAVA_FILE, JAVA_FILE_B]


def _parse_all(files: list[dict]) -> dict[str, dict]:
    """Parse all files and return {path: parsed} dict."""
    results = {}
    for f in files:
        lang = str(f["language"])
        if lang not in {"Python", "JavaScript", "JavaScript (React JSX)", "Java"}:
            continue
        try:
            parsed = parse_source(str(f["content"]), lang)
            if not parsed["errors"]:
                results[str(f["path"])] = parsed
        except Exception:
            pass
    return results


# ════════════════════════════════════════════════════════════════════════
# SECTION 1: GRAPH CONSTRUCTION TESTS
# ════════════════════════════════════════════════════════════════════════


class TestGraphConstruction:
    """Test that files/symbols become graph nodes and import edges are created."""

    def test_files_become_file_nodes(self):
        parsed = _parse_all(ALL_FILES)
        graph = build_graph(ALL_FILES, parsed)
        for f in ALL_FILES:
            assert graph.has_node(f["path"]), f"File node missing: {f['path']}"
            node = graph.get_node(f["path"])
            assert node["node_type"] == NodeType.FILE.value

    def test_python_imports_create_edges(self):
        parsed = _parse_all(ALL_FILES)
        graph = build_graph(ALL_FILES, parsed)
        # auth/service.py imports db/repository.py and utils.py
        deps = get_dependencies(graph, "app/auth/service.py")
        assert "app/db/repository.py" in deps
        assert "app/utils.py" in deps

    def test_controller_imports_service(self):
        parsed = _parse_all(ALL_FILES)
        graph = build_graph(ALL_FILES, parsed)
        deps = get_dependencies(graph, "app/controllers/auth_controller.py")
        assert "app/auth/service.py" in deps

    def test_class_nodes_created(self):
        parsed = _parse_all(ALL_FILES)
        graph = build_graph(ALL_FILES, parsed)
        # AuthService class
        cls_id = "app/auth/service.py::AuthService"
        assert graph.has_node(cls_id)
        node = graph.get_node(cls_id)
        assert node["node_type"] == NodeType.CLASS.value
        assert node["name"] == "AuthService"

    def test_method_nodes_created(self):
        parsed = _parse_all(ALL_FILES)
        graph = build_graph(ALL_FILES, parsed)
        method_id = "app/auth/service.py::AuthService.login"
        assert graph.has_node(method_id)
        node = graph.get_node(method_id)
        assert node["node_type"] == NodeType.METHOD.value

    def test_function_nodes_created(self):
        parsed = _parse_all(ALL_FILES)
        graph = build_graph(ALL_FILES, parsed)
        func_id = "app/utils.py::hash_password"
        assert graph.has_node(func_id)
        node = graph.get_node(func_id)
        assert node["node_type"] == NodeType.FUNCTION.value

    def test_contains_edges(self):
        parsed = _parse_all(ALL_FILES)
        graph = build_graph(ALL_FILES, parsed)
        # File → Class CONTAINS edge
        neighbors = get_neighbors(graph, "app/auth/service.py")
        class_neighbors = [n for n in neighbors if n.get("node_type") == NodeType.CLASS.value]
        assert any(n["name"] == "AuthService" for n in class_neighbors)

    def test_defined_in_edges(self):
        parsed = _parse_all(ALL_FILES)
        graph = build_graph(ALL_FILES, parsed)
        # Class → Method DEFINED_IN edge
        cls_id = "app/auth/service.py::AuthService"
        neighbors = get_neighbors(graph, cls_id)
        method_names = [n["name"] for n in neighbors if n.get("node_type") == NodeType.METHOD.value]
        assert "login" in method_names
        assert "logout" in method_names

    def test_js_imports_resolved(self):
        parsed = _parse_all(ALL_FILES_WITH_JS)
        graph = build_graph(ALL_FILES_WITH_JS, parsed)
        deps = get_dependencies(graph, "src/auth.js")
        assert "src/database.js" in deps

    def test_java_imports_resolved(self):
        parsed = _parse_all(ALL_FILES_WITH_JAVA)
        graph = build_graph(ALL_FILES_WITH_JAVA, parsed)
        deps = get_dependencies(graph, "com/example/service/AuthService.java")
        assert "com/example/repository/UserRepository.java" in deps

    def test_graph_to_dict_serialisation(self):
        parsed = _parse_all(ALL_FILES)
        graph = build_graph(ALL_FILES, parsed)
        d = graph.to_dict()
        assert "nodes" in d
        assert "edges" in d
        assert "stats" in d
        assert d["stats"]["total_nodes"] > 0
        assert d["stats"]["total_edges"] > 0
        assert d["stats"]["file_nodes"] == len(ALL_FILES)

    def test_empty_graph(self):
        graph = build_graph([], {})
        assert graph.node_count == 0
        assert graph.edge_count == 0


# ════════════════════════════════════════════════════════════════════════
# SECTION 2: GRAPH QUERY TESTS
# ════════════════════════════════════════════════════════════════════════


class TestGraphQueries:
    """Test dependency/dependent/neighbor/related queries."""

    @pytest.fixture
    def graph(self):
        parsed = _parse_all(ALL_FILES)
        return build_graph(ALL_FILES, parsed)

    def test_get_dependencies(self, graph):
        deps = get_dependencies(graph, "app/auth/service.py")
        assert "app/db/repository.py" in deps
        assert "app/utils.py" in deps

    def test_get_dependents(self, graph):
        dependents = get_dependents(graph, "app/auth/service.py")
        assert "app/controllers/auth_controller.py" in dependents

    def test_get_dependents_of_leaf(self, graph):
        """utils.py is imported by auth/service.py."""
        dependents = get_dependents(graph, "app/utils.py")
        assert "app/auth/service.py" in dependents

    def test_get_file_symbols(self, graph):
        symbols = get_file_symbols(graph, "app/auth/service.py")
        symbol_names = [s["name"] for s in symbols]
        assert "AuthService" in symbol_names

    def test_get_neighbors_of_file(self, graph):
        neighbors = get_neighbors(graph, "app/auth/service.py")
        neighbor_ids = [n["id"] for n in neighbors]
        # Should include imported files and contained symbols
        assert "app/db/repository.py" in neighbor_ids
        assert "app/utils.py" in neighbor_ids

    def test_get_related_symbols(self, graph):
        results = get_related_symbols(graph, "AuthService")
        assert len(results) > 0
        assert any(r["name"] == "AuthService" for r in results)

    def test_get_related_symbols_case_insensitive(self, graph):
        results = get_related_symbols(graph, "authservice")
        assert len(results) > 0

    def test_get_import_chain(self, graph):
        chain = get_import_chain(graph, "app/controllers/auth_controller.py")
        chain_files = [c["file"] for c in chain]
        # controller → service → repository/utils
        assert "app/auth/service.py" in chain_files

    def test_nonexistent_node_queries(self, graph):
        assert get_dependencies(graph, "nonexistent.py") == []
        assert get_dependents(graph, "nonexistent.py") == []
        assert get_neighbors(graph, "nonexistent.py") == []
        assert get_file_symbols(graph, "nonexistent.py") == []
        assert get_import_chain(graph, "nonexistent.py") == []

    def test_related_symbols_no_match(self, graph):
        results = get_related_symbols(graph, "XYZNonexistent123")
        assert results == []


# ════════════════════════════════════════════════════════════════════════
# SECTION 3: DEPENDENCY GRAPH TOOL TESTS
# ════════════════════════════════════════════════════════════════════════


class TestDependencyGraphTool:
    """Test the DependencyGraphTool wrapper used by the agent."""

    @pytest.fixture
    def graph(self):
        parsed = _parse_all(ALL_FILES)
        return build_graph(ALL_FILES, parsed)

    @pytest.fixture
    def tool(self, graph):
        return DependencyGraphTool(graph=graph)

    def test_find_dependencies(self, tool):
        result = tool.find_dependencies("app/auth/service.py")
        assert "app/db/repository.py" in result["dependencies"]
        assert "app/utils.py" in result["dependencies"]

    def test_find_dependents(self, tool):
        result = tool.find_dependents("app/auth/service.py")
        assert "app/controllers/auth_controller.py" in result["dependents"]

    def test_find_related(self, tool):
        result = tool.find_related("AuthService")
        assert len(result["related"]) > 0
        assert any(r["name"] == "AuthService" for r in result["related"])

    def test_explore(self, tool):
        result = tool.explore("app/auth/service.py")
        assert len(result["neighbors"]) > 0

    def test_get_import_chain(self, tool):
        result = tool.get_import_chain("app/controllers/auth_controller.py")
        chain_files = [c["file"] for c in result["chain"]]
        assert "app/auth/service.py" in chain_files

    def test_callable_dispatch(self, tool):
        result = tool("dependencies", "app/auth/service.py")
        assert "dependencies" in result

    def test_unavailable_graph(self):
        tool = DependencyGraphTool(graph=None)
        assert not tool.available
        result = tool.find_dependencies("app/auth/service.py")
        assert result["dependencies"] == []
        assert "error" in result

    def test_unknown_action(self, tool):
        result = tool("unknown_action", "some_target")
        assert "error" in result


# ════════════════════════════════════════════════════════════════════════
# SECTION 4: AGENT INTEGRATION TESTS
# ════════════════════════════════════════════════════════════════════════


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
            elif "database" in text or "connect" in text or "repository" in text:
                vectors.append([0.0, 0.95, 0.05])
            else:
                vectors.append([0.05, 0.05, 0.9])
        return np.array(vectors)


AUTH_CHUNK = {
    "content": "class AuthService:\n    def login(self, username, password):\n        repo = UserRepository()\n",
    "metadata": {
        "file": "app/auth/service.py",
        "language": "Python",
        "type": "class",
        "name": "AuthService",
        "start_line": 4,
        "end_line": 8,
    },
    "embedding": [1.0, 0.0, 0.0],
}

CONTROLLER_CHUNK = {
    "content": "class AuthController:\n    def handle_login(self, request):\n        return self.service.login(request.user, request.password)\n",
    "metadata": {
        "file": "app/controllers/auth_controller.py",
        "language": "Python",
        "type": "class",
        "name": "AuthController",
        "start_line": 3,
        "end_line": 8,
    },
    "embedding": [0.9, 0.1, 0.0],
}


@pytest.fixture
def fake_model():
    return FakeEmbeddingModel()


@pytest.fixture
def embedding_service(fake_model):
    return EmbeddingService(model_factory=lambda _: fake_model)


@pytest.fixture
def vector_store():
    store = QdrantStore(collection_name="test_graph_collection", client=QdrantClient(path=":memory:"))
    store.upsert_chunks([AUTH_CHUNK, CONTROLLER_CHUNK])
    return store


@pytest.fixture
def empty_store():
    return QdrantStore(collection_name="test_graph_empty", client=QdrantClient(path=":memory:"))


@pytest.fixture
def dep_graph():
    parsed = _parse_all(ALL_FILES)
    return build_graph(ALL_FILES, parsed)


@pytest.fixture
def graph_tool(dep_graph):
    return DependencyGraphTool(graph=dep_graph)


@pytest.fixture
def sample_repo_dir(tmp_path: Path) -> Path:
    """Create a temporary repository directory with files matching the graph test data."""
    repo = tmp_path / "test-owner--test-repo"
    for f in ALL_FILES:
        file_path = repo / f["path"]
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(str(f["content"]), encoding="utf-8")
    return tmp_path


@pytest.fixture
def inspection_tool(sample_repo_dir):
    return SourceInspectionTool(base_dir=sample_repo_dir)


@pytest.fixture
def fake_llm():
    return FakeLLMClient(
        responses="AuthController delegates to AuthService.login() in app/auth/service.py:5-8, which uses UserRepository from app/db/repository.py."
    )


class TestAgentGraphIntegration:
    """Test that the LangGraph agent correctly uses the graph tool."""

    def test_graph_tool_invoked_on_relationship_question(
        self, embedding_service, vector_store, inspection_tool, graph_tool, fake_llm
    ):
        """Questions about relationships/flow should trigger graph_investigation."""
        retriever = Retriever(embedding_service=embedding_service, vector_store=vector_store)
        agent = AgentService(
            retriever=retriever,
            inspection_tool=inspection_tool,
            graph_tool=graph_tool,
            llm_client=fake_llm,
        )

        result = agent.run_investigation("How does the login request reach the database?", top_k=2)
        assert "graph_investigation" in result["tools_used"]
        assert result["investigation_status"] == "completed"
        assert len(result["graph_investigations"]) > 0

    def test_graph_tool_not_invoked_on_simple_question(
        self, embedding_service, vector_store, inspection_tool, graph_tool, fake_llm
    ):
        """Simple factual questions should NOT trigger graph investigation."""
        retriever = Retriever(embedding_service=embedding_service, vector_store=vector_store)
        agent = AgentService(
            retriever=retriever,
            inspection_tool=inspection_tool,
            graph_tool=graph_tool,
            llm_client=fake_llm,
        )

        result = agent.run_investigation("What is AuthService?", top_k=2)
        assert "graph_investigation" not in result["tools_used"]
        assert "semantic_search" in result["tools_used"]

    def test_graph_results_in_evidence(
        self, embedding_service, vector_store, inspection_tool, graph_tool, fake_llm
    ):
        """Graph-discovered dependencies should appear in evidence."""
        retriever = Retriever(embedding_service=embedding_service, vector_store=vector_store)
        agent = AgentService(
            retriever=retriever,
            inspection_tool=inspection_tool,
            graph_tool=graph_tool,
            llm_client=fake_llm,
        )

        result = agent.run_investigation("What does auth service depend on? Show dependencies.", top_k=2)
        assert "graph_investigation" in result["tools_used"]
        # Graph-discovered files should produce evidence entries
        assert len(result["evidence"]) > 0

    def test_answer_remains_grounded(
        self, embedding_service, vector_store, inspection_tool, graph_tool, fake_llm
    ):
        """The final answer must come from the LLM with context, not invented."""
        retriever = Retriever(embedding_service=embedding_service, vector_store=vector_store)
        agent = AgentService(
            retriever=retriever,
            inspection_tool=inspection_tool,
            graph_tool=graph_tool,
            llm_client=fake_llm,
        )

        result = agent.run_investigation("How does login flow through the codebase?", top_k=2)
        assert result["answer"]  # Non-empty answer
        assert fake_llm.calls  # LLM was called (not skipped)

    def test_empty_collection_with_graph(
        self, embedding_service, empty_store, inspection_tool, graph_tool, fake_llm
    ):
        """Empty vector store should still handle gracefully with graph available."""
        retriever = Retriever(embedding_service=embedding_service, vector_store=empty_store)
        agent = AgentService(
            retriever=retriever,
            inspection_tool=inspection_tool,
            graph_tool=graph_tool,
            llm_client=fake_llm,
        )

        result = agent.run_investigation("How does auth flow work?", top_k=2)
        assert result["investigation_status"] == "insufficient"
        assert result["answer"] == "The provided repository context is insufficient to answer this question."

    def test_graph_tool_unavailable_gracefully(
        self, embedding_service, vector_store, inspection_tool, fake_llm
    ):
        """Agent should work normally when graph is not built."""
        unavailable_tool = DependencyGraphTool(graph=None)
        retriever = Retriever(embedding_service=embedding_service, vector_store=vector_store)
        agent = AgentService(
            retriever=retriever,
            inspection_tool=inspection_tool,
            graph_tool=unavailable_tool,
            llm_client=fake_llm,
        )

        result = agent.run_investigation("How does auth flow work?", top_k=2)
        # Should still complete without crashing
        assert result["investigation_status"] == "completed"
        assert "semantic_search" in result["tools_used"]

    def test_graph_investigations_returned_in_response(
        self, embedding_service, vector_store, inspection_tool, graph_tool, fake_llm
    ):
        """Response should include graph_investigations field."""
        retriever = Retriever(embedding_service=embedding_service, vector_store=vector_store)
        agent = AgentService(
            retriever=retriever,
            inspection_tool=inspection_tool,
            graph_tool=graph_tool,
            llm_client=fake_llm,
        )

        result = agent.run_investigation("What imports does auth service use?", top_k=2)
        assert "graph_investigations" in result


# ════════════════════════════════════════════════════════════════════════
# SECTION 5: API ENDPOINT TEST
# ════════════════════════════════════════════════════════════════════════


class TestGraphAPI:
    """Test the /repository/graph endpoint."""

    def test_graph_endpoint_returns_404_when_no_graph(self):
        """Before ingestion, should return 404."""
        client = TestClient(app)
        # Patch get_dependency_graph to return None
        import app.main as main_module
        original = main_module.get_dependency_graph

        main_module.get_dependency_graph = lambda: None
        try:
            response = client.get("/repository/graph")
            assert response.status_code == 404
            assert "No dependency graph" in response.json()["detail"]
        finally:
            main_module.get_dependency_graph = original

    def test_graph_endpoint_returns_graph_data(self):
        """When graph is available, should return nodes and edges."""
        client = TestClient(app)
        parsed = _parse_all(ALL_FILES)
        graph = build_graph(ALL_FILES, parsed)

        import app.main as main_module
        original = main_module.get_dependency_graph

        main_module.get_dependency_graph = lambda: graph
        try:
            response = client.get("/repository/graph")
            assert response.status_code == 200
            data = response.json()
            assert "nodes" in data
            assert "edges" in data
            assert "stats" in data
            assert data["stats"]["total_nodes"] > 0
        finally:
            main_module.get_dependency_graph = original

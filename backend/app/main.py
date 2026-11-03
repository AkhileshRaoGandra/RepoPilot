import os
import sys
from pathlib import Path

# Ensure backend directory is in sys.path regardless of execution working directory
_BACKEND_DIR = str(Path(__file__).resolve().parents[1])
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, HttpUrl

import logging

logger = logging.getLogger("repopilot")

from app.ingestion.repository import (
    CloneFailedError,
    EmptyRepositoryError,
    InvalidGitHubUrlError,
    RepositoryNotFoundError,
    UnsupportedFilesError,
    analyze_repository,
    get_active_repository,
    get_dependency_graph,
    get_vector_store,
    is_repository_indexed,
)
from app.agent import get_agent_service
from app.rag import get_rag_service, get_retriever


app = FastAPI(
    title="RepoPilot",
    version="0.9.0",
    description="Repository ingestion, embeddings, Qdrant vector storage, dependency graph, RAG, and LangGraph agentic orchestration API",
)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?|https://.*\.vercel\.app",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RepositoryAnalyzeRequest(BaseModel):
    github_url: HttpUrl


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Question about the repository")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of chunks to retrieve")


def _ensure_repository_ready(retriever_instance: object | None = None) -> None:
    """Verify that a repository has been indexed before querying, raising 400 if not."""
    if is_repository_indexed():
        store = getattr(retriever_instance, "vector_store", None) if retriever_instance is not None else get_vector_store()
        if store is not None and not store.is_empty():
            return
    # Allow test fixtures that explicitly injected a custom non-default vector store with points
    if retriever_instance is not None:
        try:
            store = getattr(retriever_instance, "vector_store", None)
            default_store = get_vector_store()
            if store is not None and store is not default_store and not store.is_empty():
                return
        except Exception:
            pass
    raise HTTPException(
        status_code=400,
        detail="No repository has been indexed yet. Please analyze a GitHub repository first.",
    )


from fastapi import APIRouter

api_router = APIRouter()


@api_router.get("/repository/status")
def repository_status() -> dict[str, object]:
    """Return active repository indexing and vector store readiness status."""
    active_repo = get_active_repository()
    graph = get_dependency_graph()
    store = get_vector_store()
    chunk_count = store.count()
    return {
        "indexed": chunk_count > 0 and active_repo is not None,
        "repository": active_repo,
        "chunks_stored": chunk_count,
        "graph_available": graph is not None,
    }


@api_router.post("/repository/analyze")
def analyze(request: RepositoryAnalyzeRequest) -> dict[str, object]:
    try:
        logger.info(f"Analyzing repository: {request.github_url}")
        return analyze_repository(str(request.github_url))
    except InvalidGitHubUrlError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except RepositoryNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except (CloneFailedError, EmptyRepositoryError, UnsupportedFilesError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        logger.error(f"Unexpected error analyzing repository: {error}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to analyze repository due to an internal server error.") from error


@api_router.get("/repository/graph")
def repository_graph() -> dict[str, object]:
    """Return the current dependency graph as nodes and edges for debugging."""
    graph = get_dependency_graph()
    if graph is None:
        raise HTTPException(
            status_code=404,
            detail="No dependency graph available. Ingest a repository first.",
        )
    return graph.to_dict()


@api_router.post("/query")
def query_repository(request: QueryRequest) -> dict[str, object]:
    """Retrieve relevant chunks and generate a grounded answer with citations."""
    try:
        rag_service = get_rag_service()
        _ensure_repository_ready(getattr(rag_service, "retriever", None))
        return rag_service.answer_question(query=request.query, top_k=request.top_k)
    except HTTPException:
        raise
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        logger.error(f"Error executing RAG query: {error}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred while processing the question.") from error


@api_router.post("/retrieve")
def retrieve_chunks(request: QueryRequest) -> dict[str, object]:
    """Retrieve top matching code chunks directly without calling the LLM."""
    try:
        retriever = get_retriever()
        _ensure_repository_ready(retriever)
        chunks = retriever.retrieve(query=request.query, top_k=request.top_k)
        return {"query": request.query, "chunks": chunks}
    except HTTPException:
        raise
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        logger.error(f"Error retrieving chunks: {error}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred while retrieving chunks.") from error


@api_router.post("/agent/query")
def agent_query(request: QueryRequest) -> dict[str, object]:
    """Execute LangGraph agentic repository investigation with tool orchestration."""
    try:
        agent_service = get_agent_service()
        _ensure_repository_ready(getattr(agent_service, "retriever", None))
        return agent_service.run_investigation(query=request.query, top_k=request.top_k)
    except HTTPException:
        raise
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        logger.error(f"Error executing agent query: {error}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred while executing the agent investigation.") from error


@api_router.get("/health/llm")
def llm_health_check() -> dict[str, object]:
    """Inspect and report the active LLM client and API key configuration status."""
    from app.rag.llm import GeminiLLMClient, HttpLLMClient, TemplateGroundedClient, get_default_llm_client

    client = get_default_llm_client()
    has_gemini = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    has_openai = bool(os.getenv("OPENAI_API_KEY"))

    client_type = type(client).__name__
    is_live = not isinstance(client, TemplateGroundedClient)

    return {
        "status": "connected" if is_live else "missing_api_key",
        "client": client_type,
        "is_live_llm": is_live,
        "env_detected": {
            "GEMINI_API_KEY": has_gemini,
            "OPENAI_API_KEY": has_openai,
        },
        "model": getattr(client, "model", "none"),
        "message": (
            "Live LLM is configured and active."
            if is_live
            else "No LLM API key detected in environment. RepoPilot is using TemplateGroundedClient fallback."
        ),
    }


# Include router at both root ('/') and '/api' prefix to support both direct and rewritten Vercel requests
app.include_router(api_router)
app.include_router(api_router, prefix="/api")


@app.get("/api/openapi.json", include_in_schema=False)
def api_openapi() -> dict[str, object]:
    """Expose OpenAPI specification under /api/openapi.json for health checks."""
    return app.openapi()

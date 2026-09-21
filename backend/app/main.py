import sys
from pathlib import Path

# Ensure backend directory is in sys.path regardless of execution working directory
_BACKEND_DIR = str(Path(__file__).resolve().parents[1])
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, HttpUrl

from app.ingestion.repository import (
    CloneFailedError,
    EmptyRepositoryError,
    InvalidGitHubUrlError,
    RepositoryNotFoundError,
    UnsupportedFilesError,
    analyze_repository,
)
from app.agent import get_agent_service
from app.rag import get_rag_service, get_retriever


app = FastAPI(
    title="RepoPilot",
    version="0.6.0",
    description="Repository ingestion, embeddings, Qdrant vector storage, RAG, and LangGraph agentic orchestration API",
)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RepositoryAnalyzeRequest(BaseModel):
    github_url: HttpUrl


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Question about the repository")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of chunks to retrieve")


@app.post("/repository/analyze")
def analyze(request: RepositoryAnalyzeRequest) -> dict[str, object]:
    try:
        return analyze_repository(str(request.github_url))
    except InvalidGitHubUrlError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except RepositoryNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except (CloneFailedError, EmptyRepositoryError, UnsupportedFilesError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/query")
def query_repository(request: QueryRequest) -> dict[str, object]:
    """Retrieve relevant chunks and generate a grounded answer with citations."""
    try:
        rag_service = get_rag_service()
        return rag_service.answer_question(query=request.query, top_k=request.top_k)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/retrieve")
def retrieve_chunks(request: QueryRequest) -> dict[str, object]:
    """Retrieve top matching code chunks directly without calling the LLM."""
    try:
        retriever = get_retriever()
        chunks = retriever.retrieve(query=request.query, top_k=request.top_k)
        return {"query": request.query, "chunks": chunks}
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/agent/query")
def agent_query(request: QueryRequest) -> dict[str, object]:
    """Execute LangGraph agentic repository investigation with tool orchestration."""
    try:
        agent_service = get_agent_service()
        return agent_service.run_investigation(query=request.query, top_k=request.top_k)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error



"""FastAPI entry point for RepoPilot's Day 1 ingestion API."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl

from app.ingestion.repository import (
    CloneFailedError,
    EmptyRepositoryError,
    InvalidGitHubUrlError,
    RepositoryNotFoundError,
    UnsupportedFilesError,
    analyze_repository,
)

app = FastAPI(title="RepoPilot", version="0.1.0", description="Day 1 repository ingestion API")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=False,
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)


class RepositoryAnalyzeRequest(BaseModel):
    github_url: HttpUrl


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

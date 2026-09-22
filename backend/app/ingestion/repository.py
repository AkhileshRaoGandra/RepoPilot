"""Clone and extract supported files from public GitHub repositories."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

from git import GitCommandError, Repo

from app.code.chunker import create_chunks
from app.code.parser import parse_source
from app.embeddings import EmbeddingService
from app.graph.builder import DependencyGraph, build_graph
from app.vectorstore import QdrantStore


PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPOSITORIES_DIR = PROJECT_ROOT / "data" / "repositories"

# This object is cheap to create. It loads BGE-M3 lazily on the first embedding
# request, then reuses that model for later repository analyses in this process.
embedding_service = EmbeddingService()


@lru_cache
def get_vector_store() -> QdrantStore:
    """Create the local Qdrant client only when chunks need to be persisted."""
    return QdrantStore(path=PROJECT_ROOT / "data" / "qdrant")


# Module-level dependency graph, populated during analyse_repository().
_dependency_graph: DependencyGraph | None = None


def get_dependency_graph() -> DependencyGraph | None:
    """Return the most recently built dependency graph (or None)."""
    return _dependency_graph


def __getattr__(name: str) -> Any:
    if name == "vector_store":
        return get_vector_store()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


SUPPORTED_EXTENSIONS = {
    ".py": "Python",
    ".js": "JavaScript",
    ".jsx": "JavaScript (React JSX)",
    ".ts": "TypeScript",
    ".tsx": "TypeScript (React TSX)",
    ".java": "Java",
    ".cpp": "C++",
    ".c": "C",
    ".h": "C/C++ Header",
    ".hpp": "C++ Header",
    ".md": "Markdown",
    ".json": "JSON",
    ".yml": "YAML",
    ".yaml": "YAML",
}

IGNORED_DIRECTORIES = {
    ".git",
    "node_modules",
    "venv",
    ".venv",
    "__pycache__",
    "dist",
    "build",
    "target",
    ".next",
    "coverage",
}
SENSITIVE_FILE_NAMES = {".env"}
SENSITIVE_SUFFIXES = {".pem", ".key", ".crt"}


class RepositoryError(Exception):
    """Base error for repository ingestion failures."""


class InvalidGitHubUrlError(RepositoryError):
    pass


class RepositoryNotFoundError(RepositoryError):
    pass


class CloneFailedError(RepositoryError):
    pass


class EmptyRepositoryError(RepositoryError):
    pass


class UnsupportedFilesError(RepositoryError):
    pass


@dataclass(frozen=True)
class RepositoryReference:
    owner: str
    name: str

    @property
    def directory_name(self) -> str:
        return f"{self.owner}--{self.name}"


def parse_github_url(github_url: str) -> RepositoryReference:
    """Validate an HTTPS GitHub repository URL and return its owner/name."""
    parsed = urlparse(github_url.strip())
    if (
        parsed.scheme != "https"
        or parsed.netloc.lower() != "github.com"
        or parsed.params
        or parsed.query
        or parsed.fragment
        or parsed.username
        or parsed.password
    ):
        raise InvalidGitHubUrlError("Provide an HTTPS GitHub repository URL.")

    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) != 2:
        raise InvalidGitHubUrlError("URL must have the form https://github.com/owner/repository.")

    owner, repository = parts
    if repository.endswith(".git"):
        repository = repository[:-4]
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    if not owner or not repository or any(char not in allowed for char in owner + repository):
        raise InvalidGitHubUrlError("GitHub owner and repository names contain unsupported characters.")
    return RepositoryReference(owner=owner, name=repository)


def is_sensitive_file(path: Path) -> bool:
    name = path.name.lower()
    return name in SENSITIVE_FILE_NAMES or name.startswith(".env.") or path.suffix.lower() in SENSITIVE_SUFFIXES


def is_supported_file(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTENSIONS and not is_sensitive_file(path)


def detect_language(path: Path) -> str:
    return SUPPORTED_EXTENSIONS.get(path.suffix.lower(), "Unknown")


def clone_or_open_repository(github_url: str, repositories_dir: Path = REPOSITORIES_DIR) -> tuple[RepositoryReference, Path]:
    """Clone a public repository once, or reuse its existing local clone."""
    reference = parse_github_url(github_url)
    repositories_dir.mkdir(parents=True, exist_ok=True)
    destination = repositories_dir / reference.directory_name

    if destination.exists():
        if (destination / ".git").is_dir():
            return reference, destination
        raise CloneFailedError(f"Local path exists but is not a Git repository: {destination.name}")

    try:
        Repo.clone_from(github_url, destination)
    except GitCommandError as error:
        if destination.exists():
            shutil.rmtree(destination, ignore_errors=True)
        # GitHub uses the same clone failure for absent and private repositories.
        message = str(error).lower()
        if "repository not found" in message or "not found" in message:
            raise RepositoryNotFoundError("Repository was not found or is inaccessible.") from error
        raise CloneFailedError("Could not clone the repository.") from error
    except Exception as error:  # Covers network and local Git failures consistently.
        if destination.exists():
            shutil.rmtree(destination, ignore_errors=True)
        raise CloneFailedError("Could not clone the repository.") from error

    return reference, destination


def extract_file_metadata(repository_path: Path, file_path: Path) -> dict[str, object] | None:
    """Read a supported UTF-8 file, returning None when it cannot be decoded."""
    try:
        content = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None

    return {
        "path": file_path.relative_to(repository_path).as_posix(),
        "language": detect_language(file_path),
        "extension": file_path.suffix.lower(),
        "size": file_path.stat().st_size,
        "lines": len(content.splitlines()),
        "content": content,
    }


def scan_repository(repository_path: Path) -> tuple[list[dict[str, object]], int, bool]:
    """Recursively collect supported files while excluding generated and sensitive paths."""
    files: list[dict[str, object]] = []
    unreadable_files = 0
    has_non_ignored_file = False

    for directory, subdirectories, filenames in os.walk(repository_path):
        subdirectories[:] = sorted(name for name in subdirectories if name not in IGNORED_DIRECTORIES)
        filenames.sort()
        current_directory = Path(directory)
        for filename in filenames:
            file_path = current_directory / filename
            if is_sensitive_file(file_path):
                continue
            has_non_ignored_file = True
            if not is_supported_file(file_path):
                continue
            metadata = extract_file_metadata(repository_path, file_path)
            if metadata is None:
                unreadable_files += 1
            else:
                files.append(metadata)

    return files, unreadable_files, has_non_ignored_file


def analyze_repository(github_url: str) -> dict[str, object]:
    """Clone/open a repository and return extracted files plus embedded code chunks."""
    global _dependency_graph

    reference, repository_path = clone_or_open_repository(github_url)
    files, unreadable_files, has_non_ignored_file = scan_repository(repository_path)
    if not has_non_ignored_file:
        raise EmptyRepositoryError("Repository contains no files to analyze.")
    if not files:
        raise UnsupportedFilesError("Repository contains no supported readable files.")

    documentation_extensions = {".md"}
    documentation_files = sum(file["extension"] in documentation_extensions for file in files)
    chunks: list[dict[str, object]] = []
    parsing_failures: list[dict[str, str]] = []
    parsed_files = 0
    parsed_results: dict[str, dict[str, object]] = {}
    for file in files:
        language = str(file["language"])
        if language not in {"Python", "JavaScript", "JavaScript (React JSX)", "Java"}:
            continue
        try:
            parsed = parse_source(str(file["content"]), language)
            if parsed["errors"]:
                parsing_failures.append({"path": str(file["path"]), "reason": "parse error"})
                continue
            parsed_files += 1
            parsed_results[str(file["path"])] = parsed
            chunks.extend(create_chunks(str(file["content"]), str(file["path"]), language, parsed))
        except Exception as error:  # A parser failure must not stop repository ingestion.
            parsing_failures.append({"path": str(file["path"]), "reason": str(error)})

    # Build the dependency graph from parsed AST data
    _dependency_graph = build_graph(files, parsed_results)

    embedded_chunks = embedding_service.embed_chunks(chunks)
    stored_point_ids = get_vector_store().upsert_chunks(embedded_chunks)
    return {
        "repository": reference.name,
        "files_found": len(files),
        "code_files": len(files) - documentation_files,
        "documentation_files": documentation_files,
        "skipped_unreadable_files": unreadable_files,
        "parsed_files": parsed_files,
        "chunks_created": len(chunks),
        "chunks_stored": len(stored_point_ids),
        "graph_nodes": _dependency_graph.node_count,
        "graph_edges": _dependency_graph.edge_count,
        "parsing_failures": parsing_failures,
        "chunks": embedded_chunks,
        "files": files,
    }

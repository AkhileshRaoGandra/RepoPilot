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


_vector_store: QdrantStore | None = None


def get_vector_store() -> QdrantStore:
    """Return the shared local Qdrant client."""
    global _vector_store
    if _vector_store is None:
        _vector_store = QdrantStore(path=PROJECT_ROOT / "data" / "qdrant")
    return _vector_store


# Module-level dependency graph and active repository reference
_dependency_graph: DependencyGraph | None = None
_active_repository: str | None = None


def get_dependency_graph() -> DependencyGraph | None:
    """Return the most recently built dependency graph (or None)."""
    return _dependency_graph


def get_active_repository() -> str | None:
    """Return the name of the currently indexed repository (or None)."""
    return _active_repository


def is_repository_indexed() -> bool:
    """Return True if an ingested repository is active with indexed chunks."""
    if _active_repository is None and _dependency_graph is None:
        return False
    try:
        return not get_vector_store().is_empty()
    except Exception:
        return False


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
    ".cc": "C++",
    ".cxx": "C++",
    ".h": "C/C++ Header",
    ".hpp": "C++ Header",
    ".hh": "C++ Header",
    ".md": "Markdown",
    ".markdown": "Markdown",
    ".rst": "Markdown",
    ".json": "JSON",
    ".txt": "Text",
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
IGNORED_FILE_NAMES = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "composer.lock",
    "cargo.lock",
    "poetry.lock",
    "pipfile.lock",
}
MAX_FILE_SIZE_BYTES = 500 * 1024  # Skip individual files > 500 KB to avoid freezing memory
MAX_CHUNKS_PER_REPO = 500         # Embed up to 500 chunks so CPU embedding completes in <30 seconds
MAX_SCANNED_FILES = 2000          # Upper bound on discovered files to prevent memory exhaustion
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
    name = path.name.lower()
    if name in IGNORED_FILE_NAMES or is_sensitive_file(path):
        return False
    # Explicitly support documentation files (e.g. README without extension or LICENSE)
    if name in {"readme", "license", "contributing", "architecture"}:
        return True
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def detect_language(path: Path) -> str:
    name = path.name.lower()
    if name in {"readme", "readme.md", "readme.markdown", "readme.rst", "readme.txt"}:
        return "Markdown"
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
        print(f"[Ingestion] Cloning {github_url} (shallow clone depth=1)...")
        Repo.clone_from(
            github_url,
            destination,
            depth=1,
            single_branch=True,
            env={"GIT_TERMINAL_PROMPT": "0"},
        )
        print(f"[Ingestion] Successfully cloned into {destination.name}")
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
    """Read a supported UTF-8 file, returning None when it cannot be decoded or exceeds max size."""
    try:
        file_size = file_path.stat().st_size
        if file_size > MAX_FILE_SIZE_BYTES:
            return None
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
        if len(files) >= MAX_SCANNED_FILES:
            break
        subdirectories[:] = sorted(name for name in subdirectories if name not in IGNORED_DIRECTORIES)
        filenames.sort()
        current_directory = Path(directory)
        for filename in filenames:
            if len(files) >= MAX_SCANNED_FILES:
                break
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


def analyze_repository(
    github_url: str,
    progress_callback: Callable[[int, str], None] | None = None,
) -> dict[str, object]:
    """Clone/open a repository and return extracted files plus embedded code chunks."""
    global _dependency_graph, _active_repository

    def _report(percent: int, message: str) -> None:
        if progress_callback is not None:
            try:
                progress_callback(percent, message)
            except Exception:
                pass

    _report(10, "Cloning or opening repository from GitHub...")
    reference, repository_path = clone_or_open_repository(github_url)

    _report(25, "Scanning repository files and filtering directories...")
    files, unreadable_files, has_non_ignored_file = scan_repository(repository_path)
    if not has_non_ignored_file:
        raise EmptyRepositoryError("Repository contains no files to analyze.")
    if not files:
        raise UnsupportedFilesError("Repository contains no supported readable files.")

    _report(40, f"Discovered {len(files)} files. Parsing AST syntax trees and chunking content...")
    documentation_extensions = {".md", ".markdown", ".rst", ".txt"}
    documentation_files = sum(file["extension"] in documentation_extensions for file in files)
    chunks: list[dict[str, object]] = []
    parsing_failures: list[dict[str, str]] = []
    parsed_files = 0
    parsed_results: dict[str, dict[str, object]] = {}
    for file in files:
        language = str(file["language"])
        content_str = str(file["content"])
        path_str = str(file["path"])

        # 1. Parse AST and extract symbols for supported code languages
        if language in {"Python", "JavaScript", "JavaScript (React JSX)", "TypeScript", "TypeScript (React TSX)", "Java"}:
            try:
                parsed = parse_source(content_str, language)
                if not parsed.get("errors"):
                    parsed_files += 1
                    parsed_results[path_str] = parsed
                    chunks.extend(create_chunks(content_str, path_str, language, parsed))
                    continue
                else:
                    parsing_failures.append({"path": path_str, "reason": "parse error"})
            except Exception as error:  # A parser failure must not stop repository ingestion.
                parsing_failures.append({"path": path_str, "reason": str(error)})

        # 2. Chunk Markdown (including README.md), C++, C, JSON, Text, YAML, etc.
        try:
            chunks.extend(create_chunks(content_str, path_str, language, {"classes": [], "functions": [], "methods": [], "errors": []}))
        except Exception as error:
            parsing_failures.append({"path": path_str, "reason": str(error)})

    # Build the dependency graph from parsed AST data
    _report(60, f"Building dependency relationship graph ({parsed_files} parsed files)...")
    print(f"[Ingestion] Building dependency graph for {len(files)} files...")
    _dependency_graph = build_graph(files, parsed_results)

    # Cap chunks to keep embedding fast and memory usage minimal
    total_chunks_created = len(chunks)
    if total_chunks_created > MAX_CHUNKS_PER_REPO:
        print(f"[Ingestion] Capping chunks from {total_chunks_created} to {MAX_CHUNKS_PER_REPO} to keep embedding fast.")
        chunks = chunks[:MAX_CHUNKS_PER_REPO]

    _report(75, f"Generating vector embeddings for {len(chunks)} chunks...")
    print(f"[Ingestion] Embedding {len(chunks)} chunks with embedding service...")
    embedded_chunks = embedding_service.embed_chunks(chunks)

    # Reset collection so the vector store is clean and isolated for this repository
    _report(90, f"Indexing {len(embedded_chunks)} chunks into Qdrant vector store...")
    vector_store = get_vector_store()
    vector_store.reset_collection()

    print(f"[Ingestion] Upserting {len(embedded_chunks)} chunks to Qdrant...")
    stored_point_ids = vector_store.upsert_chunks(embedded_chunks)
    _active_repository = reference.name
    print(f"[Ingestion] Successfully ingested repository {reference.name}!")

    _report(100, f"Successfully indexed repository {reference.name}!")

    # Invalidate cached services so agent and RAG query immediately switch to the new repository
    try:
        from app.agent import get_agent_service
        from app.rag import get_rag_service, get_retriever
        get_agent_service.cache_clear()
        get_rag_service.cache_clear()
        get_retriever.cache_clear()
    except Exception:
        pass

    return {
        "repository": reference.name,
        "files_found": len(files),
        "code_files": len(files) - documentation_files,
        "documentation_files": documentation_files,
        "skipped_unreadable_files": unreadable_files,
        "parsed_files": parsed_files,
        "chunks_created": total_chunks_created,
        "chunks_stored": len(stored_point_ids),
        "graph_nodes": _dependency_graph.node_count,
        "graph_edges": _dependency_graph.edge_count,
        "parsing_failures": parsing_failures,
        "chunks": embedded_chunks,
        "files": files,
    }

"""Repository investigation tools for the LangGraph agent."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.rag.retriever import Retriever

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REPOSITORIES_DIR = PROJECT_ROOT / "data" / "repositories"


class SemanticSearchTool:
    """Tool 1: Wrap existing Retriever to search repository chunks in Qdrant with BGE-M3."""

    def __init__(self, retriever: Retriever) -> None:
        self.retriever = retriever

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Retrieve relevant repository chunks for a query using cosine similarity."""
        return self.retriever.retrieve(query=query, top_k=top_k)

    def __call__(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        return self.search(query=query, top_k=top_k)


class SourceInspectionTool:
    """Tool 2: Inspect a specific source file and line range in the cloned repository."""

    def __init__(self, base_dir: Path = DEFAULT_REPOSITORIES_DIR) -> None:
        self.base_dir = Path(base_dir)

    def _resolve_file(self, file_path: str) -> Path | None:
        """Find the target file inside the repositories directory safely."""
        norm_path = Path(file_path.strip().replace("\\", "/"))
        if norm_path.is_absolute() or ".." in norm_path.parts:
            return None

        # Direct check relative to base_dir
        direct = self.base_dir / norm_path
        if direct.is_file():
            return direct

        # Search inside subdirectories (e.g. data/repositories/owner--repo/path)
        if self.base_dir.exists():
            for repo_dir in self.base_dir.iterdir():
                if repo_dir.is_dir():
                    candidate = repo_dir / norm_path
                    if candidate.is_file():
                        return candidate

        return None

    def inspect(
        self,
        file_path: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> dict[str, Any]:
        """Read and slice lines from a repository file."""
        target_path = self._resolve_file(file_path)
        if target_path is None:
            return {
                "file": file_path,
                "found": False,
                "content": "",
                "error": f"File '{file_path}' not found in repository storage.",
            }

        try:
            content = target_path.read_text(encoding="utf-8", errors="replace")
            all_lines = content.splitlines(keepends=True)
            total_lines = len(all_lines)

            # Determine 1-indexed line bounds
            actual_start = max(1, start_line) if start_line is not None else 1
            actual_end = min(total_lines, end_line) if end_line is not None else total_lines

            if actual_start > total_lines:
                return {
                    "file": file_path,
                    "found": True,
                    "start_line": actual_start,
                    "end_line": actual_end,
                    "lines": f"{actual_start}-{actual_end}",
                    "content": "",
                    "total_lines": total_lines,
                }

            slice_lines = all_lines[actual_start - 1 : actual_end]
            return {
                "file": file_path,
                "found": True,
                "start_line": actual_start,
                "end_line": actual_end,
                "lines": f"{actual_start}-{actual_end}",
                "content": "".join(slice_lines),
                "total_lines": total_lines,
            }
        except OSError as err:
            return {
                "file": file_path,
                "found": False,
                "content": "",
                "error": f"Unable to read file '{file_path}': {err}",
            }

    def __call__(
        self,
        file_path: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> dict[str, Any]:
        return self.inspect(file_path, start_line, end_line)

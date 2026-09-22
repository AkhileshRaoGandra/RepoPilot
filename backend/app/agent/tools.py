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


class DependencyGraphTool:
    """Tool 3: Investigate repository structure via the dependency graph."""

    def __init__(self, graph: Any | None = None) -> None:
        self._graph = graph

    @property
    def graph(self) -> Any | None:
        return self._graph

    @graph.setter
    def graph(self, value: Any) -> None:
        self._graph = value

    @property
    def available(self) -> bool:
        return self._graph is not None

    def find_dependencies(self, file_path: str) -> dict[str, Any]:
        """Return files that *file_path* imports."""
        if not self.available:
            return {"file": file_path, "dependencies": [], "error": "Graph not built yet."}
        from app.graph.queries import get_dependencies, get_file_symbols

        deps = get_dependencies(self._graph, file_path)
        symbols = get_file_symbols(self._graph, file_path)
        return {
            "file": file_path,
            "dependencies": deps,
            "symbols": [
                {"name": s.get("name", ""), "type": s.get("node_type", ""), "lines": f"{s.get('start_line', '?')}-{s.get('end_line', '?')}"}
                for s in symbols
            ],
        }

    def find_dependents(self, file_path: str) -> dict[str, Any]:
        """Return files that import *file_path*."""
        if not self.available:
            return {"file": file_path, "dependents": [], "error": "Graph not built yet."}
        from app.graph.queries import get_dependents

        return {"file": file_path, "dependents": get_dependents(self._graph, file_path)}

    def find_related(self, symbol_or_file: str) -> dict[str, Any]:
        """Find symbols/files related to the given name."""
        if not self.available:
            return {"query": symbol_or_file, "related": [], "error": "Graph not built yet."}
        from app.graph.queries import get_related_symbols

        results = get_related_symbols(self._graph, symbol_or_file)
        return {
            "query": symbol_or_file,
            "related": [
                {
                    "id": r.get("id", ""),
                    "name": r.get("name", r.get("label", "")),
                    "type": r.get("node_type", ""),
                    "file": r.get("file", ""),
                    "start_line": r.get("start_line"),
                    "end_line": r.get("end_line"),
                }
                for r in results
            ],
        }

    def explore(self, node_id: str) -> dict[str, Any]:
        """Return all direct neighbors of a graph node."""
        if not self.available:
            return {"node": node_id, "neighbors": [], "error": "Graph not built yet."}
        from app.graph.queries import get_neighbors

        neighbors = get_neighbors(self._graph, node_id)
        return {
            "node": node_id,
            "neighbors": [
                {
                    "id": n.get("id", ""),
                    "name": n.get("name", n.get("label", "")),
                    "type": n.get("node_type", ""),
                    "direction": n.get("direction", ""),
                    "edge_type": n.get("edge_type", ""),
                }
                for n in neighbors
            ],
        }

    def get_import_chain(self, file_path: str, max_depth: int = 3) -> dict[str, Any]:
        """Follow imports from *file_path* up to *max_depth* hops."""
        if not self.available:
            return {"file": file_path, "chain": [], "error": "Graph not built yet."}
        from app.graph.queries import get_import_chain

        chain = get_import_chain(self._graph, file_path, max_depth=max_depth)
        return {"file": file_path, "chain": chain}

    def __call__(self, action: str, target: str, **kwargs: Any) -> dict[str, Any]:
        actions = {
            "dependencies": self.find_dependencies,
            "dependents": self.find_dependents,
            "related": self.find_related,
            "explore": self.explore,
            "import_chain": self.get_import_chain,
        }
        handler = actions.get(action)
        if handler is None:
            return {"error": f"Unknown graph action: {action}. Available: {list(actions)}"}
        return handler(target, **kwargs)


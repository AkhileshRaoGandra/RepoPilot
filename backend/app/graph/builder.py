"""Build an in-memory dependency graph from parsed repository files."""

from __future__ import annotations

import os
import re
from typing import Any

import networkx as nx

from app.graph.models import EdgeType, NodeType


class DependencyGraph:
    """Lightweight wrapper around a networkx DiGraph for repository relationships."""

    def __init__(self) -> None:
        self.graph: nx.DiGraph = nx.DiGraph()

    # ── Node helpers ──────────────────────────────────────────────────────

    def add_file_node(self, file_path: str, *, language: str = "") -> None:
        self.graph.add_node(
            file_path,
            node_type=NodeType.FILE.value,
            label=os.path.basename(file_path),
            language=language,
        )

    def add_symbol_node(
        self,
        symbol_id: str,
        *,
        node_type: NodeType,
        name: str,
        file_path: str,
        start_line: int | None = None,
        end_line: int | None = None,
        class_name: str | None = None,
    ) -> None:
        self.graph.add_node(
            symbol_id,
            node_type=node_type.value,
            name=name,
            label=name,
            file=file_path,
            start_line=start_line,
            end_line=end_line,
            class_name=class_name,
        )

    # ── Edge helpers ──────────────────────────────────────────────────────

    def add_edge(self, source: str, target: str, edge_type: EdgeType) -> None:
        self.graph.add_edge(source, target, edge_type=edge_type.value)

    # ── Accessors ─────────────────────────────────────────────────────────

    @property
    def node_count(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self.graph.number_of_edges()

    def has_node(self, node_id: str) -> bool:
        return self.graph.has_node(node_id)

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        if not self.graph.has_node(node_id):
            return None
        return dict(self.graph.nodes[node_id])

    def to_dict(self) -> dict[str, Any]:
        """Serialise the graph to a JSON-friendly dict of nodes and edges."""
        nodes = [
            {"id": nid, **data}
            for nid, data in self.graph.nodes(data=True)
        ]
        edges = [
            {"source": u, "target": v, **data}
            for u, v, data in self.graph.edges(data=True)
        ]
        return {
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "total_nodes": self.node_count,
                "total_edges": self.edge_count,
                "file_nodes": sum(1 for _, d in self.graph.nodes(data=True) if d.get("node_type") == NodeType.FILE.value),
                "symbol_nodes": sum(1 for _, d in self.graph.nodes(data=True) if d.get("node_type") != NodeType.FILE.value),
            },
        }


# ── Import resolution helpers ─────────────────────────────────────────────


def _normalise_path(path: str) -> str:
    """Normalise path separators to forward slashes."""
    return path.replace("\\", "/")


def _build_file_index(file_paths: list[str]) -> dict[str, str]:
    """Create a lookup from various import-style keys to actual file paths.

    Each file is indexed by:
    - its full relative path (normalised)
    - path without extension
    - dotted-module form (``app.auth.service`` ↔ ``app/auth/service.py``)
    - basename without extension
    """
    index: dict[str, str] = {}
    for fp in file_paths:
        norm = _normalise_path(fp)
        index[norm] = fp

        # Without extension
        no_ext, _ = os.path.splitext(norm)
        index[no_ext] = fp

        # Dotted form (for Python / Java imports)
        dotted = no_ext.replace("/", ".")
        index[dotted] = fp

        # Basename without extension (last-resort match)
        basename = os.path.basename(no_ext)
        if basename not in index:  # Don't overwrite a more specific match
            index[basename] = fp

    return index


def _resolve_import(import_str: str, file_index: dict[str, str], source_file: str) -> str | None:
    """Best-effort resolution of an import string to a file path in the repo.

    Handles:
    - Python: ``app.auth.service`` → ``app/auth/service.py``
    - Java: ``com.example.AuthService`` → any file ending with matching path
    - JS/TS: ``./auth/service`` → relative resolution + extension guessing
    """
    import_str = import_str.strip()
    if not import_str:
        return None

    # Direct match in file index
    if import_str in file_index:
        target = file_index[import_str]
        return target if target != source_file else None

    # Dotted → slash conversion for Python/Java
    slash_form = import_str.replace(".", "/")
    if slash_form in file_index:
        target = file_index[slash_form]
        return target if target != source_file else None

    # JS-style relative path resolution
    if import_str.startswith("."):
        source_dir = os.path.dirname(_normalise_path(source_file))
        candidate = os.path.normpath(os.path.join(source_dir, import_str)).replace("\\", "/")
        # Try with common extensions
        for ext in ("", ".js", ".ts", ".jsx", ".tsx", "/index.js", "/index.ts"):
            key = candidate + ext
            if key in file_index:
                target = file_index[key]
                return target if target != source_file else None

    # Try truncating rightmost segment (e.g. `app.auth.service.AuthService` → `app.auth.service`)
    parts = import_str.rsplit(".", 1)
    if len(parts) == 2:
        parent = parts[0]
        if parent in file_index:
            target = file_index[parent]
            return target if target != source_file else None
        # Also try slash form of parent
        parent_slash = parent.replace(".", "/")
        if parent_slash in file_index:
            target = file_index[parent_slash]
            return target if target != source_file else None

    return None


# ── Main builder ──────────────────────────────────────────────────────────


def build_graph(
    files: list[dict[str, Any]],
    parsed_results: dict[str, dict[str, Any]],
) -> DependencyGraph:
    """Build a dependency graph from repository files and their parsed AST data.

    Parameters
    ----------
    files:
        List of file metadata dicts as produced by ``scan_repository``.
        Each must contain at least ``"path"`` and ``"language"``.
    parsed_results:
        Mapping from file path to the output of ``parse_source()``.
        Keys that are not in *files* are silently skipped.

    Returns
    -------
    A populated :class:`DependencyGraph`.
    """
    dep_graph = DependencyGraph()
    all_paths = [str(f["path"]) for f in files]
    file_index = _build_file_index(all_paths)

    # 1. Add file nodes
    for file_info in files:
        fp = str(file_info["path"])
        dep_graph.add_file_node(fp, language=str(file_info.get("language", "")))

    # 2. Add symbols and containment edges
    for fp, parsed in parsed_results.items():
        if not dep_graph.has_node(fp):
            continue

        for cls in parsed.get("classes", []):
            cls_name = cls["name"]
            cls_id = f"{fp}::{cls_name}"
            dep_graph.add_symbol_node(
                cls_id,
                node_type=NodeType.CLASS,
                name=cls_name,
                file_path=fp,
                start_line=cls.get("start_line"),
                end_line=cls.get("end_line"),
            )
            dep_graph.add_edge(fp, cls_id, EdgeType.CONTAINS)

        for func in parsed.get("functions", []):
            func_name = func["name"]
            func_id = f"{fp}::{func_name}"
            dep_graph.add_symbol_node(
                func_id,
                node_type=NodeType.FUNCTION,
                name=func_name,
                file_path=fp,
                start_line=func.get("start_line"),
                end_line=func.get("end_line"),
            )
            dep_graph.add_edge(fp, func_id, EdgeType.CONTAINS)

        for method in parsed.get("methods", []):
            method_name = method["name"]
            class_name = method.get("class_name", "")
            method_id = f"{fp}::{class_name}.{method_name}"
            dep_graph.add_symbol_node(
                method_id,
                node_type=NodeType.METHOD,
                name=method_name,
                file_path=fp,
                start_line=method.get("start_line"),
                end_line=method.get("end_line"),
                class_name=class_name,
            )
            dep_graph.add_edge(fp, method_id, EdgeType.CONTAINS)

            # Link method to its class node if the class exists
            cls_id = f"{fp}::{class_name}"
            if dep_graph.has_node(cls_id):
                dep_graph.add_edge(cls_id, method_id, EdgeType.DEFINED_IN)

    # 3. Resolve imports and add IMPORTS edges between file nodes
    for fp, parsed in parsed_results.items():
        if not dep_graph.has_node(fp):
            continue
        for import_str in parsed.get("imports", []):
            target = _resolve_import(import_str, file_index, fp)
            if target is not None and dep_graph.has_node(target):
                dep_graph.add_edge(fp, target, EdgeType.IMPORTS)

    return dep_graph

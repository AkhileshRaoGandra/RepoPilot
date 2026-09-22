"""Query operations over the repository dependency graph."""

from __future__ import annotations

from typing import Any

from app.graph.builder import DependencyGraph
from app.graph.models import EdgeType, NodeType


def get_dependencies(graph: DependencyGraph, file_path: str) -> list[str]:
    """Return file paths that *file_path* imports (outgoing IMPORTS edges)."""
    if not graph.has_node(file_path):
        return []
    return [
        target
        for _, target, data in graph.graph.out_edges(file_path, data=True)
        if data.get("edge_type") == EdgeType.IMPORTS.value
    ]


def get_dependents(graph: DependencyGraph, file_path: str) -> list[str]:
    """Return file paths that import *file_path* (incoming IMPORTS edges)."""
    if not graph.has_node(file_path):
        return []
    return [
        source
        for source, _, data in graph.graph.in_edges(file_path, data=True)
        if data.get("edge_type") == EdgeType.IMPORTS.value
    ]


def get_file_symbols(graph: DependencyGraph, file_path: str) -> list[dict[str, Any]]:
    """Return symbols (classes, functions, methods) defined in *file_path*."""
    if not graph.has_node(file_path):
        return []
    symbols: list[dict[str, Any]] = []
    for _, target, data in graph.graph.out_edges(file_path, data=True):
        if data.get("edge_type") == EdgeType.CONTAINS.value:
            node_data = graph.get_node(target)
            if node_data:
                symbols.append({"id": target, **node_data})
    return symbols


def get_neighbors(graph: DependencyGraph, node_id: str) -> list[dict[str, Any]]:
    """Return all directly connected nodes with edge information."""
    if not graph.has_node(node_id):
        return []

    neighbors: list[dict[str, Any]] = []
    seen: set[str] = set()

    # Outgoing edges
    for _, target, data in graph.graph.out_edges(node_id, data=True):
        if target not in seen:
            seen.add(target)
            node_data = graph.get_node(target) or {}
            neighbors.append({
                "id": target,
                "direction": "outgoing",
                "edge_type": data.get("edge_type", ""),
                **node_data,
            })

    # Incoming edges
    for source, _, data in graph.graph.in_edges(node_id, data=True):
        if source not in seen:
            seen.add(source)
            node_data = graph.get_node(source) or {}
            neighbors.append({
                "id": source,
                "direction": "incoming",
                "edge_type": data.get("edge_type", ""),
                **node_data,
            })

    return neighbors


def get_related_symbols(graph: DependencyGraph, symbol_name: str) -> list[dict[str, Any]]:
    """Find nodes whose *name* or *label* matches *symbol_name* (case-insensitive).

    For each match, also return the file it belongs to and sibling symbols in
    that file, so the caller can understand the surrounding context.
    """
    results: list[dict[str, Any]] = []
    query_lower = symbol_name.lower()

    for nid, data in graph.graph.nodes(data=True):
        node_name = str(data.get("name", data.get("label", ""))).lower()
        if query_lower in node_name or node_name in query_lower:
            entry: dict[str, Any] = {"id": nid, **data}
            # If it's a symbol node, include the owning file
            file_path = data.get("file")
            if file_path:
                entry["file"] = file_path
            results.append(entry)

    return results


def get_import_chain(
    graph: DependencyGraph,
    start_file: str,
    max_depth: int = 4,
) -> list[dict[str, Any]]:
    """Follow IMPORTS edges from *start_file* up to *max_depth* hops.

    Returns an ordered list of ``{"file": ..., "depth": ...}`` entries
    representing the dependency chain.
    """
    if not graph.has_node(start_file):
        return []

    chain: list[dict[str, Any]] = []
    visited: set[str] = {start_file}
    frontier = [start_file]

    for depth in range(1, max_depth + 1):
        next_frontier: list[str] = []
        for fp in frontier:
            for dep in get_dependencies(graph, fp):
                if dep not in visited:
                    visited.add(dep)
                    chain.append({"file": dep, "depth": depth, "imported_by": fp})
                    next_frontier.append(dep)
        frontier = next_frontier
        if not frontier:
            break

    return chain

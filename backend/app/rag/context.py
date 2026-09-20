"""Format retrieved code chunks into structured context and citations for the LLM."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def format_symbol_name(metadata: Mapping[str, Any]) -> str:
    """Format symbol name with its enclosing class and type indicator."""
    name = str(metadata.get("name", "anonymous"))
    symbol_type = str(metadata.get("type", "")).lower()
    class_name = metadata.get("class_name")

    if class_name:
        symbol = f"{class_name}.{name}"
    else:
        symbol = name

    if symbol_type in {"function", "method"} and not symbol.endswith("()"):
        return f"{symbol}()"
    return symbol


def format_lines_range(metadata: Mapping[str, Any]) -> str:
    """Format line numbers as start_line-end_line or single line."""
    start = metadata.get("start_line")
    end = metadata.get("end_line")
    if start is not None and end is not None:
        return f"{start}-{end}" if start != end else str(start)
    if start is not None:
        return str(start)
    return "Unknown"


def format_citation(metadata: Mapping[str, Any]) -> str:
    """Format a standard file:line citation (e.g. auth/service.py:25-48)."""
    file_path = str(metadata.get("file", "unknown"))
    lines = format_lines_range(metadata)
    return f"{file_path}:{lines}" if lines != "Unknown" else file_path


def extract_evidence(chunks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Extract structured evidence citations and metadata from retrieved chunks."""
    evidence: list[dict[str, Any]] = []
    seen: set[str] = set()

    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        citation = format_citation(metadata)
        if citation in seen:
            continue
        seen.add(citation)

        evidence.append(
            {
                "file": str(metadata.get("file", "unknown")),
                "symbol": format_symbol_name(metadata),
                "start_line": metadata.get("start_line"),
                "end_line": metadata.get("end_line"),
                "lines": format_lines_range(metadata),
                "citation": citation,
                "score": chunk.get("score"),
            }
        )
    return evidence


def build_rag_context(chunks: Sequence[Mapping[str, Any]]) -> str:
    """Construct a clear, structured context string from retrieved chunks for the LLM.

    Each block provides the file path, symbol identity, line span, and exact source code.
    """
    if not chunks:
        return ""

    blocks: list[str] = []
    for chunk in chunks:
        content = str(chunk.get("content", "")).strip()
        metadata = chunk.get("metadata", {})
        file_path = str(metadata.get("file", "Unknown"))
        symbol = format_symbol_name(metadata)
        lines = format_lines_range(metadata)

        header = f"File: {file_path}\nSymbol: {symbol}\nLines: {lines}"
        blocks.append(f"{header}\n\n{content}")

    return "\n\n---\n\n".join(blocks)

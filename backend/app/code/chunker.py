"""Create bounded, code-aware chunks from Tree-sitter parser output."""

from __future__ import annotations

from typing import Any


DEFAULT_MAX_LINES = 200


def _split_content(content: str, max_lines: int) -> list[tuple[int, str]]:
    lines = content.splitlines(keepends=True)
    return [
        (offset, "".join(lines[offset : offset + max_lines]))
        for offset in range(0, len(lines), max_lines)
    ] or [(0, "")]


def create_chunks(source: str, file_path: str, language: str, parsed: dict[str, object], max_lines: int = DEFAULT_MAX_LINES) -> list[dict[str, object]]:
    """Build chunks at class/function/method boundaries, splitting only large nodes."""
    if max_lines < 1:
        raise ValueError("max_lines must be at least 1")

    source_lines = source.splitlines(keepends=True)
    chunks: list[dict[str, object]] = []
    symbols: list[dict[str, Any]] = [
        *parsed.get("classes", []),
        *parsed.get("functions", []),
        *parsed.get("methods", []),
    ]
    for symbol in symbols:
        start_line, end_line = symbol["start_line"], symbol["end_line"]
        content = "".join(source_lines[start_line - 1 : end_line])
        sections = _split_content(content, max_lines)
        for index, (offset, section) in enumerate(sections, start=1):
            metadata = {
                "file": file_path,
                "language": language,
                "type": symbol["type"],
                "name": symbol["name"],
                "start_line": start_line + offset,
                "end_line": min(end_line, start_line + offset + max_lines - 1),
            }
            if "class_name" in symbol:
                metadata["class_name"] = symbol["class_name"]
            if len(sections) > 1:
                metadata.update({"part": index, "parts": len(sections)})
            chunks.append({"content": section, "metadata": metadata})
    return chunks

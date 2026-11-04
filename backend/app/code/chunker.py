"""Create bounded, code-aware and documentation-aware chunks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

DEFAULT_MAX_LINES = 200


def _split_content(content: str, max_lines: int) -> list[tuple[int, str]]:
    lines = content.splitlines(keepends=True)
    return [
        (offset, "".join(lines[offset : offset + max_lines]))
        for offset in range(0, len(lines), max_lines)
    ] or [(0, "")]


def create_markdown_chunks(
    source: str,
    file_path: str,
    max_lines: int = DEFAULT_MAX_LINES,
) -> list[dict[str, object]]:
    """Split markdown / documentation by sections and headers (# , ## , ### )."""
    lines = source.splitlines(keepends=True)
    if not lines or not source.strip():
        return []

    doc_name = Path(file_path).name
    sections: list[dict[str, Any]] = []
    current_title = f"{doc_name} Overview"
    current_lines: list[str] = []
    current_start = 1

    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith("#"):
            if current_lines:
                sec_content = "".join(current_lines)
                if sec_content.strip():
                    sections.append({
                        "name": current_title,
                        "start_line": current_start,
                        "end_line": idx - 1,
                        "content": sec_content,
                    })
                current_lines = []
            current_title = stripped.lstrip("#").strip() or f"{doc_name} Section"
            current_start = idx
        current_lines.append(line)

    if current_lines:
        sec_content = "".join(current_lines)
        if sec_content.strip():
            sections.append({
                "name": current_title,
                "start_line": current_start,
                "end_line": len(lines),
                "content": sec_content,
            })

    if not sections:
        sections = [{
            "name": doc_name,
            "start_line": 1,
            "end_line": len(lines),
            "content": source,
        }]

    chunks: list[dict[str, object]] = []
    for sec in sections:
        content = sec["content"]
        sec_splits = _split_content(content, max_lines)
        for part_idx, (offset, part_content) in enumerate(sec_splits, start=1):
            if not part_content.strip():
                continue
            meta = {
                "file": file_path,
                "language": "Markdown",
                "type": "documentation",
                "name": sec["name"],
                "start_line": sec["start_line"] + offset,
                "end_line": min(sec["end_line"], sec["start_line"] + offset + max_lines - 1),
            }
            if len(sec_splits) > 1:
                meta.update({"part": part_idx, "parts": len(sec_splits)})
            chunks.append({"content": part_content, "metadata": meta})

    return chunks


def create_chunks(
    source: str,
    file_path: str,
    language: str,
    parsed: dict[str, object],
    max_lines: int = DEFAULT_MAX_LINES,
) -> list[dict[str, object]]:
    """Build chunks at class/function/method boundaries, splitting only large nodes."""
    if max_lines < 1:
        raise ValueError("max_lines must be at least 1")

    # If the file is Markdown documentation, use section-aware chunking
    if language.lower() in {"markdown", "documentation"}:
        return create_markdown_chunks(source, file_path, max_lines)

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

    # If no functions or classes exist in the file, create module/file/config chunks
    # so scripts, settings, constants, JSON, YAML, C/C++ files, and entry points are not silently lost.
    if not symbols and source.strip():
        stem_name = Path(file_path).stem or "module"
        sections = _split_content(source, max_lines)
        total_lines = len(source_lines)
        if language.lower() in {"json", "yaml", "toml", "ini"}:
            chunk_type = "config"
        elif language.lower() in {"text"}:
            chunk_type = "text"
        else:
            chunk_type = "module"
        for index, (offset, section) in enumerate(sections, start=1):
            if not section.strip():
                continue
            metadata = {
                "file": file_path,
                "language": language,
                "type": chunk_type,
                "name": stem_name,
                "start_line": offset + 1,
                "end_line": min(total_lines, offset + max_lines),
            }
            if len(sections) > 1:
                metadata.update({"part": index, "parts": len(sections)})
            chunks.append({"content": section, "metadata": metadata})

    return chunks

"""Tree-sitter based structural parsing for Python, JavaScript, and Java."""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any

from tree_sitter import Language, Parser
import tree_sitter_java as tsjava
import tree_sitter_javascript as tsjavascript
import tree_sitter_python as tspython


class ParserError(Exception):
    """Base error for source parsing."""


class UnsupportedLanguageError(ParserError):
    """Raised when no Tree-sitter grammar is available for a language."""


_LANGUAGES = {
    "python": tspython,
    "javascript": tsjavascript,
    "java": tsjava,
}


def _normalise_language(language: str) -> str:
    normalised = language.lower().strip()
    aliases = {"javascript (react jsx)": "javascript", "jsx": "javascript"}
    return aliases.get(normalised, normalised)


def _parser_for(language: str) -> Parser:
    grammar = _LANGUAGES.get(_normalise_language(language))
    if grammar is None:
        raise UnsupportedLanguageError(f"Tree-sitter does not support language: {language}")

    parser = Parser()
    # Current grammar wheels expose a PyCapsule.  The fallback keeps the module
    # compatible with earlier Tree-sitter Python bindings.
    grammar_language = Language(grammar.language())
    try:
        parser.language = grammar_language
    except AttributeError:  # pragma: no cover - legacy binding compatibility
        parser.set_language(grammar_language)
    return parser


def _walk(node: Any) -> Iterator[Any]:
    yield node
    for child in node.children:
        yield from _walk(child)


def _text(node: Any, source_bytes: bytes) -> str:
    return source_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _name(node: Any, source_bytes: bytes) -> str | None:
    name_node = node.child_by_field_name("name")
    return _text(name_node, source_bytes) if name_node is not None else None


def _symbol(node: Any, name: str, symbol_type: str, source_bytes: bytes, **extra: str) -> dict[str, object]:
    return {
        "name": name,
        "type": symbol_type,
        "start_line": node.start_point.row + 1,
        "end_line": node.end_point.row + 1,
        **extra,
    }


def _ancestor_class(node: Any, parents: dict[int, Any]) -> Any | None:
    current = parents.get(id(node))
    while current is not None:
        if current.type in {"class_definition", "class_declaration"}:
            return current
        current = parents.get(id(current))
    return None


def _python_imports(root: Any, source_bytes: bytes) -> list[str]:
    imports: list[str] = []
    for node in _walk(root):
        text = _text(node, source_bytes).strip()
        if node.type == "import_statement":
            for item in text.removeprefix("import ").split(","):
                imports.append(item.strip().split(" as ")[0].strip())
        elif node.type == "import_from_statement":
            match = re.match(r"from\s+([.\w]+)\s+import\s+(.+)", text, re.DOTALL)
            if match:
                module, names = match.groups()
                for item in names.split(","):
                    imported = item.strip().split(" as ")[0].strip()
                    imports.append(module if imported == "*" else f"{module}.{imported}")
    return imports


def _imports(language: str, root: Any, source_bytes: bytes) -> list[str]:
    if language == "python":
        return _python_imports(root, source_bytes)

    imports: list[str] = []
    for node in _walk(root):
        if language == "javascript" and node.type == "import_statement":
            text = _text(node, source_bytes)
            match = re.search(r"(?:from\s+)?['\"]([^'\"]+)['\"]", text)
            if match:
                imports.append(match.group(1))
        elif language == "java" and node.type == "import_declaration":
            text = _text(node, source_bytes).strip().removeprefix("import ").removeprefix("static ").rstrip(";")
            imports.append(text)
    return imports


def _javascript_variable_function(node: Any, source_bytes: bytes) -> tuple[Any, str] | None:
    """Return the function node and variable name for const name = () => {}."""
    if node.type != "variable_declarator":
        return None
    value = node.child_by_field_name("value")
    name = node.child_by_field_name("name")
    if value is not None and name is not None and value.type in {"arrow_function", "function_expression"}:
        return value, _text(name, source_bytes)
    return None


def parse_source(source: str, language: str) -> dict[str, object]:
    """Parse source and return classes, functions, methods, imports, and errors.

    Source coordinates are one-based and end lines are inclusive, matching the
    lines copied by :mod:`app.code.chunker`.
    """
    normalised = _normalise_language(language)
    parser = _parser_for(normalised)
    source_bytes = source.encode("utf-8")
    root = parser.parse(source_bytes).root_node
    parents = {id(child): parent for parent in _walk(root) for child in parent.children}

    classes: list[dict[str, object]] = []
    functions: list[dict[str, object]] = []
    methods: list[dict[str, object]] = []

    for node in _walk(root):
        if node.type in {"class_definition", "class_declaration"}:
            name = _name(node, source_bytes)
            if name:
                classes.append(_symbol(node, name, "class", source_bytes))
            continue

        is_function = node.type == "function_definition" or (
            normalised == "javascript" and node.type == "function_declaration"
        )
        variable_function = _javascript_variable_function(node, source_bytes) if normalised == "javascript" else None
        is_method = node.type == "method_definition" or (
            normalised == "java" and node.type in {"method_declaration", "constructor_declaration"}
        )

        if is_function:
            name = _name(node, source_bytes)
            if not name:
                continue
            class_node = _ancestor_class(node, parents)
            if class_node is None:
                functions.append(_symbol(node, name, "function", source_bytes))
            else:
                methods.append(_symbol(node, name, "method", source_bytes, class_name=_name(class_node, source_bytes) or "<anonymous>"))
        elif variable_function is not None:
            function_node, name = variable_function
            functions.append(_symbol(function_node, name, "function", source_bytes))
        elif is_method:
            name = _name(node, source_bytes)
            class_node = _ancestor_class(node, parents)
            if name and class_node is not None:
                methods.append(_symbol(node, name, "method", source_bytes, class_name=_name(class_node, source_bytes) or "<anonymous>"))

    errors = [
        {"start_line": node.start_point.row + 1, "end_line": node.end_point.row + 1, "reason": node.type}
        for node in _walk(root)
        if node.type == "ERROR" or node.is_missing
    ]
    return {
        "language": normalised,
        "classes": classes,
        "functions": functions,
        "methods": methods,
        "imports": list(dict.fromkeys(item for item in _imports(normalised, root, source_bytes) if item)),
        "errors": errors,
    }

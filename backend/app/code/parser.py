"""Tree-sitter and AST based structural parsing for Python, JavaScript, and Java."""

from __future__ import annotations

import ast
import bisect
import re
from typing import Any

from tree_sitter import Language, Parser
import tree_sitter_java as tsjava
import tree_sitter_javascript as tsjavascript
import tree_sitter_python as tspython

try:
    import tree_sitter_cpp as tscpp
except ImportError:  # pragma: no cover
    tscpp = None


class ParserError(Exception):
    """Base error for source parsing."""


class UnsupportedLanguageError(ParserError):
    """Raised when no parser grammar is available for a language."""


_LANGUAGES = {
    "python": tspython,
    "javascript": tsjavascript,
    "java": tsjava,
}
if tscpp is not None:
    _LANGUAGES["cpp"] = tscpp

_CONTAINERS = {
    "program",
    "translation_unit",
    "export_statement",
    "statement_block",
    "class_declaration",
    "class_definition",
    "class_specifier",
    "struct_specifier",
    "class_body",
    "field_declaration_list",
    "function_declaration",
    "function_definition",
    "method_definition",
    "variable_declaration",
    "variable_declarator",
    "lexical_declaration",
    "expression_statement",
    "package_declaration",
    "import_declaration",
    "interface_declaration",
    "method_declaration",
    "constructor_declaration",
    "block",
    "compound_statement",
    "namespace_definition",
    "declaration",
}


def _normalise_language(language: str) -> str:
    normalised = language.lower().strip()
    aliases = {
        "javascript (react jsx)": "javascript",
        "jsx": "javascript",
        "typescript": "javascript",
        "typescript (react tsx)": "javascript",
        "tsx": "javascript",
        "c++": "cpp",
        "cpp": "cpp",
        "c": "cpp",
        "c/c++ header": "cpp",
        "c++ header": "cpp",
    }
    return aliases.get(normalised, normalised)


def _parser_for(language: str) -> Parser:
    grammar = _LANGUAGES.get(_normalise_language(language))
    if grammar is None:
        raise UnsupportedLanguageError(f"Tree-sitter does not support language: {language}")

    parser = Parser()
    grammar_language = Language(grammar.language())
    try:
        parser.language = grammar_language
    except AttributeError:  # pragma: no cover - legacy binding compatibility
        parser.set_language(grammar_language)
    return parser


def _walk(node: Any) -> list[Any]:
    """Collect relevant container and declaration AST nodes into a flat list."""
    nodes: list[Any] = [node]
    for current in nodes:
        if current.type in _CONTAINERS:
            nodes.extend(current.children)
    return nodes


def _text(node: Any, source_bytes: bytes) -> str:
    return source_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _name(node: Any, source_bytes: bytes) -> str | None:
    name_node = node.child_by_field_name("name")
    if name_node is not None:
        return _text(name_node, source_bytes)
    # C/C++ function definitions identify name via declarator tree
    if node.type == "function_definition":
        decl = node.child_by_field_name("declarator")
        while decl is not None and decl.type in {"function_declarator", "pointer_declarator", "reference_declarator"}:
            inner = decl.child_by_field_name("declarator")
            if inner is None:
                break
            decl = inner
        if decl is not None:
            return _text(decl, source_bytes)
    return None


def _javascript_variable_function(node: Any, source_bytes: bytes) -> tuple[Any, str] | None:
    """Return the function node and variable name for const name = () => {}."""
    if node.type != "variable_declarator":
        return None
    value = node.child_by_field_name("value")
    name = node.child_by_field_name("name")
    if value is not None and name is not None and value.type in {"arrow_function", "function_expression"}:
        return value, _text(name, source_bytes)
    return None


def _parse_python(source: str) -> dict[str, object]:
    """Parse Python source using Python's native AST parser with 100% C-level memory safety."""
    try:
        tree = ast.parse(source)
    except SyntaxError as err:
        return {
            "language": "python",
            "classes": [],
            "functions": [],
            "methods": [],
            "imports": [],
            "errors": [
                {
                    "start_line": err.lineno or 1,
                    "end_line": err.end_lineno or err.lineno or 1,
                    "reason": err.msg or "syntax error",
                }
            ],
        }

    classes: list[dict[str, object]] = []
    functions: list[dict[str, object]] = []
    methods: list[dict[str, object]] = []
    imports: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            classes.append({
                "name": node.name,
                "type": "class",
                "start_line": node.lineno,
                "end_line": node.end_lineno or node.lineno,
            })
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods.append({
                        "name": item.name,
                        "type": "method",
                        "start_line": item.lineno,
                        "end_line": item.end_lineno or item.lineno,
                        "class_name": node.name,
                    })
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for alias in node.names:
                imports.append(mod if alias.name == "*" else f"{mod}.{alias.name}")

    class_spans = [(c["start_line"], c["end_line"]) for c in classes]
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            in_class = any(s <= node.lineno <= e for s, e in class_spans)
            if not in_class:
                functions.append({
                    "name": node.name,
                    "type": "function",
                    "start_line": node.lineno,
                    "end_line": node.end_lineno or node.lineno,
                })

    classes.sort(key=lambda x: int(x["start_line"]))
    functions.sort(key=lambda x: int(x["start_line"]))
    methods.sort(key=lambda x: int(x["start_line"]))

    return {
        "language": "python",
        "classes": classes,
        "functions": functions,
        "methods": methods,
        "imports": list(dict.fromkeys(item for item in imports if item)),
        "errors": [],
    }


def parse_source(source: str, language: str) -> dict[str, object]:
    """Parse source and return classes, functions, methods, imports, and errors.

    Source coordinates are one-based and end lines are inclusive, matching the
    lines copied by :mod:`app.code.chunker`.
    """
    normalised = _normalise_language(language)
    if normalised == "python":
        return _parse_python(source)

    parser = _parser_for(normalised)
    source_bytes = source.encode("utf-8")
    tree = parser.parse(source_bytes)
    root = tree.root_node

    line_offsets = [0] + [m.end() for m in re.finditer(rb"\n", source_bytes)]

    def line_num(byte_offset: int) -> int:
        return bisect.bisect_right(line_offsets, byte_offset)

    classes: list[dict[str, object]] = []
    functions: list[dict[str, object]] = []
    methods: list[dict[str, object]] = []
    imports: list[str] = []
    errors: list[dict[str, object]] = []

    try:
        nodes = _walk(root)

        for node in nodes:
            if node.type in {"class_definition", "class_declaration", "class_specifier", "struct_specifier"}:
                name = _name(node, source_bytes)
                if name:
                    classes.append({
                        "name": name,
                        "type": "class",
                        "start_line": line_num(node.start_byte),
                        "end_line": line_num(node.end_byte),
                    })

        def _enclosing_class(node: Any) -> str | None:
            nr1 = line_num(node.start_byte)
            nr2 = line_num(node.end_byte)
            best_name = None
            best_span = float("inf")
            for c in classes:
                cr1 = int(c["start_line"])
                cr2 = int(c["end_line"])
                if cr1 <= nr1 and nr2 <= cr2:
                    span = cr2 - cr1
                    if span < best_span:
                        best_span = span
                        best_name = str(c["name"])
            return best_name

        for node in nodes:
            if node.type in {"class_definition", "class_declaration", "class_specifier", "struct_specifier"}:
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
                cls = _enclosing_class(node)
                entry = {
                    "name": name,
                    "start_line": line_num(node.start_byte),
                    "end_line": line_num(node.end_byte),
                }
                if cls is None:
                    entry["type"] = "function"
                    functions.append(entry)
                else:
                    entry["type"] = "method"
                    entry["class_name"] = cls
                    methods.append(entry)

            elif variable_function is not None:
                function_node, name = variable_function
                cls = _enclosing_class(function_node)
                entry = {
                    "name": name,
                    "start_line": line_num(function_node.start_byte),
                    "end_line": line_num(function_node.end_byte),
                }
                if cls is None:
                    entry["type"] = "function"
                    functions.append(entry)
                else:
                    entry["type"] = "method"
                    entry["class_name"] = cls
                    methods.append(entry)

            elif is_method:
                name = _name(node, source_bytes)
                if name:
                    cls = _enclosing_class(node)
                    methods.append({
                        "name": name,
                        "type": "method",
                        "start_line": line_num(node.start_byte),
                        "end_line": line_num(node.end_byte),
                        "class_name": cls or "<anonymous>",
                    })

        for node in nodes:
            if normalised == "javascript" and node.type == "import_statement":
                text = _text(node, source_bytes)
                match = re.search(r"(?:from\s+)?['\"]([^'\"]+)['\"]", text)
                if match:
                    imports.append(match.group(1))
            elif normalised == "java" and node.type == "import_declaration":
                text = _text(node, source_bytes).strip().removeprefix("import ").removeprefix("static ").rstrip(";")
                imports.append(text)
            elif normalised == "cpp" and node.type == "preproc_include":
                path_node = node.child_by_field_name("path")
                if path_node is not None:
                    inc_text = _text(path_node, source_bytes).strip('<>"')
                    imports.append(inc_text)

        if root.has_error:
            for node in nodes:
                if node.type == "ERROR" or node.is_missing:
                    errors.append({
                        "start_line": line_num(node.start_byte),
                        "end_line": line_num(node.end_byte),
                        "reason": node.type,
                    })
    finally:
        del nodes
        del root
        del tree

    return {
        "language": normalised,
        "classes": classes,
        "functions": functions,
        "methods": methods,
        "imports": list(dict.fromkeys(item for item in imports if item)),
        "errors": errors,
    }

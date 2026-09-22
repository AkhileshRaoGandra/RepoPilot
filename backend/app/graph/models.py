"""Node and edge type definitions for the repository dependency graph."""

from __future__ import annotations

from enum import Enum


class NodeType(str, Enum):
    """Types of nodes in the dependency graph."""

    FILE = "file"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"


class EdgeType(str, Enum):
    """Types of edges (relationships) in the dependency graph."""

    IMPORTS = "imports"          # File A imports File B
    CONTAINS = "contains"        # File contains a symbol (class/function/method)
    DEFINED_IN = "defined_in"    # Method is defined inside a class

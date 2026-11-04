import pytest

from app.code.chunker import create_chunks
from app.code.parser import UnsupportedLanguageError, parse_source


PYTHON_SOURCE = '''import jwt
from database import Database

class AuthService:
    def login(self, token: str):
        return token

    def logout(self):
        return None

def validate_token(token):
    return bool(token)
'''

JAVASCRIPT_SOURCE = '''import React from "react";
import { helper } from "./helper";

export class UserService {
  save(user) {
    return user;
  }
}

export function validateUser(user) {
  return !!user;
}

export const formatUser = (user) => user.name;
'''

JAVA_SOURCE = '''import java.util.List;
import com.example.User;

public class UserService {
    public User createUser(String name) {
        return new User();
    }

    public void deleteUser(User user) {
    }
}
'''


def test_python_classes_functions_methods_imports_and_lines() -> None:
    parsed = parse_source(PYTHON_SOURCE, "Python")
    assert parsed["classes"] == [{"name": "AuthService", "type": "class", "start_line": 4, "end_line": 9}]
    assert [item["name"] for item in parsed["functions"]] == ["validate_token"]
    assert [item["name"] for item in parsed["methods"]] == ["login", "logout"]
    assert parsed["methods"][0]["start_line"] == 5
    assert parsed["methods"][0]["end_line"] == 6
    assert parsed["imports"] == ["jwt", "database.Database"]
    assert parsed["errors"] == []


def test_javascript_classes_functions_methods_and_imports() -> None:
    parsed = parse_source(JAVASCRIPT_SOURCE, "JavaScript")
    assert [item["name"] for item in parsed["classes"]] == ["UserService"]
    assert [item["name"] for item in parsed["methods"]] == ["save"]
    assert [item["name"] for item in parsed["functions"]] == ["validateUser", "formatUser"]
    assert parsed["imports"] == ["react", "./helper"]


def test_java_classes_methods_and_imports() -> None:
    parsed = parse_source(JAVA_SOURCE, "Java")
    assert [item["name"] for item in parsed["classes"]] == ["UserService"]
    assert [item["name"] for item in parsed["methods"]] == ["createUser", "deleteUser"]
    assert parsed["imports"] == ["java.util.List", "com.example.User"]


def test_chunk_metadata_and_large_symbol_splitting() -> None:
    source = "def long_task():\n" + "".join(f"    step_{number}()\n" for number in range(5))
    parsed = parse_source(source, "Python")
    chunks = create_chunks(source, "tasks.py", "Python", parsed, max_lines=2)
    assert len(chunks) == 3
    assert chunks[0]["metadata"] == {
        "file": "tasks.py", "language": "Python", "type": "function", "name": "long_task",
        "start_line": 1, "end_line": 2, "part": 1, "parts": 3,
    }
    assert chunks[-1]["metadata"]["end_line"] == 6


def test_unsupported_language_is_explicit() -> None:
    with pytest.raises(UnsupportedLanguageError):
        parse_source("package main", "Go")


def test_markdown_and_readme_chunking() -> None:
    readme_content = """# RepoPilot
An AI onboarding assistant.

## Architecture
The system contains a FastAPI backend and a React frontend.

## Installation
Run pip install -r requirements.txt
"""
    chunks = create_chunks(readme_content, "README.md", "Markdown", {})
    assert len(chunks) >= 3
    section_names = [c["metadata"]["name"] for c in chunks]
    assert "RepoPilot" in section_names
    assert "Architecture" in section_names
    assert "Installation" in section_names
    for c in chunks:
        assert c["metadata"]["language"] == "Markdown"
        assert c["metadata"]["type"] == "documentation"
        assert c["metadata"]["file"] == "README.md"


def test_cpp_json_txt_chunking() -> None:
    cpp_source = """#include <iostream>
int main() {
    std::cout << "Hello C++" << std::endl;
    return 0;
}
"""
    chunks_cpp = create_chunks(cpp_source, "main.cpp", "C++", {})
    assert len(chunks_cpp) >= 1
    assert chunks_cpp[0]["metadata"]["language"] == "C++"
    assert chunks_cpp[0]["metadata"]["type"] == "module"

    json_source = '{"name": "repopilot", "version": "1.0.0"}'
    chunks_json = create_chunks(json_source, "package.json", "JSON", {})
    assert len(chunks_json) >= 1
    assert chunks_json[0]["metadata"]["language"] == "JSON"
    assert chunks_json[0]["metadata"]["type"] == "config"

    txt_source = "System requirements:\n- Python 3.12+\n- Node.js 20+\n"
    chunks_txt = create_chunks(txt_source, "requirements.txt", "Text", {})
    assert len(chunks_txt) >= 1
    assert chunks_txt[0]["metadata"]["language"] == "Text"
    assert chunks_txt[0]["metadata"]["type"] == "text"


def test_cpp_ast_parsing_classes_methods_and_functions() -> None:
    cpp_source = """#include <iostream>
#include "calculator.h"

class Calculator {
public:
    int add(int a, int b) {
        return a + b;
    }
};

int main() {
    Calculator calc;
    return calc.add(2, 3);
}
"""
    parsed = parse_source(cpp_source, "C++")
    assert parsed["errors"] == []
    assert len(parsed["classes"]) == 1
    assert parsed["classes"][0]["name"] == "Calculator"
    assert len(parsed["methods"]) == 1
    assert parsed["methods"][0]["name"] == "add"
    assert parsed["methods"][0]["class_name"] == "Calculator"
    assert len(parsed["functions"]) == 1
    assert parsed["functions"][0]["name"] == "main"
    assert "iostream" in parsed["imports"]
    assert "calculator.h" in parsed["imports"]

    # Verify AST chunking on C++ parsed output
    chunks = create_chunks(cpp_source, "calculator.cpp", "C++", parsed)
    types = [c["metadata"]["type"] for c in chunks]
    names = [c["metadata"]["name"] for c in chunks]
    assert "Calculator" in names
    assert "add" in names
    assert "main" in names
    assert "class" in types
    assert "method" in types
    assert "function" in types


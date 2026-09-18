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

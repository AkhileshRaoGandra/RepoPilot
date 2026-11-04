from pathlib import Path

import pytest

from app.ingestion.repository import (
    IGNORED_DIRECTORIES,
    InvalidGitHubUrlError,
    detect_language,
    extract_file_metadata,
    parse_github_url,
    scan_repository,
)


def test_valid_github_repository_url() -> None:
    reference = parse_github_url("https://github.com/octocat/Hello-World.git")
    assert reference.owner == "octocat"
    assert reference.name == "Hello-World"


def test_invalid_github_repository_url() -> None:
    with pytest.raises(InvalidGitHubUrlError):
        parse_github_url("https://gitlab.com/octocat/Hello-World")


def test_language_detection() -> None:
    assert detect_language(Path("component.tsx")) == "TypeScript (React TSX)"
    assert detect_language(Path("notes.md")) == "Markdown"
    assert detect_language(Path("README")) == "Markdown"
    assert detect_language(Path("main.cpp")) == "C++"
    assert detect_language(Path("config.json")) == "JSON"
    assert detect_language(Path("info.txt")) == "Text"
    assert detect_language(Path("image.png")) == "Unknown"


def test_file_filtering_ignored_directories_and_metadata(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hello')\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# Demo\n", encoding="utf-8")
    (tmp_path / ".env").write_text("TOKEN=secret", encoding="utf-8")
    (tmp_path / "notes.bin").write_text("ignored", encoding="utf-8")
    (tmp_path / "node_modules" / "package.js").write_text("ignored", encoding="utf-8")

    files, unreadable, has_files = scan_repository(tmp_path)

    assert IGNORED_DIRECTORIES.intersection({"node_modules"})
    assert has_files is True
    assert unreadable == 0
    assert [file["path"] for file in files] == ["README.md", "src/main.py"]
    python_file = next(file for file in files if file["path"] == "src/main.py")
    assert python_file["language"] == "Python"
    assert python_file["extension"] == ".py"
    assert python_file["size"] == (tmp_path / "src" / "main.py").stat().st_size
    assert python_file["lines"] == 1
    assert python_file["content"] == "print('hello')\n"


def test_undecodable_file_is_skipped(tmp_path: Path) -> None:
    binary_python = tmp_path / "bad.py"
    binary_python.write_bytes(b"\xff\xfe")
    assert extract_file_metadata(tmp_path, binary_python) is None


def test_serverless_git_suppression_and_clone_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify that when Git is unavailable, clone_or_open_repository falls back to zipball download."""
    import app.ingestion.repository as repo_mod

    # Simulate Repo being None (git not installed on serverless)
    monkeypatch.setattr(repo_mod, "Repo", None)

    # Mock _download_and_extract_zipball to create sample file in destination
    def mock_download(ref, dest):
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "main.py").write_text("print('fallback works')", encoding="utf-8")
        return dest

    monkeypatch.setattr(repo_mod, "_download_and_extract_zipball", mock_download)

    ref, dest = repo_mod.clone_or_open_repository("https://github.com/octocat/Hello-World", repositories_dir=tmp_path)
    assert ref.name == "Hello-World"
    assert (dest / "main.py").exists()
    assert (dest / "main.py").read_text() == "print('fallback works')"

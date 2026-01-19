# tests/utils/test_directory_watcher.py
"""Tests for directory watcher utilities."""

import os
import pytest
from pathlib import Path

from app.utils.directory_watcher import (
    generate_file_id,
    compute_file_hash,
    get_file_metadata,
    is_supported_file,
    should_ignore,
    discover_files,
    SUPPORTED_EXTENSIONS,
)


class TestFileIdGeneration:
    """Tests for generate_file_id function."""

    def test_deterministic_file_id(self):
        """Same filepath + entity produces same file_id."""
        id1 = generate_file_id("/path/to/file.pdf", "user123")
        id2 = generate_file_id("/path/to/file.pdf", "user123")
        assert id1 == id2

    def test_different_entity_different_id(self):
        """Different entity produces different file_id."""
        id1 = generate_file_id("/path/to/file.pdf", "user123")
        id2 = generate_file_id("/path/to/file.pdf", "user456")
        assert id1 != id2

    def test_different_path_different_id(self):
        """Different path produces different file_id."""
        id1 = generate_file_id("/path/to/file1.pdf", "user123")
        id2 = generate_file_id("/path/to/file2.pdf", "user123")
        assert id1 != id2

    def test_file_id_length(self):
        """file_id should be 32 chars (truncated SHA256)."""
        file_id = generate_file_id("/any/path", "any_user")
        assert len(file_id) == 32

    def test_file_id_is_hex(self):
        """file_id should be hexadecimal string."""
        file_id = generate_file_id("/any/path", "any_user")
        assert all(c in "0123456789abcdef" for c in file_id)

    def test_normalized_paths(self, tmp_path):
        """Paths are normalized to absolute paths."""
        # Create a real path for normalization
        subdir = tmp_path / "subdir"
        subdir.mkdir()

        # These should produce the same ID after normalization
        id1 = generate_file_id(str(subdir / "file.txt"), "user")
        id2 = generate_file_id(str(tmp_path / "subdir" / "file.txt"), "user")
        assert id1 == id2


class TestFileHash:
    """Tests for compute_file_hash function."""

    def test_compute_file_hash(self, tmp_path):
        """Returns valid MD5 hash."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("Hello, World!")

        hash_value = compute_file_hash(str(file_path))

        assert len(hash_value) == 32
        assert all(c in "0123456789abcdef" for c in hash_value)

    def test_hash_changes_with_content(self, tmp_path):
        """Different content produces different hash."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file1.write_text("Content A")
        file2.write_text("Content B")

        hash1 = compute_file_hash(str(file1))
        hash2 = compute_file_hash(str(file2))

        assert hash1 != hash2

    def test_hash_stable_for_same_content(self, tmp_path):
        """Same content produces same hash."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file1.write_text("Identical content")
        file2.write_text("Identical content")

        hash1 = compute_file_hash(str(file1))
        hash2 = compute_file_hash(str(file2))

        assert hash1 == hash2

    def test_hash_large_file(self, tmp_path):
        """Handles large files with chunked reading."""
        file_path = tmp_path / "large.txt"
        # Create a file larger than the chunk size (8192 bytes)
        content = "x" * 50000
        file_path.write_text(content)

        hash_value = compute_file_hash(str(file_path))

        assert len(hash_value) == 32

    def test_hash_binary_file(self, tmp_path):
        """Handles binary files."""
        file_path = tmp_path / "binary.bin"
        file_path.write_bytes(b"\x00\x01\x02\x03\xff\xfe\xfd")

        hash_value = compute_file_hash(str(file_path))

        assert len(hash_value) == 32


class TestFileMetadata:
    """Tests for get_file_metadata function."""

    def test_get_file_metadata(self, tmp_path):
        """Returns mtime and size."""
        file_path = tmp_path / "test.txt"
        content = "Test content"
        file_path.write_text(content)

        metadata = get_file_metadata(str(file_path))

        assert "mtime" in metadata
        assert "size" in metadata
        assert metadata["size"] == len(content)

    def test_metadata_mtime_is_datetime(self, tmp_path):
        """mtime is a timezone-aware datetime."""
        from datetime import datetime, timezone

        file_path = tmp_path / "test.txt"
        file_path.write_text("content")

        metadata = get_file_metadata(str(file_path))

        assert isinstance(metadata["mtime"], datetime)
        assert metadata["mtime"].tzinfo == timezone.utc


class TestSupportedFiles:
    """Tests for is_supported_file function."""

    @pytest.mark.parametrize(
        "filename",
        ["document.pdf", "spreadsheet.xlsx", "presentation.pptx", "text.txt", "code.py"],
    )
    def test_supported_extensions(self, filename):
        """Common file types are supported."""
        assert is_supported_file(filename) is True

    @pytest.mark.parametrize(
        "filename",
        ["image.png", "video.mp4", "audio.mp3", "archive.zip", "binary.exe"],
    )
    def test_unsupported_extensions(self, filename):
        """Binary/media files are not supported."""
        assert is_supported_file(filename) is False

    def test_supported_extensions_pdf(self):
        """PDF files are supported."""
        assert is_supported_file("document.pdf") is True

    def test_supported_extensions_docx(self):
        """DOCX files are supported."""
        assert is_supported_file("document.docx") is True

    def test_supported_extensions_txt(self):
        """TXT files are supported."""
        assert is_supported_file("readme.txt") is True

    def test_supported_extensions_code_files(self):
        """Code files (.py, .js, etc.) are supported."""
        assert is_supported_file("script.py") is True
        assert is_supported_file("app.js") is True
        assert is_supported_file("main.go") is True
        assert is_supported_file("style.css") is True

    def test_case_insensitive(self):
        """Extension matching is case-insensitive."""
        assert is_supported_file("DOCUMENT.PDF") is True
        assert is_supported_file("File.TXT") is True
        assert is_supported_file("Code.PY") is True

    def test_custom_extensions_filter(self):
        """Can filter by specific extensions."""
        assert is_supported_file("doc.pdf", extensions=["pdf"]) is True
        assert is_supported_file("doc.txt", extensions=["pdf"]) is False
        assert is_supported_file("doc.pdf", extensions=["txt", "md"]) is False

    def test_custom_extensions_with_dot(self):
        """Extensions can be specified with or without dot."""
        assert is_supported_file("doc.pdf", extensions=[".pdf"]) is True
        assert is_supported_file("doc.pdf", extensions=["pdf"]) is True


class TestShouldIgnore:
    """Tests for should_ignore function."""

    def test_ignore_git_directory(self):
        """Ignores .git directory."""
        assert should_ignore(".git", [".git"]) is True
        assert should_ignore("/path/to/.git", [".git"]) is True
        assert should_ignore("/path/to/.git/config", [".git"]) is True

    def test_ignore_pycache(self):
        """Ignores __pycache__ directory."""
        assert should_ignore("__pycache__", ["__pycache__"]) is True
        assert should_ignore("/path/__pycache__/module.pyc", ["__pycache__"]) is True

    def test_ignore_node_modules(self):
        """Ignores node_modules directory."""
        assert should_ignore("node_modules", ["node_modules"]) is True
        assert should_ignore("/path/node_modules/package", ["node_modules"]) is True

    def test_ignore_ds_store(self):
        """Ignores .DS_Store file."""
        assert should_ignore(".DS_Store", [".DS_Store"]) is True
        assert should_ignore("/path/.DS_Store", [".DS_Store"]) is True

    def test_custom_ignore_patterns(self):
        """Supports custom ignore patterns."""
        assert should_ignore("temp", ["temp", "*.tmp"]) is True
        assert should_ignore("file.tmp", ["temp", "*.tmp"]) is True

    def test_glob_pattern_matching(self):
        """Supports glob patterns."""
        assert should_ignore("test.log", ["*.log"]) is True
        assert should_ignore("debug.log", ["*.log"]) is True
        assert should_ignore("test.txt", ["*.log"]) is False

    def test_no_match_returns_false(self):
        """Returns False when no patterns match."""
        assert should_ignore("important.txt", [".git", "__pycache__"]) is False

    def test_empty_patterns(self):
        """Empty pattern list returns False."""
        assert should_ignore("anything", []) is False


class TestDiscoverFiles:
    """Tests for discover_files function."""

    def test_discover_files_recursive(self, tmp_path):
        """Finds files in subdirectories when recursive=True."""
        # Create nested structure
        (tmp_path / "subdir").mkdir()
        (tmp_path / "test.txt").write_text("content")
        (tmp_path / "subdir" / "nested.txt").write_text("nested content")

        files = discover_files(str(tmp_path), recursive=True)

        assert len(files) == 2
        filenames = [os.path.basename(f) for f in files]
        assert "test.txt" in filenames
        assert "nested.txt" in filenames

    def test_discover_files_non_recursive(self, tmp_path):
        """Only finds top-level files when recursive=False."""
        (tmp_path / "subdir").mkdir()
        (tmp_path / "test.txt").write_text("content")
        (tmp_path / "subdir" / "nested.txt").write_text("nested content")

        files = discover_files(str(tmp_path), recursive=False)

        assert len(files) == 1
        assert os.path.basename(files[0]) == "test.txt"

    def test_ignore_patterns_directory(self, tmp_path):
        """Skips directories matching ignore patterns."""
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "config").write_text("git config")
        (tmp_path / "test.txt").write_text("content")

        files = discover_files(
            str(tmp_path), recursive=True, ignore_patterns=[".git"]
        )

        assert len(files) == 1
        assert ".git" not in files[0]

    def test_ignore_patterns_file(self, tmp_path):
        """Skips files matching ignore patterns."""
        (tmp_path / "test.txt").write_text("content")
        (tmp_path / ".DS_Store").write_text("ds store")

        files = discover_files(
            str(tmp_path), recursive=True, ignore_patterns=[".DS_Store"]
        )

        assert len(files) == 1
        assert ".DS_Store" not in files[0]

    def test_filter_by_extension(self, tmp_path):
        """Filters files by extension."""
        (tmp_path / "doc.pdf").write_bytes(b"%PDF")
        (tmp_path / "text.txt").write_text("text")
        (tmp_path / "data.csv").write_text("a,b,c")

        files = discover_files(str(tmp_path), extensions=["pdf", "csv"])

        assert len(files) == 2
        extensions = {Path(f).suffix for f in files}
        assert extensions == {".pdf", ".csv"}

    def test_empty_directory(self, tmp_path):
        """Returns empty list for empty directory."""
        files = discover_files(str(tmp_path))

        assert files == []

    def test_deeply_nested_directory(self, tmp_path):
        """Works with deeply nested directories."""
        deep_path = tmp_path / "a" / "b" / "c" / "d"
        deep_path.mkdir(parents=True)
        (deep_path / "deep.txt").write_text("deep content")

        files = discover_files(str(tmp_path), recursive=True)

        assert len(files) == 1
        assert "deep.txt" in files[0]

    def test_returns_absolute_paths(self, tmp_path):
        """All returned paths are absolute."""
        (tmp_path / "test.txt").write_text("content")

        files = discover_files(str(tmp_path))

        for f in files:
            assert os.path.isabs(f)

    def test_filters_unsupported_files(self, tmp_path):
        """Doesn't return files with unsupported extensions."""
        (tmp_path / "image.png").write_bytes(b"PNG")
        (tmp_path / "document.pdf").write_bytes(b"%PDF")

        files = discover_files(str(tmp_path))

        assert len(files) == 1
        assert files[0].endswith(".pdf")

    def test_multiple_ignore_patterns(self, tmp_path):
        """Supports multiple ignore patterns."""
        (tmp_path / ".git").mkdir()
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / ".git" / "config").write_text("config")
        (tmp_path / "__pycache__" / "module.pyc").write_text("pyc")
        (tmp_path / "main.py").write_text("code")

        files = discover_files(
            str(tmp_path),
            recursive=True,
            ignore_patterns=[".git", "__pycache__"],
        )

        assert len(files) == 1
        assert files[0].endswith("main.py")


class TestSupportedExtensions:
    """Tests for SUPPORTED_EXTENSIONS constant."""

    def test_contains_document_types(self):
        """Contains common document extensions."""
        assert "pdf" in SUPPORTED_EXTENSIONS
        assert "docx" in SUPPORTED_EXTENSIONS
        assert "doc" in SUPPORTED_EXTENSIONS
        assert "txt" in SUPPORTED_EXTENSIONS
        assert "md" in SUPPORTED_EXTENSIONS

    def test_contains_spreadsheet_types(self):
        """Contains spreadsheet extensions."""
        assert "xlsx" in SUPPORTED_EXTENSIONS
        assert "xls" in SUPPORTED_EXTENSIONS
        assert "csv" in SUPPORTED_EXTENSIONS

    def test_contains_code_types(self):
        """Contains code file extensions."""
        assert "py" in SUPPORTED_EXTENSIONS
        assert "js" in SUPPORTED_EXTENSIONS
        assert "ts" in SUPPORTED_EXTENSIONS
        assert "go" in SUPPORTED_EXTENSIONS
        assert "java" in SUPPORTED_EXTENSIONS

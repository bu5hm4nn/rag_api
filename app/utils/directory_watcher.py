# app/utils/directory_watcher.py
"""
Directory watching utilities for automatic document indexing.

This module provides:
- File ID generation (deterministic hashing)
- File content hashing for change detection
- Directory traversal with filtering
- Filesystem monitoring with watchdog
- Async event handling with debouncing
"""

import os
import hashlib
import asyncio
import fnmatch
import threading
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List, Dict, Callable, Awaitable
from dataclasses import dataclass, field

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent

from app.config import (
    known_source_ext,
    logger,
    ALLOWED_LOCAL_PATHS,
    MAX_PENDING_CHANGES_PER_WATCH,
    FOLLOW_SYMLINKS,
    MAX_FILES_PER_OPERATION,
)


class PathSecurityError(Exception):
    """Raised when a path fails security validation."""

    pass


def validate_path_security(path: str, check_exists: bool = True) -> str:
    """
    Validate that a path is allowed and safe to access.

    Performs the following security checks:
    1. Resolves to absolute canonical path (no symlinks, no ..)
    2. Verifies path is under an allowed base directory
    3. Optionally checks symlink status

    Args:
        path: Path to validate
        check_exists: Whether to check if path exists

    Returns:
        Canonicalized absolute path

    Raises:
        PathSecurityError: If path fails any security check
    """
    if not ALLOWED_LOCAL_PATHS:
        raise PathSecurityError(
            "Local directory operations are disabled. "
            "Set RAG_ALLOWED_LOCAL_PATHS environment variable to enable."
        )

    # Resolve to canonical absolute path
    try:
        # First check for symlink before resolving
        if not FOLLOW_SYMLINKS and os.path.islink(path):
            raise PathSecurityError(f"Symbolic links are not allowed: {path}")

        # Use realpath to resolve symlinks and .. components
        canonical_path = os.path.realpath(path)
    except OSError as e:
        raise PathSecurityError(f"Invalid path: {e}")

    # Check if path is under any allowed directory
    is_allowed = False
    for allowed_base in ALLOWED_LOCAL_PATHS:
        # Ensure consistent comparison with trailing separator
        allowed_base_normalized = os.path.realpath(allowed_base)
        if canonical_path == allowed_base_normalized or canonical_path.startswith(
            allowed_base_normalized + os.sep
        ):
            is_allowed = True
            break

    if not is_allowed:
        raise PathSecurityError(
            "Path is not under any allowed directory. "
            "Contact administrator if this path should be accessible."
        )

    # Check existence if required
    if check_exists and not os.path.exists(canonical_path):
        raise PathSecurityError(f"Path does not exist: {canonical_path}")

    return canonical_path


def is_path_contained(child_path: str, parent_path: str) -> bool:
    """
    Check if child_path is contained within parent_path.

    Uses canonical paths to prevent path traversal attacks.

    Args:
        child_path: Path to check
        parent_path: Expected parent directory

    Returns:
        True if child is under parent
    """
    try:
        child_canonical = os.path.realpath(child_path)
        parent_canonical = os.path.realpath(parent_path)

        return child_canonical == parent_canonical or child_canonical.startswith(
            parent_canonical + os.sep
        )
    except OSError:
        return False


# Supported file extensions for indexing (document types + code files)
SUPPORTED_EXTENSIONS: set[str] = {
    "pdf", "csv", "rst", "xml", "ppt", "pptx", "md",
    "epub", "doc", "docx", "xls", "xlsx", "json", "txt"
} | set(known_source_ext)


def generate_file_id(filepath: str, entity_id: str) -> str:
    """
    Generate a deterministic file_id from filepath and entity_id.

    Uses SHA256 truncated to 32 chars for uniqueness while keeping
    IDs reasonably short.

    Args:
        filepath: Path to the file (will be normalized to absolute path)
        entity_id: User/entity identifier for namespacing

    Returns:
        32-character hexadecimal string
    """
    normalized_path = os.path.abspath(filepath)
    content = f"{normalized_path}:{entity_id}"
    return hashlib.sha256(content.encode()).hexdigest()[:32]


def compute_file_hash(filepath: str) -> str:
    """
    Compute MD5 hash of file content.

    Uses chunked reading to handle large files efficiently.

    Args:
        filepath: Path to the file

    Returns:
        32-character MD5 hexadecimal string
    """
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def get_file_metadata(filepath: str) -> dict:
    """
    Get file metadata including modification time and size.

    Args:
        filepath: Path to the file

    Returns:
        Dict with 'mtime' (datetime) and 'size' (int) keys
    """
    stat = os.stat(filepath)
    return {
        "mtime": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
        "size": stat.st_size,
    }


def is_supported_file(filepath: str, extensions: Optional[List[str]] = None) -> bool:
    """
    Check if file type is supported for indexing.

    Args:
        filepath: Path to the file
        extensions: Optional list of extensions to filter by.
                   If None, uses SUPPORTED_EXTENSIONS.

    Returns:
        True if file should be indexed
    """
    ext = Path(filepath).suffix.lower().lstrip(".")
    if extensions:
        return ext in [e.lower().lstrip(".") for e in extensions]
    return ext in SUPPORTED_EXTENSIONS


def should_ignore(path: str, ignore_patterns: List[str]) -> bool:
    """
    Check if path matches any ignore patterns.

    Args:
        path: File or directory path to check
        ignore_patterns: List of glob patterns to ignore

    Returns:
        True if path should be ignored
    """
    path_parts = Path(path).parts
    for pattern in ignore_patterns:
        # Check if any part of the path matches the pattern
        for part in path_parts:
            if fnmatch.fnmatch(part, pattern):
                return True
        # Also check the full path
        if fnmatch.fnmatch(path, pattern):
            return True
        # Check basename
        if fnmatch.fnmatch(os.path.basename(path), pattern):
            return True
    return False


def discover_files(
    directory: str,
    recursive: bool = True,
    extensions: Optional[List[str]] = None,
    ignore_patterns: Optional[List[str]] = None,
    max_files: Optional[int] = None,
) -> List[str]:
    """
    Discover all supported files in a directory.

    Args:
        directory: Directory path to scan
        recursive: Whether to scan subdirectories
        extensions: Optional list of extensions to filter by
        ignore_patterns: Patterns for files/directories to ignore
        max_files: Maximum number of files to return (default: MAX_FILES_PER_OPERATION)

    Returns:
        List of absolute file paths

    Raises:
        PathSecurityError: If max_files limit is exceeded
    """
    ignore_patterns = ignore_patterns or []
    files = []
    limit = max_files if max_files is not None else MAX_FILES_PER_OPERATION

    def _check_and_add_file(filepath: str) -> bool:
        """Add file if valid, return False if limit reached."""
        # Skip symlinks unless explicitly allowed
        if not FOLLOW_SYMLINKS and os.path.islink(filepath):
            return True  # Continue processing

        if should_ignore(filepath, ignore_patterns):
            return True

        if is_supported_file(filepath, extensions):
            if limit > 0 and len(files) >= limit:
                raise PathSecurityError(
                    f"File limit exceeded: found more than {limit} files. "
                    f"Increase RAG_MAX_FILES_PER_OPERATION or narrow your search."
                )
            files.append(os.path.abspath(filepath))

        return True

    if recursive:
        for root, dirs, filenames in os.walk(directory, followlinks=FOLLOW_SYMLINKS):
            # Filter out ignored directories (modifies in-place for os.walk)
            # Also skip symlinked directories if not allowed
            if FOLLOW_SYMLINKS:
                dirs[:] = [d for d in dirs if not should_ignore(d, ignore_patterns)]
            else:
                dirs[:] = [
                    d
                    for d in dirs
                    if not should_ignore(d, ignore_patterns)
                    and not os.path.islink(os.path.join(root, d))
                ]

            for filename in filenames:
                filepath = os.path.join(root, filename)
                _check_and_add_file(filepath)
    else:
        for entry in os.scandir(directory):
            if entry.is_file(follow_symlinks=FOLLOW_SYMLINKS):
                _check_and_add_file(entry.path)

    return files


@dataclass
class PendingChange:
    """Represents a pending file change to be processed."""

    filepath: str
    event_type: str  # 'created', 'modified', 'deleted', 'moved'
    timestamp: datetime
    dest_path: Optional[str] = None  # For move events


@dataclass
class DirectoryWatch:
    """Configuration and state for a watched directory."""

    watch_id: int
    directory_path: str
    entity_id: str
    recursive: bool = True
    extensions: Optional[List[str]] = None
    ignore_patterns: List[str] = field(default_factory=list)
    debounce_seconds: int = 5
    enabled: bool = True
    pending_changes: Dict[str, PendingChange] = field(default_factory=dict)
    last_debounce_task: Optional[asyncio.Task] = None


class AsyncFileEventHandler(FileSystemEventHandler):
    """
    Watchdog event handler that bridges to async processing.

    Collects filesystem events and debounces before triggering
    async callbacks. This allows batching rapid changes and
    processing them efficiently.
    """

    def __init__(
        self,
        watch: DirectoryWatch,
        loop: asyncio.AbstractEventLoop,
        on_changes_callback: Callable[
            [DirectoryWatch, List[PendingChange]], Awaitable[None]
        ],
    ):
        """
        Initialize the event handler.

        Args:
            watch: Watch configuration
            loop: Async event loop for scheduling callbacks
            on_changes_callback: Async function to call with accumulated changes
        """
        super().__init__()
        self.watch = watch
        self.loop = loop
        self.on_changes_callback = on_changes_callback
        self._lock = threading.Lock()

    def _should_process(self, path: str) -> bool:
        """Check if this path should be processed."""
        # Security: Verify path is contained within watched directory
        if not is_path_contained(path, self.watch.directory_path):
            logger.warning(
                f"Path traversal attempt blocked: {path} is not under {self.watch.directory_path}"
            )
            return False

        # Security: Block symlinks unless explicitly allowed
        if not FOLLOW_SYMLINKS and os.path.islink(path):
            return False

        if should_ignore(path, self.watch.ignore_patterns):
            return False
        # Always check extension filter (for created, modified, and deleted events)
        if not is_supported_file(path, self.watch.extensions):
            return False
        return True

    def _add_pending_change(
        self, filepath: str, event_type: str, dest_path: Optional[str] = None
    ):
        """Add a pending change with debouncing."""
        if not self._should_process(filepath):
            return

        with self._lock:
            # Use absolute path as key
            abs_path = os.path.abspath(filepath)

            # Security: Enforce pending changes limit
            if (
                MAX_PENDING_CHANGES_PER_WATCH > 0
                and len(self.watch.pending_changes) >= MAX_PENDING_CHANGES_PER_WATCH
                and abs_path not in self.watch.pending_changes
            ):
                # Drop oldest change to make room
                oldest_key = min(
                    self.watch.pending_changes.keys(),
                    key=lambda k: self.watch.pending_changes[k].timestamp,
                )
                del self.watch.pending_changes[oldest_key]
                logger.warning(
                    f"Pending changes limit ({MAX_PENDING_CHANGES_PER_WATCH}) reached for watch {self.watch.watch_id}, dropped oldest"
                )

            self.watch.pending_changes[abs_path] = PendingChange(
                filepath=abs_path,
                event_type=event_type,
                timestamp=datetime.now(timezone.utc),
                dest_path=os.path.abspath(dest_path) if dest_path else None,
            )

        # Schedule debounced processing in the async loop
        self.loop.call_soon_threadsafe(self._schedule_debounce)

    def _schedule_debounce(self):
        """Schedule debounced processing of pending changes."""
        # Cancel existing debounce task if any
        if (
            self.watch.last_debounce_task
            and not self.watch.last_debounce_task.done()
        ):
            self.watch.last_debounce_task.cancel()

        # Create new debounce task
        self.watch.last_debounce_task = asyncio.create_task(self._debounced_process())

    async def _debounced_process(self):
        """Wait for debounce interval then process all pending changes."""
        try:
            await asyncio.sleep(self.watch.debounce_seconds)

            # Collect and clear pending changes
            with self._lock:
                if not self.watch.pending_changes:
                    return
                changes = list(self.watch.pending_changes.values())
                self.watch.pending_changes.clear()

            # Call the async callback
            await self.on_changes_callback(self.watch, changes)

        except asyncio.CancelledError:
            # Expected when new events come in during debounce
            pass
        except Exception as e:
            logger.error(f"Error processing file changes: {e}")

    def on_created(self, event: FileSystemEvent):
        """Handle file creation events."""
        if not event.is_directory:
            self._add_pending_change(event.src_path, "created")

    def on_modified(self, event: FileSystemEvent):
        """Handle file modification events."""
        if not event.is_directory:
            self._add_pending_change(event.src_path, "modified")

    def on_deleted(self, event: FileSystemEvent):
        """Handle file deletion events."""
        if not event.is_directory:
            self._add_pending_change(event.src_path, "deleted")

    def on_moved(self, event: FileSystemEvent):
        """Handle file move events (treated as delete + create)."""
        if not event.is_directory:
            self._add_pending_change(event.src_path, "deleted")
            self._add_pending_change(event.dest_path, "created")


class DirectoryWatchManager:
    """
    Manages multiple directory watches with a single Observer.

    Singleton pattern - one manager per application.
    """

    _instance: Optional["DirectoryWatchManager"] = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self._observer: Optional[Observer] = None
        self._watches: Dict[int, DirectoryWatch] = {}
        self._handlers: Dict[int, AsyncFileEventHandler] = {}
        self._observer_watches: Dict[int, object] = {}  # watch_id -> observer watch handle
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._on_changes_callback: Optional[Callable] = None
        self._started = False

    @classmethod
    def reset_instance(cls):
        """Reset the singleton instance. Used for testing."""
        with cls._lock:
            if cls._instance is not None:
                if cls._instance._started:
                    cls._instance.stop()
                cls._instance = None

    def initialize(
        self,
        loop: asyncio.AbstractEventLoop,
        on_changes_callback: Callable[
            [DirectoryWatch, List[PendingChange]], Awaitable[None]
        ],
    ):
        """
        Initialize the manager with the event loop and callback.

        Args:
            loop: Async event loop for scheduling callbacks
            on_changes_callback: Function to call when changes are detected
        """
        self._loop = loop
        self._on_changes_callback = on_changes_callback
        self._observer = Observer()

    def start(self):
        """Start the filesystem observer."""
        if not self._started and self._observer:
            self._observer.start()
            self._started = True
            logger.info("Directory watch manager started")

    def stop(self):
        """Stop the filesystem observer gracefully."""
        if self._started and self._observer:
            self._observer.stop()
            self._observer.join(timeout=5.0)
            self._started = False
            logger.info("Directory watch manager stopped")

    def add_watch(self, watch: DirectoryWatch) -> bool:
        """
        Add a new directory watch.

        Args:
            watch: Watch configuration

        Returns:
            True if watch was added, False if already exists
        """
        if watch.watch_id in self._watches:
            return False

        if not self._loop or not self._on_changes_callback:
            raise RuntimeError("Manager not initialized. Call initialize() first.")

        handler = AsyncFileEventHandler(
            watch=watch, loop=self._loop, on_changes_callback=self._on_changes_callback
        )

        obs_watch = self._observer.schedule(
            handler, watch.directory_path, recursive=watch.recursive
        )

        self._watches[watch.watch_id] = watch
        self._handlers[watch.watch_id] = handler
        self._observer_watches[watch.watch_id] = obs_watch

        logger.info(f"Added watch {watch.watch_id} for {watch.directory_path}")
        return True

    def remove_watch(self, watch_id: int) -> bool:
        """
        Remove a directory watch.

        Args:
            watch_id: ID of the watch to remove

        Returns:
            True if watch was removed, False if not found
        """
        if watch_id not in self._watches:
            return False

        obs_watch = self._observer_watches.get(watch_id)
        if obs_watch:
            self._observer.unschedule(obs_watch)

        del self._watches[watch_id]
        del self._handlers[watch_id]
        del self._observer_watches[watch_id]

        logger.info(f"Removed watch {watch_id}")
        return True

    def get_watch(self, watch_id: int) -> Optional[DirectoryWatch]:
        """Get a watch by ID."""
        return self._watches.get(watch_id)

    def get_all_watches(self) -> List[DirectoryWatch]:
        """Get all active watches."""
        return list(self._watches.values())

    def get_pending_change_count(self, watch_id: int) -> int:
        """Get count of pending changes for a watch."""
        watch = self._watches.get(watch_id)
        return len(watch.pending_changes) if watch else 0

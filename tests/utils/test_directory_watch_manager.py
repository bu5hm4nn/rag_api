# tests/utils/test_directory_watch_manager.py
"""Tests for DirectoryWatchManager and related classes."""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timezone

from app.utils.directory_watcher import (
    DirectoryWatch,
    DirectoryWatchManager,
    AsyncFileEventHandler,
    PendingChange,
)


class TestDirectoryWatch:
    """Tests for DirectoryWatch dataclass."""

    def test_directory_watch_dataclass(self):
        """Proper initialization with required fields."""
        watch = DirectoryWatch(
            watch_id=1,
            directory_path="/path/to/dir",
            entity_id="user123",
        )

        assert watch.watch_id == 1
        assert watch.directory_path == "/path/to/dir"
        assert watch.entity_id == "user123"
        assert watch.recursive is True  # Default
        assert watch.debounce_seconds == 5  # Default
        assert watch.enabled is True  # Default

    def test_directory_watch_with_options(self):
        """Initialization with custom options."""
        watch = DirectoryWatch(
            watch_id=1,
            directory_path="/path/to/dir",
            entity_id="user123",
            recursive=False,
            extensions=["pdf", "txt"],
            ignore_patterns=[".git", "__pycache__"],
            debounce_seconds=10,
            enabled=False,
        )

        assert watch.recursive is False
        assert watch.extensions == ["pdf", "txt"]
        assert watch.ignore_patterns == [".git", "__pycache__"]
        assert watch.debounce_seconds == 10
        assert watch.enabled is False

    def test_pending_changes_dict(self):
        """Pending changes dict is initialized empty."""
        watch = DirectoryWatch(
            watch_id=1,
            directory_path="/path",
            entity_id="user",
        )

        assert watch.pending_changes == {}
        assert isinstance(watch.pending_changes, dict)


class TestPendingChange:
    """Tests for PendingChange dataclass."""

    def test_pending_change_creation(self):
        """Create a pending change."""
        now = datetime.now(timezone.utc)
        change = PendingChange(
            filepath="/path/to/file.txt",
            event_type="created",
            timestamp=now,
        )

        assert change.filepath == "/path/to/file.txt"
        assert change.event_type == "created"
        assert change.timestamp == now
        assert change.dest_path is None

    def test_pending_change_with_dest_path(self):
        """Create a pending change for move event."""
        change = PendingChange(
            filepath="/old/path.txt",
            event_type="moved",
            timestamp=datetime.now(timezone.utc),
            dest_path="/new/path.txt",
        )

        assert change.dest_path == "/new/path.txt"


class TestDirectoryWatchManager:
    """Tests for DirectoryWatchManager singleton."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        DirectoryWatchManager.reset_instance()
        yield
        DirectoryWatchManager.reset_instance()

    def test_singleton_pattern(self):
        """Only one instance is created."""
        manager1 = DirectoryWatchManager()
        manager2 = DirectoryWatchManager()

        assert manager1 is manager2

    def test_reset_instance(self):
        """reset_instance creates a new singleton."""
        manager1 = DirectoryWatchManager()
        DirectoryWatchManager.reset_instance()
        manager2 = DirectoryWatchManager()

        assert manager1 is not manager2

    @pytest.mark.asyncio
    async def test_initialize(self):
        """Initialize sets loop and callback."""
        manager = DirectoryWatchManager()
        loop = asyncio.get_running_loop()
        callback = AsyncMock()

        manager.initialize(loop, callback)

        assert manager._loop is loop
        assert manager._on_changes_callback is callback
        assert manager._observer is not None

    @pytest.mark.asyncio
    async def test_start_stop(self, tmp_path):
        """Observer can be started and stopped."""
        manager = DirectoryWatchManager()
        loop = asyncio.get_running_loop()
        callback = AsyncMock()

        manager.initialize(loop, callback)
        manager.start()

        assert manager._started is True

        manager.stop()

        assert manager._started is False

    @pytest.mark.asyncio
    async def test_add_watch(self, tmp_path):
        """Add a watch to the manager."""
        manager = DirectoryWatchManager()
        loop = asyncio.get_running_loop()
        callback = AsyncMock()

        manager.initialize(loop, callback)
        manager.start()

        watch = DirectoryWatch(
            watch_id=1,
            directory_path=str(tmp_path),
            entity_id="user123",
        )

        result = manager.add_watch(watch)

        assert result is True
        assert manager.get_watch(1) is watch
        assert len(manager.get_all_watches()) == 1

        manager.stop()

    @pytest.mark.asyncio
    async def test_add_watch_duplicate(self, tmp_path):
        """Adding duplicate watch returns False."""
        manager = DirectoryWatchManager()
        loop = asyncio.get_running_loop()
        callback = AsyncMock()

        manager.initialize(loop, callback)
        manager.start()

        watch = DirectoryWatch(
            watch_id=1,
            directory_path=str(tmp_path),
            entity_id="user123",
        )

        manager.add_watch(watch)
        result = manager.add_watch(watch)

        assert result is False

        manager.stop()

    @pytest.mark.asyncio
    async def test_remove_watch(self, tmp_path):
        """Remove a watch from the manager."""
        manager = DirectoryWatchManager()
        loop = asyncio.get_running_loop()
        callback = AsyncMock()

        manager.initialize(loop, callback)
        manager.start()

        watch = DirectoryWatch(
            watch_id=1,
            directory_path=str(tmp_path),
            entity_id="user123",
        )

        manager.add_watch(watch)
        result = manager.remove_watch(1)

        assert result is True
        assert manager.get_watch(1) is None
        assert len(manager.get_all_watches()) == 0

        manager.stop()

    @pytest.mark.asyncio
    async def test_remove_watch_not_found(self, tmp_path):
        """Removing non-existent watch returns False."""
        manager = DirectoryWatchManager()
        loop = asyncio.get_running_loop()
        callback = AsyncMock()

        manager.initialize(loop, callback)
        manager.start()

        result = manager.remove_watch(999)

        assert result is False

        manager.stop()

    @pytest.mark.asyncio
    async def test_get_watch(self, tmp_path):
        """Get a watch by ID."""
        manager = DirectoryWatchManager()
        loop = asyncio.get_running_loop()
        callback = AsyncMock()

        manager.initialize(loop, callback)
        manager.start()

        watch = DirectoryWatch(
            watch_id=42,
            directory_path=str(tmp_path),
            entity_id="user123",
        )

        manager.add_watch(watch)

        assert manager.get_watch(42) is watch
        assert manager.get_watch(999) is None

        manager.stop()

    @pytest.mark.asyncio
    async def test_get_all_watches(self, tmp_path):
        """Get all active watches."""
        manager = DirectoryWatchManager()
        loop = asyncio.get_running_loop()
        callback = AsyncMock()

        manager.initialize(loop, callback)
        manager.start()

        # Create subdirectories for separate watches
        dir1 = tmp_path / "dir1"
        dir2 = tmp_path / "dir2"
        dir1.mkdir()
        dir2.mkdir()

        watch1 = DirectoryWatch(
            watch_id=1,
            directory_path=str(dir1),
            entity_id="user1",
        )
        watch2 = DirectoryWatch(
            watch_id=2,
            directory_path=str(dir2),
            entity_id="user2",
        )

        manager.add_watch(watch1)
        manager.add_watch(watch2)

        watches = manager.get_all_watches()

        assert len(watches) == 2
        assert watch1 in watches
        assert watch2 in watches

        manager.stop()

    @pytest.mark.asyncio
    async def test_get_pending_change_count(self, tmp_path):
        """Get count of pending changes for a watch."""
        manager = DirectoryWatchManager()
        loop = asyncio.get_running_loop()
        callback = AsyncMock()

        manager.initialize(loop, callback)
        manager.start()

        watch = DirectoryWatch(
            watch_id=1,
            directory_path=str(tmp_path),
            entity_id="user123",
        )

        manager.add_watch(watch)

        # Initially no pending changes
        assert manager.get_pending_change_count(1) == 0

        # Add a pending change directly
        watch.pending_changes["/path/file.txt"] = PendingChange(
            filepath="/path/file.txt",
            event_type="created",
            timestamp=datetime.now(timezone.utc),
        )

        assert manager.get_pending_change_count(1) == 1

        manager.stop()

    def test_add_watch_without_initialize(self):
        """Adding watch without initialize raises error."""
        manager = DirectoryWatchManager()

        watch = DirectoryWatch(
            watch_id=1,
            directory_path="/path",
            entity_id="user",
        )

        with pytest.raises(RuntimeError, match="Manager not initialized"):
            manager.add_watch(watch)


class TestAsyncFileEventHandler:
    """Tests for AsyncFileEventHandler class."""

    @pytest.fixture
    def mock_watch(self, tmp_path):
        """Create a mock watch for testing."""
        return DirectoryWatch(
            watch_id=1,
            directory_path=str(tmp_path),
            entity_id="user123",
            extensions=["txt", "pdf"],
            ignore_patterns=[".git", "__pycache__"],
            debounce_seconds=1,
        )

    @pytest.fixture
    def mock_event(self):
        """Create a mock filesystem event."""
        event = Mock()
        event.is_directory = False
        event.src_path = "/path/to/file.txt"
        return event

    @pytest.mark.asyncio
    async def test_on_created_adds_pending(self, mock_watch, mock_event):
        """Created event adds pending change."""
        loop = asyncio.get_running_loop()
        callback = AsyncMock()

        handler = AsyncFileEventHandler(mock_watch, loop, callback)
        mock_event.src_path = str(mock_watch.directory_path) + "/newfile.txt"

        handler.on_created(mock_event)

        # Wait a bit for the event to be processed
        await asyncio.sleep(0.1)

        assert len(mock_watch.pending_changes) == 1

    @pytest.mark.asyncio
    async def test_on_modified_adds_pending(self, mock_watch, mock_event):
        """Modified event adds pending change."""
        loop = asyncio.get_running_loop()
        callback = AsyncMock()

        handler = AsyncFileEventHandler(mock_watch, loop, callback)
        mock_event.src_path = str(mock_watch.directory_path) + "/existing.txt"

        handler.on_modified(mock_event)

        await asyncio.sleep(0.1)

        assert len(mock_watch.pending_changes) == 1

    @pytest.mark.asyncio
    async def test_on_deleted_adds_pending(self, mock_watch, mock_event):
        """Deleted event adds pending change."""
        loop = asyncio.get_running_loop()
        callback = AsyncMock()

        handler = AsyncFileEventHandler(mock_watch, loop, callback)
        mock_event.src_path = str(mock_watch.directory_path) + "/deleted.txt"

        handler.on_deleted(mock_event)

        await asyncio.sleep(0.1)

        assert len(mock_watch.pending_changes) == 1

    @pytest.mark.asyncio
    async def test_on_moved_creates_two_changes(self, mock_watch):
        """Move event creates delete + create changes."""
        loop = asyncio.get_running_loop()
        callback = AsyncMock()

        handler = AsyncFileEventHandler(mock_watch, loop, callback)

        event = Mock()
        event.is_directory = False
        event.src_path = str(mock_watch.directory_path) + "/old.txt"
        event.dest_path = str(mock_watch.directory_path) + "/new.txt"

        handler.on_moved(event)

        await asyncio.sleep(0.1)

        # Should have both old and new paths
        assert len(mock_watch.pending_changes) == 2

    @pytest.mark.asyncio
    async def test_filters_unsupported_files(self, mock_watch, mock_event):
        """Ignores files with unsupported extensions."""
        loop = asyncio.get_running_loop()
        callback = AsyncMock()

        handler = AsyncFileEventHandler(mock_watch, loop, callback)
        mock_event.src_path = str(mock_watch.directory_path) + "/image.png"

        handler.on_created(mock_event)

        await asyncio.sleep(0.1)

        assert len(mock_watch.pending_changes) == 0

    @pytest.mark.asyncio
    async def test_filters_ignored_patterns(self, mock_watch, mock_event):
        """Ignores files matching ignore patterns."""
        loop = asyncio.get_running_loop()
        callback = AsyncMock()

        handler = AsyncFileEventHandler(mock_watch, loop, callback)
        mock_event.src_path = str(mock_watch.directory_path) + "/.git/config"

        handler.on_created(mock_event)

        await asyncio.sleep(0.1)

        assert len(mock_watch.pending_changes) == 0

    @pytest.mark.asyncio
    async def test_ignores_directory_events(self, mock_watch, mock_event):
        """Ignores directory events."""
        loop = asyncio.get_running_loop()
        callback = AsyncMock()

        handler = AsyncFileEventHandler(mock_watch, loop, callback)
        mock_event.is_directory = True
        mock_event.src_path = str(mock_watch.directory_path) + "/newdir"

        handler.on_created(mock_event)

        await asyncio.sleep(0.1)

        assert len(mock_watch.pending_changes) == 0

    @pytest.mark.asyncio
    async def test_debounce_coalesces_changes(self, mock_watch):
        """Multiple rapid changes are coalesced."""
        loop = asyncio.get_running_loop()
        callback = AsyncMock()

        # Use a short debounce for testing
        mock_watch.debounce_seconds = 0.2

        handler = AsyncFileEventHandler(mock_watch, loop, callback)

        # Create multiple events for the same file
        for i in range(5):
            event = Mock()
            event.is_directory = False
            event.src_path = str(mock_watch.directory_path) + "/file.txt"
            handler.on_modified(event)
            await asyncio.sleep(0.05)

        # Should only have one pending change (same file)
        assert len(mock_watch.pending_changes) == 1

        # Wait for debounce to trigger callback
        await asyncio.sleep(0.3)

        # Callback should be called once with the changes
        callback.assert_called_once()

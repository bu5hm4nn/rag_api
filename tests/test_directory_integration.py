# tests/test_directory_integration.py
"""End-to-end integration tests for directory watching functionality."""

import asyncio
import os
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timezone

from app.utils.directory_watcher import (
    DirectoryWatch,
    DirectoryWatchManager,
    AsyncFileEventHandler,
    PendingChange,
)


class TestDirectoryWatchingIntegration:
    """End-to-end tests for directory watching."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        DirectoryWatchManager.reset_instance()
        yield
        DirectoryWatchManager.reset_instance()

    @pytest.mark.asyncio
    async def test_file_created_triggers_indexing(self, tmp_path):
        """Adding a file triggers automatic indexing."""
        indexed_files = []

        async def mock_callback(watch, changes):
            for change in changes:
                indexed_files.append(change)

        # Setup manager
        loop = asyncio.get_running_loop()
        manager = DirectoryWatchManager()
        manager.initialize(loop, mock_callback)
        manager.start()

        watch = DirectoryWatch(
            watch_id=1,
            directory_path=str(tmp_path),
            entity_id="test-user",
            debounce_seconds=0.1,  # Short debounce for testing
        )
        manager.add_watch(watch)

        # Create a file
        test_file = tmp_path / "new_file.txt"
        test_file.write_text("New content")

        # Wait for debounce + processing
        await asyncio.sleep(0.3)

        manager.stop()

        # Verify the file was detected
        assert len(indexed_files) >= 1
        assert any("new_file.txt" in change.filepath for change in indexed_files)

    @pytest.mark.asyncio
    async def test_file_modified_triggers_reindex(self, tmp_path):
        """Modifying a file triggers re-indexing."""
        changes_detected = []

        async def mock_callback(watch, changes):
            changes_detected.extend(changes)

        # Create initial file
        test_file = tmp_path / "existing.txt"
        test_file.write_text("Initial content")

        # Setup manager
        loop = asyncio.get_running_loop()
        manager = DirectoryWatchManager()
        manager.initialize(loop, mock_callback)
        manager.start()

        watch = DirectoryWatch(
            watch_id=1,
            directory_path=str(tmp_path),
            entity_id="test-user",
            debounce_seconds=0.1,
        )
        manager.add_watch(watch)

        # Modify the file
        await asyncio.sleep(0.1)  # Give time for watch to start
        test_file.write_text("Modified content")

        # Wait for debounce + processing
        await asyncio.sleep(0.3)

        manager.stop()

        # Verify modification was detected
        assert len(changes_detected) >= 1
        modified_changes = [c for c in changes_detected if c.event_type == "modified"]
        assert len(modified_changes) >= 1

    @pytest.mark.asyncio
    async def test_file_deleted_triggers_removal(self, tmp_path):
        """Deleting a file triggers removal from index."""
        changes_detected = []

        async def mock_callback(watch, changes):
            changes_detected.extend(changes)

        # Create initial file
        test_file = tmp_path / "to_delete.txt"
        test_file.write_text("Content")

        # Setup manager
        loop = asyncio.get_running_loop()
        manager = DirectoryWatchManager()
        manager.initialize(loop, mock_callback)
        manager.start()

        watch = DirectoryWatch(
            watch_id=1,
            directory_path=str(tmp_path),
            entity_id="test-user",
            debounce_seconds=0.1,
        )
        manager.add_watch(watch)

        # Delete the file
        await asyncio.sleep(0.1)
        test_file.unlink()

        # Wait for debounce + processing
        await asyncio.sleep(0.3)

        manager.stop()

        # Verify deletion was detected
        deleted_changes = [c for c in changes_detected if c.event_type == "deleted"]
        assert len(deleted_changes) >= 1

    @pytest.mark.asyncio
    async def test_debounce_prevents_rapid_reindex(self, tmp_path):
        """Multiple rapid changes result in single callback."""
        callback_count = 0

        async def mock_callback(watch, changes):
            nonlocal callback_count
            callback_count += 1

        # Setup manager
        loop = asyncio.get_running_loop()
        manager = DirectoryWatchManager()
        manager.initialize(loop, mock_callback)
        manager.start()

        watch = DirectoryWatch(
            watch_id=1,
            directory_path=str(tmp_path),
            entity_id="test-user",
            debounce_seconds=0.2,  # 200ms debounce
        )
        manager.add_watch(watch)

        # Make rapid changes
        test_file = tmp_path / "rapid.txt"
        for i in range(5):
            test_file.write_text(f"Content {i}")
            await asyncio.sleep(0.05)  # 50ms between changes

        # Wait for debounce to complete
        await asyncio.sleep(0.4)

        manager.stop()

        # Should have only one callback (or at most 2 if timing varies)
        assert callback_count <= 2

    @pytest.mark.asyncio
    async def test_nested_directory_changes(self, tmp_path):
        """Changes in subdirectories are detected with recursive=True."""
        changes_detected = []

        async def mock_callback(watch, changes):
            changes_detected.extend(changes)

        # Create nested structure
        subdir = tmp_path / "subdir"
        subdir.mkdir()

        # Setup manager
        loop = asyncio.get_running_loop()
        manager = DirectoryWatchManager()
        manager.initialize(loop, mock_callback)
        manager.start()

        watch = DirectoryWatch(
            watch_id=1,
            directory_path=str(tmp_path),
            entity_id="test-user",
            recursive=True,
            debounce_seconds=0.1,
        )
        manager.add_watch(watch)

        # Create file in subdirectory
        await asyncio.sleep(0.1)
        nested_file = subdir / "nested.txt"
        nested_file.write_text("Nested content")

        # Wait for debounce
        await asyncio.sleep(0.3)

        manager.stop()

        # Verify nested file was detected
        assert any("nested.txt" in change.filepath for change in changes_detected)


class TestWatchPersistence:
    """Tests for watch persistence across restarts."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        DirectoryWatchManager.reset_instance()
        yield
        DirectoryWatchManager.reset_instance()

    @pytest.mark.asyncio
    async def test_watch_restored_from_database(self, tmp_path):
        """Watches are restored from database on startup."""
        mock_watches = [
            {
                "id": 1,
                "directory_path": str(tmp_path),
                "entity_id": "user1",
                "recursive": True,
                "file_extensions": None,
                "ignore_patterns": [".git"],
                "debounce_seconds": 5,
                "enabled": True,
            }
        ]

        with patch(
            "app.services.directory_service.get_all_enabled_watches",
            new_callable=AsyncMock,
            return_value=mock_watches,
        ):
            from app.services.directory_service import get_all_enabled_watches

            watches = await get_all_enabled_watches()

            # Simulate what main.py does on startup
            manager = DirectoryWatchManager()
            loop = asyncio.get_running_loop()
            manager.initialize(loop, AsyncMock())
            manager.start()

            for watch_data in watches:
                if os.path.isdir(watch_data["directory_path"]):
                    watch = DirectoryWatch(
                        watch_id=watch_data["id"],
                        directory_path=watch_data["directory_path"],
                        entity_id=watch_data["entity_id"],
                        recursive=watch_data["recursive"],
                        extensions=watch_data.get("file_extensions"),
                        ignore_patterns=watch_data.get("ignore_patterns") or [],
                        debounce_seconds=watch_data.get("debounce_seconds", 5),
                        enabled=True,
                    )
                    manager.add_watch(watch)

            # Verify watch was restored
            assert len(manager.get_all_watches()) == 1
            restored_watch = manager.get_watch(1)
            assert restored_watch.directory_path == str(tmp_path)
            assert restored_watch.entity_id == "user1"

            manager.stop()

    @pytest.mark.asyncio
    async def test_missing_directory_skipped(self, tmp_path):
        """Watches for non-existent directories are skipped."""
        mock_watches = [
            {
                "id": 1,
                "directory_path": "/nonexistent/path",
                "entity_id": "user1",
                "recursive": True,
                "file_extensions": None,
                "ignore_patterns": [],
                "debounce_seconds": 5,
                "enabled": True,
            },
            {
                "id": 2,
                "directory_path": str(tmp_path),
                "entity_id": "user2",
                "recursive": True,
                "file_extensions": None,
                "ignore_patterns": [],
                "debounce_seconds": 5,
                "enabled": True,
            },
        ]

        manager = DirectoryWatchManager()
        loop = asyncio.get_running_loop()
        manager.initialize(loop, AsyncMock())
        manager.start()

        for watch_data in mock_watches:
            if os.path.isdir(watch_data["directory_path"]):
                watch = DirectoryWatch(
                    watch_id=watch_data["id"],
                    directory_path=watch_data["directory_path"],
                    entity_id=watch_data["entity_id"],
                    recursive=watch_data["recursive"],
                    extensions=watch_data.get("file_extensions"),
                    ignore_patterns=watch_data.get("ignore_patterns") or [],
                    debounce_seconds=watch_data.get("debounce_seconds", 5),
                    enabled=True,
                )
                manager.add_watch(watch)

        # Only the existing directory should be watched
        assert len(manager.get_all_watches()) == 1
        assert manager.get_watch(2) is not None
        assert manager.get_watch(1) is None

        manager.stop()


class TestHandleFileChanges:
    """Tests for the handle_file_changes function in main.py."""

    @pytest.mark.asyncio
    async def test_handle_created_file(self, tmp_path):
        """Created file triggers sync_single_file."""
        from main import handle_file_changes

        watch = DirectoryWatch(
            watch_id=1,
            directory_path=str(tmp_path),
            entity_id="test-user",
        )

        changes = [
            PendingChange(
                filepath=str(tmp_path / "new.txt"),
                event_type="created",
                timestamp=datetime.now(timezone.utc),
            )
        ]

        mock_executor = MagicMock()

        with patch(
            "main.sync_single_file",
            new_callable=AsyncMock,
        ) as mock_sync, patch(
            "main.update_watch_last_sync",
            new_callable=AsyncMock,
        ):
            await handle_file_changes(watch, changes, mock_executor)

        mock_sync.assert_called_once()
        call_kwargs = mock_sync.call_args[1]
        assert call_kwargs["delete_if_missing"] is False

    @pytest.mark.asyncio
    async def test_handle_deleted_file(self, tmp_path):
        """Deleted file triggers sync with delete_if_missing=True."""
        from main import handle_file_changes

        watch = DirectoryWatch(
            watch_id=1,
            directory_path=str(tmp_path),
            entity_id="test-user",
        )

        changes = [
            PendingChange(
                filepath=str(tmp_path / "deleted.txt"),
                event_type="deleted",
                timestamp=datetime.now(timezone.utc),
            )
        ]

        mock_executor = MagicMock()

        with patch(
            "main.sync_single_file",
            new_callable=AsyncMock,
        ) as mock_sync, patch(
            "main.update_watch_last_sync",
            new_callable=AsyncMock,
        ):
            await handle_file_changes(watch, changes, mock_executor)

        mock_sync.assert_called_once()
        call_kwargs = mock_sync.call_args[1]
        assert call_kwargs["delete_if_missing"] is True

    @pytest.mark.asyncio
    async def test_handle_multiple_changes(self, tmp_path):
        """Multiple changes are all processed."""
        from main import handle_file_changes

        watch = DirectoryWatch(
            watch_id=1,
            directory_path=str(tmp_path),
            entity_id="test-user",
        )

        changes = [
            PendingChange(
                filepath=str(tmp_path / "file1.txt"),
                event_type="created",
                timestamp=datetime.now(timezone.utc),
            ),
            PendingChange(
                filepath=str(tmp_path / "file2.txt"),
                event_type="modified",
                timestamp=datetime.now(timezone.utc),
            ),
            PendingChange(
                filepath=str(tmp_path / "file3.txt"),
                event_type="deleted",
                timestamp=datetime.now(timezone.utc),
            ),
        ]

        mock_executor = MagicMock()

        with patch(
            "main.sync_single_file",
            new_callable=AsyncMock,
        ) as mock_sync, patch(
            "main.update_watch_last_sync",
            new_callable=AsyncMock,
        ):
            await handle_file_changes(watch, changes, mock_executor)

        assert mock_sync.call_count == 3

    @pytest.mark.asyncio
    async def test_handle_changes_error_handling(self, tmp_path):
        """Errors in one file don't prevent processing others."""
        from main import handle_file_changes

        watch = DirectoryWatch(
            watch_id=1,
            directory_path=str(tmp_path),
            entity_id="test-user",
        )

        changes = [
            PendingChange(
                filepath=str(tmp_path / "file1.txt"),
                event_type="created",
                timestamp=datetime.now(timezone.utc),
            ),
            PendingChange(
                filepath=str(tmp_path / "file2.txt"),
                event_type="created",
                timestamp=datetime.now(timezone.utc),
            ),
        ]

        mock_executor = MagicMock()

        with patch(
            "main.sync_single_file",
            new_callable=AsyncMock,
            side_effect=[Exception("Error"), None],  # First fails, second succeeds
        ) as mock_sync, patch(
            "main.update_watch_last_sync",
            new_callable=AsyncMock,
        ):
            # Should not raise, should continue processing
            await handle_file_changes(watch, changes, mock_executor)

        # Both files should have been attempted
        assert mock_sync.call_count == 2

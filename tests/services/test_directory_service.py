# tests/services/test_directory_service.py
"""Tests for directory service operations."""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone

from app.models import FileIndexResult, FileSyncResult, SyncAction
from app.utils.directory_watcher import DirectoryWatch


@contextmanager
def mock_path_security():
    """Mock validate_path_security to allow all paths in tests."""
    # The function is imported inside index_single_file and sync_single_file,
    # so we need to patch it at the source module
    with patch(
        "app.utils.directory_watcher.validate_path_security",
        side_effect=lambda path, check_exists=True: path,
    ):
        yield


def create_mock_pool(mock_conn):
    """Create a mock pool with proper async context manager for acquire()."""
    mock_pool = MagicMock()

    @asynccontextmanager
    async def mock_acquire():
        yield mock_conn

    mock_pool.acquire = mock_acquire
    return mock_pool


class TestEnsureDirectoryTables:
    """Tests for ensure_directory_tables function."""

    @pytest.mark.asyncio
    async def test_creates_indexed_files_table(self):
        """Creates indexed_files table if not exists."""
        mock_conn = AsyncMock()
        mock_pool = create_mock_pool(mock_conn)

        with patch(
            "app.services.directory_service.PSQLDatabase.get_pool",
            new_callable=AsyncMock,
            return_value=mock_pool,
        ):
            from app.services.directory_service import ensure_directory_tables

            await ensure_directory_tables()

        # Verify CREATE TABLE was called
        calls = mock_conn.execute.call_args_list
        sql_calls = [str(call) for call in calls]
        assert any("indexed_files" in str(call) for call in sql_calls)

    @pytest.mark.asyncio
    async def test_creates_watched_directories_table(self):
        """Creates watched_directories table if not exists."""
        mock_conn = AsyncMock()
        mock_pool = create_mock_pool(mock_conn)

        with patch(
            "app.services.directory_service.PSQLDatabase.get_pool",
            new_callable=AsyncMock,
            return_value=mock_pool,
        ):
            from app.services.directory_service import ensure_directory_tables

            await ensure_directory_tables()

        calls = mock_conn.execute.call_args_list
        assert any("watched_directories" in str(call) for call in calls)

    @pytest.mark.asyncio
    async def test_creates_indexes(self):
        """Creates indexes on tracking tables."""
        mock_conn = AsyncMock()
        mock_pool = create_mock_pool(mock_conn)

        with patch(
            "app.services.directory_service.PSQLDatabase.get_pool",
            new_callable=AsyncMock,
            return_value=mock_pool,
        ):
            from app.services.directory_service import ensure_directory_tables

            await ensure_directory_tables()

        calls = mock_conn.execute.call_args_list
        assert any("CREATE INDEX" in str(call) for call in calls)


class TestIndexedFileCRUD:
    """Tests for indexed_files CRUD operations."""

    @pytest.mark.asyncio
    async def test_upsert_new_file(self):
        """Insert a new indexed file record."""
        mock_conn = AsyncMock()
        mock_pool = create_mock_pool(mock_conn)

        with patch(
            "app.services.directory_service.PSQLDatabase.get_pool",
            new_callable=AsyncMock,
            return_value=mock_pool,
        ):
            from app.services.directory_service import upsert_indexed_file

            await upsert_indexed_file(
                file_id="abc123",
                filepath="/path/to/file.txt",
                entity_id="user123",
                file_hash="hash123",
                file_mtime=datetime.now(timezone.utc),
                file_size=1024,
                status="indexed",
            )

        mock_conn.execute.assert_called_once()
        call_args = str(mock_conn.execute.call_args)
        assert "INSERT INTO indexed_files" in call_args

    @pytest.mark.asyncio
    async def test_upsert_update_existing(self):
        """Update an existing indexed file record."""
        mock_conn = AsyncMock()
        mock_pool = create_mock_pool(mock_conn)

        with patch(
            "app.services.directory_service.PSQLDatabase.get_pool",
            new_callable=AsyncMock,
            return_value=mock_pool,
        ):
            from app.services.directory_service import upsert_indexed_file

            await upsert_indexed_file(
                file_id="abc123",
                filepath="/path/to/file.txt",
                entity_id="user123",
                file_hash="newhash",
                file_mtime=datetime.now(timezone.utc),
                file_size=2048,
                status="indexed",
            )

        call_args = str(mock_conn.execute.call_args)
        assert "ON CONFLICT" in call_args
        assert "DO UPDATE" in call_args

    @pytest.mark.asyncio
    async def test_get_indexed_file_exists(self):
        """Get an existing indexed file record."""
        mock_row = {
            "id": 1,
            "file_id": "abc123",
            "filepath": "/path/to/file.txt",
            "entity_id": "user123",
            "file_hash": "hash123",
            "file_mtime": datetime.now(timezone.utc),
            "file_size": 1024,
            "indexed_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
            "status": "indexed",
        }

        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = mock_row
        mock_pool = create_mock_pool(mock_conn)

        with patch(
            "app.services.directory_service.PSQLDatabase.get_pool",
            new_callable=AsyncMock,
            return_value=mock_pool,
        ):
            from app.services.directory_service import get_indexed_file

            result = await get_indexed_file("/path/to/file.txt", "user123")

        assert result == mock_row

    @pytest.mark.asyncio
    async def test_get_indexed_file_not_found(self):
        """Get returns None for non-existent file."""
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = None
        mock_pool = create_mock_pool(mock_conn)

        with patch(
            "app.services.directory_service.PSQLDatabase.get_pool",
            new_callable=AsyncMock,
            return_value=mock_pool,
        ):
            from app.services.directory_service import get_indexed_file

            result = await get_indexed_file("/nonexistent/file.txt", "user123")

        assert result is None

    @pytest.mark.asyncio
    async def test_delete_indexed_file(self):
        """Delete an indexed file record."""
        mock_conn = AsyncMock()
        mock_pool = create_mock_pool(mock_conn)

        with patch(
            "app.services.directory_service.PSQLDatabase.get_pool",
            new_callable=AsyncMock,
            return_value=mock_pool,
        ):
            from app.services.directory_service import delete_indexed_file

            await delete_indexed_file("/path/to/file.txt", "user123")

        mock_conn.execute.assert_called_once()
        call_args = str(mock_conn.execute.call_args)
        assert "DELETE FROM indexed_files" in call_args

    @pytest.mark.asyncio
    async def test_get_indexed_files_in_directory(self):
        """Get all indexed files under a directory."""
        mock_rows = [
            {"filepath": "/dir/file1.txt", "file_id": "id1"},
            {"filepath": "/dir/file2.txt", "file_id": "id2"},
        ]

        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = mock_rows
        mock_pool = create_mock_pool(mock_conn)

        with patch(
            "app.services.directory_service.PSQLDatabase.get_pool",
            new_callable=AsyncMock,
            return_value=mock_pool,
        ):
            from app.services.directory_service import get_indexed_files_in_directory

            result = await get_indexed_files_in_directory("/dir", "user123")

        assert len(result) == 2
        assert result[0]["filepath"] == "/dir/file1.txt"


class TestWatchCRUD:
    """Tests for watched_directories CRUD operations."""

    @pytest.mark.asyncio
    async def test_save_new_watch(self):
        """Save a new watch configuration."""
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = {"id": 1}
        mock_pool = create_mock_pool(mock_conn)

        watch = DirectoryWatch(
            watch_id=0,
            directory_path="/path/to/dir",
            entity_id="user123",
            recursive=True,
            debounce_seconds=5,
        )

        with patch(
            "app.services.directory_service.PSQLDatabase.get_pool",
            new_callable=AsyncMock,
            return_value=mock_pool,
        ):
            from app.services.directory_service import save_watch_to_db

            result = await save_watch_to_db(watch)

        assert result == 1

    @pytest.mark.asyncio
    async def test_get_watch_exists(self):
        """Get an existing watch configuration."""
        mock_row = {
            "id": 1,
            "directory_path": "/path/to/dir",
            "entity_id": "user123",
            "recursive": True,
            "file_extensions": None,
            "ignore_patterns": [".git"],
            "debounce_seconds": 5,
            "enabled": True,
            "last_sync_at": None,
        }

        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = mock_row
        mock_pool = create_mock_pool(mock_conn)

        with patch(
            "app.services.directory_service.PSQLDatabase.get_pool",
            new_callable=AsyncMock,
            return_value=mock_pool,
        ):
            from app.services.directory_service import get_watch_from_db

            result = await get_watch_from_db("/path/to/dir", "user123")

        assert result == mock_row

    @pytest.mark.asyncio
    async def test_get_watch_not_found(self):
        """Get returns None for non-existent watch."""
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = None
        mock_pool = create_mock_pool(mock_conn)

        with patch(
            "app.services.directory_service.PSQLDatabase.get_pool",
            new_callable=AsyncMock,
            return_value=mock_pool,
        ):
            from app.services.directory_service import get_watch_from_db

            result = await get_watch_from_db("/nonexistent/dir", "user123")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_all_enabled_watches(self):
        """Get all enabled watch configurations."""
        mock_rows = [
            {"id": 1, "directory_path": "/dir1", "entity_id": "user1"},
            {"id": 2, "directory_path": "/dir2", "entity_id": "user2"},
        ]

        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = mock_rows
        mock_pool = create_mock_pool(mock_conn)

        with patch(
            "app.services.directory_service.PSQLDatabase.get_pool",
            new_callable=AsyncMock,
            return_value=mock_pool,
        ):
            from app.services.directory_service import get_all_enabled_watches

            result = await get_all_enabled_watches()

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_delete_watch(self):
        """Delete a watch configuration."""
        mock_conn = AsyncMock()
        mock_pool = create_mock_pool(mock_conn)

        with patch(
            "app.services.directory_service.PSQLDatabase.get_pool",
            new_callable=AsyncMock,
            return_value=mock_pool,
        ):
            from app.services.directory_service import delete_watch_from_db

            await delete_watch_from_db("/path/to/dir", "user123")

        mock_conn.execute.assert_called_once()
        call_args = str(mock_conn.execute.call_args)
        assert "DELETE FROM watched_directories" in call_args


class TestIndexSingleFile:
    """Tests for index_single_file function."""

    @pytest.mark.asyncio
    async def test_index_success(self, tmp_path):
        """File is indexed successfully."""
        # Create a test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")

        mock_executor = MagicMock()
        mock_store_result = {"message": "Success", "ids": ["id1", "id2"]}

        # Create mock documents
        mock_doc = Mock()
        mock_doc.page_content = "content"
        mock_doc.metadata = {}

        with mock_path_security(), patch(
            "app.services.directory_service.get_loader"
        ) as mock_get_loader, patch(
            "app.services.directory_service.cleanup_temp_encoding_file"
        ), patch(
            "app.routes.document_routes.store_data_in_vector_db",
            new_callable=AsyncMock,
            return_value=mock_store_result,
        ), patch(
            "app.services.directory_service.run_in_executor",
            new_callable=AsyncMock,
            return_value=[mock_doc],
        ):
            mock_loader = Mock()
            mock_loader.load.return_value = [mock_doc]
            mock_get_loader.return_value = (mock_loader, True, "txt")

            from app.services.directory_service import index_single_file

            result = await index_single_file(
                filepath=str(test_file),
                entity_id="user123",
                executor=mock_executor,
                track_in_db=False,
            )

        assert result.status == "indexed"
        assert result.filepath == str(test_file)
        assert "2 chunks" in result.message

    @pytest.mark.asyncio
    async def test_index_file_not_found(self, tmp_path):
        """Non-existent file returns error."""
        mock_executor = MagicMock()

        with mock_path_security():
            from app.services.directory_service import index_single_file

            result = await index_single_file(
                filepath=str(tmp_path / "nonexistent.txt"),
                entity_id="user123",
                executor=mock_executor,
                track_in_db=False,
            )

            assert result.status == "error"
            assert "No such file" in result.message or "not found" in result.message.lower()

    @pytest.mark.asyncio
    async def test_index_tracks_in_db(self, tmp_path):
        """When track_in_db=True, records are saved to database."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")

        mock_executor = MagicMock()
        mock_store_result = {"message": "Success", "ids": ["id1"]}

        # Create mock document
        mock_doc = Mock()
        mock_doc.page_content = "content"
        mock_doc.metadata = {}

        # Mock pool for upsert
        mock_conn = AsyncMock()
        mock_pool = create_mock_pool(mock_conn)

        with mock_path_security(), patch(
            "app.services.directory_service.get_loader"
        ) as mock_get_loader, patch(
            "app.services.directory_service.cleanup_temp_encoding_file"
        ), patch(
            "app.routes.document_routes.store_data_in_vector_db",
            new_callable=AsyncMock,
            return_value=mock_store_result,
        ), patch(
            "app.services.directory_service.run_in_executor",
            new_callable=AsyncMock,
            return_value=[mock_doc],
        ), patch(
            "app.services.directory_service.upsert_indexed_file",
            new_callable=AsyncMock,
        ) as mock_upsert:
            mock_loader = Mock()
            mock_loader.load.return_value = [mock_doc]
            mock_get_loader.return_value = (mock_loader, True, "txt")

            from app.services.directory_service import index_single_file

            result = await index_single_file(
                filepath=str(test_file),
                entity_id="user123",
                executor=mock_executor,
                track_in_db=True,
            )

        assert result.status == "indexed"
        mock_upsert.assert_called_once()


class TestSyncSingleFile:
    """Tests for sync_single_file function."""

    @pytest.mark.asyncio
    async def test_sync_new_file(self, tmp_path):
        """New file is indexed."""
        test_file = tmp_path / "new.txt"
        test_file.write_text("New content")

        mock_executor = MagicMock()

        with mock_path_security(), patch(
            "app.services.directory_service.run_in_executor",
            new_callable=AsyncMock,
            return_value="hash123",  # Mock hash computation
        ), patch(
            "app.services.directory_service.get_indexed_file",
            new_callable=AsyncMock,
            return_value=None,
        ), patch(
            "app.services.directory_service.index_single_file",
            new_callable=AsyncMock,
            return_value=FileIndexResult(
                filepath=str(test_file),
                file_id="new_id",
                status="indexed",
                message="Success",
            ),
        ):
            from app.services.directory_service import sync_single_file

            result = await sync_single_file(
                filepath=str(test_file),
                entity_id="user123",
                executor=mock_executor,
            )

        assert result.action == SyncAction.INDEXED

    @pytest.mark.asyncio
    async def test_sync_unchanged_file(self, tmp_path):
        """Unchanged file returns UNCHANGED action."""
        test_file = tmp_path / "existing.txt"
        test_file.write_text("Content")

        mock_executor = MagicMock()
        existing_record = {
            "file_id": "existing_id",
            "file_hash": "hash123",
        }

        with mock_path_security(), patch(
            "app.services.directory_service.get_indexed_file",
            new_callable=AsyncMock,
            return_value=existing_record,
        ), patch(
            "app.services.directory_service.run_in_executor",
            new_callable=AsyncMock,
            return_value="hash123",  # Same hash
        ):
            from app.services.directory_service import sync_single_file

            result = await sync_single_file(
                filepath=str(test_file),
                entity_id="user123",
                executor=mock_executor,
            )

        assert result.action == SyncAction.UNCHANGED

    @pytest.mark.asyncio
    async def test_sync_modified_file(self, tmp_path):
        """Modified file is re-indexed."""
        test_file = tmp_path / "modified.txt"
        test_file.write_text("New content")

        mock_executor = MagicMock()
        existing_record = {
            "file_id": "existing_id",
            "file_hash": "old_hash",
        }

        with mock_path_security(), patch(
            "app.services.directory_service.get_indexed_file",
            new_callable=AsyncMock,
            return_value=existing_record,
        ), patch(
            "app.services.directory_service.run_in_executor",
            new_callable=AsyncMock,
            return_value="new_hash",  # Different hash
        ), patch(
            "app.services.directory_service.delete_file_from_vector_store",
            new_callable=AsyncMock,
        ), patch(
            "app.services.directory_service.index_single_file",
            new_callable=AsyncMock,
            return_value=FileIndexResult(
                filepath=str(test_file),
                file_id="new_id",
                status="indexed",
                message="Success",
            ),
        ):
            from app.services.directory_service import sync_single_file

            result = await sync_single_file(
                filepath=str(test_file),
                entity_id="user123",
                executor=mock_executor,
            )

        assert result.action == SyncAction.UPDATED

    @pytest.mark.asyncio
    async def test_sync_deleted_file(self, tmp_path):
        """Missing file is removed from index."""
        mock_executor = MagicMock()
        existing_record = {
            "file_id": "existing_id",
            "file_hash": "hash123",
        }

        with mock_path_security(), patch(
            "app.services.directory_service.get_indexed_file",
            new_callable=AsyncMock,
            return_value=existing_record,
        ), patch(
            "app.services.directory_service.delete_file_from_vector_store",
            new_callable=AsyncMock,
        ), patch(
            "app.services.directory_service.delete_indexed_file",
            new_callable=AsyncMock,
        ):
            from app.services.directory_service import sync_single_file

            result = await sync_single_file(
                filepath=str(tmp_path / "deleted.txt"),
                entity_id="user123",
                executor=mock_executor,
                delete_if_missing=True,
            )

        assert result.action == SyncAction.DELETED

    @pytest.mark.asyncio
    async def test_sync_deleted_no_remove(self, tmp_path):
        """With delete_if_missing=False, missing file returns UNCHANGED."""
        mock_executor = MagicMock()

        with mock_path_security():
            from app.services.directory_service import sync_single_file

            result = await sync_single_file(
                filepath=str(tmp_path / "deleted.txt"),
                entity_id="user123",
                executor=mock_executor,
                delete_if_missing=False,
            )

        assert result.action == SyncAction.UNCHANGED


class TestDeleteFileFromVectorStore:
    """Tests for delete_file_from_vector_store function."""

    @pytest.mark.asyncio
    async def test_delete_success(self):
        """Successfully deletes from vector store."""
        mock_executor = MagicMock()
        mock_vector_store = AsyncMock()

        with patch(
            "app.services.directory_service.vector_store", mock_vector_store
        ), patch(
            "app.services.directory_service.AsyncPgVector", type(mock_vector_store)
        ):
            from app.services.directory_service import delete_file_from_vector_store

            result = await delete_file_from_vector_store("file_id", mock_executor)

        assert result is True
        mock_vector_store.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_failure(self):
        """Returns False on deletion failure."""
        mock_executor = MagicMock()
        mock_vector_store = AsyncMock()
        mock_vector_store.delete.side_effect = Exception("DB error")

        with patch(
            "app.services.directory_service.vector_store", mock_vector_store
        ), patch(
            "app.services.directory_service.AsyncPgVector", type(mock_vector_store)
        ):
            from app.services.directory_service import delete_file_from_vector_store

            result = await delete_file_from_vector_store("file_id", mock_executor)

        assert result is False

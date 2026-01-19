# app/services/directory_service.py
"""
Directory indexing service for managing file tracking and sync operations.

This module provides:
- Database table creation for tracking indexed files and watches
- CRUD operations for indexed_files and watched_directories tables
- File indexing logic that integrates with the vector store
- Sync operations for detecting and handling file changes
"""

import os
import traceback
from datetime import datetime, timezone
from typing import Optional, List

from langchain_core.runnables import run_in_executor

from app.config import logger, vector_store
from app.services.database import PSQLDatabase
from app.services.vector_store.async_pg_vector import AsyncPgVector
from app.utils.directory_watcher import (
    generate_file_id,
    compute_file_hash,
    get_file_metadata,
    discover_files,
    DirectoryWatch,
)
from app.utils.document_loader import get_loader, cleanup_temp_encoding_file
from app.models import FileIndexResult, FileSyncResult, SyncAction


# Import store_data_in_vector_db - this needs to be done carefully to avoid circular imports
# We'll import it inside the function that needs it


async def ensure_directory_tables():
    """
    Create directory tracking tables if they don't exist.

    Creates:
    - indexed_files: Tracks individual files that have been indexed
    - watched_directories: Persists watch configurations across restarts
    """
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        # indexed_files table
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS indexed_files (
                id SERIAL PRIMARY KEY,
                file_id VARCHAR(64) NOT NULL,
                filepath TEXT NOT NULL,
                entity_id VARCHAR(255) NOT NULL,
                file_hash VARCHAR(32),
                file_mtime TIMESTAMP WITH TIME ZONE,
                file_size BIGINT,
                indexed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                status VARCHAR(20) DEFAULT 'indexed',
                error_message TEXT,
                UNIQUE(filepath, entity_id)
            );
            """
        )

        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_indexed_files_file_id
            ON indexed_files(file_id);
            """
        )

        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_indexed_files_filepath
            ON indexed_files(filepath);
            """
        )

        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_indexed_files_entity_id
            ON indexed_files(entity_id);
            """
        )

        # watched_directories table
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS watched_directories (
                id SERIAL PRIMARY KEY,
                directory_path TEXT NOT NULL,
                entity_id VARCHAR(255) NOT NULL,
                recursive BOOLEAN DEFAULT TRUE,
                file_extensions TEXT[],
                ignore_patterns TEXT[] DEFAULT ARRAY['.git', '__pycache__', 'node_modules'],
                debounce_seconds INTEGER DEFAULT 5,
                enabled BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                last_sync_at TIMESTAMP WITH TIME ZONE,
                UNIQUE(directory_path, entity_id)
            );
            """
        )

        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_watched_directories_entity_id
            ON watched_directories(entity_id);
            """
        )

        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_watched_directories_enabled
            ON watched_directories(enabled);
            """
        )

        logger.info("Directory tracking tables ensured")


# --- Indexed Files CRUD ---


async def get_indexed_file(filepath: str, entity_id: str) -> Optional[dict]:
    """
    Get indexed file record from database.

    Args:
        filepath: Absolute path to the file
        entity_id: User/entity identifier

    Returns:
        Dict with file record or None if not found
    """
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, file_id, filepath, entity_id, file_hash,
                   file_mtime, file_size, indexed_at, updated_at, status
            FROM indexed_files
            WHERE filepath = $1 AND entity_id = $2
            """,
            filepath,
            entity_id,
        )

        if row:
            return dict(row)
        return None


async def upsert_indexed_file(
    file_id: str,
    filepath: str,
    entity_id: str,
    file_hash: Optional[str],
    file_mtime: Optional[datetime],
    file_size: Optional[int],
    status: str = "indexed",
    error_message: Optional[str] = None,
):
    """
    Insert or update indexed file record.

    Args:
        file_id: Generated file identifier
        filepath: Absolute path to the file
        entity_id: User/entity identifier
        file_hash: MD5 hash of file content
        file_mtime: File modification time
        file_size: File size in bytes
        status: Status of indexing ('indexed', 'error')
        error_message: Error message if status is 'error'
    """
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO indexed_files
                (file_id, filepath, entity_id, file_hash, file_mtime, file_size, status, error_message)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (filepath, entity_id)
            DO UPDATE SET
                file_id = EXCLUDED.file_id,
                file_hash = EXCLUDED.file_hash,
                file_mtime = EXCLUDED.file_mtime,
                file_size = EXCLUDED.file_size,
                status = EXCLUDED.status,
                error_message = EXCLUDED.error_message,
                updated_at = NOW()
            """,
            file_id,
            filepath,
            entity_id,
            file_hash,
            file_mtime,
            file_size,
            status,
            error_message,
        )


async def delete_indexed_file(filepath: str, entity_id: str):
    """
    Delete indexed file record.

    Args:
        filepath: Absolute path to the file
        entity_id: User/entity identifier
    """
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            DELETE FROM indexed_files
            WHERE filepath = $1 AND entity_id = $2
            """,
            filepath,
            entity_id,
        )


async def get_indexed_files_in_directory(directory: str, entity_id: str) -> List[dict]:
    """
    Get all indexed files under a directory.

    Args:
        directory: Directory path prefix
        entity_id: User/entity identifier

    Returns:
        List of file records
    """
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, file_id, filepath, entity_id, file_hash,
                   file_mtime, file_size, indexed_at, updated_at, status
            FROM indexed_files
            WHERE filepath LIKE $1 AND entity_id = $2
            """,
            f"{directory}%",
            entity_id,
        )

        return [dict(row) for row in rows]


async def get_indexed_files_by_entity(entity_id: str) -> List[dict]:
    """
    Get all indexed files for an entity.

    Args:
        entity_id: User/entity identifier

    Returns:
        List of file records with file_id, filepath, entity_id, indexed_at,
        file_mtime, file_size, updated_at
    """
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT file_id, filepath, entity_id, indexed_at,
                   file_mtime, file_size, updated_at
            FROM indexed_files
            WHERE entity_id = $1 AND status = 'indexed'
            ORDER BY filepath
            """,
            entity_id,
        )
        return [dict(row) for row in rows]


# --- Watched Directories CRUD ---


async def save_watch_to_db(watch: DirectoryWatch) -> int:
    """
    Save or update watch configuration in database.

    Args:
        watch: Watch configuration

    Returns:
        Database ID of the watch
    """
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO watched_directories
                (directory_path, entity_id, recursive, file_extensions,
                 ignore_patterns, debounce_seconds, enabled)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (directory_path, entity_id)
            DO UPDATE SET
                recursive = EXCLUDED.recursive,
                file_extensions = EXCLUDED.file_extensions,
                ignore_patterns = EXCLUDED.ignore_patterns,
                debounce_seconds = EXCLUDED.debounce_seconds,
                enabled = EXCLUDED.enabled,
                updated_at = NOW()
            RETURNING id
            """,
            watch.directory_path,
            watch.entity_id,
            watch.recursive,
            watch.extensions,
            watch.ignore_patterns,
            watch.debounce_seconds,
            watch.enabled,
        )
        return row["id"]


async def get_watch_from_db(directory_path: str, entity_id: str) -> Optional[dict]:
    """
    Get watch configuration from database.

    Args:
        directory_path: Absolute path to the watched directory
        entity_id: User/entity identifier

    Returns:
        Dict with watch record or None if not found
    """
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, directory_path, entity_id, recursive, file_extensions,
                   ignore_patterns, debounce_seconds, enabled, last_sync_at
            FROM watched_directories
            WHERE directory_path = $1 AND entity_id = $2
            """,
            directory_path,
            entity_id,
        )

        return dict(row) if row else None


async def get_all_enabled_watches() -> List[dict]:
    """
    Get all enabled watches from database.

    Returns:
        List of watch records
    """
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, directory_path, entity_id, recursive, file_extensions,
                   ignore_patterns, debounce_seconds, enabled, last_sync_at
            FROM watched_directories
            WHERE enabled = TRUE
            """
        )
        return [dict(row) for row in rows]


async def delete_watch_from_db(directory_path: str, entity_id: str):
    """
    Delete watch configuration from database.

    Args:
        directory_path: Absolute path to the watched directory
        entity_id: User/entity identifier
    """
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            DELETE FROM watched_directories
            WHERE directory_path = $1 AND entity_id = $2
            """,
            directory_path,
            entity_id,
        )


async def update_watch_last_sync(watch_id: int):
    """
    Update the last_sync_at timestamp for a watch.

    Args:
        watch_id: Database ID of the watch
    """
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE watched_directories
            SET last_sync_at = NOW()
            WHERE id = $1
            """,
            watch_id,
        )


# --- File Indexing Operations ---


async def delete_file_from_vector_store(file_id: str, executor) -> bool:
    """
    Delete a file's documents from the vector store.

    Args:
        file_id: File identifier
        executor: Thread pool executor

    Returns:
        True if deletion succeeded
    """
    try:
        if isinstance(vector_store, AsyncPgVector):
            await vector_store.delete(ids=[file_id], executor=executor)
        else:
            await run_in_executor(executor, vector_store.delete, [file_id])
        return True
    except Exception as e:
        logger.error(f"Failed to delete file_id {file_id} from vector store: {e}")
        return False


async def index_single_file(
    filepath: str,
    entity_id: str,
    executor,
    track_in_db: bool = False,
) -> FileIndexResult:
    """
    Index a single file into the vector store.

    Args:
        filepath: Absolute path to the file
        entity_id: User/entity ID
        executor: Thread pool executor
        track_in_db: Whether to track in indexed_files table

    Returns:
        FileIndexResult with status

    Note: Caller is responsible for validating filepath is under an allowed
    directory before calling this function. This function performs a defensive
    re-validation to mitigate TOCTOU race conditions.
    """
    # Import here to avoid circular imports
    from app.routes.document_routes import store_data_in_vector_db
    from app.utils.directory_watcher import validate_path_security, PathSecurityError

    # Defensive re-validation to mitigate TOCTOU
    try:
        filepath = validate_path_security(filepath, check_exists=True)
    except PathSecurityError as e:
        return FileIndexResult(
            filepath=filepath,
            file_id=generate_file_id(filepath, entity_id),
            status="error",
            message=f"Security validation failed: {e}",
        )

    file_id = generate_file_id(filepath, entity_id)
    filename = os.path.basename(filepath)

    try:
        # Get file metadata
        metadata = get_file_metadata(filepath)

        # Determine content type from extension
        ext = os.path.splitext(filename)[1].lower()
        content_type_map = {
            ".pdf": "application/pdf",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".doc": "application/msword",
            ".txt": "text/plain",
            ".md": "text/markdown",
            ".csv": "text/csv",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ".xls": "application/vnd.ms-excel",
            ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            ".ppt": "application/vnd.ms-powerpoint",
            ".json": "application/json",
            ".xml": "application/xml",
            ".epub": "application/epub+zip",
            ".rst": "text/x-rst",
        }
        content_type = content_type_map.get(ext, "text/plain")

        # Load document
        loader, known_type, file_ext = get_loader(filename, content_type, filepath)
        data = await run_in_executor(executor, loader.load)
        cleanup_temp_encoding_file(loader)

        # Store in vector DB
        result = await store_data_in_vector_db(
            data=data,
            file_id=file_id,
            user_id=entity_id,
            clean_content=(file_ext == "pdf"),
            executor=executor,
        )

        if result and "error" not in result:
            # Track in database if requested
            if track_in_db:
                file_hash = await run_in_executor(executor, compute_file_hash, filepath)
                await upsert_indexed_file(
                    file_id=file_id,
                    filepath=filepath,
                    entity_id=entity_id,
                    file_hash=file_hash,
                    file_mtime=metadata["mtime"],
                    file_size=metadata["size"],
                    status="indexed",
                )

            return FileIndexResult(
                filepath=filepath,
                file_id=file_id,
                status="indexed",
                message=f"Successfully indexed with {len(result.get('ids', []))} chunks",
            )
        else:
            error_msg = result.get("error", "Unknown error") if result else "No result"
            if track_in_db:
                await upsert_indexed_file(
                    file_id=file_id,
                    filepath=filepath,
                    entity_id=entity_id,
                    file_hash=None,
                    file_mtime=metadata["mtime"],
                    file_size=metadata["size"],
                    status="error",
                    error_message=str(error_msg),
                )
            return FileIndexResult(
                filepath=filepath,
                file_id=file_id,
                status="error",
                message=str(error_msg),
            )

    except Exception as e:
        logger.error(f"Failed to index {filepath}: {e}\n{traceback.format_exc()}")
        if track_in_db:
            try:
                await upsert_indexed_file(
                    file_id=file_id,
                    filepath=filepath,
                    entity_id=entity_id,
                    file_hash=None,
                    file_mtime=None,
                    file_size=None,
                    status="error",
                    error_message=str(e),
                )
            except Exception:
                pass
        return FileIndexResult(
            filepath=filepath,
            file_id=file_id,
            status="error",
            message=str(e),
        )


async def sync_single_file(
    filepath: str,
    entity_id: str,
    executor,
    delete_if_missing: bool = True,
) -> FileSyncResult:
    """
    Sync a single file - index if new/modified, delete if missing.

    Args:
        filepath: Absolute path to the file
        entity_id: User/entity ID
        executor: Thread pool executor
        delete_if_missing: Whether to delete from index if file is missing

    Returns:
        FileSyncResult with action taken

    Note: Caller is responsible for validating filepath is under an allowed
    directory before calling this function.
    """
    from app.utils.directory_watcher import validate_path_security, PathSecurityError

    file_id = generate_file_id(filepath, entity_id)

    # Defensive security re-validation (TOCTOU mitigation)
    try:
        filepath = validate_path_security(filepath, check_exists=False)
    except PathSecurityError as e:
        return FileSyncResult(
            filepath=filepath,
            file_id=file_id,
            action=SyncAction.ERROR,
            message=f"Security validation failed: {e}",
        )

    # Check if file exists
    if not os.path.exists(filepath):
        if delete_if_missing:
            # Get existing record and delete from vector store
            existing = await get_indexed_file(filepath, entity_id)
            if existing:
                await delete_file_from_vector_store(existing["file_id"], executor)
                await delete_indexed_file(filepath, entity_id)
                return FileSyncResult(
                    filepath=filepath,
                    file_id=existing["file_id"],
                    action=SyncAction.DELETED,
                    message="File no longer exists, removed from index",
                )
        return FileSyncResult(
            filepath=filepath,
            file_id=file_id,
            action=SyncAction.UNCHANGED,
            message="File does not exist",
        )

    # Get current file metadata
    try:
        current_hash = await run_in_executor(executor, compute_file_hash, filepath)
    except Exception as e:
        return FileSyncResult(
            filepath=filepath,
            file_id=file_id,
            action=SyncAction.ERROR,
            message=f"Failed to read file: {e}",
        )

    # Check existing record
    existing = await get_indexed_file(filepath, entity_id)

    if existing:
        # Compare hash to detect changes
        if existing["file_hash"] == current_hash:
            return FileSyncResult(
                filepath=filepath,
                file_id=file_id,
                action=SyncAction.UNCHANGED,
                message="File unchanged",
            )
        else:
            # File modified - delete old and re-index
            await delete_file_from_vector_store(existing["file_id"], executor)
            result = await index_single_file(
                filepath, entity_id, executor, track_in_db=True
            )
            return FileSyncResult(
                filepath=filepath,
                file_id=file_id,
                action=SyncAction.UPDATED if result.status == "indexed" else SyncAction.ERROR,
                message=result.message,
            )
    else:
        # New file - index it
        result = await index_single_file(
            filepath, entity_id, executor, track_in_db=True
        )
        return FileSyncResult(
            filepath=filepath,
            file_id=file_id,
            action=SyncAction.INDEXED if result.status == "indexed" else SyncAction.ERROR,
            message=result.message,
        )

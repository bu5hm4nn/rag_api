# app/routes/directory_routes.py
"""
Directory indexing endpoints for batch embedding and watching.

This module provides:
- POST /local/embed-directory - Index all files in a directory
- POST /local/sync-directory - Sync directory with change detection
- POST /local/watch-directory - Start watching for changes
- DELETE /local/watch-directory - Stop watching
- GET /local/watch-status - List active watches
"""

import os
from typing import List

from fastapi import APIRouter, Request, HTTPException, status, BackgroundTasks

from app.config import logger, MAX_WATCHES_PER_ENTITY
from app.models import (
    EmbedDirectoryRequest,
    EmbedDirectoryResponse,
    FileIndexResult,
    SyncDirectoryRequest,
    SyncDirectoryResponse,
    FileSyncResult,
    SyncAction,
    WatchDirectoryRequest,
    WatchDirectoryResponse,
    UnwatchDirectoryRequest,
    WatchStatus,
    WatchStatusResponse,
)
from app.utils.directory_watcher import (
    discover_files,
    generate_file_id,
    DirectoryWatch,
    DirectoryWatchManager,
    validate_path_security,
    PathSecurityError,
)
from app.services.directory_service import (
    index_single_file,
    sync_single_file,
    get_indexed_files_in_directory,
    delete_indexed_file,
    delete_file_from_vector_store,
    save_watch_to_db,
    get_watch_from_db,
    delete_watch_from_db,
    update_watch_last_sync,
)
from app.routes.document_routes import get_user_id as _get_user_id_base

router = APIRouter(prefix="/local", tags=["directory"])


async def ensure_directory_watch(
    canonical_path: str,
    entity_id: str,
    manager: DirectoryWatchManager,
    recursive: bool = True,
    file_extensions=None,
    ignore_patterns=None,
    debounce_seconds: int = 10,
) -> int:
    """Ensure a directory has both a persisted watch and an active runtime watch."""
    ignore_patterns = ignore_patterns or [".git", "__pycache__", "node_modules", ".DS_Store"]

    existing = await get_watch_from_db(canonical_path, entity_id)
    watch = DirectoryWatch(
        watch_id=existing["id"] if existing else 0,
        directory_path=canonical_path,
        entity_id=entity_id,
        recursive=recursive,
        extensions=file_extensions,
        ignore_patterns=ignore_patterns,
        debounce_seconds=debounce_seconds,
        enabled=True,
    )
    watch.watch_id = await save_watch_to_db(watch)

    if not manager.get_watch(watch.watch_id):
        manager.add_watch(watch)
        logger.info(
            f"Auto-watch enabled for {canonical_path} -> {entity_id} (watch_id={watch.watch_id})"
        )

    return watch.watch_id


def _sanitize_path_for_error(path: str) -> str:
    """
    Sanitize a path for inclusion in error messages.

    Only shows the basename and immediate parent to avoid leaking
    full directory structure.
    """
    import os.path
    parts = path.split(os.sep)
    if len(parts) <= 2:
        return path
    return os.path.join("...", parts[-2], parts[-1])


def get_user_id(request: Request, entity_id: str = None) -> str:
    """
    Extract and validate user ID for directory operations.

    Security: When a user is authenticated via JWT, they can only access
    their own entity_id. Specifying a different entity_id will use the
    authenticated user's ID instead (with a warning).

    Args:
        request: FastAPI request object
        entity_id: Optional entity ID from request

    Returns:
        Validated entity ID
    """
    # Get base user ID
    base_id = _get_user_id_base(request, None)

    # If user is authenticated (has user state), enforce entity_id matches
    if hasattr(request.state, "user") and request.state.user:
        authenticated_id = request.state.user.get("id")
        if authenticated_id:
            if entity_id and entity_id != authenticated_id:
                logger.warning(
                    f"Entity ID mismatch: requested {entity_id}, authenticated as {authenticated_id}. "
                    f"Using authenticated ID for security."
                )
            return authenticated_id

    # No authentication - use provided entity_id or default
    return entity_id if entity_id else "public"


def validate_directory(directory_path: str) -> str:
    """
    Validate that directory exists, is accessible, and passes security checks.

    Security checks:
    - Path is under an allowed directory (RAG_ALLOWED_LOCAL_PATHS)
    - Path is resolved to canonical form (no symlinks, no ..)
    - Symlinks are blocked unless RAG_FOLLOW_SYMLINKS=true

    Args:
        directory_path: Path to validate

    Returns:
        Canonicalized absolute path

    Raises:
        HTTPException: If directory is invalid or fails security checks
    """
    # Security validation first
    try:
        canonical_path = validate_path_security(directory_path, check_exists=True)
    except PathSecurityError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e),
        )

    if not os.path.isdir(canonical_path):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Path is not a directory",
        )
    if not os.access(canonical_path, os.R_OK):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Directory not readable",
        )

    return canonical_path


@router.post("/embed-directory", response_model=EmbedDirectoryResponse)
async def embed_directory(
    request: Request,
    body: EmbedDirectoryRequest,
):
    """
    Index all supported files in a directory.

    This endpoint walks through the directory (optionally recursively),
    identifies supported file types, and indexes them into the vector store.

    Note: This does NOT track files for sync - use /sync-directory for that.
    """
    canonical_path = validate_directory(body.directory_path)

    entity_id = get_user_id(request, body.entity_id)
    executor = request.app.state.thread_pool

    # Discover files using canonical path
    try:
        files = discover_files(
            directory=canonical_path,
            recursive=body.recursive,
            extensions=body.file_extensions,
            ignore_patterns=body.ignore_patterns,
        )
    except PathSecurityError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    logger.info(
        f"Embedding directory {canonical_path}: found {len(files)} files"
    )

    results: List[FileIndexResult] = []
    indexed = 0
    errors = 0

    for filepath in files:
        try:
            result = await index_single_file(
                filepath=filepath,
                entity_id=entity_id,
                executor=executor,
                track_in_db=False,  # Simple embed, no tracking
            )
            results.append(result)

            if result.status == "indexed":
                indexed += 1
            else:
                errors += 1

        except Exception as e:
            logger.error(f"Failed to index {filepath}: {e}")
            results.append(
                FileIndexResult(
                    filepath=filepath,
                    file_id=generate_file_id(filepath, entity_id),
                    status="error",
                    message=str(e),
                )
            )
            errors += 1

    return EmbedDirectoryResponse(
        directory_path=canonical_path,
        total_files=len(files),
        indexed=indexed,
        skipped=0,  # Reserved for future use
        errors=errors,
        files=results,
    )


@router.post("/sync-directory", response_model=SyncDirectoryResponse)
async def sync_directory(
    request: Request,
    body: SyncDirectoryRequest,
):
    """
    Sync a directory with the vector store.

    This endpoint:
    1. Compares current directory state with tracked indexed files
    2. Indexes new files
    3. Re-indexes modified files (based on content hash)
    4. Optionally removes documents for deleted files

    Files are tracked in the database for efficient future syncs.
    """
    canonical_path = validate_directory(body.directory_path)

    entity_id = get_user_id(request, body.entity_id)
    executor = request.app.state.thread_pool

    # Ensure any synced directory is also watched for future automatic re-indexing.
    try:
        manager = DirectoryWatchManager()
        await ensure_directory_watch(
            canonical_path=canonical_path,
            entity_id=entity_id,
            manager=manager,
            recursive=body.recursive,
            file_extensions=body.file_extensions,
            ignore_patterns=body.ignore_patterns,
            debounce_seconds=10,
        )
    except Exception as e:
        logger.error(f"Failed to auto-enable watch for {canonical_path}: {e}")

    # Discover current files
    try:
        current_files = set(
            discover_files(
                directory=canonical_path,
                recursive=body.recursive,
                extensions=body.file_extensions,
                ignore_patterns=body.ignore_patterns,
            )
        )
    except PathSecurityError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    # Get previously indexed files
    indexed_records = await get_indexed_files_in_directory(
        canonical_path, entity_id
    )
    indexed_paths = {r["filepath"] for r in indexed_records}

    logger.info(
        f"Syncing directory {canonical_path}: "
        f"{len(current_files)} current files, {len(indexed_paths)} tracked"
    )

    results: List[FileSyncResult] = []
    indexed = 0
    updated = 0
    deleted = 0
    unchanged = 0
    errors = 0

    # Process current files (new or modified)
    for filepath in current_files:
        result = await sync_single_file(
            filepath=filepath,
            entity_id=entity_id,
            executor=executor,
            delete_if_missing=False,
        )
        results.append(result)

        if result.action == SyncAction.INDEXED:
            indexed += 1
        elif result.action == SyncAction.UPDATED:
            updated += 1
        elif result.action == SyncAction.UNCHANGED:
            unchanged += 1
        elif result.action == SyncAction.ERROR:
            errors += 1

    # Handle deleted files
    if body.delete_removed:
        deleted_paths = indexed_paths - current_files
        for filepath in deleted_paths:
            record = next(
                (r for r in indexed_records if r["filepath"] == filepath), None
            )
            if record:
                await delete_file_from_vector_store(record["file_id"], executor)
                await delete_indexed_file(filepath, entity_id)
                results.append(
                    FileSyncResult(
                        filepath=filepath,
                        file_id=record["file_id"],
                        action=SyncAction.DELETED,
                        message="File no longer exists",
                    )
                )
                deleted += 1

    return SyncDirectoryResponse(
        directory_path=canonical_path,
        total_files=len(current_files),
        indexed=indexed,
        updated=updated,
        deleted=deleted,
        unchanged=unchanged,
        errors=errors,
        files=results,
    )


async def _get_entity_watch_count(entity_id: str) -> int:
    """Get count of active watches for an entity."""
    manager = DirectoryWatchManager()
    watches = manager.get_all_watches()
    return sum(1 for w in watches if w.entity_id == entity_id)


@router.post("/watch-directory", response_model=WatchDirectoryResponse)
async def watch_directory(
    request: Request,
    body: WatchDirectoryRequest,
    background_tasks: BackgroundTasks,
):
    """
    Start watching a directory for changes.

    This endpoint:
    1. Registers the directory for filesystem monitoring
    2. Optionally performs an initial sync
    3. Auto-indexes files when they change (with debouncing)

    Watch registrations persist across restarts.
    """
    canonical_path = validate_directory(body.directory_path)

    entity_id = get_user_id(request, body.entity_id)
    executor = request.app.state.thread_pool

    # Check if already watching
    existing = await get_watch_from_db(canonical_path, entity_id)

    # Enforce watch limit per entity
    if not existing and MAX_WATCHES_PER_ENTITY > 0:
        current_count = await _get_entity_watch_count(entity_id)
        if current_count >= MAX_WATCHES_PER_ENTITY:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Maximum watches ({MAX_WATCHES_PER_ENTITY}) per entity reached",
            )

    # Create watch configuration
    watch = DirectoryWatch(
        watch_id=existing["id"] if existing else 0,
        directory_path=canonical_path,
        entity_id=entity_id,
        recursive=body.recursive,
        extensions=body.file_extensions,
        ignore_patterns=body.ignore_patterns,
        debounce_seconds=body.debounce_seconds,
        enabled=True,
    )

    # Save to database (get ID for new watches)
    watch.watch_id = await save_watch_to_db(watch)

    # Get the watch manager and add watch
    manager = DirectoryWatchManager()

    if existing and manager.get_watch(existing["id"]):
        return WatchDirectoryResponse(
            watch_id=watch.watch_id,
            directory_path=watch.directory_path,
            entity_id=entity_id,
            status="already_watching",
            message="Directory is already being watched",
        )

    manager.add_watch(watch)

    # Perform initial sync if requested
    if body.initial_sync:
        background_tasks.add_task(_perform_initial_sync, watch, executor)

    return WatchDirectoryResponse(
        watch_id=watch.watch_id,
        directory_path=watch.directory_path,
        entity_id=entity_id,
        status="started",
        message="Directory watch started"
        + (" (initial sync in progress)" if body.initial_sync else ""),
    )


async def _perform_initial_sync(watch: DirectoryWatch, executor):
    """Background task to perform initial sync for a new watch."""
    try:
        files = discover_files(
            directory=watch.directory_path,
            recursive=watch.recursive,
            extensions=watch.extensions,
            ignore_patterns=watch.ignore_patterns,
        )

        for filepath in files:
            await sync_single_file(
                filepath=filepath,
                entity_id=watch.entity_id,
                executor=executor,
                delete_if_missing=False,
            )

        await update_watch_last_sync(watch.watch_id)
        logger.info(f"Initial sync completed for watch {watch.watch_id}")

    except Exception as e:
        logger.error(f"Initial sync failed for watch {watch.watch_id}: {e}")


@router.delete("/watch-directory")
async def unwatch_directory(
    request: Request,
    body: UnwatchDirectoryRequest,
):
    """
    Stop watching a directory.

    Optionally removes all indexed documents from the vector store.
    """
    entity_id = get_user_id(request, body.entity_id)
    executor = request.app.state.thread_pool

    # Validate path (but don't require it to exist - it might have been deleted)
    try:
        canonical_path = validate_path_security(body.directory_path, check_exists=False)
    except PathSecurityError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e),
        )

    # Get existing watch
    existing = await get_watch_from_db(canonical_path, entity_id)
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Directory is not being watched",
        )

    # Remove from watch manager
    manager = DirectoryWatchManager()
    manager.remove_watch(existing["id"])

    # Remove indexed documents if requested
    removed_count = 0
    if body.remove_documents:
        indexed_files = await get_indexed_files_in_directory(canonical_path, entity_id)
        for record in indexed_files:
            await delete_file_from_vector_store(record["file_id"], executor)
            await delete_indexed_file(record["filepath"], entity_id)
            removed_count += 1

    # Remove from database
    await delete_watch_from_db(canonical_path, entity_id)

    return {
        "message": "Directory watch stopped",
        "directory_path": canonical_path,
        "documents_removed": removed_count,
    }


@router.get("/watch-status", response_model=WatchStatusResponse)
async def get_watch_status(request: Request, entity_id: str = None):
    """
    Get status of all watched directories for an entity.
    """
    user_entity_id = get_user_id(request, entity_id)

    manager = DirectoryWatchManager()
    watches = manager.get_all_watches()

    # Filter by entity_id
    user_watches = [w for w in watches if w.entity_id == user_entity_id]

    statuses = []
    for watch in user_watches:
        # Get file count from database
        indexed_files = await get_indexed_files_in_directory(
            watch.directory_path, watch.entity_id
        )

        # Get last_sync_at from database
        watch_data = await get_watch_from_db(watch.directory_path, watch.entity_id)
        last_sync = watch_data.get("last_sync_at") if watch_data else None

        statuses.append(
            WatchStatus(
                watch_id=watch.watch_id,
                directory_path=watch.directory_path,
                entity_id=watch.entity_id,
                recursive=watch.recursive,
                enabled=watch.enabled,
                file_count=len(indexed_files),
                last_sync_at=last_sync,
                pending_changes=manager.get_pending_change_count(watch.watch_id),
            )
        )

    return WatchStatusResponse(watches=statuses)

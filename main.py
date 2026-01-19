# main.py
import asyncio
import os
from typing import List

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

from starlette.responses import JSONResponse

from app.config import (
    VectorDBType,
    debug_mode,
    RAG_HOST,
    RAG_PORT,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    PDF_EXTRACT_IMAGES,
    VECTOR_DB_TYPE,
    LogMiddleware,
    logger,
    vector_store,
)
from app.middleware import security_middleware
from app.routes import document_routes, pgvector_routes, directory_routes
from app.services.database import PSQLDatabase, ensure_vector_indexes
from app.services.vector_store.factory import close_vector_store_connections
from app.services.directory_service import (
    ensure_directory_tables,
    get_all_enabled_watches,
    sync_single_file,
    update_watch_last_sync,
)
from app.utils.directory_watcher import (
    DirectoryWatch,
    DirectoryWatchManager,
    PendingChange,
)


async def handle_file_changes(
    watch: DirectoryWatch, changes: List[PendingChange], executor
):
    """
    Callback for processing file changes from the watch manager.
    This runs in the main async event loop.
    """
    for change in changes:
        try:
            if change.event_type == "deleted":
                await sync_single_file(
                    filepath=change.filepath,
                    entity_id=watch.entity_id,
                    executor=executor,
                    delete_if_missing=True,
                )
            else:
                await sync_single_file(
                    filepath=change.filepath,
                    entity_id=watch.entity_id,
                    executor=executor,
                    delete_if_missing=False,
                )
        except Exception as e:
            logger.error(f"Failed to process change for {change.filepath}: {e}")

    await update_watch_last_sync(watch.watch_id)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic goes here
    # Create bounded thread pool executor based on CPU cores
    max_workers = min(
        int(os.getenv("RAG_THREAD_POOL_SIZE", str(os.cpu_count()))), 8
    )  # Cap at 8
    app.state.thread_pool = ThreadPoolExecutor(
        max_workers=max_workers, thread_name_prefix="rag-worker"
    )
    logger.info(
        f"Initialized thread pool with {max_workers} workers (CPU cores: {os.cpu_count()})"
    )

    watch_manager = None

    if VECTOR_DB_TYPE == VectorDBType.PGVECTOR:
        await PSQLDatabase.get_pool()  # Initialize the pool
        await ensure_vector_indexes()
        await ensure_directory_tables()

        # Initialize and start directory watch manager
        loop = asyncio.get_running_loop()
        watch_manager = DirectoryWatchManager()

        # Create a callback that captures the executor
        async def on_changes(watch: DirectoryWatch, changes: List[PendingChange]):
            await handle_file_changes(watch, changes, app.state.thread_pool)

        watch_manager.initialize(loop, on_changes)
        watch_manager.start()

        # Restore persisted watches from database
        try:
            enabled_watches = await get_all_enabled_watches()
            for watch_data in enabled_watches:
                # Only restore if directory still exists
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
                    watch_manager.add_watch(watch)
                else:
                    logger.warning(
                        f"Skipping watch {watch_data['id']}: "
                        f"directory {watch_data['directory_path']} no longer exists"
                    )

            logger.info(f"Restored {len(watch_manager.get_all_watches())} directory watches")
        except Exception as e:
            logger.error(f"Failed to restore directory watches: {e}")

    yield

    # Cleanup logic
    if watch_manager:
        logger.info("Stopping directory watch manager")
        watch_manager.stop()

    if VECTOR_DB_TYPE == VectorDBType.PGVECTOR:
        try:
            logger.info("Closing asyncpg connection pool")
            await PSQLDatabase.close_pool()
            logger.info("asyncpg connection pool closed")
        except Exception as e:
            logger.warning("Failed to close asyncpg pool: %s", e)

    logger.info("Shutting down thread pool")
    app.state.thread_pool.shutdown(wait=True)
    logger.info("Thread pool shutdown complete")

    # Close vector store connections (MongoDB client / SQLAlchemy engine)
    try:
        close_vector_store_connections(vector_store)
    except Exception as e:
        logger.warning("Failed to close vector store connections: %s", e)


app = FastAPI(lifespan=lifespan, debug=debug_mode)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(LogMiddleware)

app.middleware("http")(security_middleware)

# Set state variables for use in routes
app.state.CHUNK_SIZE = CHUNK_SIZE
app.state.CHUNK_OVERLAP = CHUNK_OVERLAP
app.state.PDF_EXTRACT_IMAGES = PDF_EXTRACT_IMAGES

# Include routers
app.include_router(document_routes.router)
app.include_router(directory_routes.router)
if debug_mode:
    app.include_router(router=pgvector_routes.router)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.debug("Validation error: %s", exc.errors())
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "message": "Request validation failed"},
    )


if __name__ == "__main__":
    uvicorn.run(app, host=RAG_HOST, port=RAG_PORT, log_config=None)

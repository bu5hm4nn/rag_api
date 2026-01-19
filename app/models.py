# app/models.py
import hashlib
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional, List


class DocumentResponse(BaseModel):
    page_content: str
    metadata: dict


class DocumentModel(BaseModel):
    page_content: str
    metadata: Optional[dict] = {}

    def generate_digest(self):
        hash_obj = hashlib.md5(self.page_content.encode())
        return hash_obj.hexdigest()


class StoreDocument(BaseModel):
    filepath: str
    filename: str
    file_content_type: str
    file_id: str


class QueryRequestBody(BaseModel):
    query: str
    file_id: str
    k: int = 4
    entity_id: Optional[str] = None


class CleanupMethod(str, Enum):
    incremental = "incremental"
    full = "full"


class QueryMultipleBody(BaseModel):
    query: str
    file_ids: List[str]
    k: int = 4


# --- Directory Embedding Models ---


class EmbedDirectoryRequest(BaseModel):
    """Request body for POST /local/embed-directory"""

    directory_path: str = Field(..., description="Absolute path to directory")
    entity_id: Optional[str] = None
    recursive: bool = Field(default=True, description="Walk subdirectories")
    file_extensions: Optional[List[str]] = Field(
        default=None,
        description="Filter by extensions (e.g., ['pdf', 'docx']). None = all supported",
    )
    ignore_patterns: List[str] = Field(
        default=[".git", "__pycache__", "node_modules", ".DS_Store"],
        description="Directory/file patterns to ignore",
    )


class FileIndexResult(BaseModel):
    """Result for a single file indexing operation"""

    filepath: str
    file_id: str
    status: str  # 'indexed' or 'error'
    message: Optional[str] = None


class EmbedDirectoryResponse(BaseModel):
    """Response for POST /local/embed-directory"""

    directory_path: str
    total_files: int
    indexed: int
    skipped: int
    errors: int
    files: List[FileIndexResult]


# --- Directory Sync Models ---


class SyncAction(str, Enum):
    """Action taken during sync operation"""

    INDEXED = "indexed"  # New file indexed
    UPDATED = "updated"  # Modified file re-indexed
    DELETED = "deleted"  # File removed from vector store
    UNCHANGED = "unchanged"  # No changes needed
    ERROR = "error"


class FileSyncResult(BaseModel):
    """Result for a single file sync operation"""

    filepath: str
    file_id: str
    action: SyncAction
    message: Optional[str] = None


class SyncDirectoryRequest(BaseModel):
    """Request body for POST /local/sync-directory"""

    directory_path: str = Field(..., description="Absolute path to directory")
    entity_id: Optional[str] = None
    recursive: bool = True
    file_extensions: Optional[List[str]] = None
    ignore_patterns: List[str] = Field(
        default=[".git", "__pycache__", "node_modules", ".DS_Store"]
    )
    delete_removed: bool = Field(
        default=True, description="Remove documents for files that no longer exist"
    )


class SyncDirectoryResponse(BaseModel):
    """Response for POST /local/sync-directory"""

    directory_path: str
    total_files: int
    indexed: int
    updated: int
    deleted: int
    unchanged: int
    errors: int
    files: List[FileSyncResult]


# --- Directory Watch Models ---


class WatchDirectoryRequest(BaseModel):
    """Request body for POST /local/watch-directory"""

    directory_path: str = Field(..., description="Absolute path to directory")
    entity_id: Optional[str] = None
    recursive: bool = True
    file_extensions: Optional[List[str]] = None
    ignore_patterns: List[str] = Field(
        default=[".git", "__pycache__", "node_modules", ".DS_Store"]
    )
    debounce_seconds: int = Field(
        default=5,
        ge=1,
        le=300,
        description="Seconds to wait before processing accumulated changes",
    )
    initial_sync: bool = Field(
        default=True, description="Perform initial sync when watch starts"
    )


class WatchDirectoryResponse(BaseModel):
    """Response for POST /local/watch-directory"""

    watch_id: int
    directory_path: str
    entity_id: str
    status: str  # 'started', 'already_watching'
    message: str


class UnwatchDirectoryRequest(BaseModel):
    """Request body for DELETE /local/watch-directory"""

    directory_path: str
    entity_id: Optional[str] = None
    remove_documents: bool = Field(
        default=False, description="Also remove indexed documents from vector store"
    )


class WatchStatus(BaseModel):
    """Status of a watched directory"""

    watch_id: int
    directory_path: str
    entity_id: str
    recursive: bool
    enabled: bool
    file_count: int
    last_sync_at: Optional[datetime]
    pending_changes: int


class WatchStatusResponse(BaseModel):
    """Response for GET /local/watch-status"""

    watches: List[WatchStatus]

from __future__ import annotations

import mimetypes
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

from fastapi import HTTPException, UploadFile, status

DEFAULT_MAX_UPLOAD_SIZE_BYTES = 50 * 1024 * 1024
ALLOWED_MIME_TYPES = {
    'text/plain',
    'text/markdown',
    'application/pdf',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'image/jpeg',
    'image/png',
    'image/webp',
    'video/mp4',
    'video/quicktime',
    'video/x-msvideo',
    'video/x-matroska',
}
ALLOWED_EXTENSIONS_TO_MIME = {
    '.txt': 'text/plain',
    '.md': 'text/markdown',
    '.pdf': 'application/pdf',
    '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.png': 'image/png',
    '.webp': 'image/webp',
    '.mp4': 'video/mp4',
    '.mov': 'video/quicktime',
    '.avi': 'video/x-msvideo',
    '.mkv': 'video/x-matroska',
}

UPLOADS_DIR = Path(os.getenv('UPLOADS_DIR', 'data/uploads'))


@dataclass
class StoredFile:
    """Stored file metadata returned after saving upload."""

    file_id: str
    filename: str
    mime: str
    size: int
    storage_path: str


def _safe_filename(filename: str) -> str:
    return Path(filename).name.replace(' ', '_')


def validate_mime(mime: str | None, filename: str) -> str:
    """Validate and normalize MIME type."""
    guessed_mime, _ = mimetypes.guess_type(filename)
    file_extension = Path(filename).suffix.lower()
    by_extension = ALLOWED_EXTENSIONS_TO_MIME.get(file_extension)
    normalized_mime = (mime or '').lower().strip()
    resolved_mime = normalized_mime or guessed_mime or by_extension or ''

    # Browsers often send application/octet-stream for uploads from drag/drop.
    if resolved_mime == 'application/octet-stream' and by_extension:
        resolved_mime = by_extension

    if resolved_mime not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f'Unsupported MIME type: {resolved_mime or "unknown"} '
                f'for extension: {file_extension or "unknown"}'
            ),
        )
    return resolved_mime


async def save_upload_file(
    upload: UploadFile,
    max_upload_size_bytes: int = DEFAULT_MAX_UPLOAD_SIZE_BYTES,
) -> StoredFile:
    """Persist an uploaded file to local storage with validation."""
    if not upload.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='filename is required',
        )

    mime = validate_mime(upload.content_type, upload.filename)

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    extension = Path(upload.filename).suffix.lower()
    file_id = str(uuid.uuid4())
    disk_name = f'{file_id}{extension}'
    destination = UPLOADS_DIR / disk_name

    total_size = 0
    with destination.open('wb') as out:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            total_size += len(chunk)
            if total_size > max_upload_size_bytes:
                destination.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail='File is too large',
                )
            out.write(chunk)

    return StoredFile(
        file_id=file_id,
        filename=_safe_filename(upload.filename),
        mime=mime,
        size=total_size,
        storage_path=str(destination),
    )


def delete_stored_file(storage_path: str) -> None:
    """Delete a file from local storage if it exists."""
    path = Path(storage_path)
    if path.exists() and path.is_file():
        path.unlink()


def delete_uploads_dir() -> None:
    """Remove uploads directory if needed by maintenance jobs."""
    if UPLOADS_DIR.exists() and UPLOADS_DIR.is_dir():
        shutil.rmtree(UPLOADS_DIR)

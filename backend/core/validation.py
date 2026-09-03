"""
SIH26166 — Upload file validation.

Each validation function raises an ``HTTPException`` on failure so the
caller can simply await them in sequence.  The module is designed for
easy extension — GeoTIFF metadata, raster dimensions, CRS checks, and
IIRS hyperspectral validation can be added as additional functions
without modifying existing logic.
"""

from __future__ import annotations

import os
from fastapi import HTTPException, UploadFile, status

from backend.core.config import (
    ALLOWED_EXTENSION_SET,
    ALLOWED_EXTENSIONS,
    ALLOWED_MIME_TYPES,
    MAX_UPLOAD_SIZE_BYTES,
)


def _safe_extension(filename: str | None) -> str:
    """Return the lowercased file extension, or empty string."""
    if not filename:
        return ""
    _, ext = os.path.splitext(filename)
    return ext.lower()


def validate_extension(filename: str | None) -> str:
    """Validate and return the file extension.

    Raises 415 (Unsupported Media Type) if the extension is not allowed.
    """
    ext = _safe_extension(filename)
    if ext not in ALLOWED_EXTENSION_SET:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported file extension '{ext or '(none)'}'. "
                f"Allowed: {', '.join(sorted(ALLOWED_EXTENSION_SET))}"
            ),
        )
    return ext


def validate_content_type(content_type: str | None, ext: str) -> str:
    """Validate the MIME / content-type header.

    Some clients send ``application/octet-stream`` for TIFF files, so we
    accept that as a fallback for TIFF extensions only.  Returns the
    validated content type string.

    Raises 415 if the content type is entirely invalid.
    """
    ct = (content_type or "").lower().strip()

    # Direct match
    if ct in ALLOWED_MIME_TYPES:
        return ct

    # Fallback: many HTTP clients report TIFF as octet-stream
    if ct == "application/octet-stream" and ext in (".tif", ".tiff"):
        return "image/tiff"

    raise HTTPException(
        status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        detail=(
            f"Unsupported content type '{ct}'. "
            f"Allowed: {', '.join(sorted(ALLOWED_MIME_TYPES))}"
        ),
    )


async def validate_not_empty(file: UploadFile) -> None:
    """Ensure the uploaded file is not empty.

    Reads the first byte to check and then seeks back to the start.
    Raises 400 if empty.
    """
    first_chunk = await file.read(1)
    if not first_chunk:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Uploaded file '{file.filename}' is empty.",
        )
    # Seek back so subsequent reads start from the beginning.
    await file.seek(0)


async def validate_file_size(file: UploadFile) -> int:
    """Stream-read the file to determine its size without loading it
    entirely into memory.  Raises 413 if the file exceeds the limit.

    Returns the file size in bytes and seeks back to 0.
    """
    size = 0
    chunk_size = 1024 * 1024  # 1 MB chunks
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        size += len(chunk)
        if size > MAX_UPLOAD_SIZE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=(
                    f"File '{file.filename}' exceeds the maximum upload size "
                    f"of {MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)} MB."
                ),
            )
    await file.seek(0)
    return size


async def validate_upload(file: UploadFile) -> tuple[str, str, int]:
    """Run all validations on a single ``UploadFile``.

    Returns ``(extension, content_type, size_bytes)`` on success.
    """
    ext = validate_extension(file.filename)
    content_type = validate_content_type(file.content_type, ext)
    await validate_not_empty(file)
    size_bytes = await validate_file_size(file)
    return ext, content_type, size_bytes

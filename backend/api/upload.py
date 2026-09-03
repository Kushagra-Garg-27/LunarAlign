"""
SIH26166 — Image upload endpoint.

Accepts two lunar images (reference + target) via multipart/form-data,
validates them, stores them safely, and returns structured metadata.
"""

from __future__ import annotations

import os
import uuid
import logging
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from backend.api.models import ImageMetadata, UploadResponse
from backend.core.config import UPLOAD_DIR
from backend.core.validation import validate_upload

logger = logging.getLogger("sih26166.upload")

router = APIRouter()


def _ensure_upload_dir() -> Path:
    """Create the upload directory if it does not exist."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    return UPLOAD_DIR


def _safe_stored_filename(ext: str) -> str:
    """Generate a unique, filesystem-safe filename.

    Uses UUID4 so there is no collision risk and no dependence on
    the original (untrusted) filename.
    """
    return f"{uuid.uuid4().hex}{ext}"


async def _save_file(file: UploadFile, dest: Path) -> None:
    """Stream the upload to disk in chunks.

    If writing fails partway through, the partial file is removed.
    """
    try:
        with open(dest, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)  # 1 MB
                if not chunk:
                    break
                f.write(chunk)
    except Exception:
        # Remove partial/corrupted file
        if dest.exists():
            dest.unlink()
        raise


async def _process_upload(file: UploadFile, label: str) -> ImageMetadata:
    """Validate, store, and build metadata for a single uploaded image.

    ``label`` is a human-readable identifier (e.g. "reference", "target")
    used only in log/error messages.
    """
    ext, content_type, size_bytes = await validate_upload(file)

    upload_dir = _ensure_upload_dir()
    stored_name = _safe_stored_filename(ext)
    dest = upload_dir / stored_name

    await _save_file(file, dest)
    logger.info(
        "Stored %s image: %s -> %s (%d bytes)",
        label,
        file.filename,
        stored_name,
        size_bytes,
    )

    # Extract lightweight image metadata (dimensions, bands, dtype).
    # This reads only the file header — no expensive pixel decoding.
    # Failures are non-fatal; fields default to None.
    img_meta: dict = {}
    try:
        from backend.preprocessing.metadata import extract_image_metadata
        img_meta = extract_image_metadata(dest)
    except Exception as exc:
        logger.warning("Could not extract image metadata for %s: %s", stored_name, exc)

    return ImageMetadata(
        original_filename=file.filename or "unknown",
        stored_filename=stored_name,
        size_bytes=size_bytes,
        content_type=content_type,
        extension=ext,
        width=img_meta.get("width"),
        height=img_meta.get("height"),
        num_bands=img_meta.get("num_bands"),
        image_dtype=img_meta.get("image_dtype"),
    )


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_200_OK,
    summary="Upload reference and target lunar images",
    description=(
        "Accepts two lunar images via multipart/form-data. "
        "Both images are validated for extension, MIME type, size, and "
        "non-emptiness, then stored with safe unique filenames."
    ),
)
async def upload_images(
    reference: UploadFile = File(..., description="Reference lunar image"),
    target: UploadFile = File(..., description="Target lunar image"),
) -> UploadResponse:
    """Upload a reference and a target image for registration."""
    ref_meta = await _process_upload(reference, "reference")
    tgt_meta = await _process_upload(target, "target")

    return UploadResponse(
        status="success",
        reference=ref_meta,
        target=tgt_meta,
    )

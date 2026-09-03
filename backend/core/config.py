"""
SIH26166 — Backend configuration constants.

Centralised configuration for upload settings, allowed formats, and paths.
This module is imported by validation and upload logic so that limits and
allowed types can be changed in a single place.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Upload directory
# ---------------------------------------------------------------------------
# Resolve relative to the project root (two levels up from this file).
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"

# ---------------------------------------------------------------------------
# File-size limits
# ---------------------------------------------------------------------------
# 100 MB per individual image (generous for TIFF imagery).
MAX_UPLOAD_SIZE_BYTES: int = 100 * 1024 * 1024  # 100 MB

# ---------------------------------------------------------------------------
# Allowed image formats
# ---------------------------------------------------------------------------
# Map of lowercase extension -> list of acceptable MIME/content-type strings.
# This is intentionally restrictive for now; GeoTIFF, HDF5, and IIRS formats
# will be added when the preprocessing layer is implemented.
ALLOWED_EXTENSIONS: dict[str, list[str]] = {
    ".png": ["image/png"],
    ".jpg": ["image/jpeg"],
    ".jpeg": ["image/jpeg"],
    ".tif": ["image/tiff"],
    ".tiff": ["image/tiff"],
}

# Flat set of allowed extensions for quick membership checks.
ALLOWED_EXTENSION_SET: set[str] = set(ALLOWED_EXTENSIONS.keys())

# Flat set of allowed MIME types for quick membership checks.
ALLOWED_MIME_TYPES: set[str] = {
    mime for mimes in ALLOWED_EXTENSIONS.values() for mime in mimes
}

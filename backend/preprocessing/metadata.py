"""
SIH26166 — Lightweight image metadata extraction.

Extracts basic image properties (dimensions, band count, dtype) from
a stored file without fully decoding the pixel data into memory.

This is used by the upload endpoint to populate metadata fields
without triggering expensive preprocessing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PIL import Image as PILImage

logger = logging.getLogger("sih26166.preprocessing.metadata")

# Pillow mode → approximate numpy dtype string
_PIL_MODE_DTYPE: dict[str, str] = {
    "1": "bool",
    "L": "uint8",
    "P": "uint8",
    "RGB": "uint8",
    "RGBA": "uint8",
    "CMYK": "uint8",
    "I": "int32",
    "I;16": "uint16",
    "I;16B": "uint16",
    "F": "float32",
}

# Pillow mode → number of channels
_PIL_MODE_BANDS: dict[str, int] = {
    "1": 1,
    "L": 1,
    "P": 1,
    "RGB": 3,
    "RGBA": 4,
    "CMYK": 4,
    "I": 1,
    "I;16": 1,
    "I;16B": 1,
    "F": 1,
}


def extract_image_metadata(path: str | Path) -> dict[str, Any]:
    """Extract lightweight metadata from an image file.

    Returns a dict with keys:
    - ``width`` (int)
    - ``height`` (int)
    - ``num_bands`` (int)
    - ``image_dtype`` (str)

    For TIFF files, attempts rasterio first for accurate multi-band info,
    then falls back to Pillow.

    This function is intentionally lightweight — it reads only the file
    header, not the full pixel data.
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext in (".tif", ".tiff"):
        try:
            return _extract_rasterio(path)
        except Exception:
            pass  # fall through to Pillow

    return _extract_pillow(path)


def _extract_pillow(path: Path) -> dict[str, Any]:
    """Extract metadata via Pillow (header-only)."""
    try:
        with PILImage.open(path) as img:
            width, height = img.size
            mode = img.mode
            num_bands = _PIL_MODE_BANDS.get(mode, len(img.getbands()))
            dtype = _PIL_MODE_DTYPE.get(mode, "unknown")
    except Exception as exc:
        logger.warning("Could not extract metadata via Pillow for %s: %s", path.name, exc)
        return {"width": None, "height": None, "num_bands": None, "image_dtype": None}

    return {
        "width": width,
        "height": height,
        "num_bands": num_bands,
        "image_dtype": dtype,
    }


def _extract_rasterio(path: Path) -> dict[str, Any]:
    """Extract metadata via rasterio (GeoTIFF / multi-band)."""
    import rasterio

    with rasterio.open(path) as src:
        return {
            "width": src.width,
            "height": src.height,
            "num_bands": src.count,
            "image_dtype": str(src.dtypes[0]) if src.dtypes else "unknown",
        }

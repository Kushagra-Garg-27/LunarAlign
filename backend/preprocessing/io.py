"""
SIH26166 — Image I/O layer.

Loads images from disk into a ``RawImage`` representation.

Supported formats
-----------------
- **PNG / JPEG** — loaded via Pillow.
- **TIFF** — attempted first with rasterio (which supports multi-band
  GeoTIFF and scientific rasters), falling back to Pillow for simple
  TIFF files.
- **PDS4 XML** — loaded via rasterio GDAL PDS4 driver (TMC-2 and OHRC).

Design notes
------------
- All data is converted to NumPy arrays on load.
- Multi-band rasters retain *all* bands in the ``RawImage.data`` array
  (shape ``(H, W, C)`` with bands-last ordering).
- Single-band images are stored as ``(H, W)`` arrays.
- dtype is preserved from the source file; no silent normalisation.
- Filesystem paths are not stored in the returned metadata to avoid
  leaking internal paths through the API.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image as PILImage

from backend.preprocessing.datamodel import RawImage

logger = logging.getLogger("sih26166.preprocessing.io")

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_image(path: str | Path) -> RawImage:
    """Load an image from *path* and return a ``RawImage``.

    Parameters
    ----------
    path : str | Path
        Absolute or relative filesystem path to a supported image file.

    Returns
    -------
    RawImage
        Full-fidelity loaded image.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file format is unsupported or the image is corrupt.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    ext = path.suffix.lower()

    if ext in (".tif", ".tiff"):
        return _load_tiff(path)
    elif ext in (".png", ".jpg", ".jpeg"):
        return _load_pillow(path, ext)
    elif ext == ".xml":
        return _load_pds4(path)
    else:
        raise ValueError(f"Unsupported image format: '{ext}'")


# ---------------------------------------------------------------------------
# PDS4 loader (TMC-2 and OHRC)
# ---------------------------------------------------------------------------


def _load_pds4(path: Path) -> RawImage:
    """Load a PDS4 Product_Observational raster image (TMC-2 and OHRC).

    Parameters
    ----------
    path : Path
        Filesystem path to a PDS4 .xml label file.

    Returns
    -------
    RawImage
        Full-fidelity loaded image with source_format="PDS4".

    Raises
    ------
    ValueError
        If the file is not a valid PDS4 label or the instrument is not supported.
    """
    import xml.etree.ElementTree as ET
    from backend.preprocessing.pds4 import (
        identify_instrument,
        is_pds4_label,
        load_pds4_raster,
    )

    if not is_pds4_label(path):
        raise ValueError(f"File '{path.name}' is not a valid PDS4 label.")

    try:
        tree = ET.parse(str(path))
    except ET.ParseError as exc:
        raise ValueError(f"Malformed XML in {path.name}: {exc}") from exc

    instrument = identify_instrument(tree)
    if instrument not in ("TMC2", "OHRC"):
        raise ValueError(
            f"Unsupported PDS4 instrument: '{instrument}'. Only TMC-2 and OHRC products are supported in this pipeline."
        )

    raw_img = load_pds4_raster(path)
    raw_img.metadata["instrument"] = instrument
    return raw_img


# ---------------------------------------------------------------------------
# Pillow loader (PNG, JPEG, simple TIFF fallback)
# ---------------------------------------------------------------------------


def _load_pillow(path: Path, ext: str) -> RawImage:
    """Load an image via Pillow."""
    try:
        img = PILImage.open(path)
        img.load()  # force full decode
    except Exception as exc:
        raise ValueError(f"Failed to open image with Pillow: {exc}") from exc

    arr = np.asarray(img)
    metadata: dict[str, Any] = {}

    # Preserve useful TIFF/EXIF metadata if available
    tag_v2 = getattr(img, "tag_v2", None)
    if tag_v2 is not None:
        # Pillow TiffImagePlugin stores tags in tag_v2 as {tag_id: value}
        metadata["tiff_tags"] = {
            k: _safe_meta_value(v) for k, v in tag_v2.items()
        }
    if hasattr(img, "info") and img.info:
        # Generic Pillow info dict (DPI, palette, etc.)
        metadata["pillow_info"] = {
            k: _safe_meta_value(v) for k, v in img.info.items()
        }

    # Determine shape
    if arr.ndim == 2:
        height, width = arr.shape
        num_bands = 1
    elif arr.ndim == 3:
        height, width, num_bands = arr.shape
    else:
        raise ValueError(f"Unexpected image array dimensions: {arr.ndim}")

    source_format = {".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg"}.get(ext, ext.lstrip("."))

    logger.info(
        "Loaded %s image via Pillow: %dx%d, %d band(s), dtype=%s",
        source_format, width, height, num_bands, arr.dtype,
    )

    return RawImage(
        data=arr,
        width=width,
        height=height,
        num_bands=num_bands,
        dtype=arr.dtype,
        source_format=source_format,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# TIFF loader — rasterio first, Pillow fallback
# ---------------------------------------------------------------------------


def _load_tiff(path: Path) -> RawImage:
    """Load a TIFF file.

    Attempts rasterio first (supports multi-band GeoTIFF / scientific
    rasters).  Falls back to Pillow for simple TIFF files if rasterio
    fails or is unavailable.
    """
    try:
        return _load_rasterio(path)
    except Exception as rio_exc:
        logger.debug(
            "Rasterio failed for %s (%s), falling back to Pillow.",
            path.name, rio_exc,
        )
        try:
            return _load_pillow(path, ".tiff")
        except Exception as pil_exc:
            raise ValueError(
                f"Failed to load TIFF with both rasterio ({rio_exc}) "
                f"and Pillow ({pil_exc})."
            ) from pil_exc


def _load_rasterio(path: Path) -> RawImage:
    """Load an image via rasterio (multi-band / GeoTIFF support)."""
    import rasterio  # lazy import — only needed for TIFF

    with rasterio.open(path) as src:
        num_bands = src.count
        width = src.width
        height = src.height
        dtype = np.dtype(src.dtypes[0])  # dtype of first band

        # Read all bands → shape (bands, H, W)
        data_bhw = src.read()

        # Convert to bands-last: (H, W, C)  or  (H, W) for single-band
        if num_bands == 1:
            data = data_bhw[0]  # (H, W)
        else:
            data = np.moveaxis(data_bhw, 0, -1)  # (H, W, C)

        # Collect safe metadata
        metadata: dict[str, Any] = {
            "rasterio_driver": src.driver,
            "rasterio_crs": str(src.crs) if src.crs else None,
            "rasterio_transform": list(src.transform) if src.transform else None,
            "rasterio_band_count": num_bands,
            "rasterio_dtypes": list(src.dtypes),
        }
        if src.descriptions and any(src.descriptions):
            metadata["rasterio_band_descriptions"] = list(src.descriptions)
        if src.tags():
            metadata["rasterio_tags"] = dict(src.tags())

    logger.info(
        "Loaded TIFF via rasterio: %dx%d, %d band(s), dtype=%s",
        width, height, num_bands, dtype,
    )

    return RawImage(
        data=data,
        width=width,
        height=height,
        num_bands=num_bands,
        dtype=dtype,
        source_format="rasterio",
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_meta_value(v: Any) -> Any:
    """Convert metadata values to JSON-safe types."""
    if isinstance(v, bytes):
        try:
            return v.decode("utf-8", errors="replace")
        except Exception:
            return repr(v)
    if isinstance(v, (tuple, list)):
        return [_safe_meta_value(x) for x in v]
    if isinstance(v, np.generic):
        return v.item()
    return v

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


def extract_pds4_file_metadata(xml_path: str | Path) -> dict[str, Any]:
    """Extract metadata from a PDS4 label file without loading pixel data.

    Combines the basic raster properties (width, height, num_bands, dtype)
    obtained by opening the label via rasterio with the rich PDS4 metadata
    (instrument, isda: fields, corner coordinates, etc.) extracted from the
    label XML.

    Follows the same contract as :func:`extract_image_metadata` — returns a
    plain dict suitable for API responses.

    Parameters
    ----------
    xml_path : str | Path
        Path to the PDS4 ``.xml`` label file.

    Returns
    -------
    dict[str, Any]
        Keys include at minimum:

        - ``width``, ``height``, ``num_bands``, ``image_dtype`` (from rasterio)
        - ``pds4_instrument`` — canonical instrument identifier
        - ``pds4_logical_identifier``, ``pds4_title``
        - ``pds4_start_date_time``, ``pds4_stop_date_time``
        - ``pds4_isda_product_params`` — generic key-value dict
        - ``pds4_isda_geometry_params`` — generic key-value dict
        - ``pds4_tmc2_product_params`` — named TMC-2 fields (or ``None``)
        - ``pds4_tmc2_geometry_params`` — TMC-2 corner coords (or ``None``)
        - ``pds4_wavelength_info`` — IIRS wavelength note (or ``None``)

    Raises
    ------
    FileNotFoundError
        If *xml_path* does not exist.
    ValueError
        If *xml_path* is not a valid PDS4 Product_Observational label.
    """
    from backend.preprocessing.pds4 import extract_pds4_metadata

    xml_path = Path(xml_path)

    # Raster geometry via rasterio (reads label header, no pixel decode)
    raster_info: dict[str, Any] = {}
    try:
        raster_info = _extract_rasterio(xml_path)
    except Exception as exc:
        logger.warning(
            "Could not extract raster properties via rasterio for %s: %s",
            xml_path.name, exc,
        )
        raster_info = {"width": None, "height": None, "num_bands": None, "image_dtype": None}

    # PDS4 label metadata via XML parse
    pds4_meta = extract_pds4_metadata(xml_path)

    return {
        **raster_info,
        "pds4_instrument": pds4_meta.instrument,
        "pds4_logical_identifier": pds4_meta.logical_identifier,
        "pds4_title": pds4_meta.title,
        "pds4_start_date_time": pds4_meta.start_date_time,
        "pds4_stop_date_time": pds4_meta.stop_date_time,
        "pds4_investigation_name": pds4_meta.investigation_name,
        "pds4_isda_product_params": pds4_meta.isda_product_params,
        "pds4_isda_geometry_params": pds4_meta.isda_geometry_params,
        "pds4_tmc2_product_params": pds4_meta.tmc2_product_params,
        "pds4_tmc2_geometry_params": pds4_meta.tmc2_geometry_params,
        "pds4_wavelength_info": pds4_meta.wavelength_info,
    }

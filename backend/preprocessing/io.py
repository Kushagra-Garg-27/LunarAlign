"""
SIH26166 — Image I/O layer.

Loads images from disk into a ``RawImage`` representation.

Supported formats
-----------------
- **PNG / JPEG** — loaded via Pillow.
- **TIFF** — attempted first with rasterio (which supports multi-band
  GeoTIFF and scientific rasters), falling back to Pillow for simple
  TIFF files.
- **PDS4 XML** — loaded via rasterio GDAL PDS4 driver (TMC-2 and OHRC)
  or hyperspectral solar band-reduction pipeline (IIRS).

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
# PDS4 loader (TMC-2, OHRC, and IIRS)
# ---------------------------------------------------------------------------


def _load_pds4(path: Path) -> RawImage:
    """Load a PDS4 Product_Observational raster image (TMC-2, OHRC, and IIRS).

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
    from backend.preprocessing.band_reduction import (
        reduce_bands_mean,
        select_bands,
        select_solar_reflective_bands,
    )
    from backend.preprocessing.grayscale import _normalize_dynamic_range
    from backend.preprocessing.pds4 import (
        extract_iirs_band_wavelengths,
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
    if instrument not in ("TMC2", "OHRC", "IIRS"):
        raise ValueError(
            f"Unsupported PDS4 instrument: '{instrument}'. Only TMC-2, OHRC, and IIRS products are supported in this pipeline."
        )

    raw_img = load_pds4_raster(path)
    raw_img.metadata["instrument"] = instrument

    if instrument in ("TMC2", "OHRC"):
        return raw_img

    # --- IIRS multi-band hyperspectral reduction path ---
    # 1. Extract per-band wavelength metadata and determine solar-reflective bands (< 2500nm).
    wavelengths = extract_iirs_band_wavelengths(path)
    if not wavelengths:
        raise ValueError(
            f"IIRS product '{path.name}' has no Band_Bin wavelength metadata; cannot determine solar-reflective bands."
        )
    solar_indices = select_solar_reflective_bands(wavelengths)
    if not solar_indices:
        raise ValueError(
            f"{path.name}: No solar-reflective bands found below cutoff."
        )

    # 2. Subset spectral cube to solar-reflective bands.
    # Note on axis ordering: load_pds4_raster() returns raw_img.data in (H, W, C)
    # bands-last layout (per the RawImage dataclass specification). However,
    # select_bands() and reduce_bands_mean() expect BSQ layout (bands, H, W).
    # We transpose the data from (H, W, C) to (C, H, W) before subsetting.
    cube_bhw = np.moveaxis(raw_img.data, -1, 0)
    solar_cube = select_bands(cube_bhw, solar_indices)

    # 3. Reduce 3D spectral cube to single 2D image.
    # Stated reasoning for reduction method choice (reduce_bands_mean vs reduce_bands_pca):
    # - reduce_bands_mean() collapses bands via per-pixel mean averaging into a strictly 2D
    #   (H, W) array. Mean radiance preserves physical additive energy and positive monotonicity,
    #   ensuring stable spatial gradients for classical feature detectors (SIFT).
    # - In contrast, reduce_bands_pca() returns (n_components, H, W) (a 3D array requiring component
    #   selection), has eigenvector sign ambiguity across different views/transforms (which can
    #   invert contrast and break SIFT descriptors), and can yield negative unbounded values.
    # - Therefore, reduce_bands_mean() is directly compatible with the 2D single-channel FeatureImage
    #   and downstream registration pipeline.
    reduced_2d = reduce_bands_mean(solar_cube)

    # 4. Dynamic range normalization.
    # Stated reasoning for normalization placement (after band reduction vs before):
    # - Applying dynamic range normalization (_normalize_dynamic_range) AFTER band reduction
    #   preserves true physical relative spectral radiance across bands during averaging and
    #   averages out uncorrelated per-band sensor noise (enhancing SNR).
    # - Normalizing before reduction would artificially re-scale individual bands independently,
    #   amplifying noise in low-signal bands and distorting physical spectral weighting.
    # - Stretching the single aggregated 2D array to [0, 255] float32 produces the exact contrast
    #   range expected downstream by prepare_for_sift() with minimal computational overhead.
    normalized_2d = _normalize_dynamic_range(reduced_2d)

    metadata = dict(raw_img.metadata)
    metadata["solar_reflective_bands"] = solar_indices
    metadata["band_reduction_method"] = "mean"

    return RawImage(
        data=normalized_2d,
        width=raw_img.width,
        height=raw_img.height,
        num_bands=1,
        dtype=normalized_2d.dtype,
        source_format="PDS4",
        metadata=metadata,
    )


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

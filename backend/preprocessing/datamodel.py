"""
SIH26166 — Internal image representation.

Defines typed data structures that carry image data through the pipeline.

Three distinct representations are maintained:

1. **RawImage** — the full-fidelity loaded image.  All original bands,
   dtype, and metadata are preserved.  This is what the I/O layer produces.

2. **FeatureImage** — a 2-D single-channel floating-point array ready for
   feature extraction / classical CV algorithms.  Derived from a RawImage
   via an explicit conversion step (grayscale + optional normalisation).

3. **PDS4Metadata** — structured metadata extracted from a PDS4 label
   (``.xml`` file) for Chandrayaan-2 instruments (TMC-2, OHRC, IIRS).
   Carries both the confirmed named fields for TMC-2 and a generic
   key-value dict for all instruments (isda: namespace).

Downstream code should accept one of these explicit types rather than raw
``ndarray`` to keep the boundary between "original data" and "processing
data" visible and auditable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=False)
class RawImage:
    """Full-fidelity loaded image.

    Attributes
    ----------
    data : np.ndarray
        Image pixel data.  Shape is ``(H, W)`` for single-band images or
        ``(H, W, C)`` for multi-band images (bands-last).  dtype is
        preserved from the source file.
    width : int
        Number of columns.
    height : int
        Number of rows.
    num_bands : int
        Number of spectral bands / channels (1 for grayscale, 3 for RGB,
        arbitrary for multi-band raster data).
    dtype : np.dtype
        NumPy dtype of the pixel values (e.g. ``uint8``, ``uint16``,
        ``float32``).
    source_format : str
        File format identifier: ``"png"``, ``"jpeg"``, ``"tiff"``,
        ``"rasterio"`` etc.
    metadata : dict[str, Any]
        Arbitrary source metadata.  For Pillow images this may contain
        EXIF/TIFF tags; for rasterio images it may contain CRS, transform,
        and band descriptions.  Never contains filesystem paths.
    """

    data: np.ndarray
    width: int
    height: int
    num_bands: int
    dtype: np.dtype
    source_format: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=False)
class FeatureImage:
    """2-D single-channel image ready for feature extraction.

    This is always:
    - 2-D ``(H, W)``
    - ``float32`` or ``float64``
    - derived from a ``RawImage`` via an explicit conversion

    Attributes
    ----------
    data : np.ndarray
        2-D float array, shape ``(H, W)``.
    width : int
    height : int
    source_dtype : np.dtype
        The dtype of the ``RawImage`` this was derived from, kept for
        provenance so later stages can reason about the original dynamic
        range.
    conversion_method : str
        Human-readable label describing how the conversion was performed
        (e.g. ``"luminance_bt709"``, ``"band_mean"``, ``"passthrough"``).
    """

    data: np.ndarray
    width: int
    height: int
    source_dtype: np.dtype
    conversion_method: str


@dataclass(frozen=False)
class PDS4Metadata:
    """Structured metadata extracted from a Chandrayaan-2 PDS4 label.

    Populated by :func:`backend.preprocessing.pds4.extract_pds4_metadata`.

    Attributes
    ----------
    logical_identifier : str | None
        PDS4 logical identifier from ``Identification_Area``.
    title : str | None
        Product title from ``Identification_Area``.
    start_date_time : str | None
        Observation start time (ISO 8601) from ``Time_Coordinates``.
    stop_date_time : str | None
        Observation stop time (ISO 8601) from ``Time_Coordinates``.
    investigation_name : str | None
        Mission name from ``Investigation_Area`` (e.g. ``"chandrayaan-2"``).
    instrument : str
        Canonical instrument identifier: ``"TMC2"``, ``"OHRC"``,
        ``"IIRS"``, or ``"UNKNOWN"``.
    isda_product_params : dict[str, Any]
        Generic key-value extraction of all children under
        ``Mission_Area/isda:Product_Parameters``.  Keys are the local
        tag names (or dotted paths for nested containers); values are
        ``{"value": str, "unit": str | None}`` dicts.  Present for all
        instruments.  For OHRC/IIRS these are the *observed* fields from
        a single real sample, not a validated schema.
    isda_geometry_params : dict[str, Any]
        Same structure as above, for ``isda:Geometry_Parameters``.
    tmc2_product_params : dict[str, Any] | None
        TMC-2 only: subset of ``isda_product_params`` containing the
        confirmed named fields (pixel_resolution, sun_azimuth, etc.).
        ``None`` for non-TMC-2 products.
    tmc2_geometry_params : dict[str, Any] | None
        TMC-2 only: corner coordinates from both
        ``System_Level_Coordinates`` and ``Refined_Corner_Coordinates``,
        extracted from ``isda_geometry_params`` with dotted-path keys.
        ``None`` for non-TMC-2 products.
    wavelength_info : str | None
        IIRS only: human-readable note describing the location of
        per-band wavelength metadata in the label XML.  ``None`` for
        non-IIRS products.  Extraction/use of wavelength values is out
        of scope for Stage 2a.
    """

    logical_identifier: str | None
    title: str | None
    start_date_time: str | None
    stop_date_time: str | None
    investigation_name: str | None
    instrument: str
    isda_product_params: dict[str, Any] = field(default_factory=dict)
    isda_geometry_params: dict[str, Any] = field(default_factory=dict)
    tmc2_product_params: dict[str, Any] | None = None
    tmc2_geometry_params: dict[str, Any] | None = None
    wavelength_info: str | None = None


@dataclass(frozen=False)
class PreprocessedImage:
    """Preprocessed 2-D float32 image ready for feature detection.

    Attributes
    ----------
    data : np.ndarray
        Preprocessed 2-D array of float32 values in [0, 1].
    instrument : str
        Source instrument identifier (e.g. "TMC-2", "OHRC", "IIRS").
    original_shape : tuple
        Shape of the input raster prior to preprocessing.
    pixel_resolution : float
        Effective spatial resolution in meters per pixel.
    preprocessing_steps : list[str]
        Ordered audit trail of preprocessing steps applied.
    """

    data: np.ndarray
    instrument: str
    original_shape: tuple
    pixel_resolution: float
    preprocessing_steps: list[str] = field(default_factory=list)


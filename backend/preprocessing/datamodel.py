"""
SIH26166 — Internal image representation.

Defines typed data structures that carry image data through the pipeline.

Two distinct representations are maintained:

1. **RawImage** — the full-fidelity loaded image.  All original bands,
   dtype, and metadata are preserved.  This is what the I/O layer produces.

2. **FeatureImage** — a 2-D single-channel floating-point array ready for
   feature extraction / classical CV algorithms.  Derived from a RawImage
   via an explicit conversion step (grayscale + optional normalisation).

Downstream code should accept one of these two types explicitly rather
than raw ``ndarray`` to keep the boundary between "original data" and
"processing data" visible and auditable.
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

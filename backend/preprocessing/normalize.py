"""
SIH26166 — Radiometric Normalization (Module 01).

Provides intensity normalization and cross-temporal histogram matching
for multi-sensor lunar imagery (TMC-2, OHRC, IIRS).
"""

from __future__ import annotations

import logging
from typing import Literal

import cv2
import numpy as np
from skimage import exposure

logger = logging.getLogger("sih26166.preprocessing.normalize")


def normalize_intensity(
    image: np.ndarray,
    method: Literal["clahe", "histogram_eq", "minmax"] | str = "clahe",
) -> np.ndarray:
    """Normalize intensity of a 2D grayscale image to float32 in [0, 1].

    Parameters
    ----------
    image : np.ndarray
        2D grayscale array of any numerical dtype.
    method : str, default "clahe"
        Normalization strategy:
        - "clahe": Contrast Limited Adaptive Histogram Equalization (OpenCV).
        - "histogram_eq": Global histogram equalization (scikit-image).
        - "minmax": Linear min-max scaling to [0, 1].

    Returns
    -------
    np.ndarray
        Float32 2D array with values in [0.0, 1.0].
    """
    norm_method = method.lower()
    if norm_method not in ("clahe", "histogram_eq", "minmax"):
        raise ValueError(
            f"Unsupported normalization method: '{method}'. "
            f"Expected 'clahe', 'histogram_eq', or 'minmax'."
        )

    if image.size == 0:
        return np.zeros(image.shape, dtype=np.float32)

    # Clean NaNs and infinite values
    arr = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

    # Initial scaling to [0, 1] based on source dtype
    if image.dtype == np.uint16:
        # Scale uint16 (TMC-2) by max range
        scaled = arr.astype(np.float32) / 65535.0
    elif image.dtype == np.uint8:
        # Scale uint8 (OHRC) by max range
        scaled = arr.astype(np.float32) / 255.0
    else:
        # Floating point or other integer types (e.g. IIRS bands)
        f_arr = arr.astype(np.float32)
        lo, hi = float(f_arr.min()), float(f_arr.max())
        if lo == hi:
            return np.zeros(image.shape, dtype=np.float32)
        scaled = (f_arr - lo) / (hi - lo)

    scaled = np.clip(scaled, 0.0, 1.0)
    lo, hi = float(scaled.min()), float(scaled.max())
    if lo == hi:
        return np.zeros(image.shape, dtype=np.float32)

    if norm_method == "minmax":
        # Ensure exact [0, 1] stretch
        if lo == hi:
            return np.zeros(image.shape, dtype=np.float32)
        out = (scaled - lo) / (hi - lo)
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    elif norm_method == "histogram_eq":
        # Global histogram equalization via skimage
        eq = exposure.equalize_hist(scaled)
        return eq.astype(np.float32)

    elif norm_method == "clahe":
        # Convert scaled [0, 1] to uint8 for OpenCV CLAHE
        u8 = (scaled * 255.0).astype(np.uint8)
        h, w = u8.shape[:2]
        # Adjust tile grid size if image dimensions are small
        tile_w = max(1, min(8, w))
        tile_h = max(1, min(8, h))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(tile_w, tile_h))
        equalized = clahe.apply(u8)
        out = equalized.astype(np.float32) / 255.0
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    return scaled.astype(np.float32)


def match_histograms(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Adjust pixel values of source image to match reference image histogram.

    Parameters
    ----------
    source : np.ndarray
        2D source image to be adjusted.
    reference : np.ndarray
        2D reference image template.

    Returns
    -------
    np.ndarray
        Float32 image with intensity distribution matching reference.
    """
    if source.size == 0 or reference.size == 0:
        return np.zeros(source.shape, dtype=np.float32)

    src_clean = np.nan_to_num(source, nan=0.0, posinf=0.0, neginf=0.0)
    ref_clean = np.nan_to_num(reference, nan=0.0, posinf=0.0, neginf=0.0)

    matched = exposure.match_histograms(src_clean, ref_clean)
    return matched.astype(np.float32)


def normalize_minmax(
    image: np.ndarray,
    *,
    out_range: tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    """Min-max normalization to out_range (backwards compatibility)."""
    arr = np.nan_to_num(image.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = float(arr.min()), float(arr.max())

    if lo == hi:
        return np.full_like(arr, out_range[0], dtype=np.float32)

    out_lo, out_hi = out_range
    return ((arr - lo) / (hi - lo)) * (out_hi - out_lo) + out_lo


def normalize_percentile(
    image: np.ndarray,
    *,
    low_pct: float = 2.0,
    high_pct: float = 98.0,
    out_range: tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    """Percentile-based robust normalization (backwards compatibility)."""
    arr = np.nan_to_num(image.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(np.percentile(arr, low_pct))
    hi = float(np.percentile(arr, high_pct))

    if lo == hi:
        return np.full_like(arr, out_range[0], dtype=np.float32)

    clipped = np.clip(arr, lo, hi)
    out_lo, out_hi = out_range
    return ((clipped - lo) / (hi - lo)) * (out_hi - out_lo) + out_lo

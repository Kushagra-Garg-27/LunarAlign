"""
SIH26166 — Intensity normalization utilities.

All functions operate on **copies** of the input data — the original
array is never modified.

Available strategies
--------------------
- ``normalize_minmax`` — linear stretch to [0, 1] using min/max.
- ``normalize_percentile`` — robust linear stretch using percentile
  clipping (e.g. 2nd–98th percentile) to reduce the influence of
  outlier pixels.

Both are explicit, opt-in operations.  The pipeline never auto-normalises
raw scientific data.
"""

from __future__ import annotations

import numpy as np


def normalize_minmax(
    image: np.ndarray,
    *,
    out_range: tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    """Min–max normalization to ``out_range``.

    Parameters
    ----------
    image : np.ndarray
        Input array (any shape / dtype).
    out_range : tuple[float, float]
        Desired output range, default ``(0.0, 1.0)``.

    Returns
    -------
    np.ndarray
        Float32 array with values in ``[out_range[0], out_range[1]]``.
        If the image is constant (min == max), returns an array filled
        with ``out_range[0]``.
    """
    arr = image.astype(np.float32)
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
    """Percentile-based robust normalization.

    Clips pixel values to the ``[low_pct, high_pct]`` percentile range
    before performing linear stretch.  This reduces the effect of extreme
    outliers (e.g. hot pixels, saturated pixels in lunar imagery).

    Parameters
    ----------
    image : np.ndarray
        Input array.
    low_pct, high_pct : float
        Lower and upper percentile for clipping (0–100).
    out_range : tuple[float, float]
        Desired output range.

    Returns
    -------
    np.ndarray
        Float32 array with values clipped and stretched to ``out_range``.
    """
    arr = image.astype(np.float32)
    lo = float(np.percentile(arr, low_pct))
    hi = float(np.percentile(arr, high_pct))

    if lo == hi:
        return np.full_like(arr, out_range[0], dtype=np.float32)

    clipped = np.clip(arr, lo, hi)
    out_lo, out_hi = out_range
    return ((clipped - lo) / (hi - lo)) * (out_hi - out_lo) + out_lo

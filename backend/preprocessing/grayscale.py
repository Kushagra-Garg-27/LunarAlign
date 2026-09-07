"""
SIH26166 — Grayscale / feature-image conversion.

Converts a ``RawImage`` into a ``FeatureImage`` — a 2-D single-channel
float array suitable for classical feature extraction (SIFT, Phase
Congruency, etc.).

Conversion strategies
---------------------
- **Single-band / grayscale** — direct passthrough to float.
- **RGB (3-band uint8)** — standard ITU-R BT.709 luminance weights.
- **RGBA (4-band)** — alpha channel is dropped, then RGB→luminance.
- **Multi-band (> 4 bands)** — per-pixel mean across bands.  This is a
  documented *generic* strategy that does not claim domain-specific
  correctness for IIRS hyperspectral data.  Later stages will replace
  this with PCA, specific band selection, or domain-aware reduction.

The raw data is never modified.
"""

from __future__ import annotations

import numpy as np

from backend.preprocessing.datamodel import FeatureImage, RawImage

# ITU-R BT.709 luminance coefficients  (standard for sRGB)
_BT709_R = 0.2126
_BT709_G = 0.7152
_BT709_B = 0.0722


def to_feature_image(raw: RawImage) -> FeatureImage:
    """Convert a ``RawImage`` to a 2-D ``FeatureImage``.

    Parameters
    ----------
    raw : RawImage
        Full-fidelity loaded image.

    Returns
    -------
    FeatureImage
        2-D float32 array in [0, 255] range suitable for feature extraction.

    Notes
    -----
    A min-max dynamic range stretch is applied when pixel values fall
    outside the [0, 255] range (e.g. uint16 TMC-2 data with values in
    [500, 40000]).  Without this stretch, ``prepare_for_sift()`` would
    clip all values to 255, destroying contrast information.
    """
    data = raw.data
    num_bands = raw.num_bands

    if num_bands == 1:
        # Already single-band — just ensure 2-D float
        out = data.astype(np.float32, copy=True)
        if out.ndim == 3:
            out = out[:, :, 0]
        method = "passthrough"

    elif num_bands == 3:
        out = _rgb_to_gray(data)
        method = "luminance_bt709"

    elif num_bands == 4:
        # Drop alpha, then luminance
        out = _rgb_to_gray(data[:, :, :3])
        method = "luminance_bt709_alpha_dropped"

    else:
        # Generic multi-band: per-pixel mean
        out = np.mean(data.astype(np.float64), axis=2).astype(np.float32)
        method = f"band_mean_{num_bands}bands"

    # Dynamic range normalization: stretch to [0, 255] if needed.
    # This is critical for uint16 data (e.g. TMC-2 with values in
    # [500, 40000]) that would otherwise be clipped to 255 by
    # prepare_for_sift(), destroying all contrast.
    out = _normalize_dynamic_range(out)

    return FeatureImage(
        data=out,
        width=raw.width,
        height=raw.height,
        source_dtype=raw.dtype,
        conversion_method=method,
    )


def _normalize_dynamic_range(img: np.ndarray) -> np.ndarray:
    """Stretch pixel values to [0, 255] float32 if they fall outside uint8 range.

    This handles high-dynamic-range inputs (uint16 TMC-2 data, float32
    data with arbitrary range) by applying a linear min-max stretch.
    For data already within [0, 255], this is effectively a no-op
    (values are preserved as-is apart from NaN/inf cleanup).

    Equivalent to the ``normalize_intensity(method="minmax")`` fix in
    ``normalize.py`` but scaled to [0, 255] for direct SIFT consumption.
    """
    arr = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(arr.min())
    hi = float(arr.max())

    # Only stretch if data exceeds uint8 range
    if lo < 0.0 or hi > 255.0:
        if lo == hi:
            # Constant image outside [0, 255] — no contrast to stretch
            return np.zeros_like(arr, dtype=np.float32)
        stretched = (arr - lo) / (hi - lo) * 255.0
        return np.clip(stretched, 0.0, 255.0).astype(np.float32)

    # Data is within [0, 255] — preserve as-is
    return arr.astype(np.float32)


def _rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    """Convert an ``(H, W, 3)`` RGB array to ``(H, W)`` float32 grayscale.

    Uses BT.709 luminance weights.
    """
    # Work in float64 for precision, output float32
    r = rgb[:, :, 0].astype(np.float64)
    g = rgb[:, :, 1].astype(np.float64)
    b = rgb[:, :, 2].astype(np.float64)
    gray = _BT709_R * r + _BT709_G * g + _BT709_B * b
    return gray.astype(np.float32)

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
        2-D float32 array suitable for feature extraction.
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

    return FeatureImage(
        data=out,
        width=raw.width,
        height=raw.height,
        source_dtype=raw.dtype,
        conversion_method=method,
    )


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

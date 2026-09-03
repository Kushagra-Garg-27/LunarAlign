"""
SIH26166 — CLAHE (Contrast Limited Adaptive Histogram Equalisation).

This module wraps OpenCV's CLAHE as an **optional** preprocessing step
for illumination / contrast normalisation.

Key design decisions
--------------------
- CLAHE is applied to a **FeatureImage** (2-D float), never directly to
  a ``RawImage``.
- The input array is never modified in place; a new array is returned.
- The caller decides whether to apply CLAHE.  It is not automatically
  applied during image loading.
- For lunar imagery with strong sun-angle variations, CLAHE can improve
  feature repeatability by equalising local contrast.

Parameters
----------
The default ``clip_limit=2.0`` and ``tile_grid_size=(8, 8)`` are
standard values that work well for most imagery.  They can be tuned
per pair type by the pipeline router in later stages.
"""

from __future__ import annotations

import numpy as np
import cv2

from backend.preprocessing.datamodel import FeatureImage


def apply_clahe(
    feature_img: FeatureImage,
    *,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> FeatureImage:
    """Apply CLAHE to a ``FeatureImage`` and return a new ``FeatureImage``.

    Parameters
    ----------
    feature_img : FeatureImage
        Input 2-D single-channel feature image (float32).
    clip_limit : float
        Contrast limit for local histogram equalisation.
    tile_grid_size : tuple[int, int]
        Number of tiles in each direction for the adaptive grid.

    Returns
    -------
    FeatureImage
        A new ``FeatureImage`` with CLAHE applied.  The original is
        not modified.
    """
    data = feature_img.data

    # OpenCV CLAHE operates on uint8/uint16.  We normalise to uint8
    # for CLAHE, then convert back to float32 preserving [0, 255] range.
    dmin, dmax = float(data.min()), float(data.max())
    if dmin == dmax:
        # Constant image — CLAHE is a no-op
        return FeatureImage(
            data=data.copy(),
            width=feature_img.width,
            height=feature_img.height,
            source_dtype=feature_img.source_dtype,
            conversion_method=feature_img.conversion_method + "+clahe",
        )

    # Scale to 0–255 uint8 for CLAHE
    scaled = ((data - dmin) / (dmax - dmin) * 255.0).astype(np.uint8)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size,
    )
    equalised = clahe.apply(scaled)

    # Convert back to float32 (0–255 range)
    result = equalised.astype(np.float32)

    return FeatureImage(
        data=result,
        width=feature_img.width,
        height=feature_img.height,
        source_dtype=feature_img.source_dtype,
        conversion_method=feature_img.conversion_method + "+clahe",
    )

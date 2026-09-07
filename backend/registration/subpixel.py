"""
SIH26166 — Sub-Pixel Refinement (Module 12).

Refines image alignment to sub-pixel precision using Normalized Cross-Correlation (NCC)
and 1D parabolic peak interpolation across local spatial windows.
"""

from __future__ import annotations

import logging
from typing import Sequence

import cv2
import numpy as np

logger = logging.getLogger("sih26166.registration.subpixel")


def refine_subpixel(
    img1: np.ndarray,
    img2_warped: np.ndarray,
    initial_transform: np.ndarray,
    window_size: int = 32,
    search_radius: int = 2,
) -> np.ndarray:
    """Refine transformation matrix below pixel resolution via NCC peak parabolic fitting.

    Parameters
    ----------
    img1 : np.ndarray
        2D reference image (float32).
    img2_warped : np.ndarray
        2D source image already warped onto reference frame (float32).
    initial_transform : np.ndarray
        Initial 2x3 affine or 3x3 homography transformation matrix.
    window_size : int, default 32
        Correlation patch template size.
    search_radius : int, default 2
        Search displacement radius in pixels around patch center.

    Returns
    -------
    np.ndarray
        Refined transformation matrix with identical shape as input.
    """
    refined = initial_transform.copy().astype(np.float64)
    h1, w1 = img1.shape[:2]
    h2, w2 = img2_warped.shape[:2]
    h = min(h1, h2)
    w = min(w1, w2)

    half_w = window_size // 2
    margin = half_w + search_radius + 2

    if h < 2 * margin or w < 2 * margin:
        return refined

    # Sample a grid of test patch centers across the valid frame
    xs = np.linspace(margin, w - margin, 6, dtype=int)
    ys = np.linspace(margin, h - margin, 6, dtype=int)

    offsets_x: list[float] = []
    offsets_y: list[float] = []

    for y in ys:
        for x in xs:
            # Extract template patch from reference
            p1 = img1[y - half_w : y + half_w, x - half_w : x + half_w].astype(np.float32)
            if float(np.std(p1)) < 1e-3:
                continue

            # Extract search area from warped image
            r2 = img2_warped[
                y - half_w - search_radius : y + half_w + search_radius,
                x - half_w - search_radius : x + half_w + search_radius,
            ].astype(np.float32)

            expected_shape = (window_size + 2 * search_radius, window_size + 2 * search_radius)
            if r2.shape != expected_shape or float(np.std(r2)) < 1e-3:
                continue

            try:
                ncc = cv2.matchTemplate(r2, p1, cv2.TM_CCOEFF_NORMED)
            except cv2.error:
                continue

            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(ncc)
            px, py = max_loc

            # Integer shift relative to search window center
            dx = float(px - search_radius)
            dy = float(py - search_radius)

            # Subpixel parabolic interpolation along X axis
            if 0 < px < ncc.shape[1] - 1:
                denom_x = float(ncc[py, px - 1] - 2 * ncc[py, px] + ncc[py, px + 1])
                if abs(denom_x) > 1e-6:
                    sub_x = float(ncc[py, px - 1] - ncc[py, px + 1]) / (2.0 * denom_x)
                    if abs(sub_x) <= 1.0:
                        dx += sub_x

            # Subpixel parabolic interpolation along Y axis
            if 0 < py < ncc.shape[0] - 1:
                denom_y = float(ncc[py - 1, px] - 2 * ncc[py, px] + ncc[py + 1, px])
                if abs(denom_y) > 1e-6:
                    sub_y = float(ncc[py - 1, px] - ncc[py + 1, px]) / (2.0 * denom_y)
                    if abs(sub_y) <= 1.0:
                        dy += sub_y

            offsets_x.append(dx)
            offsets_y.append(dy)

    if not offsets_x or not offsets_y:
        return refined

    med_dx = float(np.median(offsets_x))
    med_dy = float(np.median(offsets_y))

    # Apply translation correction to transformation
    if refined.shape == (2, 3):
        refined[0, 2] += med_dx
        refined[1, 2] += med_dy
    elif refined.shape == (3, 3):
        corr = np.array(
            [[1.0, 0.0, med_dx], [0.0, 1.0, med_dy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        refined = corr @ refined

    logger.debug("Subpixel refinement: median correction (dx=%.3f, dy=%.3f)", med_dx, med_dy)
    return refined

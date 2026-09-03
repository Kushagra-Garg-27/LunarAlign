"""
SIH26166 — Match and registration visualization utilities.

Produces diagnostic images for:
- Raw/filtered correspondence lines
- Inlier/outlier annotated matches
- Registration before/after overlay

All functions return NumPy arrays (BGR uint8 images).  They do NOT
save files — file export is a future API/frontend concern.

Color convention: OpenCV BGR throughout.

This module is part of the **CLASSICAL REGISTRATION BASELINE**.
Visualization is diagnostic, not a substitute for quantitative
evaluation.
"""

from __future__ import annotations

import logging
from typing import Sequence

import cv2
import numpy as np

from backend.features.models import FilteredMatch

logger = logging.getLogger("sih26166.evaluation.visualization")

# ---------------------------------------------------------------------------
# Color palette (BGR)
# ---------------------------------------------------------------------------
COLOR_INLIER = (0, 255, 0)      # Green
COLOR_OUTLIER = (0, 0, 255)     # Red
COLOR_MATCH = (255, 200, 0)     # Cyan-ish
COLOR_POINT = (255, 255, 0)     # Cyan
POINT_RADIUS = 4
LINE_THICKNESS = 1


# ===================================================================
# Image preparation helpers
# ===================================================================

def _to_bgr_uint8(img: np.ndarray) -> np.ndarray:
    """Convert an image to BGR uint8 for drawing.

    Handles grayscale (2-D) and multi-channel (3-D) inputs.
    Float images are clipped to [0, 255] and cast to uint8.
    """
    if img is None or img.size == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    # Handle float → uint8
    if img.dtype in (np.float32, np.float64):
        out = np.clip(img, 0, 255).astype(np.uint8)
    else:
        out = img.copy()

    # Ensure uint8
    if out.dtype != np.uint8:
        out = out.astype(np.uint8)

    # Grayscale → BGR
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    elif out.ndim == 3 and out.shape[2] == 1:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

    return out


def _make_side_by_side(
    ref_img: np.ndarray,
    tgt_img: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Create a side-by-side canvas from two images.

    The images are vertically centered.  Returns the canvas and the
    x-offset for the target image.

    Returns
    -------
    canvas : np.ndarray
        BGR uint8 canvas with both images.
    x_offset : int
        Horizontal offset for target-image coordinates.
    """
    ref = _to_bgr_uint8(ref_img)
    tgt = _to_bgr_uint8(tgt_img)

    h1, w1 = ref.shape[:2]
    h2, w2 = tgt.shape[:2]

    max_h = max(h1, h2)
    canvas = np.zeros((max_h, w1 + w2, 3), dtype=np.uint8)

    # Place reference (left) and target (right), vertically centered
    y_off1 = (max_h - h1) // 2
    y_off2 = (max_h - h2) // 2
    canvas[y_off1:y_off1 + h1, :w1] = ref
    canvas[y_off2:y_off2 + h2, w1:w1 + w2] = tgt

    return canvas, w1


# ===================================================================
# Match visualization
# ===================================================================

def draw_filtered_matches(
    ref_img: np.ndarray,
    tgt_img: np.ndarray,
    matches: list[FilteredMatch],
    *,
    max_matches: int = 200,
    color: tuple[int, int, int] = COLOR_MATCH,
) -> np.ndarray:
    """Draw filtered correspondences between two images.

    Parameters
    ----------
    ref_img : np.ndarray
        Reference image (grayscale or BGR).
    tgt_img : np.ndarray
        Target image (grayscale or BGR).
    matches : list[FilteredMatch]
        Filtered correspondences.
    max_matches : int
        Maximum matches to draw (for readability).
    color : tuple
        BGR color for match lines and points.

    Returns
    -------
    np.ndarray
        BGR uint8 side-by-side image with match lines drawn.
    """
    canvas, x_offset = _make_side_by_side(ref_img, tgt_img)

    if not matches:
        return canvas

    for m in matches[:max_matches]:
        ref_pt = (int(round(m.ref_pt[0])), int(round(m.ref_pt[1])))
        tgt_pt = (int(round(m.tgt_pt[0])) + x_offset, int(round(m.tgt_pt[1])))

        cv2.circle(canvas, ref_pt, POINT_RADIUS, color, -1)
        cv2.circle(canvas, tgt_pt, POINT_RADIUS, color, -1)
        cv2.line(canvas, ref_pt, tgt_pt, color, LINE_THICKNESS)

    return canvas


def draw_inlier_outlier_matches(
    ref_img: np.ndarray,
    tgt_img: np.ndarray,
    matches: list[FilteredMatch],
    inlier_mask: np.ndarray,
    *,
    max_matches: int = 200,
    inlier_color: tuple[int, int, int] = COLOR_INLIER,
    outlier_color: tuple[int, int, int] = COLOR_OUTLIER,
) -> np.ndarray:
    """Draw correspondences with inliers and outliers visually distinguished.

    Parameters
    ----------
    ref_img : np.ndarray
        Reference image (grayscale or BGR).
    tgt_img : np.ndarray
        Target image (grayscale or BGR).
    matches : list[FilteredMatch]
        All filtered correspondences.
    inlier_mask : np.ndarray
        Boolean array of length ``len(matches)``.
    max_matches : int
        Maximum matches to draw.
    inlier_color : tuple
        BGR color for inlier matches.
    outlier_color : tuple
        BGR color for outlier matches.

    Returns
    -------
    np.ndarray
        BGR uint8 side-by-side image with colored match annotations.
    """
    canvas, x_offset = _make_side_by_side(ref_img, tgt_img)

    if not matches:
        return canvas

    if inlier_mask is None:
        inlier_mask = np.ones(len(matches), dtype=bool)

    # Draw outliers first (so inliers are on top)
    drawn = 0
    for m, is_inlier in zip(matches, inlier_mask):
        if drawn >= max_matches:
            break
        if is_inlier:
            continue  # draw inliers second pass
        ref_pt = (int(round(m.ref_pt[0])), int(round(m.ref_pt[1])))
        tgt_pt = (int(round(m.tgt_pt[0])) + x_offset, int(round(m.tgt_pt[1])))
        cv2.circle(canvas, ref_pt, POINT_RADIUS, outlier_color, -1)
        cv2.circle(canvas, tgt_pt, POINT_RADIUS, outlier_color, -1)
        cv2.line(canvas, ref_pt, tgt_pt, outlier_color, LINE_THICKNESS)
        drawn += 1

    for m, is_inlier in zip(matches, inlier_mask):
        if drawn >= max_matches:
            break
        if not is_inlier:
            continue
        ref_pt = (int(round(m.ref_pt[0])), int(round(m.ref_pt[1])))
        tgt_pt = (int(round(m.tgt_pt[0])) + x_offset, int(round(m.tgt_pt[1])))
        cv2.circle(canvas, ref_pt, POINT_RADIUS, inlier_color, -1)
        cv2.circle(canvas, tgt_pt, POINT_RADIUS, inlier_color, -1)
        cv2.line(canvas, ref_pt, tgt_pt, inlier_color, LINE_THICKNESS)
        drawn += 1

    return canvas


# ===================================================================
# Registration overlay
# ===================================================================

def draw_registration_overlay(
    target_img: np.ndarray,
    registered_img: np.ndarray,
    *,
    alpha: float = 0.5,
) -> np.ndarray:
    """Create an alpha-blended overlay of target and registered images.

    Parameters
    ----------
    target_img : np.ndarray
        Target (reference frame) image.
    registered_img : np.ndarray
        Warped reference image in the target coordinate frame.
    alpha : float
        Blend factor for the registered image (0–1).
        0.0 = only target, 1.0 = only registered.

    Returns
    -------
    np.ndarray
        BGR uint8 alpha-blended overlay image.
    """
    tgt = _to_bgr_uint8(target_img)
    reg = _to_bgr_uint8(registered_img)

    # Ensure same dimensions
    h1, w1 = tgt.shape[:2]
    h2, w2 = reg.shape[:2]

    # Use the smaller common area
    h = min(h1, h2)
    w = min(w1, w2)

    tgt_crop = tgt[:h, :w]
    reg_crop = reg[:h, :w]

    alpha = max(0.0, min(1.0, alpha))
    blended = cv2.addWeighted(reg_crop, alpha, tgt_crop, 1.0 - alpha, 0.0)

    return blended


def draw_side_by_side(
    img_a: np.ndarray,
    img_b: np.ndarray,
    *,
    label_a: str = "",
    label_b: str = "",
) -> np.ndarray:
    """Create a side-by-side comparison of two images.

    Parameters
    ----------
    img_a : np.ndarray
        Left image.
    img_b : np.ndarray
        Right image.
    label_a, label_b : str
        Optional text labels drawn on the images.

    Returns
    -------
    np.ndarray
        BGR uint8 side-by-side image.
    """
    canvas, x_offset = _make_side_by_side(img_a, img_b)

    if label_a:
        cv2.putText(canvas, label_a, (10, 25),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if label_b:
        cv2.putText(canvas, label_b, (x_offset + 10, 25),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return canvas

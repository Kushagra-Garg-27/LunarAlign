"""
SIH26166 — SIFT Feature Detection & Description (Modules 04, 05, 09).

Implements standard SIFT extraction and grid-bucketed SIFT for spatially
uniform keypoint distribution across lunar crater morphology.
"""

from __future__ import annotations

import logging
from typing import Sequence

import cv2
import numpy as np

logger = logging.getLogger("sih26166.matching.sift")


def detect_sift_features(
    image: np.ndarray,
    max_keypoints: int = 5000,
    contrast_threshold: float = 0.04,
    edge_threshold: float = 10.0,
) -> tuple[list[cv2.KeyPoint], np.ndarray]:
    """Detect SIFT keypoints and compute 128-dimensional floating point descriptors.

    Parameters
    ----------
    image : np.ndarray
        2D preprocessed grayscale image (float32, uint8, or uint16).
    max_keypoints : int, default 5000
        Maximum number of keypoints to retain.
    contrast_threshold : float, default 0.04
        Contrast threshold to filter out weak features in semi-uniform regions.
    edge_threshold : float, default 10.0
        Threshold to filter out edge-like features.

    Returns
    -------
    tuple[list[cv2.KeyPoint], np.ndarray]
        - keypoints: List of cv2.KeyPoint objects.
        - descriptors: Array of shape (N, 128) in float32.
    """
    if image.size == 0 or image.ndim != 2:
        return [], np.zeros((0, 128), dtype=np.float32)

    h, w = image.shape
    if h < 4 or w < 4:
        return [], np.zeros((0, 128), dtype=np.float32)

    # Convert to uint8 for OpenCV SIFT
    u8 = _to_uint8(image)

    sift = cv2.SIFT_create(
        nfeatures=max_keypoints,
        contrastThreshold=contrast_threshold,
        edgeThreshold=edge_threshold,
    )

    try:
        keypoints, descriptors = sift.detectAndCompute(u8, None)
    except cv2.error as exc:
        logger.error("OpenCV SIFT detectAndCompute failed: %s", exc)
        return [], np.zeros((0, 128), dtype=np.float32)

    if keypoints is None or descriptors is None or len(keypoints) == 0:
        return [], np.zeros((0, 128), dtype=np.float32)

    # Enforce max_keypoints if needed
    if max_keypoints > 0 and len(keypoints) > max_keypoints:
        # Sort by response strength descending
        indices = np.argsort([-kp.response for kp in keypoints])[:max_keypoints]
        keypoints = [keypoints[i] for i in indices]
        descriptors = descriptors[indices]

    return list(keypoints), descriptors.astype(np.float32)


def detect_sift_bucketed(
    image: np.ndarray,
    grid_size: tuple[int, int] = (8, 8),
    keypoints_per_cell: int = 100,
    contrast_threshold: float = 0.04,
    edge_threshold: float = 10.0,
) -> tuple[list[cv2.KeyPoint], np.ndarray]:
    """Grid-bucketed SIFT detection for spatially uniform feature distribution (Module 09).

    Divides image into grid_size cells, runs SIFT independently per cell,
    and retains top keypoints_per_cell by response strength.

    Parameters
    ----------
    image : np.ndarray
        2D preprocessed grayscale image.
    grid_size : tuple[int, int], default (8, 8)
        (rows, cols) subdivision of the image grid.
    keypoints_per_cell : int, default 100
        Maximum keypoints retained per grid cell.
    contrast_threshold : float, default 0.04
        Contrast threshold for SIFT detector.
    edge_threshold : float, default 10.0
        Edge threshold for SIFT detector.

    Returns
    -------
    tuple[list[cv2.KeyPoint], np.ndarray]
        Merged keypoints and descriptors across all grid cells.
    """
    if image.size == 0 or image.ndim != 2:
        return [], np.zeros((0, 128), dtype=np.float32)

    h, w = image.shape
    grid_rows, grid_cols = grid_size
    grid_rows = max(1, grid_rows)
    grid_cols = max(1, grid_cols)

    cell_h = h / grid_rows
    cell_w = w / grid_cols

    all_kps: list[cv2.KeyPoint] = []
    desc_list: list[np.ndarray] = []

    u8 = _to_uint8(image)

    sift = cv2.SIFT_create(
        nfeatures=keypoints_per_cell * 2,
        contrastThreshold=contrast_threshold,
        edgeThreshold=edge_threshold,
    )

    for r in range(grid_rows):
        y0 = int(round(r * cell_h))
        y1 = int(round((r + 1) * cell_h)) if r < grid_rows - 1 else h
        for c in range(grid_cols):
            x0 = int(round(c * cell_w))
            x1 = int(round((c + 1) * cell_w)) if c < grid_cols - 1 else w

            cell = u8[y0:y1, x0:x1]
            if cell.shape[0] < 4 or cell.shape[1] < 4:
                continue

            try:
                kps, descs = sift.detectAndCompute(cell, None)
            except cv2.error:
                continue

            if kps is None or descs is None or len(kps) == 0:
                continue

            # Select top response keypoints in this bucket
            if len(kps) > keypoints_per_cell:
                indices = np.argsort([-kp.response for kp in kps])[:keypoints_per_cell]
                kps = [kps[i] for i in indices]
                descs = descs[indices]

            # Offset keypoint coordinates from cell local frame to global image frame
            for kp in kps:
                shifted_kp = cv2.KeyPoint(
                    x=kp.pt[0] + x0,
                    y=kp.pt[1] + y0,
                    size=kp.size,
                    angle=kp.angle,
                    response=kp.response,
                    octave=kp.octave,
                    class_id=kp.class_id,
                )
                all_kps.append(shifted_kp)

            desc_list.append(descs)

    if not all_kps or not desc_list:
        # Fallback to standard global SIFT detection if bucketing yielded no points
        return detect_sift_features(
            image,
            max_keypoints=grid_rows * grid_cols * keypoints_per_cell,
            contrast_threshold=contrast_threshold,
            edge_threshold=edge_threshold,
        )

    all_descs = np.vstack(desc_list).astype(np.float32)
    return all_kps, all_descs


def _to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert arbitrary numeric 2D image to uint8 for OpenCV."""
    clean = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    if clean.dtype == np.uint8:
        return clean

    if clean.dtype == np.uint16:
        return (clean / 256.0).astype(np.uint8)

    # Float array
    lo, hi = float(clean.min()), float(clean.max())
    if lo == hi:
        return np.zeros(clean.shape, dtype=np.uint8)

    if lo >= 0.0 and hi <= 1.01:
        return (np.clip(clean, 0.0, 1.0) * 255.0).astype(np.uint8)

    # General min-max stretch
    scaled = (clean - lo) / (hi - lo) * 255.0
    return np.clip(scaled, 0.0, 255.0).astype(np.uint8)

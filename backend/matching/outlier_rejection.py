"""
SIH26166 — MAGSAC++ & Robust Outlier Rejection (Module 10).

Performs robust geometric model estimation (Homography via USAC_MAGSAC, Affine via RANSAC)
to eliminate false correspondences caused by repetitive lunar crater patterns and shadow variations.
"""

from __future__ import annotations

import logging
from typing import Sequence

import cv2
import numpy as np

logger = logging.getLogger("sih26166.matching.outlier_rejection")


def reject_outliers_magsac(
    kp1: list[cv2.KeyPoint],
    kp2: list[cv2.KeyPoint],
    matches: list[cv2.DMatch],
    confidence: float = 0.999,
    max_iters: int = 5000,
    threshold: float = 3.0,
) -> tuple[list[cv2.DMatch], np.ndarray, np.ndarray]:
    """Perform MAGSAC++ robust homography estimation and outlier rejection.

    Parameters
    ----------
    kp1 : list[cv2.KeyPoint]
        Keypoints detected in image 1.
    kp2 : list[cv2.KeyPoint]
        Keypoints detected in image 2.
    matches : list[cv2.DMatch]
        Initial candidate matches.
    confidence : float, default 0.999
        RANSAC/MAGSAC confidence parameter.
    max_iters : int, default 5000
        Maximum MAGSAC iterations.
    threshold : float, default 3.0
        Maximum reprojection error in pixels for inlier classification.

    Returns
    -------
    tuple[list[cv2.DMatch], np.ndarray, np.ndarray]
        - inlier_matches: List of geometrically consistent cv2.DMatch objects.
        - homography_matrix: 3x3 float32 homography matrix (or identity if failed).
        - inlier_mask: (N, 1) uint8 binary mask indicating inliers.
    """
    if len(matches) < 4 or len(kp1) == 0 or len(kp2) == 0:
        return [], np.eye(3, dtype=np.float32), np.zeros((len(matches), 1), dtype=np.uint8)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    h_matrix = None
    mask = None

    # 1. Try USAC_MAGSAC (marginalized sample consensus)
    try:
        h_matrix, mask = cv2.findHomography(
            src_pts,
            dst_pts,
            method=cv2.USAC_MAGSAC,
            ransacReprojThreshold=threshold,
            maxIters=max_iters,
            confidence=confidence,
        )
    except cv2.error as exc:
        logger.warning("cv2.USAC_MAGSAC failed (%s), falling back to standard RANSAC.", exc)

    # 2. Fallback to standard RANSAC if MAGSAC returned None
    if h_matrix is None or mask is None:
        try:
            h_matrix, mask = cv2.findHomography(
                src_pts,
                dst_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=threshold,
                maxIters=max_iters,
                confidence=confidence,
            )
        except cv2.error as exc:
            logger.error("Standard RANSAC homography estimation failed: %s", exc)

    if h_matrix is None or mask is None:
        return [], np.eye(3, dtype=np.float32), np.zeros((len(matches), 1), dtype=np.uint8)

    inlier_mask = mask.ravel()
    inlier_matches = [m for i, m in enumerate(matches) if inlier_mask[i] == 1]

    logger.debug(
        "MAGSAC++ geometric verification: %d / %d inliers (%.1f%%)",
        len(inlier_matches), len(matches),
        (len(inlier_matches) / max(1, len(matches))) * 100.0,
    )

    return inlier_matches, h_matrix.astype(np.float32), mask.astype(np.uint8)


def reject_outliers_affine(
    kp1: list[cv2.KeyPoint],
    kp2: list[cv2.KeyPoint],
    matches: list[cv2.DMatch],
    confidence: float = 0.999,
    threshold: float = 3.0,
    max_iters: int = 5000,
) -> tuple[list[cv2.DMatch], np.ndarray, np.ndarray]:
    """Perform Affine-model geometric verification using RANSAC.

    Suitable for same-sensor pairs with negligible perspective distortion.

    Parameters
    ----------
    kp1 : list[cv2.KeyPoint]
        Keypoints detected in image 1.
    kp2 : list[cv2.KeyPoint]
        Keypoints detected in image 2.
    matches : list[cv2.DMatch]
        Initial candidate matches.
    confidence : float, default 0.999
        RANSAC confidence parameter.
    threshold : float, default 3.0
        Maximum reprojection error in pixels.
    max_iters : int, default 5000
        Maximum RANSAC iterations.

    Returns
    -------
    tuple[list[cv2.DMatch], np.ndarray, np.ndarray]
        - inlier_matches: List of affine inliers.
        - affine_matrix: 2x3 float32 affine matrix.
        - inlier_mask: (N, 1) uint8 binary mask.
    """
    if len(matches) < 3 or len(kp1) == 0 or len(kp2) == 0:
        return [], np.zeros((2, 3), dtype=np.float32), np.zeros((len(matches), 1), dtype=np.uint8)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches])

    try:
        aff_matrix, mask = cv2.estimateAffine2D(
            src_pts,
            dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=threshold,
            maxIters=max_iters,
            confidence=confidence,
        )
    except cv2.error as exc:
        logger.error("estimateAffine2D failed: %s", exc)
        return [], np.zeros((2, 3), dtype=np.float32), np.zeros((len(matches), 1), dtype=np.uint8)

    if aff_matrix is None or mask is None:
        return [], np.zeros((2, 3), dtype=np.float32), np.zeros((len(matches), 1), dtype=np.uint8)

    inlier_mask = mask.ravel()
    inlier_matches = [m for i, m in enumerate(matches) if inlier_mask[i] == 1]

    return inlier_matches, aff_matrix.astype(np.float32), mask.astype(np.uint8)

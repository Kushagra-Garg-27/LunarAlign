"""
SIH26166 — Transformation Estimation & Model Selection (Module 11).

Selects, estimates, and refines geometric transformation models (Affine / Homography)
from match correspondences and decomposes them into physical parameters.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import cv2
import numpy as np

from backend.matching.datamodel import MatchResult

logger = logging.getLogger("sih26166.registration.transform")


def estimate_transform(
    match_result: MatchResult,
    model: Literal["auto", "affine", "homography"] | str = "auto",
) -> tuple[np.ndarray, str]:
    """Select, estimate, and refine transformation model from MatchResult.

    Parameters
    ----------
    match_result : MatchResult
        Feature match results containing inliers and candidate keypoints.
    model : str, default "auto"
        Transformation model selection ("auto", "affine", "homography").
        In "auto" mode:
        - Uses Affine if inlier_matches < 50 or spatial_distribution < 2.0.
        - Uses Homography otherwise.

    Returns
    -------
    tuple[np.ndarray, str]
        - transform_matrix: 3x3 float64 matrix for homography or 2x3 float64 matrix for affine.
        - model_type: "homography" or "affine".
    """
    model_norm = model.lower()

    # 1. Determine target model type
    if model_norm == "auto":
        if match_result.inlier_matches < 50 or match_result.spatial_distribution < 2.0:
            target_model = "affine"
        else:
            target_model = "homography"
    elif model_norm in ("affine", "homography"):
        target_model = model_norm
    else:
        raise ValueError(
            f"Unsupported transformation model: '{model}'. "
            f"Expected 'auto', 'affine', or 'homography'."
        )

    pts1 = match_result.match_points1
    pts2 = match_result.match_points2
    n_pts = len(pts1)

    # 2. Check if pre-existing transform in match_result matches requested model
    if match_result.transform_matrix is not None:
        mat = match_result.transform_matrix
        if target_model == "homography" and mat.shape == (3, 3):
            return mat.astype(np.float64), "homography"
        if target_model == "affine" and mat.shape == (2, 3):
            return mat.astype(np.float64), "affine"

    # 3. Estimate model from inlier points
    if target_model == "affine":
        if n_pts >= 3:
            try:
                aff_mat, _ = cv2.estimateAffine2D(pts1, pts2, method=cv2.RANSAC)
                if aff_mat is not None:
                    return aff_mat.astype(np.float64), "affine"
            except cv2.error as exc:
                logger.warning("estimateAffine2D failed: %s", exc)

        # Fallback identity affine
        return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64), "affine"

    else:  # Homography
        if n_pts >= 4:
            try:
                h_mat, _ = cv2.findHomography(pts1, pts2, method=cv2.USAC_MAGSAC)
                if h_mat is not None:
                    return h_mat.astype(np.float64), "homography"
            except cv2.error as exc:
                logger.warning("findHomography failed: %s", exc)

        # Fallback identity homography
        return np.eye(3, dtype=np.float64), "homography"


def decompose_transform(matrix: np.ndarray) -> dict[str, float]:
    """Decompose an affine (2x3) or homography (3x3) matrix into physical parameters.

    Parameters
    ----------
    matrix : np.ndarray
        2x3 affine matrix or 3x3 homography matrix.

    Returns
    -------
    dict[str, float]
        Dictionary with:
        - rotation_deg: Rotation angle in degrees (-180 to 180).
        - scale_x, scale_y: Scaling factors along x and y axes.
        - translation_x, translation_y: Translation offsets in pixels.
        - shear: Shear factor.
        - perspective_x, perspective_y: Perspective warping parameters (0.0 for affine).
    """
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.shape == (3, 3):
        norm_factor = mat[2, 2] if abs(mat[2, 2]) > 1e-9 else 1.0
        normalized = mat / norm_factor
        aff = normalized[:2, :]
        p_x, p_y = float(normalized[2, 0]), float(normalized[2, 1])
    elif mat.shape == (2, 3):
        aff = mat
        p_x, p_y = 0.0, 0.0
    else:
        raise ValueError(
            f"Invalid transform matrix shape {mat.shape}. Expected (2, 3) or (3, 3)."
        )

    a, b, tx = aff[0, 0], aff[0, 1], aff[0, 2]
    c, d, ty = aff[1, 0], aff[1, 1], aff[1, 2]

    # Polar decomposition
    sx = float(np.sqrt(a**2 + c**2))
    theta = float(np.arctan2(c, a))
    rotation_deg = float(np.rad2deg(theta))

    det = a * d - b * c
    sy = float(abs(det) / max(sx, 1e-9))
    shear = float((a * b + c * d) / max(sx**2, 1e-9))

    return {
        "rotation_deg": rotation_deg,
        "scale_x": sx,
        "scale_y": sy,
        "translation_x": float(tx),
        "translation_y": float(ty),
        "shear": shear,
        "perspective_x": p_x,
        "perspective_y": p_y,
    }

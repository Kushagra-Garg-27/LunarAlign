"""
SIH26166 — Robust geometric estimation using OpenCV's USAC framework.

Estimates affine or homography transformations from filtered SIFT
correspondences, using MAGSAC++ (via ``cv2.USAC_MAGSAC``) where the
installed OpenCV version supports it.

Pipeline position
-----------------
    FeatureImage → SIFT → FLANN → Lowe ratio test
        → **geometric verification** (this module)
        → transformation estimation
        → registration metrics

Estimator selection
-------------------
1. Preferred: ``cv2.USAC_MAGSAC`` (MAGSAC++ through OpenCV's USAC
   framework).  Available in OpenCV ≥ 4.5.
2. Fallback: ``cv2.RANSAC``.  Available in all OpenCV versions with
   ``estimateAffine2D`` / ``findHomography``.

The estimator actually used is recorded in the result — the code never
silently claims MAGSAC++ if RANSAC was used.

Minimum correspondences
-----------------------
- Affine (2×3): minimum 3 non-degenerate correspondences.
- Homography (3×3): minimum 4 non-degenerate correspondences.

If fewer correspondences are available, the function returns a
structured failure result rather than crashing.

Geometric assumptions
---------------------
For lunar orbital imagery, a single affine transform may not perfectly
capture all geometric distortion present — terrain relief and oblique
viewing angles can introduce perspective effects.  This limitation is
kept explicit (see ``TransformModel`` documentation).

This module is part of the **CLASSICAL REGISTRATION BASELINE**.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from backend.features.models import FilteredMatch
from backend.geometry.models import (
    ErrorMetrics,
    EstimatorCapability,
    EstimatorMethod,
    GeometricResult,
    TransformModel,
)
from backend.geometry.transform import transform_points

logger = logging.getLogger("sih26166.geometry.estimation")

# ---------------------------------------------------------------------------
# Minimum correspondences for each model
# ---------------------------------------------------------------------------
MIN_CORRESPONDENCES = {
    TransformModel.AFFINE: 3,
    TransformModel.HOMOGRAPHY: 4,
}

# ---------------------------------------------------------------------------
# Default estimation parameters
# ---------------------------------------------------------------------------
DEFAULT_REPROJ_THRESHOLD: float = 3.0   # pixels
DEFAULT_CONFIDENCE: float = 0.999
DEFAULT_MAX_ITERS: int = 2000


# ===================================================================
# Capability detection
# ===================================================================

def detect_estimator_capability() -> EstimatorCapability:
    """Detect what robust estimation facilities the installed OpenCV provides.

    Returns
    -------
    EstimatorCapability
        Detected capability record.
    """
    version = cv2.__version__

    has_usac_magsac = hasattr(cv2, "USAC_MAGSAC")
    has_estimate_affine2d = hasattr(cv2, "estimateAffine2D")
    has_find_homography = hasattr(cv2, "findHomography")

    usac_flags = [attr for attr in dir(cv2) if "USAC" in attr]

    cap = EstimatorCapability(
        opencv_version=version,
        has_usac_magsac=has_usac_magsac,
        has_estimate_affine2d=has_estimate_affine2d,
        has_find_homography=has_find_homography,
        available_usac_flags=usac_flags,
    )

    logger.info(
        "OpenCV %s: USAC_MAGSAC=%s, estimateAffine2D=%s, findHomography=%s",
        version, has_usac_magsac, has_estimate_affine2d, has_find_homography,
    )

    return cap


# ===================================================================
# Correspondence extraction
# ===================================================================

def _extract_point_arrays(
    matches: list[FilteredMatch],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract reference and target point arrays from filtered matches.

    Parameters
    ----------
    matches : list[FilteredMatch]
        Correspondences from the Lowe ratio test.

    Returns
    -------
    ref_pts : np.ndarray
        Shape ``(N, 2)`` — reference image (x, y) coordinates.
    tgt_pts : np.ndarray
        Shape ``(N, 2)`` — target image (x, y) coordinates.
    """
    ref_pts = np.array([m.ref_pt for m in matches], dtype=np.float64)
    tgt_pts = np.array([m.tgt_pt for m in matches], dtype=np.float64)
    return ref_pts, tgt_pts


# ===================================================================
# Error metrics computation
# ===================================================================

def compute_reprojection_errors(
    ref_pts: np.ndarray,
    tgt_pts: np.ndarray,
    matrix: np.ndarray,
    inlier_mask: np.ndarray,
) -> ErrorMetrics:
    """Compute reprojection error metrics.

    For each correspondence (ref_pt, tgt_pt):
    1. Transform ref_pt using the estimated transformation.
    2. Compute Euclidean distance to tgt_pt.

    Parameters
    ----------
    ref_pts : np.ndarray
        Reference points, shape ``(N, 2)``.
    tgt_pts : np.ndarray
        Target points, shape ``(N, 2)``.
    matrix : np.ndarray
        Estimated transformation matrix (2×3 or 3×3).
    inlier_mask : np.ndarray
        Boolean array of length N.

    Returns
    -------
    ErrorMetrics
        Complete error statistics.
    """
    n = len(ref_pts)

    # Transform reference points
    transformed = transform_points(ref_pts, matrix)

    # Per-match Euclidean error
    diff = transformed - tgt_pts.astype(np.float64)
    per_match_errors = np.sqrt(np.sum(diff ** 2, axis=1))

    # All-match metrics
    all_rmse = float(np.sqrt(np.mean(per_match_errors ** 2)))
    all_median = float(np.median(per_match_errors))

    # Inlier-only metrics
    inlier_errors = per_match_errors[inlier_mask]
    if len(inlier_errors) > 0:
        inlier_rmse = float(np.sqrt(np.mean(inlier_errors ** 2)))
        inlier_median = float(np.median(inlier_errors))
        inlier_max = float(np.max(inlier_errors))
    else:
        inlier_rmse = 0.0
        inlier_median = 0.0
        inlier_max = 0.0

    return ErrorMetrics(
        per_match_errors=per_match_errors,
        inlier_rmse=inlier_rmse,
        all_rmse=all_rmse,
        inlier_median_error=inlier_median,
        inlier_max_error=inlier_max,
        all_median_error=all_median,
    )


# ===================================================================
# Failure result helper
# ===================================================================

def _make_failure_result(
    reason: str,
    model: TransformModel,
    total: int,
    config: dict[str, Any],
    capability: EstimatorCapability | None = None,
) -> GeometricResult:
    """Create a structured failure GeometricResult."""
    logger.warning("Geometric estimation failed: %s", reason)
    return GeometricResult(
        success=False,
        transform_matrix=None,
        transform_model=model,
        estimator_method=None,
        inlier_mask=None,
        inlier_count=0,
        outlier_count=total,
        total_correspondences=total,
        inlier_ratio=0.0,
        error_metrics=None,
        failure_reason=reason,
        estimator_config=config,
        capability=capability,
    )


# ===================================================================
# Core estimation
# ===================================================================

def _select_method_flag(
    capability: EstimatorCapability,
) -> tuple[int, EstimatorMethod]:
    """Select the best available robust estimation method.

    Returns
    -------
    flag : int
        OpenCV method flag to pass to estimateAffine2D / findHomography.
    method : EstimatorMethod
        Record of what was selected.
    """
    if capability.has_usac_magsac:
        logger.info("Using USAC_MAGSAC (MAGSAC++) estimator")
        return cv2.USAC_MAGSAC, EstimatorMethod.MAGSAC
    else:
        logger.warning(
            "USAC_MAGSAC not available in OpenCV %s; falling back to RANSAC",
            capability.opencv_version,
        )
        return cv2.RANSAC, EstimatorMethod.RANSAC


def estimate_transform(
    matches: list[FilteredMatch],
    *,
    model: TransformModel = TransformModel.AFFINE,
    reproj_threshold: float = DEFAULT_REPROJ_THRESHOLD,
    confidence: float = DEFAULT_CONFIDENCE,
    max_iters: int = DEFAULT_MAX_ITERS,
) -> GeometricResult:
    """Estimate a geometric transformation from filtered correspondences.

    This is the main entry point for geometric verification in the
    classical registration baseline.

    Parameters
    ----------
    matches : list[FilteredMatch]
        Correspondences that passed the Lowe ratio test.
    model : TransformModel
        Which transformation model to estimate.
    reproj_threshold : float
        RANSAC/MAGSAC reprojection error threshold in pixels.
    confidence : float
        Desired confidence level (0–1).
    max_iters : int
        Maximum number of RANSAC/MAGSAC iterations.

    Returns
    -------
    GeometricResult
        Complete estimation result with transformation, inlier mask,
        and error metrics.
    """
    config: dict[str, Any] = {
        "model": model.value,
        "reproj_threshold": reproj_threshold,
        "confidence": confidence,
        "max_iters": max_iters,
    }

    # --- Capability detection ---
    capability = detect_estimator_capability()

    total = len(matches)
    min_required = MIN_CORRESPONDENCES[model]

    # --- Minimum correspondence check ---
    if total == 0:
        return _make_failure_result(
            "No correspondences provided (0 matches).",
            model, total, config, capability,
        )

    if total < min_required:
        return _make_failure_result(
            f"Insufficient correspondences: {total} provided, "
            f"{min_required} required for {model.value} estimation.",
            model, total, config, capability,
        )

    # --- Extract point arrays ---
    ref_pts, tgt_pts = _extract_point_arrays(matches)

    # --- Select robust estimator ---
    method_flag, method_enum = _select_method_flag(capability)

    # --- Perform estimation ---
    try:
        if model == TransformModel.AFFINE:
            matrix, inlier_mask_raw = cv2.estimateAffine2D(
                ref_pts.reshape(-1, 1, 2).astype(np.float64),
                tgt_pts.reshape(-1, 1, 2).astype(np.float64),
                method=method_flag,
                ransacReprojThreshold=reproj_threshold,
                confidence=confidence,
                maxIters=max_iters,
            )
        elif model == TransformModel.HOMOGRAPHY:
            matrix, inlier_mask_raw = cv2.findHomography(
                ref_pts.reshape(-1, 1, 2).astype(np.float64),
                tgt_pts.reshape(-1, 1, 2).astype(np.float64),
                method=method_flag,
                ransacReprojThreshold=reproj_threshold,
                confidence=confidence,
                maxIters=max_iters,
            )
        else:
            return _make_failure_result(
                f"Unsupported model: {model}",
                model, total, config, capability,
            )
    except cv2.error as exc:
        return _make_failure_result(
            f"OpenCV estimation failed: {exc}",
            model, total, config, capability,
        )

    # --- Validate output ---
    if matrix is None:
        return _make_failure_result(
            "Estimation returned no transformation matrix "
            "(degenerate configuration or convergence failure).",
            model, total, config, capability,
        )

    # Parse inlier mask
    if inlier_mask_raw is not None:
        inlier_mask = inlier_mask_raw.ravel().astype(bool)
    else:
        # If no mask returned, treat all as inliers (unlikely but safe)
        inlier_mask = np.ones(total, dtype=bool)

    inlier_count = int(np.sum(inlier_mask))
    outlier_count = total - inlier_count
    inlier_ratio = inlier_count / total if total > 0 else 0.0

    # --- Compute reprojection errors ---
    error_metrics = compute_reprojection_errors(
        ref_pts, tgt_pts, matrix, inlier_mask,
    )

    logger.info(
        "Geometric estimation succeeded: model=%s, method=%s, "
        "inliers=%d/%d (%.1f%%), inlier_RMSE=%.3f px",
        model.value, method_enum.value,
        inlier_count, total, inlier_ratio * 100,
        error_metrics.inlier_rmse,
    )

    return GeometricResult(
        success=True,
        transform_matrix=matrix,
        transform_model=model,
        estimator_method=method_enum,
        inlier_mask=inlier_mask,
        inlier_count=inlier_count,
        outlier_count=outlier_count,
        total_correspondences=total,
        inlier_ratio=inlier_ratio,
        error_metrics=error_metrics,
        failure_reason="",
        estimator_config=config,
        capability=capability,
    )

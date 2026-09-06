"""Evaluation metrics for image registration quality."""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_rmse(
    points_src: np.ndarray,
    points_dst: np.ndarray,
    transform_matrix: np.ndarray,
) -> float:
    """Compute Root Mean Square Error of transformed source points vs destination points.

    Parameters
    ----------
    points_src : np.ndarray
        Nx2 array of source point coordinates [x, y].
    points_dst : np.ndarray
        Nx2 array of target/destination point coordinates [x, y].
    transform_matrix : np.ndarray
        2x3 affine matrix or 3x3 homography matrix.

    Returns
    -------
    float
        RMSE in pixels.
    """
    if len(points_src) == 0 or len(points_dst) == 0:
        return 0.0

    pts_src = np.asarray(points_src, dtype=np.float64)
    pts_dst = np.asarray(points_dst, dtype=np.float64)
    mat = np.asarray(transform_matrix, dtype=np.float64)

    if mat.shape == (2, 3):
        # Affine transform
        reprojected = (mat[:2, :2] @ pts_src.T).T + mat[:2, 2]
    elif mat.shape == (3, 3):
        # Homography
        src_h = np.hstack([pts_src, np.ones((len(pts_src), 1), dtype=np.float64)])
        proj_h = (mat @ src_h.T).T
        w = np.maximum(np.abs(proj_h[:, 2:]), 1e-9)
        reprojected = proj_h[:, :2] / w
    else:
        raise ValueError(f"Invalid transform_matrix shape {mat.shape}. Expected (2, 3) or (3, 3).")

    sq_errors = np.sum((reprojected - pts_dst) ** 2, axis=1)
    rmse = float(np.sqrt(np.mean(sq_errors)))
    return rmse


def compute_inlier_stats(inlier_mask: np.ndarray, total_matches: int) -> dict[str, Any]:
    """Compute summary inlier statistics from a boolean or binary inlier mask.

    Parameters
    ----------
    inlier_mask : np.ndarray
        1D or 2D array indicating inliers (non-zero entries).
    total_matches : int
        Total candidate matches prior to outlier rejection.

    Returns
    -------
    dict[str, Any]
        Dictionary with inlier_count, total_matches, and inlier_ratio.
    """
    inlier_count = int(np.sum(np.asarray(inlier_mask) > 0))
    total = max(0, int(total_matches))
    ratio = float(inlier_count / total) if total > 0 else 0.0

    return {
        "inlier_count": inlier_count,
        "total_matches": total,
        "inlier_ratio": ratio,
    }


def compute_ncc_score(
    reference: np.ndarray,
    warped: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    """Compute Normalized Cross-Correlation in the overlapping region.

    Parameters
    ----------
    reference : np.ndarray
        Reference image.
    warped : np.ndarray
        Warped image aligned to reference.
    mask : np.ndarray | None, default None
        Optional binary mask (pixels > 0 are evaluated).

    Returns
    -------
    float
        Normalized cross-correlation score in [-1.0, 1.0].
    """
    if mask is not None:
        idx = mask > 0
    else:
        idx = (reference != 0) & (warped != 0)

    if not np.any(idx):
        return 0.0

    r = reference[idx].astype(np.float64).ravel()
    w = warped[idx].astype(np.float64).ravel()

    r_std = float(np.std(r))
    w_std = float(np.std(w))

    if r_std < 1e-9 or w_std < 1e-9:
        return 0.0

    r_mean = float(np.mean(r))
    w_mean = float(np.mean(w))

    covar = float(np.mean((r - r_mean) * (w - w_mean)))
    ncc = covar / (r_std * w_std)

    return float(np.clip(ncc, -1.0, 1.0))


def compute_ssim_score(
    reference: np.ndarray,
    warped: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    """Compute simplified Structural Similarity Index (SSIM) in overlap region.

    Parameters
    ----------
    reference : np.ndarray
        Reference image.
    warped : np.ndarray
        Warped image aligned to reference.
    mask : np.ndarray | None, default None
        Optional binary mask (pixels > 0 evaluated).

    Returns
    -------
    float
        SSIM score in [-1.0, 1.0].
    """
    if mask is not None:
        idx = mask > 0
    else:
        idx = (reference != 0) & (warped != 0)

    if not np.any(idx):
        return 0.0

    r = reference[idx].astype(np.float64).ravel()
    w = warped[idx].astype(np.float64).ravel()

    # Scale values to [0, 255] if normalized in [0, 1]
    if r.max() <= 1.0 and w.max() <= 1.0:
        r = r * 255.0
        w = w * 255.0

    C1 = (0.01 * 255.0) ** 2
    C2 = (0.03 * 255.0) ** 2

    mu1 = float(np.mean(r))
    mu2 = float(np.mean(w))
    sigma1_sq = float(np.var(r))
    sigma2_sq = float(np.var(w))
    sigma12 = float(np.mean((r - mu1) * (w - mu2)))

    numerator = (2.0 * mu1 * mu2 + C1) * (2.0 * sigma12 + C2)
    denominator = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)

    if abs(denominator) < 1e-12:
        return 0.0

    ssim = numerator / denominator
    return float(np.clip(ssim, -1.0, 1.0))


def generate_evaluation_report(
    rmse: float,
    inlier_stats: dict[str, Any],
    ncc: float,
    ssim: float,
    transform_decomposition: dict[str, Any],
) -> dict[str, Any]:
    """Generate comprehensive registration quality report.

    Parameters
    ----------
    rmse : float
        Root Mean Square Error of feature correspondences.
    inlier_stats : dict
        Inlier count and ratio statistics.
    ncc : float
        Normalized Cross-Correlation score.
    ssim : float
        Structural Similarity Index score.
    transform_decomposition : dict
        Decomposed transformation parameters.

    Returns
    -------
    dict[str, Any]
        Aggregated report dictionary including 'quality_grade'.
    """
    # Quality grade logic:
    # "excellent" if RMSE < 1.0 and NCC > 0.9
    # "good" if RMSE < 2.0 and NCC > 0.7
    # "fair" if RMSE < 5.0
    # else "poor"
    if rmse < 1.0 and ncc > 0.9:
        grade = "excellent"
    elif rmse < 2.0 and ncc > 0.7:
        grade = "good"
    elif rmse < 5.0:
        grade = "fair"
    else:
        grade = "poor"

    return {
        "rmse": float(rmse),
        "inlier_stats": inlier_stats,
        "ncc": float(ncc),
        "ssim": float(ssim),
        "transform_decomposition": transform_decomposition,
        "quality_grade": grade,
    }

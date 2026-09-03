"""
SIH26166 — Data models for geometric verification and transformation estimation.

Typed structures for geometric estimation results, error metrics, and
estimator capability detection.

Coordinate convention
---------------------
All point coordinates use **(x, y)** with origin at the **top-left**
corner of the image.

- x = column index (increases rightward)
- y = row index (increases downward)

This matches OpenCV's convention and the feature/matching layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class TransformModel(str, Enum):
    """Supported transformation model types.

    Geometric assumptions
    ---------------------
    AFFINE:
        6 degrees of freedom.  Preserves parallelism.
        Requires ≥ 3 non-collinear point correspondences.
        Appropriate when both images are near-nadir views of approximately
        planar terrain at similar viewing angles.

    HOMOGRAPHY:
        8 degrees of freedom (projective).  Preserves collinearity only.
        Requires ≥ 4 non-collinear point correspondences.
        Appropriate when viewpoint variation or non-planar terrain
        introduce projective distortion.

    For lunar orbital imagery (OHRC, TMC-2, IIRS), affine is typically
    the first choice because nadir-pointing cameras view approximately
    planar terrain.  However, terrain relief and oblique views can
    introduce perspective effects not captured by affine.  This
    limitation is explicit — a single affine transform may not perfectly
    represent all geometric distortions present in the data.
    """

    AFFINE = "affine"
    HOMOGRAPHY = "homography"


class EstimatorMethod(str, Enum):
    """Robust estimation method actually used.

    This records what was *actually* used at runtime, not what was
    requested.  If MAGSAC++ was unavailable and a fallback was used,
    this field will reflect the fallback.
    """

    MAGSAC = "USAC_MAGSAC"
    RANSAC = "RANSAC"
    LMEDS = "LMEDS"
    USAC_DEFAULT = "USAC_DEFAULT"


@dataclass
class EstimatorCapability:
    """Records the OpenCV estimator capabilities detected at runtime.

    Attributes
    ----------
    opencv_version : str
        OpenCV version string (e.g. '4.14.0').
    has_usac_magsac : bool
        Whether ``cv2.USAC_MAGSAC`` is available.
    has_estimate_affine2d : bool
        Whether ``cv2.estimateAffine2D`` is available.
    has_find_homography : bool
        Whether ``cv2.findHomography`` is available.
    available_usac_flags : list[str]
        All USAC-related flags found in the cv2 module.
    """

    opencv_version: str
    has_usac_magsac: bool
    has_estimate_affine2d: bool
    has_find_homography: bool
    available_usac_flags: list[str] = field(default_factory=list)


@dataclass
class ErrorMetrics:
    """Reprojection error metrics for a geometric estimation.

    All errors are in **pixels** (Euclidean distance).

    RMSE definition
    ---------------
    For N correspondences with per-match errors e_i:

        RMSE = sqrt( sum(e_i^2) / N )

    Two RMSE values are provided:
    - ``inlier_rmse``: computed over inlier correspondences only.
      This is the **primary** geometric accuracy metric.
    - ``all_rmse``: computed over all candidate correspondences
      (inliers + outliers).  Provided for completeness.

    Attributes
    ----------
    per_match_errors : np.ndarray
        Euclidean reprojection error for each correspondence.
        Length equals the number of input correspondences.
        Entries for outliers may have larger values.
    inlier_rmse : float
        Root mean square error over inlier correspondences.
    all_rmse : float
        Root mean square error over all correspondences.
    inlier_median_error : float
        Median error over inlier correspondences.
    inlier_max_error : float
        Maximum error over inlier correspondences.
    all_median_error : float
        Median error over all correspondences.
    """

    per_match_errors: np.ndarray
    inlier_rmse: float
    all_rmse: float
    inlier_median_error: float
    inlier_max_error: float
    all_median_error: float


@dataclass
class GeometricResult:
    """Complete result of geometric verification and transformation estimation.

    Attributes
    ----------
    success : bool
        Whether estimation succeeded.
    transform_matrix : np.ndarray | None
        Estimated transformation matrix.
        - Affine: shape (2, 3)
        - Homography: shape (3, 3)
        ``None`` if estimation failed.
    transform_model : TransformModel
        The transformation model that was used / requested.
    estimator_method : EstimatorMethod | None
        The robust estimator that was actually used.
        ``None`` if estimation was not attempted.
    inlier_mask : np.ndarray | None
        Boolean array of length N (number of input correspondences).
        ``True`` for inliers, ``False`` for outliers.
        ``None`` if estimation failed.
    inlier_count : int
        Number of geometrically consistent correspondences.
    outlier_count : int
        Number of geometrically inconsistent correspondences.
    total_correspondences : int
        Total number of input correspondences (inlier + outlier).
    inlier_ratio : float
        ``inlier_count / total_correspondences``.  0.0 if no correspondences.
    error_metrics : ErrorMetrics | None
        Reprojection error metrics.  ``None`` if estimation failed.
    failure_reason : str
        Human-readable explanation if estimation failed.  Empty string
        on success.
    estimator_config : dict[str, Any]
        Configuration parameters used for estimation (threshold,
        confidence, etc.).
    capability : EstimatorCapability | None
        Detected estimator capabilities at runtime.
    """

    success: bool
    transform_matrix: np.ndarray | None
    transform_model: TransformModel
    estimator_method: EstimatorMethod | None
    inlier_mask: np.ndarray | None
    inlier_count: int
    outlier_count: int
    total_correspondences: int
    inlier_ratio: float
    error_metrics: ErrorMetrics | None
    failure_reason: str
    estimator_config: dict[str, Any] = field(default_factory=dict)
    capability: EstimatorCapability | None = None

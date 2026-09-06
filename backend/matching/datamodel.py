"""
SIH26166 — Matching Data Models.

Defines data structures for feature matching results and geometric verification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=False)
class MatchResult:
    """Container for pair matching output and registration diagnostics.

    Attributes
    ----------
    keypoints1 : list
        Detected keypoints in image 1 (e.g. list of cv2.KeyPoint or coordinates).
    keypoints2 : list
        Detected keypoints in image 2.
    raw_matches : int
        Count of raw nearest-neighbor matches before ratio filtering.
    good_matches : int
        Count of matches passing ratio test / mutual consistency.
    inlier_matches : int
        Count of geometric inliers after MAGSAC++ / affine RANSAC.
    inlier_ratio : float
        Ratio of inlier matches to good matches (inlier_matches / max(1, good_matches)).
    transform_matrix : np.ndarray | None
        Estimated transformation matrix (3x3 homography or 2x3 affine).
    transform_type : str
        Transformation model ("homography" or "affine").
    match_points1 : np.ndarray
        (N, 2) float32 coordinates (x, y) of inlier keypoints in image 1.
    match_points2 : np.ndarray
        (N, 2) float32 coordinates (x, y) of inlier keypoints in image 2.
    spatial_distribution : float
        Shannon entropy of inlier point distribution across an 8x8 spatial grid.
    metadata : dict[str, Any]
        Additional diagnostic metadata.
    """

    keypoints1: list
    keypoints2: list
    raw_matches: int
    good_matches: int
    inlier_matches: int
    inlier_ratio: float
    transform_matrix: np.ndarray | None
    transform_type: str
    match_points1: np.ndarray
    match_points2: np.ndarray
    spatial_distribution: float
    metadata: dict[str, Any] = field(default_factory=dict)

"""
SIH26166 — Registration Data Models (Stage 4).

Defines data containers for end-to-end registration results, transformation matrices,
and evaluation metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from backend.matching.datamodel import MatchResult


@dataclass(frozen=False)
class RegistrationResult:
    """Complete output of the multi-modal lunar image registration pipeline.

    Attributes
    ----------
    warped_image : np.ndarray
        Source image warped and aligned onto the reference coordinate frame (float32).
    transform_matrix : np.ndarray
        Final estimated and refined transformation matrix (3x3 homography or 2x3 affine).
    transform_type : str
        Model type ("homography" or "affine").
    match_result : MatchResult
        Feature extraction and keypoint matching result from Stage 3.
    evaluation : dict[str, Any]
        Quantitative metrics (RMSE, NCC, SSIM, inlier ratio, quality grade).
    overlap_mask : np.ndarray
        Binary mask (uint8 0 or 255) of valid overlapping pixels between warped and reference.
    checkerboard : np.ndarray
        Visual quality assurance checkerboard composite image (float32).
    """

    warped_image: np.ndarray
    transform_matrix: np.ndarray
    transform_type: str
    match_result: MatchResult
    evaluation: dict[str, Any] = field(default_factory=dict)
    overlap_mask: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.uint8))
    checkerboard: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))

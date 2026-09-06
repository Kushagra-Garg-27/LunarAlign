"""End-to-end registration pipeline orchestrator."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from backend.matching.datamodel import MatchResult
from backend.registration.datamodel import RegistrationResult
from backend.registration.metrics import (
    compute_inlier_stats,
    compute_ncc_score,
    compute_rmse,
    compute_ssim_score,
    generate_evaluation_report,
)
from backend.registration.subpixel import refine_subpixel
from backend.registration.transform import decompose_transform, estimate_transform
from backend.registration.warper import (
    compute_overlap_mask,
    create_checkerboard_overlay,
    warp_image,
)

logger = logging.getLogger(__name__)


def register_pair(
    reference: np.ndarray,
    target: np.ndarray,
    match_result: MatchResult,
    model: str = "auto",
    enable_subpixel: bool = True,
    interpolation: str = "bicubic",
) -> RegistrationResult:
    """Execute end-to-end transformation estimation, warping, and evaluation.

    Parameters
    ----------
    reference : np.ndarray
        Reference image array (defines destination coordinate space).
    target : np.ndarray
        Target image array to be aligned and warped.
    match_result : MatchResult
        Pre-computed keypoint correspondences and inliers from Stage 3.
    model : str, default "auto"
        Geometric model: "auto", "affine", or "homography".
    enable_subpixel : bool, default True
        Whether to perform sub-pixel template refinement on the warped target.
    interpolation : str, default "bicubic"
        Interpolation method: "nearest", "bilinear", or "bicubic".

    Returns
    -------
    RegistrationResult
        Registered warped image, transform matrix, evaluation report, and QA overlays.
    """
    try:
        # Step 1: Transformation estimation
        transform_matrix, transform_type = estimate_transform(match_result, model=model)

        # Step 2: Sub-pixel refinement (if enabled)
        ref_shape = reference.shape[:2]
        if enable_subpixel:
            initial_warped = warp_image(target, transform_matrix, ref_shape)
            final_transform = refine_subpixel(reference, initial_warped, transform_matrix)
        else:
            final_transform = transform_matrix

        # Step 3: Warp target onto reference coordinate frame
        warped = warp_image(target, final_transform, ref_shape, interpolation=interpolation)

        # Step 4: Compute binary overlap mask
        overlap_mask = compute_overlap_mask(reference, warped)

        # Step 5: Create checkerboard visual inspection overlay
        checkerboard = create_checkerboard_overlay(reference, warped)

        # Step 6: Compute quality and error metrics
        pts_src = match_result.match_points1
        pts_dst = match_result.match_points2
        rmse = compute_rmse(pts_src, pts_dst, final_transform)

        inlier_mask = np.ones(match_result.inlier_matches, dtype=np.uint8)
        inlier_stats = compute_inlier_stats(inlier_mask, match_result.good_matches)

        ncc = compute_ncc_score(reference, warped, mask=overlap_mask)
        ssim = compute_ssim_score(reference, warped, mask=overlap_mask)
        transform_decomp = decompose_transform(final_transform)

        evaluation = generate_evaluation_report(
            rmse=rmse,
            inlier_stats=inlier_stats,
            ncc=ncc,
            ssim=ssim,
            transform_decomposition=transform_decomp,
        )

        logger.info(
            "Registration complete (%s): RMSE=%.3f px, NCC=%.3f, SSIM=%.3f, Grade=%s",
            transform_type,
            rmse,
            ncc,
            ssim,
            evaluation.get("quality_grade"),
        )

        # Step 7: Build and return RegistrationResult
        return RegistrationResult(
            warped_image=warped,
            transform_matrix=final_transform,
            transform_type=transform_type,
            match_result=match_result,
            evaluation=evaluation,
            overlap_mask=overlap_mask,
            checkerboard=checkerboard,
        )

    except Exception as exc:
        logger.error("Registration pipeline failed: %s", exc, exc_info=True)
        raise

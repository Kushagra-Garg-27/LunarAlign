"""
SIH26166 — Classical registration pipeline orchestration service.

Thin orchestrator that composes existing modules into the complete
classical same-modality registration pipeline:

    load images
    → preprocess (grayscale conversion)
    → SIFT feature extraction
    → FLANN kNN matching
    → Lowe ratio test
    → MAGSAC++ geometric estimation
    → image warping (registration)
    → evaluation metrics + spatial distribution
    → diagnostic visualizations

This module contains NO algorithmic logic.  Every processing step
delegates to the corresponding backend module.

Pipeline mode: ``classical_sift``

Transform direction:
    - Source image: reference
    - Matrix maps: reference → target
    - Output: reference warped into target coordinate frame
    - No inversion is performed
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from backend.evaluation.metrics import MatchQualitySummary, build_quality_summary
from backend.evaluation.spatial import compute_spatial_distribution, SpatialDistribution
from backend.evaluation.visualization import (
    draw_filtered_matches,
    draw_inlier_outlier_matches,
    draw_registration_overlay,
)
from backend.features.sift import extract_sift
from backend.geometry.estimation import estimate_transform
from backend.geometry.models import GeometricResult, TransformModel
from backend.matching.flann import flann_knn_match
from backend.matching.ratio_test import apply_ratio_test
from backend.preprocessing.datamodel import FeatureImage, RawImage
from backend.preprocessing.grayscale import to_feature_image
from backend.preprocessing.io import load_image
from backend.registration.models import WarpConfig
from backend.registration.warping import warp_image

logger = logging.getLogger("sih26166.core.registration_service")

PIPELINE_MODE = "classical_sift"


# ===================================================================
# Result container
# ===================================================================

@dataclass
class RegistrationPipelineResult:
    """Complete result from the classical registration pipeline.

    Holds all artifacts and metrics needed by the API layer.
    """

    result_id: str
    success: bool
    pipeline_mode: str

    # Metrics
    quality_summary: MatchQualitySummary | None = None
    spatial: SpatialDistribution | None = None

    # Images (NumPy arrays, not saved to disk yet)
    registered_image: np.ndarray | None = None
    match_visualization: np.ndarray | None = None
    inlier_visualization: np.ndarray | None = None
    overlay_visualization: np.ndarray | None = None

    # Timing (seconds)
    timings: dict[str, float] = field(default_factory=dict)

    # Failure
    failure_reason: str = ""
    failure_stage: str = ""


# ===================================================================
# Pipeline orchestrator
# ===================================================================

def run_classical_registration(
    ref_path: str | Path,
    tgt_path: str | Path,
    *,
    transform_model: str = "affine",
    ratio_threshold: float = 0.75,
    reproj_threshold: float = 3.0,
    confidence: float = 0.999,
) -> RegistrationPipelineResult:
    """Execute the full classical registration pipeline.

    Parameters
    ----------
    ref_path : str | Path
        Path to the reference image on disk.
    tgt_path : str | Path
        Path to the target image on disk.
    transform_model : str
        ``'affine'`` or ``'homography'``.
    ratio_threshold : float
        Lowe ratio test threshold (0–1).
    reproj_threshold : float
        MAGSAC++ reprojection threshold in pixels.
    confidence : float
        Estimation confidence (0–1).

    Returns
    -------
    RegistrationPipelineResult
        Complete pipeline result with metrics and image artifacts.
    """
    result_id = uuid.uuid4().hex[:16]
    timings: dict[str, float] = {}
    total_start = time.perf_counter()

    # --- Validate transform model ---
    try:
        model = TransformModel(transform_model)
    except ValueError:
        return RegistrationPipelineResult(
            result_id=result_id,
            success=False,
            pipeline_mode=PIPELINE_MODE,
            failure_reason=f"Unsupported transform model: '{transform_model}'. Use 'affine' or 'homography'.",
            failure_stage="configuration",
        )

    # --- 1. Load images ---
    t0 = time.perf_counter()
    try:
        ref_raw = load_image(ref_path)
        tgt_raw = load_image(tgt_path)
    except Exception as exc:
        return RegistrationPipelineResult(
            result_id=result_id,
            success=False,
            pipeline_mode=PIPELINE_MODE,
            failure_reason=f"Failed to load images: {exc}",
            failure_stage="load",
        )
    timings["load"] = time.perf_counter() - t0

    # --- 2. Preprocess (grayscale conversion) ---
    t0 = time.perf_counter()
    try:
        ref_feat = to_feature_image(ref_raw)
        tgt_feat = to_feature_image(tgt_raw)
    except Exception as exc:
        return RegistrationPipelineResult(
            result_id=result_id,
            success=False,
            pipeline_mode=PIPELINE_MODE,
            failure_reason=f"Preprocessing failed: {exc}",
            failure_stage="preprocess",
        )
    timings["preprocess"] = time.perf_counter() - t0

    # --- 3. SIFT feature extraction ---
    t0 = time.perf_counter()
    try:
        ref_sift = extract_sift(ref_feat)
        tgt_sift = extract_sift(tgt_feat)
    except Exception as exc:
        return RegistrationPipelineResult(
            result_id=result_id,
            success=False,
            pipeline_mode=PIPELINE_MODE,
            failure_reason=f"SIFT extraction failed: {exc}",
            failure_stage="sift",
        )
    timings["sift"] = time.perf_counter() - t0

    if ref_sift.num_keypoints == 0 or tgt_sift.num_keypoints == 0:
        return RegistrationPipelineResult(
            result_id=result_id,
            success=False,
            pipeline_mode=PIPELINE_MODE,
            failure_reason=(
                f"Insufficient SIFT features: reference={ref_sift.num_keypoints}, "
                f"target={tgt_sift.num_keypoints}. Need ≥1 keypoint in each image."
            ),
            failure_stage="sift",
            timings=timings,
        )

    # --- 4. FLANN matching ---
    t0 = time.perf_counter()
    try:
        raw_matches = flann_knn_match(ref_sift, tgt_sift)
    except Exception as exc:
        return RegistrationPipelineResult(
            result_id=result_id,
            success=False,
            pipeline_mode=PIPELINE_MODE,
            failure_reason=f"FLANN matching failed: {exc}",
            failure_stage="matching",
            timings=timings,
        )
    timings["matching"] = time.perf_counter() - t0

    # --- 5. Lowe ratio test ---
    t0 = time.perf_counter()
    try:
        match_result = apply_ratio_test(
            raw_matches, ref_sift, tgt_sift,
            ratio_threshold=ratio_threshold,
        )
    except Exception as exc:
        return RegistrationPipelineResult(
            result_id=result_id,
            success=False,
            pipeline_mode=PIPELINE_MODE,
            failure_reason=f"Ratio test failed: {exc}",
            failure_stage="ratio_test",
            timings=timings,
        )
    timings["ratio_test"] = time.perf_counter() - t0

    if match_result.accepted == 0:
        return RegistrationPipelineResult(
            result_id=result_id,
            success=False,
            pipeline_mode=PIPELINE_MODE,
            failure_reason=(
                f"No matches passed the Lowe ratio test "
                f"(threshold={ratio_threshold})."
            ),
            failure_stage="ratio_test",
            timings=timings,
        )

    # --- 6. Geometric estimation (MAGSAC++) ---
    t0 = time.perf_counter()
    try:
        geo_result = estimate_transform(
            match_result.matches,
            model=model,
            reproj_threshold=reproj_threshold,
            confidence=confidence,
        )
    except Exception as exc:
        return RegistrationPipelineResult(
            result_id=result_id,
            success=False,
            pipeline_mode=PIPELINE_MODE,
            failure_reason=f"Geometric estimation failed: {exc}",
            failure_stage="geometry",
            timings=timings,
        )
    timings["geometry"] = time.perf_counter() - t0

    if not geo_result.success:
        return RegistrationPipelineResult(
            result_id=result_id,
            success=False,
            pipeline_mode=PIPELINE_MODE,
            failure_reason=f"Geometric estimation failed: {geo_result.failure_reason}",
            failure_stage="geometry",
            timings=timings,
        )

    # --- 7. Image warping ---
    t0 = time.perf_counter()
    try:
        reg_result = warp_image(
            ref_raw.data,
            geo_result.transform_matrix,
            geo_result.transform_model,
            target_width=tgt_raw.width,
            target_height=tgt_raw.height,
        )
    except Exception as exc:
        return RegistrationPipelineResult(
            result_id=result_id,
            success=False,
            pipeline_mode=PIPELINE_MODE,
            failure_reason=f"Warp failed: {exc}",
            failure_stage="warp",
            timings=timings,
        )
    timings["warp"] = time.perf_counter() - t0

    if not reg_result.success:
        return RegistrationPipelineResult(
            result_id=result_id,
            success=False,
            pipeline_mode=PIPELINE_MODE,
            failure_reason=f"Warp failed: {reg_result.failure_reason}",
            failure_stage="warp",
            timings=timings,
        )

    # --- 8. Evaluation ---
    t0 = time.perf_counter()

    # Spatial distribution
    spatial = compute_spatial_distribution(
        match_result.matches,
        geo_result.inlier_mask,
        tgt_raw.width,
        tgt_raw.height,
    )

    # Quality summary
    summary = build_quality_summary(
        geo_result,
        spatial_entropy=spatial.normalized_entropy,
    )

    timings["evaluation"] = time.perf_counter() - t0

    # --- 9. Visualizations ---
    t0 = time.perf_counter()

    # Prepare display images (uint8)
    ref_display = _to_display(ref_raw.data)
    tgt_display = _to_display(tgt_raw.data)

    match_vis = draw_filtered_matches(ref_display, tgt_display, match_result.matches)
    inlier_vis = draw_inlier_outlier_matches(
        ref_display, tgt_display, match_result.matches, geo_result.inlier_mask,
    )
    overlay_vis = draw_registration_overlay(
        tgt_display, _to_display(reg_result.registered_image),
    )

    timings["visualization"] = time.perf_counter() - t0

    # --- Total ---
    timings["total"] = time.perf_counter() - total_start

    logger.info(
        "Registration complete: id=%s, inliers=%d/%d, RMSE=%.3f px, total=%.3f s",
        result_id,
        summary.inlier_count,
        summary.total_correspondences,
        summary.inlier_rmse,
        timings["total"],
    )

    return RegistrationPipelineResult(
        result_id=result_id,
        success=True,
        pipeline_mode=PIPELINE_MODE,
        quality_summary=summary,
        spatial=spatial,
        registered_image=reg_result.registered_image,
        match_visualization=match_vis,
        inlier_visualization=inlier_vis,
        overlay_visualization=overlay_vis,
        timings=timings,
    )


def _to_display(img: np.ndarray) -> np.ndarray:
    """Convert image to uint8 for visualization."""
    if img is None:
        return np.zeros((1, 1), dtype=np.uint8)
    if img.dtype in (np.float32, np.float64):
        return np.clip(img, 0, 255).astype(np.uint8)
    if img.dtype == np.uint16:
        return (img / 256).astype(np.uint8)
    return img.astype(np.uint8) if img.dtype != np.uint8 else img

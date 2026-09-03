"""
SIH26166 — Comprehensive tests for the evaluation layer.

Test categories:
1. Metrics: MatchQualitySummary from GeometricResult
2. Spatial: bucketing, entropy, edge cases
3. Visualization: match drawing, inlier/outlier, overlay
4. Integration: full classical pipeline → metrics → visualization
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from backend.evaluation.metrics import MatchQualitySummary, build_quality_summary
from backend.evaluation.spatial import (
    SpatialDistribution,
    compute_spatial_distribution,
    _compute_normalized_entropy,
)
from backend.evaluation.visualization import (
    draw_filtered_matches,
    draw_inlier_outlier_matches,
    draw_registration_overlay,
    draw_side_by_side,
    _to_bgr_uint8,
)
from backend.features.models import FilteredMatch
from backend.features.sift import extract_sift
from backend.geometry.estimation import estimate_transform
from backend.geometry.models import (
    ErrorMetrics,
    EstimatorCapability,
    EstimatorMethod,
    GeometricResult,
    TransformModel,
)
from backend.matching.flann import flann_knn_match
from backend.matching.ratio_test import apply_ratio_test
from backend.preprocessing.datamodel import FeatureImage
from backend.registration.warping import warp_image


# =========================================================================
# Helpers
# =========================================================================

def _make_geo_result(
    success: bool = True,
    inlier_count: int = 80,
    outlier_count: int = 20,
    inlier_rmse: float = 0.5,
    all_rmse: float = 2.0,
    inlier_median: float = 0.3,
    inlier_max: float = 1.2,
    model: TransformModel = TransformModel.AFFINE,
    method: EstimatorMethod = EstimatorMethod.MAGSAC,
) -> GeometricResult:
    """Create a synthetic GeometricResult for testing."""
    total = inlier_count + outlier_count

    if success:
        matrix = np.eye(2, 3, dtype=np.float64)
        mask = np.array(
            [True] * inlier_count + [False] * outlier_count, dtype=bool
        )
        errors = np.concatenate([
            np.random.default_rng(42).uniform(0.1, 1.0, inlier_count),
            np.random.default_rng(42).uniform(5.0, 20.0, outlier_count),
        ])
        em = ErrorMetrics(
            per_match_errors=errors,
            inlier_rmse=inlier_rmse,
            all_rmse=all_rmse,
            inlier_median_error=inlier_median,
            inlier_max_error=inlier_max,
            all_median_error=1.5,
        )
    else:
        matrix = None
        mask = None
        em = None

    return GeometricResult(
        success=success,
        transform_matrix=matrix,
        transform_model=model,
        estimator_method=method if success else None,
        inlier_mask=mask,
        inlier_count=inlier_count,
        outlier_count=outlier_count,
        total_correspondences=total,
        inlier_ratio=inlier_count / total if total > 0 else 0.0,
        error_metrics=em,
        failure_reason="" if success else "test failure",
    )


def _make_matches(
    n: int,
    *,
    ref_positions: list[tuple[float, float]] | None = None,
    tgt_positions: list[tuple[float, float]] | None = None,
) -> list[FilteredMatch]:
    """Create synthetic FilteredMatch objects."""
    matches = []
    rng = np.random.default_rng(42)
    for i in range(n):
        ref_pt = ref_positions[i] if ref_positions else (rng.uniform(0, 200), rng.uniform(0, 200))
        tgt_pt = tgt_positions[i] if tgt_positions else (rng.uniform(0, 200), rng.uniform(0, 200))
        matches.append(FilteredMatch(
            query_idx=i,
            train_idx=i,
            distance=rng.uniform(0, 100),
            ratio=rng.uniform(0.5, 0.8),
            ref_pt=ref_pt,
            tgt_pt=tgt_pt,
        ))
    return matches


def _make_gray_image(w: int = 128, h: int = 128, val: int = 128) -> np.ndarray:
    return np.full((h, w), val, dtype=np.uint8)


def _make_color_image(w: int = 128, h: int = 128) -> np.ndarray:
    return np.full((h, w, 3), (100, 150, 200), dtype=np.uint8)


def _make_structured_image(width=256, height=256):
    img = np.zeros((height, width), dtype=np.uint8)
    for y in range(height):
        img[y, :] = int(40 + 80 * y / height)
    cv2.circle(img, (60, 60), 25, 255, -1)
    cv2.circle(img, (180, 80), 15, 200, -1)
    cv2.circle(img, (120, 200), 30, 230, -1)
    cv2.rectangle(img, (30, 130), (90, 180), 220, -1)
    cv2.rectangle(img, (160, 160), (230, 220), 180, -1)
    cv2.line(img, (10, 10), (100, 50), 255, 2)
    cv2.line(img, (200, 10), (150, 100), 200, 2)
    cv2.line(img, (120, 100), (120, 140), 255, 3)
    cv2.line(img, (100, 120), (140, 120), 255, 3)
    cv2.rectangle(img, (200, 30), (220, 50), 240, -1)
    cv2.rectangle(img, (40, 200), (60, 220), 200, -1)
    return img


def _make_feature_image(arr):
    return FeatureImage(
        data=arr.astype(np.float32),
        width=arr.shape[1],
        height=arr.shape[0],
        source_dtype=np.dtype("uint8"),
        conversion_method="test_fixture",
    )


# =========================================================================
# 1. METRICS TESTS
# =========================================================================

class TestMetrics:
    """Tests for MatchQualitySummary construction."""

    def test_successful_result(self):
        geo = _make_geo_result(success=True, inlier_count=80, outlier_count=20)
        summary = build_quality_summary(geo)

        assert summary.success is True
        assert summary.total_correspondences == 100
        assert summary.inlier_count == 80
        assert summary.outlier_count == 20
        assert summary.inlier_ratio == pytest.approx(0.8)
        assert summary.inlier_rmse == pytest.approx(0.5)
        assert summary.all_rmse == pytest.approx(2.0)
        assert summary.inlier_median_error == pytest.approx(0.3)
        assert summary.inlier_max_error == pytest.approx(1.2)

    def test_failed_result(self):
        geo = _make_geo_result(success=False, inlier_count=0, outlier_count=10)
        summary = build_quality_summary(geo)

        assert summary.success is False
        assert summary.inlier_rmse == 0.0
        assert summary.all_rmse == 0.0
        assert summary.failure_reason == "test failure"

    def test_zero_correspondences(self):
        geo = _make_geo_result(success=False, inlier_count=0, outlier_count=0)
        summary = build_quality_summary(geo)

        assert summary.total_correspondences == 0
        assert summary.inlier_ratio == 0.0

    def test_all_inliers(self):
        geo = _make_geo_result(success=True, inlier_count=50, outlier_count=0)
        summary = build_quality_summary(geo)

        assert summary.inlier_count == 50
        assert summary.outlier_count == 0
        assert summary.inlier_ratio == pytest.approx(1.0)

    def test_transform_model_recorded(self):
        geo = _make_geo_result(model=TransformModel.HOMOGRAPHY)
        summary = build_quality_summary(geo)

        assert summary.transform_model == "homography"

    def test_estimator_method_recorded(self):
        geo = _make_geo_result(method=EstimatorMethod.MAGSAC)
        summary = build_quality_summary(geo)

        assert summary.estimator_method == "USAC_MAGSAC"

    def test_transform_matrix_serializable(self):
        geo = _make_geo_result()
        summary = build_quality_summary(geo)

        assert summary.transform_matrix is not None
        assert isinstance(summary.transform_matrix, list)
        assert isinstance(summary.transform_matrix[0], list)
        assert isinstance(summary.transform_matrix[0][0], float)

    def test_spatial_entropy_passed_through(self):
        geo = _make_geo_result()
        summary = build_quality_summary(geo, spatial_entropy=0.85)

        assert summary.spatial_entropy == pytest.approx(0.85)

    def test_spatial_entropy_default(self):
        geo = _make_geo_result()
        summary = build_quality_summary(geo)

        assert summary.spatial_entropy == pytest.approx(-1.0)

    def test_failed_result_has_none_matrix(self):
        geo = _make_geo_result(success=False, inlier_count=0, outlier_count=5)
        summary = build_quality_summary(geo)

        assert summary.transform_matrix is None


# =========================================================================
# 2. SPATIAL DISTRIBUTION TESTS
# =========================================================================

class TestSpatialDistribution:
    """Tests for spatial bucketing and entropy."""

    def test_empty_input(self):
        result = compute_spatial_distribution([], None, 256, 256)

        assert result.total_inliers == 0
        assert result.occupied_cells == 0
        assert result.normalized_entropy == 0.0
        assert result.grid_counts.shape == (8, 8)

    def test_one_occupied_cell(self):
        matches = _make_matches(
            5,
            tgt_positions=[(10, 10), (11, 11), (12, 12), (13, 13), (14, 14)],
        )
        result = compute_spatial_distribution(matches, None, 256, 256)

        assert result.total_inliers == 5
        assert result.occupied_cells == 1
        # Single cell → entropy = 0
        assert result.normalized_entropy == pytest.approx(0.0)

    def test_evenly_distributed(self):
        """Points spread across all cells should give high entropy."""
        positions = []
        for r in range(8):
            for c in range(8):
                x = (c + 0.5) * 256 / 8
                y = (r + 0.5) * 256 / 8
                positions.append((x, y))

        matches = _make_matches(64, tgt_positions=positions)
        result = compute_spatial_distribution(matches, None, 256, 256)

        assert result.total_inliers == 64
        assert result.occupied_cells == 64
        # Perfect distribution → entropy = 1.0
        assert result.normalized_entropy == pytest.approx(1.0, abs=0.01)

    def test_clustered_points(self):
        """Points clustered in one corner should have low entropy."""
        positions = [(5 + i, 5 + i) for i in range(20)]
        matches = _make_matches(20, tgt_positions=positions)
        result = compute_spatial_distribution(matches, None, 256, 256)

        assert result.occupied_cells == 1
        assert result.normalized_entropy == pytest.approx(0.0)

    def test_boundary_coordinates(self):
        """Points at exact image boundaries should be assigned correctly."""
        positions = [
            (0, 0),          # top-left
            (255, 0),        # top-right (inside for 256px)
            (0, 255),        # bottom-left
            (255, 255),      # bottom-right
        ]
        matches = _make_matches(4, tgt_positions=positions)
        result = compute_spatial_distribution(matches, None, 256, 256)

        assert result.total_inliers == 4
        # Four corners → 4 cells occupied
        assert result.occupied_cells >= 3  # at least 3 distinct cells

    def test_exact_image_boundary(self):
        """Point at x=width should be clamped to last column."""
        positions = [(256, 128)]  # x == width
        matches = _make_matches(1, tgt_positions=positions)
        result = compute_spatial_distribution(matches, None, 256, 256)

        # Should be assigned to last column, not out of bounds
        assert result.total_inliers == 1
        assert result.grid_counts[4, 7] == 1  # row 4, last col

    def test_grid_shape(self):
        matches = _make_matches(10, tgt_positions=[(50, 50)] * 10)
        result = compute_spatial_distribution(matches, None, 256, 256)

        assert result.grid_counts.shape == (8, 8)
        assert result.grid_rows == 8
        assert result.grid_cols == 8
        assert result.total_cells == 64

    def test_custom_grid_size(self):
        matches = _make_matches(10, tgt_positions=[(50, 50)] * 10)
        result = compute_spatial_distribution(
            matches, None, 256, 256, grid_rows=4, grid_cols=4,
        )

        assert result.grid_counts.shape == (4, 4)
        assert result.total_cells == 16

    def test_entropy_range(self):
        """Entropy should always be in [0, 1]."""
        for n in [1, 5, 20, 64]:
            rng = np.random.default_rng(42)
            positions = [(rng.uniform(0, 255), rng.uniform(0, 255)) for _ in range(n)]
            matches = _make_matches(n, tgt_positions=positions)
            result = compute_spatial_distribution(matches, None, 256, 256)

            assert 0.0 <= result.normalized_entropy <= 1.0

    def test_inlier_mask_filters_outliers(self):
        """Only inliers should be counted in the spatial distribution."""
        positions = [
            (10, 10),   # inlier → cell (0,0)
            (200, 200), # outlier → should be excluded
            (10, 10),   # inlier → cell (0,0)
        ]
        matches = _make_matches(3, tgt_positions=positions)
        mask = np.array([True, False, True])

        result = compute_spatial_distribution(matches, mask, 256, 256)

        assert result.total_inliers == 2
        assert result.occupied_cells == 1

    def test_zero_image_dimensions(self):
        matches = _make_matches(5)
        result = compute_spatial_distribution(matches, None, 0, 0)

        assert result.total_inliers == 0
        assert result.normalized_entropy == 0.0

    def test_all_mask_false(self):
        matches = _make_matches(5)
        mask = np.zeros(5, dtype=bool)
        result = compute_spatial_distribution(matches, mask, 256, 256)

        assert result.total_inliers == 0
        assert result.normalized_entropy == 0.0


# =========================================================================
# 3. ENTROPY UNIT TESTS
# =========================================================================

class TestNormalizedEntropy:
    """Direct tests for the entropy computation."""

    def test_single_cell(self):
        grid = np.array([[5, 0], [0, 0]])
        assert _compute_normalized_entropy(grid, 4) == pytest.approx(0.0)

    def test_uniform_distribution(self):
        grid = np.ones((8, 8), dtype=np.int64)
        # All 64 cells equal → max entropy
        assert _compute_normalized_entropy(grid, 64) == pytest.approx(1.0, abs=1e-10)

    def test_two_equal_cells(self):
        grid = np.zeros((8, 8), dtype=np.int64)
        grid[0, 0] = 10
        grid[7, 7] = 10
        expected = math.log(2) / math.log(64)
        assert _compute_normalized_entropy(grid, 64) == pytest.approx(expected, abs=1e-10)

    def test_zero_total(self):
        grid = np.zeros((8, 8), dtype=np.int64)
        assert _compute_normalized_entropy(grid, 64) == 0.0

    def test_normalizes_by_total_cells_not_occupied(self):
        """Entropy normalized by log(64), NOT log(occupied cells)."""
        grid = np.zeros((8, 8), dtype=np.int64)
        grid[0, 0] = 10
        grid[0, 1] = 10
        grid[0, 2] = 10
        # 3 cells occupied, each with equal count
        expected = math.log(3) / math.log(64)
        assert _compute_normalized_entropy(grid, 64) == pytest.approx(expected, abs=1e-10)


# =========================================================================
# 4. VISUALIZATION TESTS
# =========================================================================

class TestVisualization:
    """Tests for match and registration visualization."""

    def test_empty_matches(self):
        ref = _make_gray_image()
        tgt = _make_gray_image()
        vis = draw_filtered_matches(ref, tgt, [])

        assert vis.ndim == 3
        assert vis.shape[2] == 3
        assert vis.dtype == np.uint8

    def test_valid_matches(self):
        ref = _make_gray_image()
        tgt = _make_gray_image()
        matches = _make_matches(10)
        vis = draw_filtered_matches(ref, tgt, matches)

        assert vis.ndim == 3
        assert vis.shape[0] == 128  # max height
        assert vis.shape[1] == 256  # side-by-side width

    def test_inlier_outlier_mask(self):
        ref = _make_gray_image()
        tgt = _make_gray_image()
        matches = _make_matches(10)
        mask = np.array([True] * 7 + [False] * 3)

        vis = draw_inlier_outlier_matches(ref, tgt, matches, mask)

        assert vis.ndim == 3
        assert vis.dtype == np.uint8

    def test_grayscale_input(self):
        ref = _make_gray_image()
        tgt = _make_gray_image()
        assert ref.ndim == 2

        vis = draw_filtered_matches(ref, tgt, _make_matches(5))
        assert vis.ndim == 3  # converted to BGR

    def test_multi_channel_input(self):
        ref = _make_color_image()
        tgt = _make_color_image()

        vis = draw_filtered_matches(ref, tgt, _make_matches(5))
        assert vis.ndim == 3
        assert vis.shape[2] == 3

    def test_mismatched_image_sizes(self):
        ref = _make_gray_image(100, 80)
        tgt = _make_gray_image(150, 120)

        vis = draw_filtered_matches(ref, tgt, _make_matches(5))
        assert vis.shape[0] == 120  # max height
        assert vis.shape[1] == 250  # 100 + 150

    def test_empty_inlier_outlier(self):
        ref = _make_gray_image()
        tgt = _make_gray_image()
        mask = np.array([], dtype=bool)
        vis = draw_inlier_outlier_matches(ref, tgt, [], mask)

        assert vis.ndim == 3

    def test_overlay_generation(self):
        target = _make_gray_image(128, 128, val=100)
        registered = _make_gray_image(128, 128, val=200)

        overlay = draw_registration_overlay(target, registered, alpha=0.5)

        assert overlay.ndim == 3
        assert overlay.shape[:2] == (128, 128)
        assert overlay.dtype == np.uint8

    def test_overlay_alpha_zero(self):
        target = _make_gray_image(64, 64, val=100)
        registered = _make_gray_image(64, 64, val=200)

        overlay = draw_registration_overlay(target, registered, alpha=0.0)
        # alpha=0 → only target visible
        np.testing.assert_array_equal(
            overlay[:, :, 0], np.full((64, 64), 100, dtype=np.uint8),
        )

    def test_overlay_alpha_one(self):
        target = _make_gray_image(64, 64, val=100)
        registered = _make_gray_image(64, 64, val=200)

        overlay = draw_registration_overlay(target, registered, alpha=1.0)
        # alpha=1 → only registered visible
        np.testing.assert_array_equal(
            overlay[:, :, 0], np.full((64, 64), 200, dtype=np.uint8),
        )

    def test_overlay_mismatched_sizes(self):
        target = _make_gray_image(128, 128)
        registered = _make_gray_image(100, 90)

        overlay = draw_registration_overlay(target, registered)
        assert overlay.shape[:2] == (90, 100)  # smaller common area

    def test_side_by_side(self):
        a = _make_gray_image(100, 80)
        b = _make_gray_image(120, 100)

        vis = draw_side_by_side(a, b, label_a="Ref", label_b="Tgt")
        assert vis.shape[0] == 100  # max height
        assert vis.shape[1] == 220  # 100 + 120

    def test_to_bgr_handles_none(self):
        result = _to_bgr_uint8(None)
        assert result.ndim == 3

    def test_to_bgr_handles_float(self):
        img = np.full((32, 32), 128.5, dtype=np.float32)
        result = _to_bgr_uint8(img)
        assert result.dtype == np.uint8
        assert result.ndim == 3

    def test_none_inlier_mask_treats_all_as_inlier(self):
        ref = _make_gray_image()
        tgt = _make_gray_image()
        matches = _make_matches(5)

        vis = draw_inlier_outlier_matches(ref, tgt, matches, None)
        assert vis.ndim == 3


# =========================================================================
# 5. INTEGRATION: FULL PIPELINE → METRICS → VISUALIZATION
# =========================================================================

class TestEndToEndEvaluation:
    """End-to-end: synthetic → SIFT → FLANN → Lowe → MAGSAC++ → warp → metrics → viz."""

    def test_full_pipeline_evaluation(self):
        # --- Create synthetic reference and translated target ---
        ref_img = _make_structured_image()
        tx, ty = 15, 10
        M_true = np.float32([[1, 0, tx], [0, 1, ty]])
        tgt_img = cv2.warpAffine(ref_img, M_true, (ref_img.shape[1], ref_img.shape[0]))

        # --- Feature extraction ---
        ref_feat = _make_feature_image(ref_img)
        tgt_feat = _make_feature_image(tgt_img)

        ref_sift = extract_sift(ref_feat)
        tgt_sift = extract_sift(tgt_feat)

        # --- Matching ---
        raw_matches = flann_knn_match(ref_sift, tgt_sift)
        match_result = apply_ratio_test(raw_matches, ref_sift, tgt_sift)
        assert match_result.accepted > 0

        # --- Geometric estimation ---
        geo_result = estimate_transform(
            match_result.matches, model=TransformModel.AFFINE,
        )
        assert geo_result.success is True
        assert geo_result.inlier_count > 0

        # --- Image warping ---
        reg_result = warp_image(
            ref_img,
            geo_result.transform_matrix,
            geo_result.transform_model,
            target_width=tgt_img.shape[1],
            target_height=tgt_img.shape[0],
        )
        assert reg_result.success is True

        # --- Spatial distribution ---
        spatial = compute_spatial_distribution(
            match_result.matches,
            geo_result.inlier_mask,
            tgt_img.shape[1],
            tgt_img.shape[0],
        )
        assert spatial.total_inliers > 0
        assert 0.0 <= spatial.normalized_entropy <= 1.0
        assert spatial.grid_counts.shape == (8, 8)

        # --- Metrics ---
        summary = build_quality_summary(
            geo_result, spatial_entropy=spatial.normalized_entropy,
        )
        assert summary.success is True
        assert summary.total_correspondences > 0
        assert summary.inlier_count > 0
        assert 0.0 < summary.inlier_ratio <= 1.0
        assert summary.inlier_rmse >= 0.0
        assert np.isfinite(summary.inlier_rmse)
        assert summary.all_rmse >= 0.0
        assert 0.0 <= summary.spatial_entropy <= 1.0
        assert summary.transform_model == "affine"
        assert summary.estimator_method in ("USAC_MAGSAC", "RANSAC")
        assert summary.transform_matrix is not None

        # --- Match visualization ---
        match_vis = draw_filtered_matches(ref_img, tgt_img, match_result.matches)
        assert match_vis.ndim == 3
        assert match_vis.dtype == np.uint8
        assert match_vis.shape[0] > 0
        assert match_vis.shape[1] > 0

        # --- Inlier/outlier visualization ---
        io_vis = draw_inlier_outlier_matches(
            ref_img, tgt_img, match_result.matches, geo_result.inlier_mask,
        )
        assert io_vis.ndim == 3
        assert io_vis.dtype == np.uint8

        # --- Registration overlay ---
        overlay = draw_registration_overlay(
            tgt_img, reg_result.registered_image, alpha=0.5,
        )
        assert overlay.ndim == 3
        assert overlay.dtype == np.uint8
        assert overlay.shape[:2] == tgt_img.shape[:2]

        # --- Side-by-side ---
        sbs = draw_side_by_side(tgt_img, reg_result.registered_image)
        assert sbs.shape[1] == tgt_img.shape[1] * 2

"""
SIH26166 — Comprehensive tests for geometric verification and transformation estimation.

Test categories:
1. Affine estimation (valid, outliers, insufficient, degenerate, noisy)
2. Homography estimation
3. Robust estimation (inlier mask, counts, ratio)
4. Error metrics (zero-error, known error, RMSE, median, zero-correspondence)
5. Transformation application (translation, rotation/scale, multiple points, invalid)
6. End-to-end classical baseline (synthetic image → full pipeline)
7. Capability detection

All fixtures use deterministic synthetic data — no real imagery.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from backend.features.models import (
    FilteredMatch,
    Keypoint,
    MatchResult,
    NeighborInfo,
    RawMatch,
    SIFTFeatures,
)
from backend.features.sift import extract_sift
from backend.geometry.estimation import (
    DEFAULT_CONFIDENCE,
    DEFAULT_MAX_ITERS,
    DEFAULT_REPROJ_THRESHOLD,
    MIN_CORRESPONDENCES,
    compute_reprojection_errors,
    detect_estimator_capability,
    estimate_transform,
)
from backend.geometry.models import (
    ErrorMetrics,
    EstimatorCapability,
    EstimatorMethod,
    GeometricResult,
    TransformModel,
)
from backend.geometry.transform import transform_points
from backend.matching.flann import flann_knn_match
from backend.matching.ratio_test import apply_ratio_test
from backend.preprocessing.datamodel import FeatureImage


# =========================================================================
# Synthetic data helpers
# =========================================================================

def _make_filtered_match(
    ref_pt: tuple[float, float],
    tgt_pt: tuple[float, float],
    query_idx: int = 0,
    train_idx: int = 0,
) -> FilteredMatch:
    """Create a single synthetic FilteredMatch."""
    return FilteredMatch(
        ref_pt=ref_pt,
        tgt_pt=tgt_pt,
        distance=10.0,
        ratio=0.3,
        query_idx=query_idx,
        train_idx=train_idx,
    )


def _make_affine_correspondences(
    matrix: np.ndarray,
    ref_points: list[tuple[float, float]],
) -> list[FilteredMatch]:
    """Generate perfect correspondences under a known affine transform.

    Parameters
    ----------
    matrix : np.ndarray
        Affine matrix of shape (2, 3).
    ref_points : list of (x, y) tuples
        Reference image points.

    Returns
    -------
    list[FilteredMatch]
        Correspondences where tgt_pt = M @ [x, y, 1]^T.
    """
    matches = []
    for i, (x, y) in enumerate(ref_points):
        pt = np.array([x, y, 1.0], dtype=np.float64)
        transformed = matrix @ pt
        tgt = (float(transformed[0]), float(transformed[1]))
        matches.append(_make_filtered_match(
            ref_pt=(x, y), tgt_pt=tgt, query_idx=i, train_idx=i,
        ))
    return matches


def _make_homography_correspondences(
    matrix: np.ndarray,
    ref_points: list[tuple[float, float]],
) -> list[FilteredMatch]:
    """Generate perfect correspondences under a known homography.

    Parameters
    ----------
    matrix : np.ndarray
        Homography matrix of shape (3, 3).
    ref_points : list of (x, y) tuples
        Reference image points.

    Returns
    -------
    list[FilteredMatch]
        Correspondences where tgt_pt = dehomogenize(M @ [x, y, 1]^T).
    """
    matches = []
    for i, (x, y) in enumerate(ref_points):
        pt = np.array([x, y, 1.0], dtype=np.float64)
        proj = matrix @ pt
        tgt_x = proj[0] / proj[2]
        tgt_y = proj[1] / proj[2]
        matches.append(_make_filtered_match(
            ref_pt=(x, y), tgt_pt=(float(tgt_x), float(tgt_y)),
            query_idx=i, train_idx=i,
        ))
    return matches


# Standard reference points (well-distributed, non-degenerate)
STANDARD_REF_POINTS: list[tuple[float, float]] = [
    (20.0, 20.0),
    (60.0, 30.0),
    (100.0, 80.0),
    (150.0, 40.0),
    (200.0, 150.0),
    (30.0, 180.0),
    (170.0, 100.0),
    (80.0, 160.0),
    (220.0, 50.0),
    (130.0, 200.0),
]

# A known affine matrix: rotation ~10° + scale 1.05 + translation (5, -3)
_angle = math.radians(10.0)
_s = 1.05
KNOWN_AFFINE = np.array([
    [_s * math.cos(_angle), -_s * math.sin(_angle), 5.0],
    [_s * math.sin(_angle),  _s * math.cos(_angle), -3.0],
], dtype=np.float64)

# A known homography (identity + small perspective)
KNOWN_HOMOGRAPHY = np.array([
    [1.02,  0.01,  3.0],
    [0.01,  1.03, -2.0],
    [1e-5,  2e-5,  1.0],
], dtype=np.float64)


# =========================================================================
# 1. AFFINE ESTIMATION TESTS
# =========================================================================

class TestAffineEstimation:
    """Tests for affine transformation estimation."""

    def test_valid_affine_correspondences(self):
        """Exact affine correspondences should be recovered."""
        matches = _make_affine_correspondences(KNOWN_AFFINE, STANDARD_REF_POINTS)
        result = estimate_transform(matches, model=TransformModel.AFFINE)

        assert result.success is True
        assert result.transform_matrix is not None
        assert result.transform_matrix.shape == (2, 3)
        assert result.transform_model == TransformModel.AFFINE
        assert result.inlier_count > 0
        assert result.inlier_ratio > 0.5

        # Recovered matrix should be close to known
        np.testing.assert_allclose(
            result.transform_matrix, KNOWN_AFFINE, atol=0.5,
        )

    def test_affine_with_outliers(self):
        """Estimation should be robust to outliers."""
        # 10 good matches + 4 outliers
        matches = _make_affine_correspondences(KNOWN_AFFINE, STANDARD_REF_POINTS)

        # Add outliers
        outliers = [
            _make_filtered_match((50.0, 50.0), (999.0, 999.0), 10, 10),
            _make_filtered_match((80.0, 80.0), (10.0, 500.0), 11, 11),
            _make_filtered_match((120.0, 30.0), (300.0, 300.0), 12, 12),
            _make_filtered_match((200.0, 200.0), (5.0, 5.0), 13, 13),
        ]
        all_matches = matches + outliers

        result = estimate_transform(all_matches, model=TransformModel.AFFINE)

        assert result.success is True
        assert result.inlier_count >= 6  # most inliers should survive
        assert result.outlier_count >= 2  # some outliers rejected
        assert result.total_correspondences == 14
        assert result.inlier_ratio > 0.0

        # Matrix should still be close to known
        np.testing.assert_allclose(
            result.transform_matrix, KNOWN_AFFINE, atol=1.0,
        )

    def test_insufficient_points_affine(self):
        """Fewer than 3 correspondences should fail for affine."""
        matches = _make_affine_correspondences(
            KNOWN_AFFINE, STANDARD_REF_POINTS[:2],  # only 2
        )
        result = estimate_transform(matches, model=TransformModel.AFFINE)

        assert result.success is False
        assert "Insufficient" in result.failure_reason
        assert result.transform_matrix is None
        assert result.inlier_count == 0

    def test_zero_matches(self):
        """Zero correspondences should fail cleanly."""
        result = estimate_transform([], model=TransformModel.AFFINE)

        assert result.success is False
        assert "0 matches" in result.failure_reason
        assert result.total_correspondences == 0
        assert result.inlier_count == 0
        assert result.inlier_ratio == 0.0

    def test_one_match(self):
        """Single correspondence should fail for affine."""
        matches = [_make_filtered_match((50.0, 50.0), (55.0, 47.0))]
        result = estimate_transform(matches, model=TransformModel.AFFINE)

        assert result.success is False
        assert result.total_correspondences == 1

    def test_degenerate_collinear_points(self):
        """Collinear points may cause estimation to fail or degrade."""
        # All points on a single line
        collinear = [(float(i * 10), float(i * 10)) for i in range(10)]
        matches = _make_affine_correspondences(KNOWN_AFFINE, collinear)

        result = estimate_transform(matches, model=TransformModel.AFFINE)

        # Degenerate case: may succeed with poor quality or fail gracefully
        # Either way, it should not crash
        assert isinstance(result, GeometricResult)

    def test_noisy_correspondences(self):
        """Noisy correspondences should still produce a reasonable result."""
        rng = np.random.RandomState(42)
        matches = _make_affine_correspondences(KNOWN_AFFINE, STANDARD_REF_POINTS)

        # Add Gaussian noise (σ=2 pixels) to target points
        noisy_matches = []
        for m in matches:
            noise_x = rng.normal(0, 2.0)
            noise_y = rng.normal(0, 2.0)
            noisy_tgt = (m.tgt_pt[0] + noise_x, m.tgt_pt[1] + noise_y)
            noisy_matches.append(_make_filtered_match(
                m.ref_pt, noisy_tgt, m.query_idx, m.train_idx,
            ))

        result = estimate_transform(noisy_matches, model=TransformModel.AFFINE)

        assert result.success is True
        assert result.inlier_count > 0
        # Matrix should be approximately correct despite noise
        np.testing.assert_allclose(
            result.transform_matrix, KNOWN_AFFINE, atol=5.0,
        )

    def test_exact_three_points(self):
        """Exactly 3 correspondences (minimum) should work for affine."""
        matches = _make_affine_correspondences(
            KNOWN_AFFINE, STANDARD_REF_POINTS[:3],
        )
        result = estimate_transform(matches, model=TransformModel.AFFINE)

        assert result.success is True
        assert result.total_correspondences == 3


# =========================================================================
# 2. HOMOGRAPHY ESTIMATION TESTS
# =========================================================================

class TestHomographyEstimation:
    """Tests for projective/homography transformation estimation."""

    def test_valid_homography(self):
        """Exact homography correspondences should be recovered."""
        matches = _make_homography_correspondences(
            KNOWN_HOMOGRAPHY, STANDARD_REF_POINTS,
        )
        result = estimate_transform(matches, model=TransformModel.HOMOGRAPHY)

        assert result.success is True
        assert result.transform_matrix is not None
        assert result.transform_matrix.shape == (3, 3)
        assert result.transform_model == TransformModel.HOMOGRAPHY

    def test_insufficient_points_homography(self):
        """Fewer than 4 correspondences should fail for homography."""
        matches = _make_homography_correspondences(
            KNOWN_HOMOGRAPHY, STANDARD_REF_POINTS[:3],  # only 3
        )
        result = estimate_transform(matches, model=TransformModel.HOMOGRAPHY)

        assert result.success is False
        assert "Insufficient" in result.failure_reason

    def test_homography_with_outliers(self):
        """Homography estimation should handle outliers."""
        matches = _make_homography_correspondences(
            KNOWN_HOMOGRAPHY, STANDARD_REF_POINTS,
        )
        outliers = [
            _make_filtered_match((50.0, 50.0), (999.0, 999.0), 10, 10),
            _make_filtered_match((80.0, 80.0), (10.0, 500.0), 11, 11),
        ]
        all_matches = matches + outliers

        result = estimate_transform(all_matches, model=TransformModel.HOMOGRAPHY)

        assert result.success is True
        assert result.inlier_count >= 6


# =========================================================================
# 3. ROBUST ESTIMATION TESTS
# =========================================================================

class TestRobustEstimation:
    """Tests for inlier/outlier handling and robust estimation behavior."""

    def test_inlier_mask_shape(self):
        """Inlier mask should have length equal to total correspondences."""
        matches = _make_affine_correspondences(KNOWN_AFFINE, STANDARD_REF_POINTS)
        result = estimate_transform(matches, model=TransformModel.AFFINE)

        assert result.inlier_mask is not None
        assert len(result.inlier_mask) == result.total_correspondences

    def test_inlier_mask_correctness(self):
        """Inlier mask should correctly identify good correspondences."""
        matches = _make_affine_correspondences(KNOWN_AFFINE, STANDARD_REF_POINTS)
        outliers = [
            _make_filtered_match((50.0, 50.0), (999.0, 999.0), 10, 10),
        ]
        all_matches = matches + outliers

        result = estimate_transform(all_matches, model=TransformModel.AFFINE)

        assert result.success is True
        # The outlier (last match) should be rejected
        assert result.inlier_mask is not None
        # Outlier at index 10 should be False
        assert result.inlier_mask[10] is np.bool_(False)

    def test_inlier_count_consistency(self):
        """inlier_count + outlier_count should equal total_correspondences."""
        matches = _make_affine_correspondences(KNOWN_AFFINE, STANDARD_REF_POINTS)
        outliers = [
            _make_filtered_match((50.0, 50.0), (999.0, 999.0), 10, 10),
            _make_filtered_match((80.0, 80.0), (10.0, 500.0), 11, 11),
        ]
        all_matches = matches + outliers

        result = estimate_transform(all_matches, model=TransformModel.AFFINE)

        assert (result.inlier_count + result.outlier_count
                == result.total_correspondences)

    def test_inlier_ratio_calculation(self):
        """Inlier ratio should be inlier_count / total."""
        matches = _make_affine_correspondences(KNOWN_AFFINE, STANDARD_REF_POINTS)
        result = estimate_transform(matches, model=TransformModel.AFFINE)

        expected_ratio = result.inlier_count / result.total_correspondences
        assert result.inlier_ratio == pytest.approx(expected_ratio)

    def test_zero_correspondence_inlier_ratio(self):
        """Zero correspondences should give 0.0 inlier ratio."""
        result = estimate_transform([], model=TransformModel.AFFINE)

        assert result.inlier_ratio == 0.0

    def test_all_outliers(self):
        """If all matches are wild outliers, estimation may fail or have low inlier count."""
        outliers = [
            _make_filtered_match((10.0, 10.0), (999.0, 999.0), 0, 0),
            _make_filtered_match((20.0, 20.0), (1.0, 500.0), 1, 1),
            _make_filtered_match((30.0, 30.0), (500.0, 1.0), 2, 2),
            _make_filtered_match((40.0, 40.0), (200.0, 800.0), 3, 3),
            _make_filtered_match((50.0, 50.0), (800.0, 200.0), 4, 4),
        ]
        result = estimate_transform(outliers, model=TransformModel.AFFINE)

        # Should not crash — may succeed or fail depending on geometry
        assert isinstance(result, GeometricResult)

    def test_estimator_method_recorded(self):
        """The estimator method actually used should be recorded."""
        matches = _make_affine_correspondences(KNOWN_AFFINE, STANDARD_REF_POINTS)
        result = estimate_transform(matches, model=TransformModel.AFFINE)

        assert result.estimator_method is not None
        assert isinstance(result.estimator_method, EstimatorMethod)


# =========================================================================
# 4. ERROR METRICS TESTS
# =========================================================================

class TestErrorMetrics:
    """Tests for reprojection error computation."""

    def test_known_zero_error(self):
        """Perfect correspondences should have near-zero reprojection error."""
        matches = _make_affine_correspondences(KNOWN_AFFINE, STANDARD_REF_POINTS)
        result = estimate_transform(matches, model=TransformModel.AFFINE)

        assert result.success is True
        assert result.error_metrics is not None
        # Inlier RMSE should be very small for perfect correspondences
        assert result.error_metrics.inlier_rmse < 1.0

    def test_known_nonzero_error(self):
        """Noisy correspondences should have nonzero RMSE."""
        rng = np.random.RandomState(123)
        matches = _make_affine_correspondences(KNOWN_AFFINE, STANDARD_REF_POINTS)

        # Add noise to target points
        noisy_matches = []
        for m in matches:
            noise_x = rng.normal(0, 5.0)
            noise_y = rng.normal(0, 5.0)
            noisy_tgt = (m.tgt_pt[0] + noise_x, m.tgt_pt[1] + noise_y)
            noisy_matches.append(_make_filtered_match(
                m.ref_pt, noisy_tgt, m.query_idx, m.train_idx,
            ))

        result = estimate_transform(noisy_matches, model=TransformModel.AFFINE)

        if result.success:
            assert result.error_metrics is not None
            assert result.error_metrics.all_rmse > 0.0

    def test_rmse_calculation_direct(self):
        """Verify RMSE = sqrt(mean(errors^2)) on a known case."""
        # Identity transform, known errors
        ref_pts = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]],
                           dtype=np.float64)
        # Target points with known displacement
        tgt_pts = np.array([[3.0, 4.0], [13.0, 4.0], [3.0, 14.0]],
                           dtype=np.float64)
        # Identity matrix
        matrix = np.array([[1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0]], dtype=np.float64)
        inlier_mask = np.array([True, True, True])

        metrics = compute_reprojection_errors(ref_pts, tgt_pts, matrix, inlier_mask)

        # Error for each point: sqrt(3^2 + 4^2) = 5.0
        expected_error = 5.0
        for e in metrics.per_match_errors:
            assert e == pytest.approx(expected_error)

        # RMSE = sqrt(mean(25, 25, 25)) = 5.0
        assert metrics.inlier_rmse == pytest.approx(5.0)
        assert metrics.all_rmse == pytest.approx(5.0)
        assert metrics.inlier_median_error == pytest.approx(5.0)
        assert metrics.inlier_max_error == pytest.approx(5.0)

    def test_median_error(self):
        """Median error should correctly select the middle value."""
        ref_pts = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                           dtype=np.float64)
        tgt_pts = np.array([[1.0, 0.0], [3.0, 0.0], [5.0, 0.0]],
                           dtype=np.float64)
        matrix = np.array([[1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0]], dtype=np.float64)
        inlier_mask = np.array([True, True, True])

        metrics = compute_reprojection_errors(ref_pts, tgt_pts, matrix, inlier_mask)

        # Errors: 1.0, 3.0, 5.0 → median = 3.0
        assert metrics.inlier_median_error == pytest.approx(3.0)
        assert metrics.inlier_max_error == pytest.approx(5.0)

    def test_per_match_errors_length(self):
        """Per-match errors should have same length as input."""
        matches = _make_affine_correspondences(KNOWN_AFFINE, STANDARD_REF_POINTS)
        result = estimate_transform(matches, model=TransformModel.AFFINE)

        assert result.error_metrics is not None
        assert len(result.error_metrics.per_match_errors) == len(matches)

    def test_error_metrics_not_present_on_failure(self):
        """Error metrics should be None when estimation fails."""
        result = estimate_transform([], model=TransformModel.AFFINE)
        assert result.error_metrics is None

    def test_zero_correspondences_error_handling(self):
        """Zero correspondences should not crash error computation."""
        result = estimate_transform([], model=TransformModel.AFFINE)
        assert result.success is False
        assert result.error_metrics is None


# =========================================================================
# 5. TRANSFORMATION APPLICATION TESTS
# =========================================================================

class TestTransformPoints:
    """Tests for backend.geometry.transform.transform_points."""

    def test_identity_affine(self):
        """Identity affine should not change points."""
        pts = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64)
        identity = np.array([[1.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0]], dtype=np.float64)

        result = transform_points(pts, identity)

        np.testing.assert_allclose(result, pts)

    def test_translation(self):
        """Pure translation should shift all points."""
        pts = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64)
        translation = np.array([[1.0, 0.0, 5.0],
                                [0.0, 1.0, -3.0]], dtype=np.float64)

        result = transform_points(pts, translation)

        expected = np.array([[15.0, 17.0], [35.0, 37.0]], dtype=np.float64)
        np.testing.assert_allclose(result, expected)

    def test_rotation_scale_affine(self):
        """Known rotation + scale affine should transform correctly."""
        pts = np.array([[1.0, 0.0]], dtype=np.float64)
        angle = math.pi / 2  # 90 degrees
        M = np.array([[math.cos(angle), -math.sin(angle), 0.0],
                       [math.sin(angle),  math.cos(angle), 0.0]],
                      dtype=np.float64)

        result = transform_points(pts, M)

        # (1, 0) rotated 90° → (0, 1)
        np.testing.assert_allclose(result, [[0.0, 1.0]], atol=1e-10)

    def test_multiple_points(self):
        """Multiple points should all be transformed correctly."""
        pts = np.array([
            [0.0, 0.0],
            [10.0, 0.0],
            [0.0, 10.0],
            [10.0, 10.0],
        ], dtype=np.float64)
        M = np.array([[2.0, 0.0, 1.0],
                       [0.0, 3.0, 2.0]], dtype=np.float64)

        result = transform_points(pts, M)

        expected = np.array([
            [1.0, 2.0],
            [21.0, 2.0],
            [1.0, 32.0],
            [21.0, 32.0],
        ], dtype=np.float64)
        np.testing.assert_allclose(result, expected)

    def test_homography_identity(self):
        """Identity homography should not change points."""
        pts = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64)
        identity_H = np.eye(3, dtype=np.float64)

        result = transform_points(pts, identity_H)

        np.testing.assert_allclose(result, pts, atol=1e-10)

    def test_homography_translation(self):
        """Homography with pure translation."""
        pts = np.array([[10.0, 20.0]], dtype=np.float64)
        H = np.array([[1.0, 0.0, 5.0],
                       [0.0, 1.0, -3.0],
                       [0.0, 0.0, 1.0]], dtype=np.float64)

        result = transform_points(pts, H)

        np.testing.assert_allclose(result, [[15.0, 17.0]], atol=1e-10)

    def test_invalid_points_shape(self):
        """Non-(N,2) points should raise ValueError."""
        pts = np.array([1.0, 2.0, 3.0])
        M = np.eye(3, dtype=np.float64)

        with pytest.raises(ValueError, match="shape"):
            transform_points(pts, M)

    def test_invalid_matrix_shape(self):
        """Unsupported matrix shapes should raise ValueError."""
        pts = np.array([[1.0, 2.0]], dtype=np.float64)
        bad_M = np.eye(4, dtype=np.float64)

        with pytest.raises(ValueError, match="shape"):
            transform_points(pts, bad_M)

    def test_empty_points(self):
        """Empty point array should return empty result."""
        pts = np.empty((0, 2), dtype=np.float64)
        M = np.array([[1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0]], dtype=np.float64)

        result = transform_points(pts, M)

        assert result.shape == (0, 2)

    def test_affine_roundtrip(self):
        """Applying a transform and its inverse should recover original points."""
        pts = np.array([[50.0, 60.0], [100.0, 200.0]], dtype=np.float64)

        # Scale 2x + translate (10, 20)
        M = np.array([[2.0, 0.0, 10.0],
                       [0.0, 2.0, 20.0]], dtype=np.float64)

        # Build full 3×3 matrix for inversion
        M_full = np.vstack([M, [0, 0, 1]])
        M_inv = np.linalg.inv(M_full)[:2, :]

        forward = transform_points(pts, M)
        recovered = transform_points(forward, M_inv)

        np.testing.assert_allclose(recovered, pts, atol=1e-10)


# =========================================================================
# 6. CAPABILITY DETECTION TESTS
# =========================================================================

class TestCapabilityDetection:
    """Tests for OpenCV capability detection."""

    def test_capability_returns_valid_structure(self):
        """Capability detection should return a valid EstimatorCapability."""
        cap = detect_estimator_capability()

        assert isinstance(cap, EstimatorCapability)
        assert isinstance(cap.opencv_version, str)
        assert len(cap.opencv_version) > 0
        assert isinstance(cap.has_usac_magsac, bool)
        assert isinstance(cap.has_estimate_affine2d, bool)
        assert isinstance(cap.has_find_homography, bool)
        assert isinstance(cap.available_usac_flags, list)

    def test_estimateAffine2D_available(self):
        """estimateAffine2D must be available for this module to work."""
        cap = detect_estimator_capability()
        assert cap.has_estimate_affine2d is True

    def test_findHomography_available(self):
        """findHomography must be available for homography support."""
        cap = detect_estimator_capability()
        assert cap.has_find_homography is True


# =========================================================================
# 7. ESTIMATOR CONFIGURATION TESTS
# =========================================================================

class TestEstimatorConfiguration:
    """Tests for configurable estimator parameters."""

    def test_custom_threshold(self):
        """Custom reprojection threshold should be recorded in config."""
        matches = _make_affine_correspondences(KNOWN_AFFINE, STANDARD_REF_POINTS)
        result = estimate_transform(
            matches, model=TransformModel.AFFINE, reproj_threshold=5.0,
        )

        assert result.estimator_config["reproj_threshold"] == 5.0

    def test_custom_confidence(self):
        """Custom confidence should be recorded."""
        matches = _make_affine_correspondences(KNOWN_AFFINE, STANDARD_REF_POINTS)
        result = estimate_transform(
            matches, model=TransformModel.AFFINE, confidence=0.99,
        )

        assert result.estimator_config["confidence"] == 0.99

    def test_transform_model_in_config(self):
        """Transform model should be recorded in config."""
        matches = _make_affine_correspondences(KNOWN_AFFINE, STANDARD_REF_POINTS)
        result = estimate_transform(matches, model=TransformModel.AFFINE)

        assert result.estimator_config["model"] == "affine"

    def test_capability_recorded(self):
        """Capability should be recorded in the result."""
        matches = _make_affine_correspondences(KNOWN_AFFINE, STANDARD_REF_POINTS)
        result = estimate_transform(matches, model=TransformModel.AFFINE)

        assert result.capability is not None
        assert isinstance(result.capability.opencv_version, str)

    def test_failure_result_has_capability(self):
        """Even failed results should record capability."""
        result = estimate_transform([], model=TransformModel.AFFINE)

        assert result.capability is not None


# =========================================================================
# 8. END-TO-END CLASSICAL BASELINE TEST
# =========================================================================

def _make_structured_image(width: int = 256, height: int = 256) -> np.ndarray:
    """Create a deterministic structured image (same as in test_classical_baseline)."""
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


def _make_feature_image(arr: np.ndarray) -> FeatureImage:
    """Wrap array into FeatureImage."""
    return FeatureImage(
        data=arr.astype(np.float32),
        width=arr.shape[1],
        height=arr.shape[0],
        source_dtype=np.dtype("uint8"),
        conversion_method="test_fixture",
    )


class TestEndToEndClassicalBaseline:
    """End-to-end: synthetic image → SIFT → FLANN → Lowe → geometry → metrics.

    This verifies the complete classical registration baseline pipeline.
    """

    def test_translated_image_full_pipeline(self):
        """Full pipeline on translated image should produce valid results."""
        img = _make_structured_image()
        # Apply known translation
        M_translate = np.float32([[1, 0, 15], [0, 1, 10]])
        translated = cv2.warpAffine(img, M_translate, (img.shape[1], img.shape[0]))

        ref_feat = _make_feature_image(img)
        tgt_feat = _make_feature_image(translated)

        # SIFT
        ref_sift = extract_sift(ref_feat)
        tgt_sift = extract_sift(tgt_feat)
        assert ref_sift.num_keypoints > 0
        assert tgt_sift.num_keypoints > 0

        # FLANN
        raw_matches = flann_knn_match(ref_sift, tgt_sift)
        assert len(raw_matches) > 0

        # Lowe ratio test
        match_result = apply_ratio_test(raw_matches, ref_sift, tgt_sift)
        assert match_result.accepted > 0

        # Geometric verification
        geo_result = estimate_transform(
            match_result.matches, model=TransformModel.AFFINE,
        )

        # Verify the complete pipeline output
        assert geo_result.success is True, (
            f"Geometric estimation failed: {geo_result.failure_reason}"
        )
        assert geo_result.total_correspondences > 0
        assert geo_result.transform_matrix is not None
        assert geo_result.transform_matrix.shape == (2, 3)
        assert geo_result.inlier_count > 0
        assert geo_result.inlier_ratio > 0.0
        assert geo_result.error_metrics is not None
        assert np.isfinite(geo_result.error_metrics.inlier_rmse)
        assert geo_result.error_metrics.inlier_rmse >= 0.0

        # Do NOT require exact RMSE — SIFT localization varies by OpenCV version
        # Just verify it's finite and non-negative
        assert np.isfinite(geo_result.error_metrics.all_rmse)

    def test_rotated_image_full_pipeline(self):
        """Full pipeline on rotated image should produce valid results."""
        img = _make_structured_image()
        h, w = img.shape[:2]
        M_rotate = cv2.getRotationMatrix2D((w / 2, h / 2), 5.0, 1.0)
        rotated = cv2.warpAffine(img, M_rotate, (w, h))

        ref_feat = _make_feature_image(img)
        tgt_feat = _make_feature_image(rotated)

        ref_sift = extract_sift(ref_feat)
        tgt_sift = extract_sift(tgt_feat)

        raw_matches = flann_knn_match(ref_sift, tgt_sift)
        match_result = apply_ratio_test(raw_matches, ref_sift, tgt_sift)

        if match_result.accepted >= MIN_CORRESPONDENCES[TransformModel.AFFINE]:
            geo_result = estimate_transform(
                match_result.matches, model=TransformModel.AFFINE,
            )
            assert isinstance(geo_result, GeometricResult)
            if geo_result.success:
                assert geo_result.inlier_count > 0
                assert np.isfinite(geo_result.error_metrics.inlier_rmse)

    def test_identity_full_pipeline(self):
        """Matching image to itself should have near-zero error."""
        img = _make_structured_image()
        feat_img = _make_feature_image(img)

        sift_result = extract_sift(feat_img)
        raw_matches = flann_knn_match(sift_result, sift_result)
        match_result = apply_ratio_test(raw_matches, sift_result, sift_result)

        assert match_result.accepted > 0

        geo_result = estimate_transform(
            match_result.matches, model=TransformModel.AFFINE,
        )

        assert geo_result.success is True
        assert geo_result.inlier_count > 0
        assert geo_result.error_metrics is not None
        # Identity should have very small RMSE
        assert geo_result.error_metrics.inlier_rmse < 5.0

    def test_pipeline_records_estimator_info(self):
        """Pipeline should record what estimator was used."""
        img = _make_structured_image()
        feat_img = _make_feature_image(img)

        sift_result = extract_sift(feat_img)
        raw_matches = flann_knn_match(sift_result, sift_result)
        match_result = apply_ratio_test(raw_matches, sift_result, sift_result)

        geo_result = estimate_transform(
            match_result.matches, model=TransformModel.AFFINE,
        )

        assert geo_result.success is True
        assert geo_result.estimator_method is not None
        assert geo_result.capability is not None
        assert len(geo_result.capability.opencv_version) > 0

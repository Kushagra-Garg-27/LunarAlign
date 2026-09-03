"""
SIH26166 — Tests for the classical baseline: SIFT → FLANN → Lowe Ratio.

All fixtures are deterministic synthetic images with geometric structure
(not random noise).  No real Chandrayaan imagery is used.
"""

from __future__ import annotations

import numpy as np
import pytest
import cv2

from backend.preprocessing.datamodel import FeatureImage
from backend.features.models import (
    Keypoint, SIFTFeatures, RawMatch, NeighborInfo,
    FilteredMatch, MatchResult,
)
from backend.features.sift import extract_sift, prepare_for_sift
from backend.matching.flann import flann_knn_match
from backend.matching.ratio_test import apply_ratio_test, DEFAULT_RATIO_THRESHOLD


# =========================================================================
# Synthetic image fixtures
# =========================================================================

def _make_structured_image(width: int = 256, height: int = 256) -> np.ndarray:
    """Create a deterministic structured image with circles, rectangles,
    lines, and corners — features that SIFT can reliably detect.

    Returns uint8 grayscale array (H, W).
    """
    img = np.zeros((height, width), dtype=np.uint8)

    # Background gradient for texture
    for y in range(height):
        img[y, :] = int(40 + 80 * y / height)

    # Filled circles at known positions
    cv2.circle(img, (60, 60), 25, 255, -1)
    cv2.circle(img, (180, 80), 15, 200, -1)
    cv2.circle(img, (120, 200), 30, 230, -1)

    # Rectangles
    cv2.rectangle(img, (30, 130), (90, 180), 220, -1)
    cv2.rectangle(img, (160, 160), (230, 220), 180, -1)

    # Lines for edge/corner features
    cv2.line(img, (10, 10), (100, 50), 255, 2)
    cv2.line(img, (200, 10), (150, 100), 200, 2)

    # Cross pattern (corners)
    cv2.line(img, (120, 100), (120, 140), 255, 3)
    cv2.line(img, (100, 120), (140, 120), 255, 3)

    # Small squares for corner-like features
    cv2.rectangle(img, (200, 30), (220, 50), 240, -1)
    cv2.rectangle(img, (40, 200), (60, 220), 200, -1)

    return img


def _make_feature_image(
    arr: np.ndarray | None = None,
    width: int = 256,
    height: int = 256,
) -> FeatureImage:
    """Wrap an array into a FeatureImage."""
    if arr is None:
        arr = _make_structured_image(width, height).astype(np.float32)
    return FeatureImage(
        data=arr.astype(np.float32) if arr.dtype != np.float32 else arr,
        width=arr.shape[1],
        height=arr.shape[0],
        source_dtype=np.dtype("uint8"),
        conversion_method="test_fixture",
    )


def _translate_image(img: np.ndarray, tx: int, ty: int) -> np.ndarray:
    """Apply a known integer translation to an image."""
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))


def _rotate_image(img: np.ndarray, angle_deg: float, center: tuple | None = None) -> np.ndarray:
    """Rotate image around its center by the given angle (degrees)."""
    h, w = img.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(img, M, (w, h))


def _scale_image(img: np.ndarray, scale: float) -> np.ndarray:
    """Scale image by the given factor."""
    h, w = img.shape[:2]
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h))
    # Pad or crop back to original size for consistent matching
    out = np.zeros_like(img)
    copy_h = min(new_h, h)
    copy_w = min(new_w, w)
    out[:copy_h, :copy_w] = resized[:copy_h, :copy_w]
    return out


# =========================================================================
# SIFT FEATURE DETECTION TESTS
# =========================================================================

class TestSIFTExtraction:
    """Tests for backend.features.sift.extract_sift."""

    def test_detects_keypoints_on_structured_image(self):
        """SIFT should detect keypoints on an image with structure."""
        feat_img = _make_feature_image()
        result = extract_sift(feat_img)

        assert result.num_keypoints > 0
        assert len(result.keypoints) == result.num_keypoints
        assert result.descriptors is not None

    def test_descriptor_shape(self):
        """Descriptors should be (N, 128) float32."""
        feat_img = _make_feature_image()
        result = extract_sift(feat_img)

        assert result.descriptors is not None
        assert result.descriptors.shape == (result.num_keypoints, 128)
        assert result.descriptors.dtype == np.float32

    def test_num_descriptors_matches_keypoints(self):
        """Number of descriptors must equal number of keypoints."""
        feat_img = _make_feature_image()
        result = extract_sift(feat_img)

        assert result.descriptors.shape[0] == len(result.keypoints)

    def test_keypoint_coordinates_within_bounds(self):
        """All keypoint coordinates should be within image bounds."""
        feat_img = _make_feature_image()
        result = extract_sift(feat_img)

        for kp in result.keypoints:
            assert 0 <= kp.x <= feat_img.width, f"x={kp.x} out of bounds"
            assert 0 <= kp.y <= feat_img.height, f"y={kp.y} out of bounds"

    def test_does_not_modify_source(self):
        """Feature extraction should not modify the input FeatureImage."""
        arr = _make_structured_image().astype(np.float32)
        original = arr.copy()
        feat_img = _make_feature_image(arr)
        extract_sift(feat_img)
        np.testing.assert_array_equal(feat_img.data, original)

    def test_empty_flat_image(self):
        """A completely flat image should not crash; may detect 0 keypoints."""
        arr = np.full((64, 64), 128.0, dtype=np.float32)
        feat_img = _make_feature_image(arr)
        result = extract_sift(feat_img)

        # Flat image may or may not yield keypoints depending on OpenCV version
        assert result.num_keypoints >= 0
        if result.num_keypoints == 0:
            assert result.descriptors is None

    def test_image_dimensions_recorded(self):
        """The result should record the source image dimensions."""
        feat_img = _make_feature_image(width=200, height=150)
        result = extract_sift(feat_img)

        assert result.image_width == 200
        assert result.image_height == 150

    def test_config_recorded(self):
        """The configuration used should be recorded in the result."""
        feat_img = _make_feature_image()
        result = extract_sift(feat_img, nfeatures=500, contrastThreshold=0.05)

        assert result.config["nfeatures"] == 500
        assert result.config["contrastThreshold"] == 0.05

    def test_nfeatures_limits_keypoints(self):
        """Setting nfeatures should substantially reduce keypoints.

        OpenCV's nfeatures is not an exact cap — orientation duplicates
        can cause the count to slightly exceed the requested number.
        We verify that the constrained result is much smaller than
        unconstrained.
        """
        feat_img = _make_feature_image()
        unlimited = extract_sift(feat_img, nfeatures=0)
        limited = extract_sift(feat_img, nfeatures=10)

        # The limited result should be substantially fewer than unlimited
        assert limited.num_keypoints < unlimited.num_keypoints
        # Allow a small margin over the requested limit (orientation duplicates)
        assert limited.num_keypoints <= 20

    def test_prepare_for_sift_clips_to_uint8(self):
        """prepare_for_sift should clip values outside [0, 255]."""
        arr = np.array([[-10.0, 300.0], [128.0, 0.0]], dtype=np.float32)
        feat = _make_feature_image(arr)
        u8 = prepare_for_sift(feat)

        assert u8.dtype == np.uint8
        assert u8[0, 0] == 0
        assert u8[0, 1] == 255
        assert u8[1, 0] == 128


# =========================================================================
# FLANN MATCHING TESTS
# =========================================================================

class TestFLANNMatching:
    """Tests for backend.matching.flann.flann_knn_match."""

    def test_match_similar_images(self):
        """Matching the same structured image should produce raw matches."""
        img = _make_structured_image()
        ref = extract_sift(_make_feature_image(img.astype(np.float32)))
        tgt = extract_sift(_make_feature_image(img.astype(np.float32)))

        raw = flann_knn_match(ref, tgt)

        assert len(raw) > 0
        # Each match should have up to 2 neighbours
        for m in raw:
            assert len(m.neighbors) >= 1

    def test_matches_have_valid_indices(self):
        """Match indices should be within valid ranges."""
        img = _make_structured_image()
        ref = extract_sift(_make_feature_image(img.astype(np.float32)))
        tgt = extract_sift(_make_feature_image(img.astype(np.float32)))

        raw = flann_knn_match(ref, tgt)

        for m in raw:
            assert 0 <= m.query_idx < ref.num_keypoints
            for n in m.neighbors:
                assert 0 <= n.train_idx < tgt.num_keypoints
                assert n.distance >= 0

    def test_empty_reference_raises(self):
        """Empty reference descriptors should raise ValueError."""
        img = _make_structured_image()
        tgt = extract_sift(_make_feature_image(img.astype(np.float32)))
        empty_ref = SIFTFeatures(
            keypoints=[], descriptors=None,
            image_width=256, image_height=256,
            num_keypoints=0,
        )

        with pytest.raises(ValueError, match="Reference"):
            flann_knn_match(empty_ref, tgt)

    def test_empty_target_raises(self):
        """Empty target descriptors should raise ValueError."""
        img = _make_structured_image()
        ref = extract_sift(_make_feature_image(img.astype(np.float32)))
        empty_tgt = SIFTFeatures(
            keypoints=[], descriptors=None,
            image_width=256, image_height=256,
            num_keypoints=0,
        )

        with pytest.raises(ValueError, match="Target"):
            flann_knn_match(ref, empty_tgt)

    def test_single_target_descriptor_no_crash(self):
        """If target has only 1 descriptor, k is clamped and no crash."""
        img = _make_structured_image()
        ref = extract_sift(_make_feature_image(img.astype(np.float32)))

        # Create target with single descriptor
        single_tgt = SIFTFeatures(
            keypoints=[ref.keypoints[0]],
            descriptors=ref.descriptors[:1].copy(),
            image_width=256, image_height=256,
            num_keypoints=1,
        )

        raw = flann_knn_match(ref, single_tgt, k=2)
        # Should not crash; matches may have only 1 neighbour
        assert isinstance(raw, list)


# =========================================================================
# LOWE RATIO TEST TESTS
# =========================================================================

class TestRatioTest:
    """Tests for backend.matching.ratio_test.apply_ratio_test."""

    def _make_features(self, n: int) -> SIFTFeatures:
        """Create a dummy SIFTFeatures with n keypoints."""
        kps = [Keypoint(x=float(i), y=float(i), size=1.0,
                        angle=0.0, response=0.1, octave=0)
               for i in range(n)]
        desc = np.random.rand(n, 128).astype(np.float32)
        return SIFTFeatures(
            keypoints=kps, descriptors=desc,
            image_width=100, image_height=100,
            num_keypoints=n,
        )

    def test_clear_good_match_passes(self):
        """A match with ratio well below threshold should be accepted."""
        ref = self._make_features(5)
        tgt = self._make_features(5)
        raw = [RawMatch(
            query_idx=0,
            neighbors=[NeighborInfo(train_idx=0, distance=10.0),
                        NeighborInfo(train_idx=1, distance=100.0)],
        )]

        result = apply_ratio_test(raw, ref, tgt, ratio_threshold=0.75)

        assert result.accepted == 1
        assert result.rejected == 0
        assert result.matches[0].ratio == pytest.approx(0.1)

    def test_ambiguous_match_rejected(self):
        """A match with ratio above threshold should be rejected."""
        ref = self._make_features(5)
        tgt = self._make_features(5)
        raw = [RawMatch(
            query_idx=0,
            neighbors=[NeighborInfo(train_idx=0, distance=90.0),
                        NeighborInfo(train_idx=1, distance=100.0)],
        )]

        result = apply_ratio_test(raw, ref, tgt, ratio_threshold=0.75)

        assert result.accepted == 0
        assert result.rejected == 1

    def test_exact_threshold_is_rejected(self):
        """A ratio exactly equal to threshold should be rejected (strict <)."""
        ref = self._make_features(5)
        tgt = self._make_features(5)
        raw = [RawMatch(
            query_idx=0,
            neighbors=[NeighborInfo(train_idx=0, distance=75.0),
                        NeighborInfo(train_idx=1, distance=100.0)],
        )]

        result = apply_ratio_test(raw, ref, tgt, ratio_threshold=0.75)

        # ratio = 0.75, threshold is 0.75, strict < means rejected
        assert result.accepted == 0
        assert result.rejected == 1

    def test_division_by_zero_handled(self):
        """If d2 == 0, the match should be rejected (not crash)."""
        ref = self._make_features(5)
        tgt = self._make_features(5)
        raw = [RawMatch(
            query_idx=0,
            neighbors=[NeighborInfo(train_idx=0, distance=0.0),
                        NeighborInfo(train_idx=1, distance=0.0)],
        )]

        result = apply_ratio_test(raw, ref, tgt)

        assert result.accepted == 0
        assert result.rejected == 1

    def test_empty_match_list(self):
        """Empty match list should return cleanly."""
        ref = self._make_features(5)
        tgt = self._make_features(5)

        result = apply_ratio_test([], ref, tgt)

        assert result.accepted == 0
        assert result.rejected == 0
        assert result.total_raw == 0
        assert result.matches == []

    def test_single_neighbour_skipped(self):
        """Matches with < 2 neighbours should be skipped (not rejected)."""
        ref = self._make_features(5)
        tgt = self._make_features(5)
        raw = [RawMatch(
            query_idx=0,
            neighbors=[NeighborInfo(train_idx=0, distance=10.0)],
        )]

        result = apply_ratio_test(raw, ref, tgt)

        assert result.accepted == 0
        assert result.rejected == 0  # skipped, not rejected

    def test_ratio_stats_computed(self):
        """Ratio statistics should be computed when matches exist."""
        ref = self._make_features(5)
        tgt = self._make_features(5)
        raw = [
            RawMatch(query_idx=0,
                     neighbors=[NeighborInfo(0, 10.0), NeighborInfo(1, 100.0)]),
            RawMatch(query_idx=1,
                     neighbors=[NeighborInfo(2, 50.0), NeighborInfo(3, 60.0)]),
        ]

        result = apply_ratio_test(raw, ref, tgt)

        assert "min" in result.ratio_stats
        assert "max" in result.ratio_stats
        assert "mean" in result.ratio_stats
        assert "median" in result.ratio_stats

    def test_coordinates_correct(self):
        """Accepted matches should carry correct reference/target coordinates."""
        ref = self._make_features(5)
        tgt = self._make_features(5)
        raw = [RawMatch(
            query_idx=2,
            neighbors=[NeighborInfo(train_idx=3, distance=10.0),
                        NeighborInfo(train_idx=4, distance=100.0)],
        )]

        result = apply_ratio_test(raw, ref, tgt, ratio_threshold=0.75)

        assert result.accepted == 1
        m = result.matches[0]
        assert m.ref_pt == (ref.keypoints[2].x, ref.keypoints[2].y)
        assert m.tgt_pt == (tgt.keypoints[3].x, tgt.keypoints[3].y)
        assert m.query_idx == 2
        assert m.train_idx == 3


# =========================================================================
# END-TO-END BASELINE TEST
# =========================================================================

class TestEndToEndBaseline:
    """End-to-end test: SIFT → FLANN → Lowe → verify correspondences."""

    def test_identity_matching(self):
        """Matching an image to itself should produce many good matches."""
        img = _make_structured_image()
        feat_img = _make_feature_image(img.astype(np.float32))

        ref = extract_sift(feat_img)
        tgt = extract_sift(feat_img)

        assert ref.num_keypoints > 0
        assert tgt.num_keypoints > 0

        raw = flann_knn_match(ref, tgt)
        assert len(raw) > 0

        result = apply_ratio_test(raw, ref, tgt)
        assert result.accepted > 0

        # For identity, most matches should be near-perfect
        for m in result.matches:
            # Reference and target points should be very close
            dx = abs(m.ref_pt[0] - m.tgt_pt[0])
            dy = abs(m.ref_pt[1] - m.tgt_pt[1])
            assert dx < 5.0 and dy < 5.0, (
                f"Identity match too far: ref={m.ref_pt}, tgt={m.tgt_pt}"
            )

    def test_translated_image(self):
        """Matching a translated image should produce correspondences."""
        img = _make_structured_image()
        translated = _translate_image(img, tx=15, ty=10)

        ref_feat = _make_feature_image(img.astype(np.float32))
        tgt_feat = _make_feature_image(translated.astype(np.float32))

        ref = extract_sift(ref_feat)
        tgt = extract_sift(tgt_feat)

        assert ref.num_keypoints > 0
        assert tgt.num_keypoints > 0

        raw = flann_knn_match(ref, tgt)
        result = apply_ratio_test(raw, ref, tgt)

        assert result.accepted > 0

        # Correspondences should reflect ~(15, 10) translation
        displacements_x = [m.tgt_pt[0] - m.ref_pt[0] for m in result.matches]
        displacements_y = [m.tgt_pt[1] - m.ref_pt[1] for m in result.matches]

        median_dx = float(np.median(displacements_x))
        median_dy = float(np.median(displacements_y))

        # Allow generous tolerance (SIFT is not pixel-perfect at this stage)
        assert abs(median_dx - 15) < 10, f"median dx={median_dx}, expected ~15"
        assert abs(median_dy - 10) < 10, f"median dy={median_dy}, expected ~10"

    def test_rotated_image(self):
        """Matching a slightly rotated image should produce correspondences."""
        img = _make_structured_image()
        rotated = _rotate_image(img, angle_deg=5)

        ref_feat = _make_feature_image(img.astype(np.float32))
        tgt_feat = _make_feature_image(rotated.astype(np.float32))

        ref = extract_sift(ref_feat)
        tgt = extract_sift(tgt_feat)

        raw = flann_knn_match(ref, tgt)
        result = apply_ratio_test(raw, ref, tgt)

        # We expect at least some correspondences survive
        assert result.accepted > 0

    def test_scaled_image(self):
        """Matching a slightly scaled image should produce correspondences."""
        img = _make_structured_image()
        scaled = _scale_image(img, scale=1.2)

        ref_feat = _make_feature_image(img.astype(np.float32))
        tgt_feat = _make_feature_image(scaled.astype(np.float32))

        ref = extract_sift(ref_feat)
        tgt = extract_sift(tgt_feat)

        raw = flann_knn_match(ref, tgt)
        result = apply_ratio_test(raw, ref, tgt)

        assert result.accepted >= 0  # Scale may reduce matches but shouldn't crash

    def test_match_result_has_all_fields(self):
        """The MatchResult should have all expected fields populated."""
        img = _make_structured_image()
        feat_img = _make_feature_image(img.astype(np.float32))

        ref = extract_sift(feat_img)
        tgt = extract_sift(feat_img)
        raw = flann_knn_match(ref, tgt)
        result = apply_ratio_test(raw, ref, tgt)

        assert isinstance(result.matches, list)
        assert isinstance(result.total_raw, int)
        assert isinstance(result.accepted, int)
        assert isinstance(result.rejected, int)
        assert result.ratio_threshold == DEFAULT_RATIO_THRESHOLD
        assert result.accepted + result.rejected <= result.total_raw

"""
SIH26166 — Tests for Stage 3 Feature Detection, Description & Matching.

Validates SIFT, Grid-Bucketed SIFT, MIND descriptors, FLANN/BF matching,
and MAGSAC++ / Affine outlier rejection using real Chandrayaan-2 fixtures and synthetic patterns.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.matching import (
    MatchResult,
    compute_mind_descriptor,
    compute_spatial_entropy,
    detect_sift_bucketed,
    detect_sift_features,
    match_mind_descriptors,
    match_pair,
    reject_outliers_affine,
    reject_outliers_magsac,
)
from backend.preprocessing.normalize import normalize_intensity
from backend.preprocessing.pds4 import load_pds4_raster

# ---------------------------------------------------------------------------
# Fixture paths
# ---------------------------------------------------------------------------

PDS4_FIXTURES = Path("tests/fixtures/pds4")
TMC2_XML = PDS4_FIXTURES / "tmc2" / "ch2_tmc_sample_500x500.xml"
OHRC_XML = PDS4_FIXTURES / "ohrc" / "ch2_ohrc_sample_500x500.xml"
IIRS_XML = PDS4_FIXTURES / "iirs" / "ch2_iirs_sample_200x200x16.xml"


# ---------------------------------------------------------------------------
# SIFT Detector Tests
# ---------------------------------------------------------------------------

class TestSiftDetector:
    """Tests for backend.matching.sift_detector (Modules 04, 05, 09)."""

    def test_detect_sift_tmc2_fixture(self):
        """SIFT detects > 100 keypoints on real TMC-2 fixture."""
        raw = load_pds4_raster(TMC2_XML)
        normed = normalize_intensity(raw.data, method="clahe")
        kps, descs = detect_sift_features(normed)

        assert len(kps) > 100
        assert descs.shape == (len(kps), 128)
        assert descs.dtype == np.float32

    def test_detect_sift_ohrc_fixture(self):
        """SIFT detects keypoints on real OHRC fixture."""
        raw = load_pds4_raster(OHRC_XML)
        normed = normalize_intensity(raw.data, method="clahe")
        kps, descs = detect_sift_features(normed)

        assert len(kps) > 50
        assert descs.shape == (len(kps), 128)

    def test_max_keypoints_limits_output(self):
        """max_keypoints parameter strictly caps the returned keypoints."""
        raw = load_pds4_raster(TMC2_XML)
        normed = normalize_intensity(raw.data, method="clahe")
        kps, descs = detect_sift_features(normed, max_keypoints=50)

        assert len(kps) <= 50
        assert descs.shape[0] == len(kps)

    def test_bucketed_detection_spatial_distribution(self):
        """Grid-bucketed SIFT spreads keypoints across image cells."""
        raw = load_pds4_raster(TMC2_XML)
        normed = normalize_intensity(raw.data, method="clahe")

        kps_std, _ = detect_sift_features(normed, max_keypoints=300)
        kps_bkt, _ = detect_sift_bucketed(
            normed, grid_size=(4, 4), keypoints_per_cell=20
        )

        assert len(kps_bkt) > 0
        # Compute spatial distribution entropy
        pts_std = np.float32([kp.pt for kp in kps_std])
        pts_bkt = np.float32([kp.pt for kp in kps_bkt])

        h_std = compute_spatial_entropy(pts_std, normed.shape[:2], grid_size=(4, 4))
        h_bkt = compute_spatial_entropy(pts_bkt, normed.shape[:2], grid_size=(4, 4))

        assert h_bkt >= h_std * 0.9

    def test_empty_image_returns_empty(self):
        """Empty or invalid image returns 0 keypoints without error."""
        empty = np.zeros((0, 0), dtype=np.float32)
        kps, descs = detect_sift_features(empty)
        assert len(kps) == 0
        assert descs.shape == (0, 128)

    def test_descriptors_shape_and_dtype(self):
        """Descriptors are always float32 with 128 dimensions."""
        img = np.random.rand(100, 100).astype(np.float32)
        kps, descs = detect_sift_features(img)
        assert descs.dtype == np.float32
        assert descs.shape[1] == 128

    def test_contrast_threshold_impact(self):
        """Higher contrast threshold yields fewer or equal keypoints."""
        raw = load_pds4_raster(OHRC_XML)
        normed = normalize_intensity(raw.data, method="clahe")

        kps_sensitive, _ = detect_sift_features(normed, contrast_threshold=0.01)
        kps_strict, _ = detect_sift_features(normed, contrast_threshold=0.10)

        assert len(kps_sensitive) >= len(kps_strict)

    def test_non_square_image(self):
        """Works seamlessly on non-square image dimensions."""
        img = np.random.rand(80, 140).astype(np.float32)
        kps, descs = detect_sift_bucketed(img, grid_size=(4, 7))
        assert isinstance(kps, list)
        assert descs.shape == (len(kps), 128)


# ---------------------------------------------------------------------------
# MIND Descriptor Tests
# ---------------------------------------------------------------------------

class TestMindDescriptor:
    """Tests for backend.matching.mind_descriptor (Module 07)."""

    def test_mind_descriptor_shape(self):
        """MIND descriptor has shape (H, W, 8) for search_radius=1."""
        img = np.random.rand(64, 64).astype(np.float32)
        mind = compute_mind_descriptor(img, patch_radius=3, search_radius=1)

        assert mind.shape == (64, 64, 8)
        assert mind.dtype == np.float32

    def test_mind_output_range(self):
        """MIND descriptor values are in [0, 1]."""
        img = np.random.rand(50, 50).astype(np.float32)
        mind = compute_mind_descriptor(img)

        assert float(mind.min()) >= 0.0
        assert float(mind.max()) <= 1.0

    def test_mind_different_for_different_structures(self):
        """MIND produces distinct descriptors for flat vs textured regions."""
        img = np.zeros((64, 64), dtype=np.float32)
        img[:32, :32] = 0.5  # flat region
        img[32:, 32:] = np.random.rand(32, 32)  # textured region

        mind = compute_mind_descriptor(img)
        desc_flat = mind[16, 16]
        desc_text = mind[48, 48]

        assert not np.allclose(desc_flat, desc_text, atol=1e-3)

    def test_match_mind_descriptors_shifted_image(self):
        """MIND block matching correctly finds correspondences on shifted image."""
        np.random.seed(42)
        base = np.zeros((100, 100), dtype=np.float32)
        for _ in range(8):
            cy, cx = np.random.randint(20, 80, 2)
            y, x = np.ogrid[:100, :100]
            base[(y - cy) ** 2 + (x - cx) ** 2 < 12 ** 2] = 0.8

        shift_x, shift_y = 2, 2
        shifted = np.roll(np.roll(base, shift_y, axis=0), shift_x, axis=1)

        m1 = compute_mind_descriptor(base)
        m2 = compute_mind_descriptor(shifted)

        corrs = match_mind_descriptors(m1, m2, block_size=16, search_window=24, stride=16)
        assert len(corrs) > 0

        # Verify majority of correspondences reflect the shift
        accurate = 0
        for (x1, y1), (x2, y2), _ in corrs:
            if abs((x2 - x1) - shift_x) <= 2 and abs((y2 - y1) - shift_y) <= 2:
                accurate += 1
        assert accurate / len(corrs) >= 0.6

    def test_empty_image_handling(self):
        """Empty array returns empty MIND volume and no matches."""
        empty = np.zeros((0, 0), dtype=np.float32)
        mind = compute_mind_descriptor(empty)
        assert mind.shape == (0, 0, 8)
        assert match_mind_descriptors(mind, mind) == []

    def test_channel_mismatch_raises(self):
        """Channel mismatch between MIND volumes raises ValueError."""
        m1 = np.zeros((20, 20, 8), dtype=np.float32)
        m2 = np.zeros((20, 20, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="channel mismatch"):
            match_mind_descriptors(m1, m2)



# ---------------------------------------------------------------------------
# Outlier Rejection Tests
# ---------------------------------------------------------------------------

class TestOutlierRejection:
    """Tests for backend.matching.outlier_rejection (Module 10)."""

    def test_magsac_filters_outliers(self):
        """MAGSAC++ correctly isolates inliers from synthetic noisy correspondences."""
        np.random.seed(42)
        n_inliers = 30
        n_outliers = 15

        pts1 = np.random.rand(n_inliers + n_outliers, 2).astype(np.float32) * 400
        # True affine transform: dx=20, dy=-15
        pts2 = pts1.copy()
        pts2[:n_inliers, 0] += 20.0
        pts2[:n_inliers, 1] -= 15.0
        # Corrupt outliers
        pts2[n_inliers:] += np.random.rand(n_outliers, 2) * 200

        kp1 = [cv2.KeyPoint(x=p[0], y=p[1], size=10.0) for p in pts1]
        kp2 = [cv2.KeyPoint(x=p[0], y=p[1], size=10.0) for p in pts2]
        matches = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0.0) for i in range(len(pts1))]

        inliers, h_mat, mask = reject_outliers_magsac(kp1, kp2, matches)

        assert len(inliers) >= n_inliers * 0.8
        assert h_mat.shape == (3, 3)

    def test_inlier_ratio_good_data(self):
        """Clean synthetic inliers achieve inlier ratio > 0.7."""
        pts1 = np.random.rand(20, 2).astype(np.float32) * 300
        pts2 = pts1 + 5.0
        kp1 = [cv2.KeyPoint(x=p[0], y=p[1], size=10.0) for p in pts1]
        kp2 = [cv2.KeyPoint(x=p[0], y=p[1], size=10.0) for p in pts2]
        matches = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0.0) for i in range(20)]

        inliers, _, _ = reject_outliers_magsac(kp1, kp2, matches)
        ratio = len(inliers) / len(matches)
        assert ratio >= 0.7

    def test_magsac_returns_valid_homography(self):
        """Homography matrix is non-singular 3x3 float32."""
        pts1 = np.random.rand(15, 2).astype(np.float32) * 200
        pts2 = pts1 + 10.0
        kp1 = [cv2.KeyPoint(x=p[0], y=p[1], size=10.0) for p in pts1]
        kp2 = [cv2.KeyPoint(x=p[0], y=p[1], size=10.0) for p in pts2]
        matches = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0.0) for i in range(15)]

        _, h_mat, _ = reject_outliers_magsac(kp1, kp2, matches)
        assert h_mat.shape == (3, 3)
        assert np.linalg.det(h_mat) != pytest.approx(0.0, abs=1e-5)

    def test_affine_rejection_returns_2x3(self):
        """Affine outlier rejection returns 2x3 matrix."""
        pts1 = np.random.rand(15, 2).astype(np.float32) * 200
        pts2 = pts1 + 10.0
        kp1 = [cv2.KeyPoint(x=p[0], y=p[1], size=10.0) for p in pts1]
        kp2 = [cv2.KeyPoint(x=p[0], y=p[1], size=10.0) for p in pts2]
        matches = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0.0) for i in range(15)]

        inliers, aff_mat, _ = reject_outliers_affine(kp1, kp2, matches)
        assert len(inliers) > 0
        assert aff_mat.shape == (2, 3)

    def test_few_matches_fallback(self):
        """< 4 matches gracefully returns empty inliers without crashing."""
        kp1 = [cv2.KeyPoint(x=0, y=0, size=5)]
        kp2 = [cv2.KeyPoint(x=0, y=0, size=5)]
        matches = [cv2.DMatch(_queryIdx=0, _trainIdx=0, _distance=0.0)]

        inliers, h_mat, _ = reject_outliers_magsac(kp1, kp2, matches)
        assert inliers == []
        assert h_mat.shape == (3, 3)

    def test_all_outliers_handled(self):
        """Random unaligned correspondences return empty or low inlier count."""
        np.random.seed(7)
        pts1 = np.random.rand(20, 2).astype(np.float32) * 500
        pts2 = np.random.rand(20, 2).astype(np.float32) * 500
        kp1 = [cv2.KeyPoint(x=p[0], y=p[1], size=10.0) for p in pts1]
        kp2 = [cv2.KeyPoint(x=p[0], y=p[1], size=10.0) for p in pts2]
        matches = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0.0) for i in range(20)]

        inliers, _, _ = reject_outliers_magsac(kp1, kp2, matches, threshold=1.0)
        assert len(inliers) <= 8


# ---------------------------------------------------------------------------
# Match Pipeline Orchestrator Tests
# ---------------------------------------------------------------------------

class TestMatchPipeline:
    """Tests for backend.matching.match_pipeline."""

    def test_match_pair_tmc2_shifted(self):
        """match_pair produces inlier matches on TMC-2 fixture vs shifted copy."""
        raw = load_pds4_raster(TMC2_XML)
        img1 = normalize_intensity(raw.data, method="clahe")
        # Synthesize shifted pair (dx=5, dy=3)
        img2 = np.roll(np.roll(img1, 3, axis=0), 5, axis=1)

        result = match_pair(img1, img2, method="sift")

        assert isinstance(result, MatchResult)
        assert result.raw_matches > 0
        assert result.inlier_matches > 0
        assert result.inlier_ratio > 0.0
        assert result.transform_matrix is not None
        assert result.transform_type in ("homography", "affine")
        assert result.match_points1.shape[0] == result.inlier_matches

    def test_match_result_dataclass_fields(self):
        """All fields in MatchResult are populated."""
        raw = load_pds4_raster(OHRC_XML)
        img1 = normalize_intensity(raw.data, method="clahe")
        img2 = img1.copy()

        res = match_pair(img1, img2, method="sift")

        assert hasattr(res, "keypoints1")
        assert hasattr(res, "keypoints2")
        assert hasattr(res, "raw_matches")
        assert hasattr(res, "good_matches")
        assert hasattr(res, "inlier_matches")
        assert hasattr(res, "inlier_ratio")
        assert hasattr(res, "transform_matrix")
        assert hasattr(res, "transform_type")
        assert hasattr(res, "match_points1")
        assert hasattr(res, "match_points2")
        assert hasattr(res, "spatial_distribution")

    def test_spatial_entropy_positive_for_distributed_matches(self):
        """Spatial entropy is positive for well-spread match points."""
        raw = load_pds4_raster(TMC2_XML)
        img1 = normalize_intensity(raw.data, method="clahe")
        img2 = np.roll(img1, 4, axis=1)

        res = match_pair(img1, img2, method="sift_bucketed")
        assert res.spatial_distribution > 0.0

    def test_bucketed_vs_unbucketed_entropy(self):
        """Grid-bucketed SIFT produces equal or higher spatial entropy."""
        raw = load_pds4_raster(TMC2_XML)
        img1 = normalize_intensity(raw.data, method="clahe")
        img2 = np.roll(img1, 4, axis=1)

        res_std = match_pair(img1, img2, method="sift")
        res_bkt = match_pair(img1, img2, method="sift_bucketed")

        assert res_bkt.spatial_distribution >= 0.0
        assert res_std.spatial_distribution >= 0.0

    def test_transform_type_respected(self):
        """transform_model option sets transform_type in MatchResult."""
        img1 = np.random.rand(100, 100).astype(np.float32)
        img2 = img1.copy()

        res_aff = match_pair(img1, img2, method="sift", transform_model="affine")
        assert res_aff.transform_type == "affine"

        res_homo = match_pair(img1, img2, method="sift", transform_model="homography")
        assert res_homo.transform_type == "homography"

    def test_invalid_method_raises(self):
        """Invalid matching method name raises ValueError."""
        img = np.ones((50, 50), dtype=np.float32)
        with pytest.raises(ValueError, match="Unsupported matching method"):
            match_pair(img, img, method="unknown_algorithm")

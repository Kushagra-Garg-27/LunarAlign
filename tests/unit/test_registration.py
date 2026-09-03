"""
SIH26166 — Comprehensive tests for image warping and registration output.

Test categories:
1. Affine warping (identity, translation, rotation, scale, dimensions, grayscale, multi-channel)
2. Homography warping (identity, translation, perspective)
3. Input validation (invalid matrix, NaN/Inf, empty image, bad dimensions, bad config)
4. WarpConfig (interpolation modes, border modes, border value, explicit dimensions)
5. End-to-end classical baseline (SIFT → FLANN → Lowe → MAGSAC++ → warp → alignment)

All fixtures use deterministic synthetic data — no real imagery.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from backend.features.sift import extract_sift
from backend.geometry.estimation import estimate_transform
from backend.geometry.models import TransformModel
from backend.matching.flann import flann_knn_match
from backend.matching.ratio_test import apply_ratio_test
from backend.preprocessing.datamodel import FeatureImage
from backend.registration.models import (
    BORDER_MODES,
    INTERPOLATION_MODES,
    RegistrationResult,
    WarpConfig,
)
from backend.registration.warping import warp_image


# =========================================================================
# Synthetic image helpers
# =========================================================================

def _make_gray_image(
    width: int = 128,
    height: int = 128,
    value: int = 128,
) -> np.ndarray:
    """Create a uniform grayscale uint8 image."""
    return np.full((height, width), value, dtype=np.uint8)


def _make_gradient_image(width: int = 128, height: int = 128) -> np.ndarray:
    """Create a grayscale image with a horizontal gradient."""
    img = np.zeros((height, width), dtype=np.uint8)
    for x in range(width):
        img[:, x] = int(255 * x / max(width - 1, 1))
    return img


def _make_checkerboard(
    width: int = 128,
    height: int = 128,
    block: int = 16,
) -> np.ndarray:
    """Create a checkerboard pattern (useful for alignment verification)."""
    img = np.zeros((height, width), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            if ((x // block) + (y // block)) % 2 == 0:
                img[y, x] = 255
    return img


def _make_color_image(
    width: int = 128,
    height: int = 128,
) -> np.ndarray:
    """Create a 3-channel BGR uint8 image with distinct channel patterns."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :, 0] = 100  # Blue
    img[:, :, 1] = 150  # Green
    for x in range(width):
        img[:, x, 2] = int(255 * x / max(width - 1, 1))  # Red gradient
    return img


def _make_structured_image(width: int = 256, height: int = 256) -> np.ndarray:
    """Create a structured image with features for SIFT (same as baseline tests)."""
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


# =========================================================================
# 1. AFFINE WARPING TESTS
# =========================================================================

class TestAffineWarping:
    """Tests for affine image warping via cv2.warpAffine."""

    def test_identity_warp_preserves_image(self):
        """Identity affine should produce an identical image."""
        img = _make_gradient_image(100, 80)
        identity = np.array([[1.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0]], dtype=np.float64)

        result = warp_image(img, identity, TransformModel.AFFINE)

        assert result.success is True
        assert result.registered_image is not None
        assert result.registered_image.shape == img.shape
        np.testing.assert_array_equal(result.registered_image, img)

    def test_known_translation(self):
        """Translating a gradient image should shift pixel values."""
        img = _make_gradient_image(128, 128)
        tx, ty = 10, 5
        M = np.array([[1.0, 0.0, float(tx)],
                       [0.0, 1.0, float(ty)]], dtype=np.float64)

        result = warp_image(img, M, TransformModel.AFFINE)

        assert result.success is True
        warped = result.registered_image

        # The interior should match the original shifted by (tx, ty)
        # At position (tx + 20, ty + 20) in warped, we expect img[20, 20]
        assert warped[ty + 20, tx + 20] == img[20, 20]

        # Top-left corner should be border fill (0 by default)
        assert warped[0, 0] == 0

    def test_known_rotation(self):
        """90-degree rotation of a known image should reposition pixels."""
        # Create a simple 4x4 image with a unique pattern
        img = np.array([
            [255, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.uint8)

        # 90° counter-clockwise rotation around center (1.5, 1.5)
        center = (1.5, 1.5)
        M = cv2.getRotationMatrix2D(center, 90, 1.0)

        result = warp_image(img, M, TransformModel.AFFINE)

        assert result.success is True
        warped = result.registered_image
        assert warped.shape == img.shape
        # The bright pixel at (0,0) should have moved to (0,3) approximately
        # (rotation is interpolated, so check the corner region)
        assert warped[3, 0] > 100  # rotated bright pixel

    def test_known_scale(self):
        """2x scale should stretch the image."""
        img = _make_checkerboard(64, 64, block=8)
        M = np.array([[2.0, 0.0, 0.0],
                       [0.0, 2.0, 0.0]], dtype=np.float64)

        result = warp_image(
            img, M, TransformModel.AFFINE,
            config=WarpConfig(output_width=128, output_height=128),
        )

        assert result.success is True
        assert result.output_width == 128
        assert result.output_height == 128
        assert result.registered_image.shape == (128, 128)

    def test_output_dimensions_from_target(self):
        """Output dimensions should come from target_width/target_height."""
        img = _make_gray_image(100, 80)
        identity = np.eye(2, 3, dtype=np.float64)

        result = warp_image(
            img, identity, TransformModel.AFFINE,
            target_width=200, target_height=150,
        )

        assert result.success is True
        assert result.output_width == 200
        assert result.output_height == 150
        assert result.registered_image.shape == (150, 200)

    def test_output_dimensions_from_source(self):
        """Without explicit dims, output should match source dimensions."""
        img = _make_gray_image(100, 80)
        identity = np.eye(2, 3, dtype=np.float64)

        result = warp_image(img, identity, TransformModel.AFFINE)

        assert result.output_width == 100
        assert result.output_height == 80

    def test_config_overrides_target_dims(self):
        """Explicit config dimensions override target dimensions."""
        img = _make_gray_image(100, 80)
        identity = np.eye(2, 3, dtype=np.float64)

        result = warp_image(
            img, identity, TransformModel.AFFINE,
            config=WarpConfig(output_width=50, output_height=40),
            target_width=200, target_height=150,
        )

        assert result.output_width == 50
        assert result.output_height == 40

    def test_grayscale_image(self):
        """2-D grayscale image should be warped correctly."""
        img = _make_gray_image(64, 64)
        assert img.ndim == 2

        identity = np.eye(2, 3, dtype=np.float64)
        result = warp_image(img, identity, TransformModel.AFFINE)

        assert result.success is True
        assert result.registered_image.ndim == 2

    def test_multi_channel_image(self):
        """3-channel image should be warped with all channels preserved."""
        img = _make_color_image(64, 64)
        assert img.ndim == 3
        assert img.shape[2] == 3

        identity = np.eye(2, 3, dtype=np.float64)
        result = warp_image(img, identity, TransformModel.AFFINE)

        assert result.success is True
        assert result.registered_image.ndim == 3
        assert result.registered_image.shape[2] == 3
        np.testing.assert_array_equal(result.registered_image, img)

    def test_preserves_dtype(self):
        """Warped image should have the same dtype as the source."""
        for dtype in [np.uint8, np.uint16, np.float32]:
            img = np.full((32, 32), 100, dtype=dtype)
            identity = np.eye(2, 3, dtype=np.float64)
            result = warp_image(img, identity, TransformModel.AFFINE)

            assert result.success is True
            assert result.registered_image.dtype == dtype

    def test_source_dimensions_recorded(self):
        """Source dimensions should be recorded in the result."""
        img = _make_gray_image(100, 80)
        identity = np.eye(2, 3, dtype=np.float64)
        result = warp_image(img, identity, TransformModel.AFFINE)

        assert result.source_width == 100
        assert result.source_height == 80


# =========================================================================
# 2. HOMOGRAPHY WARPING TESTS
# =========================================================================

class TestHomographyWarping:
    """Tests for homography/perspective warping via cv2.warpPerspective."""

    def test_identity_homography(self):
        """Identity homography should preserve the image."""
        img = _make_gradient_image(100, 80)
        H = np.eye(3, dtype=np.float64)

        result = warp_image(img, H, TransformModel.HOMOGRAPHY)

        assert result.success is True
        assert result.registered_image is not None
        np.testing.assert_array_equal(result.registered_image, img)

    def test_translation_homography(self):
        """Pure translation via homography."""
        img = _make_gradient_image(128, 128)
        H = np.array([
            [1.0, 0.0, 15.0],
            [0.0, 1.0, 10.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        result = warp_image(img, H, TransformModel.HOMOGRAPHY)

        assert result.success is True
        warped = result.registered_image
        # Interior pixel check
        assert warped[10 + 30, 15 + 30] == img[30, 30]

    def test_simple_perspective(self):
        """A mild perspective transform should produce a valid image."""
        img = _make_checkerboard(128, 128)
        H = np.array([
            [1.01, 0.005, 2.0],
            [0.005, 1.02, -1.0],
            [1e-5, 2e-5, 1.0],
        ], dtype=np.float64)

        result = warp_image(img, H, TransformModel.HOMOGRAPHY)

        assert result.success is True
        assert result.registered_image is not None
        assert result.registered_image.shape == img.shape
        assert result.transform_model == TransformModel.HOMOGRAPHY

    def test_homography_multi_channel(self):
        """Homography should work with multi-channel images."""
        img = _make_color_image(64, 64)
        H = np.eye(3, dtype=np.float64)

        result = warp_image(img, H, TransformModel.HOMOGRAPHY)

        assert result.success is True
        assert result.registered_image.ndim == 3
        np.testing.assert_array_equal(result.registered_image, img)


# =========================================================================
# 3. INPUT VALIDATION TESTS
# =========================================================================

class TestInputValidation:
    """Tests for input validation and error handling."""

    def test_invalid_matrix_shape_affine(self):
        """Wrong matrix shape for affine should fail."""
        img = _make_gray_image()
        bad_M = np.eye(3, dtype=np.float64)  # 3x3 for affine is wrong

        result = warp_image(img, bad_M, TransformModel.AFFINE)

        assert result.success is False
        assert "shape" in result.failure_reason.lower()

    def test_invalid_matrix_shape_homography(self):
        """Wrong matrix shape for homography should fail."""
        img = _make_gray_image()
        bad_M = np.eye(2, 3, dtype=np.float64)  # 2x3 for homography is wrong

        result = warp_image(img, bad_M, TransformModel.HOMOGRAPHY)

        assert result.success is False
        assert "shape" in result.failure_reason.lower()

    def test_nan_matrix(self):
        """Matrix with NaN should fail."""
        img = _make_gray_image()
        M = np.array([[1.0, 0.0, float("nan")],
                       [0.0, 1.0, 0.0]], dtype=np.float64)

        result = warp_image(img, M, TransformModel.AFFINE)

        assert result.success is False
        assert "NaN" in result.failure_reason or "nan" in result.failure_reason.lower()

    def test_inf_matrix(self):
        """Matrix with Inf should fail."""
        img = _make_gray_image()
        M = np.array([[1.0, 0.0, float("inf")],
                       [0.0, 1.0, 0.0]], dtype=np.float64)

        result = warp_image(img, M, TransformModel.AFFINE)

        assert result.success is False
        assert "Inf" in result.failure_reason or "inf" in result.failure_reason.lower()

    def test_none_matrix(self):
        """None matrix should fail."""
        img = _make_gray_image()

        result = warp_image(img, None, TransformModel.AFFINE)

        assert result.success is False
        assert "None" in result.failure_reason

    def test_empty_image(self):
        """Empty image should fail."""
        img = np.empty((0, 0), dtype=np.uint8)
        M = np.eye(2, 3, dtype=np.float64)

        result = warp_image(img, M, TransformModel.AFFINE)

        assert result.success is False
        assert "empty" in result.failure_reason.lower()

    def test_none_image(self):
        """None image should fail."""
        M = np.eye(2, 3, dtype=np.float64)

        result = warp_image(None, M, TransformModel.AFFINE)

        assert result.success is False

    def test_1d_image(self):
        """1-D array should fail."""
        img = np.array([1, 2, 3], dtype=np.uint8)
        M = np.eye(2, 3, dtype=np.float64)

        result = warp_image(img, M, TransformModel.AFFINE)

        assert result.success is False

    def test_invalid_output_dimensions(self):
        """Zero or negative output dimensions should fail."""
        img = _make_gray_image()
        M = np.eye(2, 3, dtype=np.float64)

        result = warp_image(
            img, M, TransformModel.AFFINE,
            config=WarpConfig(output_width=0, output_height=100),
        )

        assert result.success is False
        assert "positive" in result.failure_reason.lower() or "dimensions" in result.failure_reason.lower()

    def test_invalid_interpolation_mode(self):
        """Unknown interpolation mode should fail."""
        img = _make_gray_image()
        M = np.eye(2, 3, dtype=np.float64)

        result = warp_image(
            img, M, TransformModel.AFFINE,
            config=WarpConfig(interpolation="bicubic_fancy"),
        )

        assert result.success is False
        assert "interpolation" in result.failure_reason.lower()

    def test_invalid_border_mode(self):
        """Unknown border mode should fail."""
        img = _make_gray_image()
        M = np.eye(2, 3, dtype=np.float64)

        result = warp_image(
            img, M, TransformModel.AFFINE,
            config=WarpConfig(border_mode="magic_border"),
        )

        assert result.success is False
        assert "border" in result.failure_reason.lower()

    def test_failure_result_structure(self):
        """Failed results should have correct structure."""
        result = warp_image(None, None, TransformModel.AFFINE)

        assert result.success is False
        assert result.registered_image is None
        assert isinstance(result.failure_reason, str)
        assert len(result.failure_reason) > 0
        assert isinstance(result.warp_config, WarpConfig)


# =========================================================================
# 4. WARP CONFIG TESTS
# =========================================================================

class TestWarpConfig:
    """Tests for WarpConfig validation and flag resolution."""

    def test_all_interpolation_modes_valid(self):
        """All documented interpolation modes should resolve to valid flags."""
        for mode_name, expected_flag in INTERPOLATION_MODES.items():
            cfg = WarpConfig(interpolation=mode_name)
            assert cfg.get_interpolation_flag() == expected_flag

    def test_all_border_modes_valid(self):
        """All documented border modes should resolve to valid flags."""
        for mode_name, expected_flag in BORDER_MODES.items():
            cfg = WarpConfig(interpolation="linear", border_mode=mode_name)
            assert cfg.get_border_flag() == expected_flag

    def test_invalid_interpolation_raises(self):
        """Unknown interpolation should raise ValueError."""
        cfg = WarpConfig(interpolation="super_cubic")
        with pytest.raises(ValueError, match="interpolation"):
            cfg.get_interpolation_flag()

    def test_invalid_border_raises(self):
        """Unknown border mode should raise ValueError."""
        cfg = WarpConfig(border_mode="fancy_border")
        with pytest.raises(ValueError, match="border"):
            cfg.get_border_flag()

    def test_custom_border_value(self):
        """Custom border value should be used in warp."""
        img = _make_gray_image(64, 64, value=200)
        # Translate image so borders are exposed
        M = np.array([[1.0, 0.0, 30.0],
                       [0.0, 1.0, 30.0]], dtype=np.float64)

        result = warp_image(
            img, M, TransformModel.AFFINE,
            config=WarpConfig(border_mode="constant", border_value=42.0),
        )

        assert result.success is True
        # Top-left corner should have border value
        assert result.registered_image[0, 0] == 42

    def test_nearest_interpolation(self):
        """Nearest-neighbor interpolation should produce sharp output."""
        img = _make_checkerboard(64, 64, block=8)
        identity = np.eye(2, 3, dtype=np.float64)

        result = warp_image(
            img, identity, TransformModel.AFFINE,
            config=WarpConfig(interpolation="nearest"),
        )

        assert result.success is True
        # With identity + nearest, result should be identical
        np.testing.assert_array_equal(result.registered_image, img)

    def test_replicate_border(self):
        """Replicate border should extend edge pixels."""
        img = _make_gradient_image(64, 64)
        M = np.array([[1.0, 0.0, 10.0],
                       [0.0, 1.0, 0.0]], dtype=np.float64)

        result = warp_image(
            img, M, TransformModel.AFFINE,
            config=WarpConfig(border_mode="replicate"),
        )

        assert result.success is True
        # Left edge should be replicated (value from column 0 of original)
        # Column 0 in the gradient is ~0, replicated into exposed area
        assert result.registered_image[32, 0] == img[32, 0]

    def test_default_config(self):
        """Default WarpConfig values should be sensible."""
        cfg = WarpConfig()
        assert cfg.interpolation == "linear"
        assert cfg.border_mode == "constant"
        assert cfg.border_value == 0.0
        assert cfg.output_width is None
        assert cfg.output_height is None


# =========================================================================
# 5. END-TO-END CLASSICAL BASELINE
# =========================================================================

class TestEndToEndRegistration:
    """End-to-end: synthetic image → SIFT → FLANN → Lowe → MAGSAC++ → warp → alignment.

    Tests the full classical registration pipeline including image output.
    """

    def test_translated_image_registration(self):
        """Register a translated image and verify geometric alignment."""
        img = _make_structured_image()
        tx, ty = 15, 10
        M_true = np.float32([[1, 0, tx], [0, 1, ty]])
        translated = cv2.warpAffine(img, M_true, (img.shape[1], img.shape[0]))

        # --- Feature extraction ---
        ref_feat = _make_feature_image(img)
        tgt_feat = _make_feature_image(translated)

        ref_sift = extract_sift(ref_feat)
        tgt_sift = extract_sift(tgt_feat)
        assert ref_sift.num_keypoints > 0
        assert tgt_sift.num_keypoints > 0

        # --- Matching ---
        raw_matches = flann_knn_match(ref_sift, tgt_sift)
        match_result = apply_ratio_test(raw_matches, ref_sift, tgt_sift)
        assert match_result.accepted > 0

        # --- Geometric estimation ---
        geo_result = estimate_transform(
            match_result.matches, model=TransformModel.AFFINE,
        )
        assert geo_result.success is True

        # --- Image warping ---
        reg_result = warp_image(
            img,  # source = reference image
            geo_result.transform_matrix,
            geo_result.transform_model,
            target_width=translated.shape[1],
            target_height=translated.shape[0],
        )

        assert reg_result.success is True
        assert reg_result.registered_image is not None
        assert reg_result.registered_image.shape == translated.shape

        # --- Verify alignment ---
        # The registered image should be close to the translated image
        # in the overlapping interior region (avoid borders).
        # Use a generous tolerance because SIFT localization + interpolation
        # introduce some error.
        margin = 30  # avoid borders where warping introduces artefacts
        interior_registered = reg_result.registered_image[
            margin:-margin, margin:-margin
        ]
        interior_target = translated[margin:-margin, margin:-margin]

        # Compute pixel-level similarity (mean absolute difference)
        mae = np.mean(np.abs(
            interior_registered.astype(np.float64)
            - interior_target.astype(np.float64)
        ))

        # MAE should be small — these are the same image under a simple translation
        assert mae < 30.0, f"Mean absolute error too high: {mae:.1f}"

    def test_rotated_image_registration(self):
        """Register a rotated image through the full pipeline."""
        img = _make_structured_image()
        h, w = img.shape[:2]
        M_true = cv2.getRotationMatrix2D((w / 2, h / 2), 5.0, 1.0)
        rotated = cv2.warpAffine(img, M_true, (w, h))

        ref_feat = _make_feature_image(img)
        tgt_feat = _make_feature_image(rotated)

        ref_sift = extract_sift(ref_feat)
        tgt_sift = extract_sift(tgt_feat)

        raw_matches = flann_knn_match(ref_sift, tgt_sift)
        match_result = apply_ratio_test(raw_matches, ref_sift, tgt_sift)

        if match_result.accepted >= 3:
            geo_result = estimate_transform(
                match_result.matches, model=TransformModel.AFFINE,
            )

            if geo_result.success:
                reg_result = warp_image(
                    img,
                    geo_result.transform_matrix,
                    geo_result.transform_model,
                    target_width=w,
                    target_height=h,
                )

                assert reg_result.success is True
                assert reg_result.registered_image is not None
                assert reg_result.registered_image.shape == rotated.shape

    def test_identity_registration_near_zero_error(self):
        """Registering an image to itself should produce near-identical output."""
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

        reg_result = warp_image(
            img,
            geo_result.transform_matrix,
            geo_result.transform_model,
        )

        assert reg_result.success is True

        # Identity match → registered image should be very close to original
        mae = np.mean(np.abs(
            reg_result.registered_image.astype(np.float64)
            - img.astype(np.float64)
        ))
        assert mae < 5.0, f"Identity registration MAE too high: {mae:.1f}"

    def test_pipeline_result_metadata(self):
        """Full pipeline result should contain expected metadata."""
        img = _make_structured_image()
        feat_img = _make_feature_image(img)

        sift_result = extract_sift(feat_img)
        raw_matches = flann_knn_match(sift_result, sift_result)
        match_result = apply_ratio_test(raw_matches, sift_result, sift_result)

        geo_result = estimate_transform(
            match_result.matches, model=TransformModel.AFFINE,
        )

        reg_result = warp_image(
            img,
            geo_result.transform_matrix,
            geo_result.transform_model,
        )

        assert reg_result.success is True
        assert reg_result.transform_model == TransformModel.AFFINE
        assert reg_result.transform_matrix is not None
        assert reg_result.output_width == img.shape[1]
        assert reg_result.output_height == img.shape[0]
        assert reg_result.source_width == img.shape[1]
        assert reg_result.source_height == img.shape[0]
        assert isinstance(reg_result.warp_config, WarpConfig)
        assert reg_result.failure_reason == ""

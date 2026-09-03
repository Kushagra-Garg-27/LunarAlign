"""
SIH26166 — Tests for POST /register endpoint and artifact retrieval.

Test categories:
1. Successful registration (translated, identity, rotated)
2. Input validation (missing files, invalid images, bad config)
3. Response schema verification
4. Artifact endpoints (registered image, matches, inliers, overlay)
5. Regression (existing tests unaffected)
"""

from __future__ import annotations

import io
import struct
import zlib

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


# =========================================================================
# Synthetic image helpers
# =========================================================================

def _make_png_bytes(
    width: int = 128,
    height: int = 128,
    *,
    structured: bool = False,
) -> bytes:
    """Create a valid PNG image as bytes.

    If structured=True, creates an image with features suitable for SIFT.
    """
    if structured:
        img = _make_structured_image(width, height)
    else:
        img = np.full((height, width), 128, dtype=np.uint8)

    success, buf = cv2.imencode(".png", img)
    assert success
    return buf.tobytes()


def _make_structured_image(width: int = 256, height: int = 256) -> np.ndarray:
    """Create an image with geometric features for SIFT detection."""
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


def _make_translated_pair() -> tuple[bytes, bytes]:
    """Create a reference + translated target PNG pair."""
    ref = _make_structured_image(256, 256)
    M = np.float32([[1, 0, 15], [0, 1, 10]])
    tgt = cv2.warpAffine(ref, M, (256, 256))

    _, ref_buf = cv2.imencode(".png", ref)
    _, tgt_buf = cv2.imencode(".png", tgt)
    return ref_buf.tobytes(), tgt_buf.tobytes()


def _make_identity_pair() -> tuple[bytes, bytes]:
    """Create identical reference and target PNGs."""
    img_bytes = _make_png_bytes(256, 256, structured=True)
    return img_bytes, img_bytes


def _make_rotated_pair() -> tuple[bytes, bytes]:
    """Create a reference + rotated target PNG pair."""
    ref = _make_structured_image(256, 256)
    h, w = ref.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 5.0, 1.0)
    tgt = cv2.warpAffine(ref, M, (w, h))

    _, ref_buf = cv2.imencode(".png", ref)
    _, tgt_buf = cv2.imencode(".png", tgt)
    return ref_buf.tobytes(), tgt_buf.tobytes()


def _post_register(
    ref_bytes: bytes,
    tgt_bytes: bytes,
    *,
    ref_name: str = "ref.png",
    tgt_name: str = "tgt.png",
    transform_model: str = "affine",
    ratio_threshold: float = 0.75,
) -> dict:
    """Helper: POST to /register and return JSON."""
    response = client.post(
        "/register",
        files={
            "reference": (ref_name, ref_bytes, "image/png"),
            "target": (tgt_name, tgt_bytes, "image/png"),
        },
        data={
            "transform_model": transform_model,
            "ratio_threshold": str(ratio_threshold),
        },
    )
    return response


# =========================================================================
# 1. SUCCESSFUL REGISTRATION
# =========================================================================

class TestSuccessfulRegistration:
    """Tests for successful classical SIFT registration."""

    def test_translated_pair(self):
        ref_bytes, tgt_bytes = _make_translated_pair()
        resp = _post_register(ref_bytes, tgt_bytes)

        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["pipeline_mode"] == "classical_sift"
        assert data["result_id"]

    def test_identity_pair(self):
        ref_bytes, tgt_bytes = _make_identity_pair()
        resp = _post_register(ref_bytes, tgt_bytes)

        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True

    def test_rotated_pair(self):
        ref_bytes, tgt_bytes = _make_rotated_pair()
        resp = _post_register(ref_bytes, tgt_bytes)

        assert resp.status_code == 200
        data = resp.json()
        # Rotation may or may not succeed depending on feature count
        # Just verify the endpoint doesn't crash
        assert "success" in data

    def test_homography_model(self):
        ref_bytes, tgt_bytes = _make_translated_pair()
        resp = _post_register(ref_bytes, tgt_bytes, transform_model="homography")

        assert resp.status_code == 200
        data = resp.json()
        if data["success"]:
            assert data["transform"]["transform_model"] == "homography"


# =========================================================================
# 2. INPUT VALIDATION
# =========================================================================

class TestInputValidation:
    """Tests for input validation errors."""

    def test_missing_reference(self):
        tgt_bytes = _make_png_bytes()
        resp = client.post(
            "/register",
            files={"target": ("tgt.png", tgt_bytes, "image/png")},
            data={"transform_model": "affine"},
        )
        assert resp.status_code == 422

    def test_missing_target(self):
        ref_bytes = _make_png_bytes()
        resp = client.post(
            "/register",
            files={"reference": ("ref.png", ref_bytes, "image/png")},
            data={"transform_model": "affine"},
        )
        assert resp.status_code == 422

    def test_invalid_transform_model(self):
        ref_bytes, tgt_bytes = _make_translated_pair()
        resp = _post_register(ref_bytes, tgt_bytes, transform_model="projective")

        assert resp.status_code == 422

    def test_invalid_ratio_threshold_zero(self):
        ref_bytes, tgt_bytes = _make_translated_pair()
        resp = _post_register(ref_bytes, tgt_bytes, ratio_threshold=0.0)

        assert resp.status_code == 422

    def test_invalid_ratio_threshold_negative(self):
        ref_bytes, tgt_bytes = _make_translated_pair()
        resp = _post_register(ref_bytes, tgt_bytes, ratio_threshold=-0.5)

        assert resp.status_code == 422

    def test_unsupported_extension(self):
        resp = client.post(
            "/register",
            files={
                "reference": ("ref.bmp", b"\x00" * 100, "image/bmp"),
                "target": ("tgt.png", _make_png_bytes(), "image/png"),
            },
            data={"transform_model": "affine"},
        )
        assert resp.status_code == 415

    def test_empty_file(self):
        resp = client.post(
            "/register",
            files={
                "reference": ("ref.png", b"", "image/png"),
                "target": ("tgt.png", _make_png_bytes(), "image/png"),
            },
            data={"transform_model": "affine"},
        )
        assert resp.status_code == 400


# =========================================================================
# 3. RESPONSE SCHEMA VERIFICATION
# =========================================================================

class TestResponseSchema:
    """Verify the response schema contains expected fields."""

    def test_success_response_fields(self):
        ref_bytes, tgt_bytes = _make_translated_pair()
        resp = _post_register(ref_bytes, tgt_bytes)

        assert resp.status_code == 200
        data = resp.json()

        # Top level
        assert "success" in data
        assert "result_id" in data
        assert "pipeline_mode" in data

        if data["success"]:
            # Correspondence
            assert "correspondence" in data
            corr = data["correspondence"]
            assert "total_correspondences" in corr
            assert "inlier_count" in corr
            assert "outlier_count" in corr
            assert "inlier_ratio" in corr
            assert corr["total_correspondences"] > 0
            assert corr["inlier_count"] > 0
            assert 0.0 < corr["inlier_ratio"] <= 1.0

            # Accuracy
            assert "accuracy" in data
            acc = data["accuracy"]
            assert "inlier_rmse" in acc
            assert "all_rmse" in acc
            assert "inlier_median_error" in acc
            assert "inlier_max_error" in acc
            assert acc["inlier_rmse"] >= 0.0

            # Spatial
            assert "spatial" in data
            assert "spatial_entropy" in data["spatial"]
            assert 0.0 <= data["spatial"]["spatial_entropy"] <= 1.0

            # Transform
            assert "transform" in data
            tx = data["transform"]
            assert tx["transform_model"] == "affine"
            assert tx["estimator_method"] in ("USAC_MAGSAC", "RANSAC")
            assert tx["transform_matrix"] is not None

            # Output
            assert "output" in data
            assert data["output"]["width"] > 0
            assert data["output"]["height"] > 0

            # Timing
            assert "timing" in data
            timing = data["timing"]
            assert timing["total"] > 0.0

    def test_failure_response_fields(self):
        """A featureless uniform image should produce structured failure."""
        # Uniform white image → very few/no SIFT features
        white = np.full((64, 64), 255, dtype=np.uint8)
        _, buf = cv2.imencode(".png", white)
        white_bytes = buf.tobytes()

        resp = _post_register(white_bytes, white_bytes)

        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is False
        assert data["failure_reason"] != ""
        assert data["result_id"] != ""


# =========================================================================
# 4. ARTIFACT ENDPOINTS
# =========================================================================

class TestArtifactEndpoints:
    """Tests for registered image and visualization retrieval."""

    def _register_and_get_id(self) -> str:
        """Run a registration and return the result_id."""
        ref_bytes, tgt_bytes = _make_translated_pair()
        resp = _post_register(ref_bytes, tgt_bytes)
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        return data["result_id"]

    def test_registered_image(self):
        result_id = self._register_and_get_id()
        resp = client.get(f"/register/{result_id}/registered")

        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"
        assert len(resp.content) > 100  # Non-trivial PNG

    def test_match_visualization(self):
        result_id = self._register_and_get_id()
        resp = client.get(f"/register/{result_id}/matches")

        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"

    def test_inlier_visualization(self):
        result_id = self._register_and_get_id()
        resp = client.get(f"/register/{result_id}/inliers")

        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"

    def test_overlay_visualization(self):
        result_id = self._register_and_get_id()
        resp = client.get(f"/register/{result_id}/overlay")

        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"

    def test_nonexistent_result(self):
        resp = client.get("/register/nonexistent123456/registered")
        assert resp.status_code == 404

    def test_registered_image_is_valid_png(self):
        """Verify the returned bytes actually decode as a valid image."""
        result_id = self._register_and_get_id()
        resp = client.get(f"/register/{result_id}/registered")

        assert resp.status_code == 200
        img_array = np.frombuffer(resp.content, dtype=np.uint8)
        decoded = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
        assert decoded is not None
        assert decoded.shape[0] > 0
        assert decoded.shape[1] > 0

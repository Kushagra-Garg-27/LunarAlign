"""
SIH26166 — Tests for the preprocessing layer.

All fixtures are tiny synthetic images created in-memory or as temp files.
No real Chandrayaan imagery is used.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image as PILImage

# ---------------------------------------------------------------------------
# Helpers — create temp image files
# ---------------------------------------------------------------------------


def _save_png(arr: np.ndarray, path: Path, mode: str | None = None) -> Path:
    """Save a NumPy array as a PNG via Pillow."""
    if mode is None:
        if arr.ndim == 2:
            mode = "L"
        elif arr.ndim == 3 and arr.shape[2] == 3:
            mode = "RGB"
        elif arr.ndim == 3 and arr.shape[2] == 4:
            mode = "RGBA"
        else:
            mode = "L"
    img = PILImage.fromarray(arr, mode=mode)
    img.save(path)
    return path


def _save_jpeg(arr: np.ndarray, path: Path) -> Path:
    """Save a NumPy array as JPEG (must be RGB uint8)."""
    img = PILImage.fromarray(arr, mode="RGB")
    img.save(path, format="JPEG")
    return path


def _save_tiff_pillow(arr: np.ndarray, path: Path, mode: str = "L") -> Path:
    """Save a NumPy array as TIFF via Pillow."""
    img = PILImage.fromarray(arr, mode=mode)
    img.save(path, format="TIFF")
    return path


def _save_multiband_tiff(arr: np.ndarray, path: Path) -> Path:
    """Save a multi-band array as GeoTIFF via rasterio.

    arr shape: (H, W, C) — bands-last.
    """
    import rasterio
    from rasterio.transform import from_bounds

    h, w, bands = arr.shape
    transform = from_bounds(0, 0, w, h, w, h)
    with rasterio.open(
        path, "w",
        driver="GTiff",
        height=h, width=w,
        count=bands,
        dtype=arr.dtype,
        transform=transform,
    ) as dst:
        for b in range(bands):
            dst.write(arr[:, :, b], b + 1)
    return path


# ---------------------------------------------------------------------------
# IMAGE I/O TESTS
# ---------------------------------------------------------------------------

class TestImageIO:
    """Tests for backend.preprocessing.io.load_image."""

    def test_load_png_rgb(self, tmp_path):
        """Load a small RGB PNG."""
        from backend.preprocessing.io import load_image

        arr = np.random.randint(0, 255, (8, 12, 3), dtype=np.uint8)
        path = _save_png(arr, tmp_path / "test.png", mode="RGB")
        raw = load_image(path)

        assert raw.width == 12
        assert raw.height == 8
        assert raw.num_bands == 3
        assert raw.dtype == np.uint8
        assert raw.source_format == "png"
        assert raw.data.shape == (8, 12, 3)
        np.testing.assert_array_equal(raw.data, arr)

    def test_load_png_grayscale(self, tmp_path):
        """Load a grayscale PNG."""
        from backend.preprocessing.io import load_image

        arr = np.random.randint(0, 255, (10, 10), dtype=np.uint8)
        path = _save_png(arr, tmp_path / "gray.png", mode="L")
        raw = load_image(path)

        assert raw.width == 10
        assert raw.height == 10
        assert raw.num_bands == 1
        assert raw.data.ndim == 2

    def test_load_jpeg(self, tmp_path):
        """Load a small JPEG image."""
        from backend.preprocessing.io import load_image

        arr = np.random.randint(0, 255, (16, 20, 3), dtype=np.uint8)
        path = _save_jpeg(arr, tmp_path / "test.jpg")
        raw = load_image(path)

        assert raw.width == 20
        assert raw.height == 16
        assert raw.num_bands == 3
        assert raw.source_format == "jpeg"
        # JPEG is lossy so we cannot assert pixel equality

    def test_load_tiff_pillow(self, tmp_path):
        """Load a simple single-band TIFF via Pillow fallback."""
        from backend.preprocessing.io import load_image

        arr = np.random.randint(0, 255, (6, 8), dtype=np.uint8)
        path = _save_tiff_pillow(arr, tmp_path / "test.tif")
        raw = load_image(path)

        assert raw.width == 8
        assert raw.height == 6
        assert raw.num_bands == 1
        assert raw.data.ndim == 2

    def test_load_tiff_rgb(self, tmp_path):
        """Load a 3-band RGB TIFF."""
        from backend.preprocessing.io import load_image

        arr = np.random.randint(0, 255, (10, 12, 3), dtype=np.uint8)
        path = _save_tiff_pillow(arr, tmp_path / "rgb.tif", mode="RGB")
        raw = load_image(path)

        assert raw.width == 12
        assert raw.height == 10
        assert raw.num_bands == 3

    def test_load_multiband_tiff_rasterio(self, tmp_path):
        """Load a 5-band GeoTIFF via rasterio."""
        from backend.preprocessing.io import load_image

        arr = np.random.randint(0, 1000, (8, 10, 5), dtype=np.uint16)
        path = _save_multiband_tiff(arr, tmp_path / "multi.tif")
        raw = load_image(path)

        assert raw.width == 10
        assert raw.height == 8
        assert raw.num_bands == 5
        assert raw.dtype == np.uint16
        assert raw.source_format == "rasterio"
        assert raw.data.shape == (8, 10, 5)
        np.testing.assert_array_equal(raw.data, arr)

    def test_load_tiff_preserves_uint16(self, tmp_path):
        """dtype should be preserved, not silently cast to uint8."""
        from backend.preprocessing.io import load_image

        arr = np.random.randint(0, 65535, (4, 4, 3), dtype=np.uint16)
        path = _save_multiband_tiff(arr, tmp_path / "u16.tif")
        raw = load_image(path)

        assert raw.dtype == np.uint16

    def test_load_nonexistent_file(self):
        """Loading a nonexistent file should raise FileNotFoundError."""
        from backend.preprocessing.io import load_image

        with pytest.raises(FileNotFoundError):
            load_image("/nonexistent/image.png")

    def test_load_unsupported_format(self, tmp_path):
        """Loading an unsupported format should raise ValueError."""
        from backend.preprocessing.io import load_image

        path = tmp_path / "test.bmp"
        path.write_bytes(b"\x00" * 100)
        with pytest.raises(ValueError, match="Unsupported"):
            load_image(path)

    def test_load_corrupted_png(self, tmp_path):
        """Loading a corrupted file should raise ValueError."""
        from backend.preprocessing.io import load_image

        path = tmp_path / "corrupt.png"
        path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)
        with pytest.raises(ValueError):
            load_image(path)

    def test_dimensions_correct(self, tmp_path):
        """Width corresponds to columns, height to rows."""
        from backend.preprocessing.io import load_image

        # Deliberately non-square: 5 rows x 13 cols
        arr = np.zeros((5, 13, 3), dtype=np.uint8)
        path = _save_png(arr, tmp_path / "rect.png", mode="RGB")
        raw = load_image(path)

        assert raw.height == 5
        assert raw.width == 13
        assert raw.data.shape == (5, 13, 3)


# ---------------------------------------------------------------------------
# GRAYSCALE CONVERSION TESTS
# ---------------------------------------------------------------------------

class TestGrayscaleConversion:
    """Tests for backend.preprocessing.grayscale.to_feature_image."""

    def test_rgb_to_gray(self):
        """RGB image should become a 2-D float32 array."""
        from backend.preprocessing.datamodel import RawImage
        from backend.preprocessing.grayscale import to_feature_image

        data = np.random.randint(0, 255, (10, 12, 3), dtype=np.uint8)
        raw = RawImage(data=data, width=12, height=10, num_bands=3,
                       dtype=np.dtype("uint8"), source_format="png")

        feat = to_feature_image(raw)

        assert feat.data.ndim == 2
        assert feat.data.shape == (10, 12)
        assert feat.data.dtype == np.float32
        assert feat.width == 12
        assert feat.height == 10
        assert feat.conversion_method == "luminance_bt709"

    def test_grayscale_remains_2d(self):
        """Single-band image should pass through as 2-D float32."""
        from backend.preprocessing.datamodel import RawImage
        from backend.preprocessing.grayscale import to_feature_image

        data = np.random.randint(0, 255, (8, 8), dtype=np.uint8)
        raw = RawImage(data=data, width=8, height=8, num_bands=1,
                       dtype=np.dtype("uint8"), source_format="tiff")

        feat = to_feature_image(raw)

        assert feat.data.ndim == 2
        assert feat.data.shape == (8, 8)
        assert feat.conversion_method == "passthrough"

    def test_rgba_drops_alpha(self):
        """RGBA should drop alpha and convert RGB to grayscale."""
        from backend.preprocessing.datamodel import RawImage
        from backend.preprocessing.grayscale import to_feature_image

        data = np.random.randint(0, 255, (6, 6, 4), dtype=np.uint8)
        raw = RawImage(data=data, width=6, height=6, num_bands=4,
                       dtype=np.dtype("uint8"), source_format="png")

        feat = to_feature_image(raw)

        assert feat.data.ndim == 2
        assert feat.data.shape == (6, 6)
        assert "alpha_dropped" in feat.conversion_method

    def test_multiband_to_2d(self):
        """Multi-band (>4) should use band mean → 2-D."""
        from backend.preprocessing.datamodel import RawImage
        from backend.preprocessing.grayscale import to_feature_image

        data = np.random.randint(0, 1000, (4, 6, 7), dtype=np.uint16)
        raw = RawImage(data=data, width=6, height=4, num_bands=7,
                       dtype=np.dtype("uint16"), source_format="rasterio")

        feat = to_feature_image(raw)

        assert feat.data.ndim == 2
        assert feat.data.shape == (4, 6)
        assert feat.data.dtype == np.float32
        assert "band_mean" in feat.conversion_method

    def test_bt709_luminance_correctness(self):
        """BT.709 weights should be applied correctly."""
        from backend.preprocessing.datamodel import RawImage
        from backend.preprocessing.grayscale import to_feature_image

        # Pure red pixel
        data = np.array([[[255, 0, 0]]], dtype=np.uint8)
        raw = RawImage(data=data, width=1, height=1, num_bands=3,
                       dtype=np.dtype("uint8"), source_format="png")
        feat = to_feature_image(raw)
        expected = 0.2126 * 255
        np.testing.assert_allclose(feat.data[0, 0], expected, atol=0.5)

    def test_original_data_not_modified(self):
        """Conversion should not mutate the raw image data."""
        from backend.preprocessing.datamodel import RawImage
        from backend.preprocessing.grayscale import to_feature_image

        data = np.random.randint(0, 255, (4, 4, 3), dtype=np.uint8)
        original = data.copy()
        raw = RawImage(data=data, width=4, height=4, num_bands=3,
                       dtype=np.dtype("uint8"), source_format="png")

        to_feature_image(raw)

        np.testing.assert_array_equal(raw.data, original)


# ---------------------------------------------------------------------------
# NORMALIZATION TESTS
# ---------------------------------------------------------------------------

class TestNormalization:

    def test_minmax_normal(self):
        """Min-max should map to [0, 1]."""
        from backend.preprocessing.normalize import normalize_minmax

        arr = np.array([10, 50, 100, 200], dtype=np.float32)
        out = normalize_minmax(arr)

        assert out.dtype == np.float32
        np.testing.assert_allclose(out.min(), 0.0)
        np.testing.assert_allclose(out.max(), 1.0)

    def test_minmax_constant_image(self):
        """Constant image should return array of out_range[0]."""
        from backend.preprocessing.normalize import normalize_minmax

        arr = np.full((4, 4), 42.0, dtype=np.float32)
        out = normalize_minmax(arr)

        np.testing.assert_array_equal(out, np.zeros((4, 4), dtype=np.float32))

    def test_minmax_custom_range(self):
        """Custom output range should be respected."""
        from backend.preprocessing.normalize import normalize_minmax

        arr = np.array([0, 255], dtype=np.float32)
        out = normalize_minmax(arr, out_range=(0.0, 255.0))

        np.testing.assert_allclose(out[0], 0.0)
        np.testing.assert_allclose(out[1], 255.0)

    def test_minmax_preserves_original(self):
        """Original array should not be modified."""
        from backend.preprocessing.normalize import normalize_minmax

        arr = np.array([10, 50, 200], dtype=np.float32)
        original = arr.copy()
        normalize_minmax(arr)
        np.testing.assert_array_equal(arr, original)

    def test_percentile_normal(self):
        """Percentile normalization should clip outliers."""
        from backend.preprocessing.normalize import normalize_percentile

        arr = np.array([0, 1, 2, 3, 4, 5, 100], dtype=np.float32)
        out = normalize_percentile(arr, low_pct=0, high_pct=80)

        # Value 100 should be clipped
        np.testing.assert_allclose(out.max(), 1.0, atol=0.05)

    def test_percentile_constant(self):
        """Constant image with percentile norm should return fill value."""
        from backend.preprocessing.normalize import normalize_percentile

        arr = np.full((3, 3), 50.0, dtype=np.float32)
        out = normalize_percentile(arr)

        np.testing.assert_array_equal(out, np.zeros((3, 3), dtype=np.float32))

    def test_percentile_preserves_original(self):
        """Original array should not be modified."""
        from backend.preprocessing.normalize import normalize_percentile

        arr = np.array([0, 50, 100, 200, 255], dtype=np.float32)
        original = arr.copy()
        normalize_percentile(arr)
        np.testing.assert_array_equal(arr, original)

    def test_minmax_2d_image(self):
        """Works on 2-D arrays (typical feature image shape)."""
        from backend.preprocessing.normalize import normalize_minmax

        arr = np.random.rand(10, 10).astype(np.float32) * 1000
        out = normalize_minmax(arr)

        assert out.shape == (10, 10)
        np.testing.assert_allclose(out.min(), 0.0, atol=1e-6)
        np.testing.assert_allclose(out.max(), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# CLAHE TESTS
# ---------------------------------------------------------------------------

class TestCLAHE:

    def test_output_dimensions_unchanged(self):
        """CLAHE should not change image dimensions."""
        from backend.preprocessing.datamodel import FeatureImage
        from backend.preprocessing.clahe import apply_clahe

        data = np.random.rand(32, 48).astype(np.float32) * 255
        feat = FeatureImage(data=data, width=48, height=32,
                            source_dtype=np.dtype("uint8"),
                            conversion_method="passthrough")

        result = apply_clahe(feat)

        assert result.data.shape == (32, 48)
        assert result.width == 48
        assert result.height == 32

    def test_output_is_valid_numeric(self):
        """CLAHE output should be finite float32."""
        from backend.preprocessing.datamodel import FeatureImage
        from backend.preprocessing.clahe import apply_clahe

        data = np.random.rand(16, 16).astype(np.float32) * 200
        feat = FeatureImage(data=data, width=16, height=16,
                            source_dtype=np.dtype("uint8"),
                            conversion_method="passthrough")

        result = apply_clahe(feat)

        assert result.data.dtype == np.float32
        assert np.all(np.isfinite(result.data))

    def test_does_not_modify_original(self):
        """CLAHE should not modify the input FeatureImage data."""
        from backend.preprocessing.datamodel import FeatureImage
        from backend.preprocessing.clahe import apply_clahe

        data = np.random.rand(16, 16).astype(np.float32) * 200
        original = data.copy()
        feat = FeatureImage(data=data, width=16, height=16,
                            source_dtype=np.dtype("uint8"),
                            conversion_method="passthrough")

        apply_clahe(feat)

        np.testing.assert_array_equal(feat.data, original)

    def test_constant_image_is_noop(self):
        """CLAHE on a constant image should not crash."""
        from backend.preprocessing.datamodel import FeatureImage
        from backend.preprocessing.clahe import apply_clahe

        data = np.full((16, 16), 100.0, dtype=np.float32)
        feat = FeatureImage(data=data, width=16, height=16,
                            source_dtype=np.dtype("uint8"),
                            conversion_method="passthrough")

        result = apply_clahe(feat)

        assert result.data.shape == (16, 16)

    def test_conversion_method_tracked(self):
        """Conversion method should include '+clahe' suffix."""
        from backend.preprocessing.datamodel import FeatureImage
        from backend.preprocessing.clahe import apply_clahe

        data = np.random.rand(16, 16).astype(np.float32) * 200
        feat = FeatureImage(data=data, width=16, height=16,
                            source_dtype=np.dtype("uint8"),
                            conversion_method="luminance_bt709")

        result = apply_clahe(feat)

        assert result.conversion_method == "luminance_bt709+clahe"


# ---------------------------------------------------------------------------
# METADATA EXTRACTION TESTS
# ---------------------------------------------------------------------------

class TestMetadataExtraction:

    def test_png_metadata(self, tmp_path):
        """Extract metadata from a PNG file."""
        from backend.preprocessing.metadata import extract_image_metadata

        arr = np.zeros((20, 30, 3), dtype=np.uint8)
        path = _save_png(arr, tmp_path / "test.png", mode="RGB")
        meta = extract_image_metadata(path)

        assert meta["width"] == 30
        assert meta["height"] == 20
        assert meta["num_bands"] == 3
        assert meta["image_dtype"] == "uint8"

    def test_grayscale_png_metadata(self, tmp_path):
        """Grayscale PNG should report 1 band."""
        from backend.preprocessing.metadata import extract_image_metadata

        arr = np.zeros((10, 10), dtype=np.uint8)
        path = _save_png(arr, tmp_path / "gray.png", mode="L")
        meta = extract_image_metadata(path)

        assert meta["num_bands"] == 1
        assert meta["image_dtype"] == "uint8"

    def test_multiband_tiff_metadata(self, tmp_path):
        """Multi-band TIFF metadata via rasterio."""
        from backend.preprocessing.metadata import extract_image_metadata

        arr = np.zeros((4, 6, 5), dtype=np.uint16)
        path = _save_multiband_tiff(arr, tmp_path / "multi.tif")
        meta = extract_image_metadata(path)

        assert meta["width"] == 6
        assert meta["height"] == 4
        assert meta["num_bands"] == 5
        assert meta["image_dtype"] == "uint16"


# ---------------------------------------------------------------------------
# END-TO-END: load → convert → normalize → clahe
# ---------------------------------------------------------------------------

class TestEndToEnd:

    def test_full_pipeline(self, tmp_path):
        """Load an image, convert to feature image, normalize, apply CLAHE."""
        from backend.preprocessing.io import load_image
        from backend.preprocessing.grayscale import to_feature_image
        from backend.preprocessing.normalize import normalize_minmax
        from backend.preprocessing.clahe import apply_clahe
        from backend.preprocessing.datamodel import FeatureImage

        # Create a non-trivial RGB image
        arr = np.random.randint(10, 240, (32, 48, 3), dtype=np.uint8)
        path = _save_png(arr, tmp_path / "e2e.png", mode="RGB")

        raw = load_image(path)
        assert raw.data.shape == (32, 48, 3)

        feat = to_feature_image(raw)
        assert feat.data.shape == (32, 48)
        assert feat.data.dtype == np.float32

        normed = normalize_minmax(feat.data)
        assert normed.min() >= 0.0
        assert normed.max() <= 1.0

        # CLAHE needs values in a reasonable range
        feat_for_clahe = FeatureImage(
            data=feat.data,
            width=feat.width, height=feat.height,
            source_dtype=feat.source_dtype,
            conversion_method=feat.conversion_method,
        )
        clahe_result = apply_clahe(feat_for_clahe)
        assert clahe_result.data.shape == (32, 48)
        assert np.all(np.isfinite(clahe_result.data))

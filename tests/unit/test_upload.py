"""
SIH26166 — Tests for POST /upload endpoint.

Uses small in-memory test fixtures (no real Chandrayaan imagery).
"""

from __future__ import annotations

import io
import os
import struct
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.core.config import UPLOAD_DIR, MAX_UPLOAD_SIZE_BYTES

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers — tiny valid image bytes
# ---------------------------------------------------------------------------

def _minimal_png() -> bytes:
    """Return the smallest valid PNG file (1×1 transparent pixel)."""
    # Minimal valid PNG: signature + IHDR + IDAT + IEND
    return (
        b"\x89PNG\r\n\x1a\n"  # PNG signature
        # IHDR chunk
        b"\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01"   # width = 1
        b"\x00\x00\x00\x01"   # height = 1
        b"\x08\x02"           # 8-bit RGB
        b"\x00\x00\x00"       # compression, filter, interlace
        b"\x90wS\xde"         # CRC
        # IDAT chunk (deflate of single row filter-byte + 3 bytes)
        b"\x00\x00\x00\x0cIDAT"
        b"\x08\xd7c\xf8\x0f\x00\x00\x01\x01\x00\x05"
        b"\x18\xd8N"          # CRC
        # IEND chunk
        b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )


def _minimal_jpeg() -> bytes:
    """Return a minimal valid JFIF/JPEG (just the SOI + EOI markers)."""
    # This is technically the smallest valid JPEG structure.
    return b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9"


def _minimal_tiff() -> bytes:
    """Return a minimal TIFF header (little-endian, 8-byte header)."""
    # II (little-endian) + magic 42 + offset to first IFD (8)
    # Followed by IFD entry count = 0 and next IFD offset = 0
    return b"II" + struct.pack("<H", 42) + struct.pack("<I", 8) + struct.pack("<H", 0) + struct.pack("<I", 0)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_uploads():
    """Remove any files created in the uploads dir during each test."""
    yield
    for f in UPLOAD_DIR.glob("*"):
        if f.name == ".gitkeep":
            continue
        f.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 1. Successful upload
# ---------------------------------------------------------------------------

def test_successful_upload_png():
    """Two valid PNG images should upload successfully."""
    ref_bytes = _minimal_png()
    tgt_bytes = _minimal_png()

    response = client.post(
        "/upload",
        files={
            "reference": ("ref.png", io.BytesIO(ref_bytes), "image/png"),
            "target": ("tgt.png", io.BytesIO(tgt_bytes), "image/png"),
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"

    for key in ("reference", "target"):
        meta = data[key]
        assert meta["extension"] == ".png"
        assert meta["content_type"] == "image/png"
        assert meta["size_bytes"] > 0
        assert meta["stored_filename"].endswith(".png")
        assert meta["original_filename"] in ("ref.png", "tgt.png")
        # Preprocessing layer now populates dimensions for valid images
        assert meta["width"] == 1
        assert meta["height"] == 1
        assert meta["num_bands"] == 3  # minimal PNG is RGB
        assert meta["image_dtype"] == "uint8"


def test_successful_upload_jpeg():
    """JPEG files should be accepted."""
    ref_bytes = _minimal_jpeg()
    tgt_bytes = _minimal_jpeg()

    response = client.post(
        "/upload",
        files={
            "reference": ("moon_a.jpg", io.BytesIO(ref_bytes), "image/jpeg"),
            "target": ("moon_b.jpeg", io.BytesIO(tgt_bytes), "image/jpeg"),
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["reference"]["extension"] == ".jpg"
    assert data["target"]["extension"] == ".jpeg"


def test_successful_upload_tiff():
    """TIFF files should be accepted."""
    ref_bytes = _minimal_tiff()
    tgt_bytes = _minimal_tiff()

    response = client.post(
        "/upload",
        files={
            "reference": ("ref.tif", io.BytesIO(ref_bytes), "image/tiff"),
            "target": ("tgt.tiff", io.BytesIO(tgt_bytes), "image/tiff"),
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["reference"]["extension"] == ".tif"
    assert data["target"]["extension"] == ".tiff"


# ---------------------------------------------------------------------------
# 2 & 3. Missing reference / target
# ---------------------------------------------------------------------------

def test_missing_reference_image():
    """Omitting the reference field should return 422."""
    response = client.post(
        "/upload",
        files={
            "target": ("tgt.png", io.BytesIO(_minimal_png()), "image/png"),
        },
    )
    assert response.status_code == 422


def test_missing_target_image():
    """Omitting the target field should return 422."""
    response = client.post(
        "/upload",
        files={
            "reference": ("ref.png", io.BytesIO(_minimal_png()), "image/png"),
        },
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# 4. Unsupported file extension
# ---------------------------------------------------------------------------

def test_unsupported_extension_bmp():
    """A .bmp file should be rejected with 415."""
    response = client.post(
        "/upload",
        files={
            "reference": ("ref.bmp", io.BytesIO(b"\x00" * 100), "image/bmp"),
            "target": ("tgt.png", io.BytesIO(_minimal_png()), "image/png"),
        },
    )
    assert response.status_code == 415


def test_unsupported_extension_exe():
    """An executable disguised as an upload should be rejected."""
    response = client.post(
        "/upload",
        files={
            "reference": ("malware.exe", io.BytesIO(b"MZ" + b"\x00" * 100), "application/octet-stream"),
            "target": ("tgt.png", io.BytesIO(_minimal_png()), "image/png"),
        },
    )
    assert response.status_code == 415


# ---------------------------------------------------------------------------
# 5. Empty file
# ---------------------------------------------------------------------------

def test_empty_file():
    """An empty file should be rejected with 400."""
    response = client.post(
        "/upload",
        files={
            "reference": ("empty.png", io.BytesIO(b""), "image/png"),
            "target": ("tgt.png", io.BytesIO(_minimal_png()), "image/png"),
        },
    )
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# 6. Oversized file
# ---------------------------------------------------------------------------

def test_oversized_file():
    """A file exceeding MAX_UPLOAD_SIZE_BYTES should be rejected with 413.

    We use a generator-backed BytesIO that reports a size just over the limit
    without actually allocating 100 MB of memory.
    """

    class FakeOversizedFile(io.RawIOBase):
        """Yields zeros up to ``total`` bytes, simulating an oversized file."""

        def __init__(self, total: int):
            self._remaining = total

        def readable(self):
            return True

        def readinto(self, b):
            n = min(len(b), self._remaining)
            b[:n] = b"\x00" * n
            self._remaining -= n
            return n

    oversized = io.BufferedReader(FakeOversizedFile(MAX_UPLOAD_SIZE_BYTES + 1))

    response = client.post(
        "/upload",
        files={
            "reference": ("big.png", oversized, "image/png"),
            "target": ("tgt.png", io.BytesIO(_minimal_png()), "image/png"),
        },
    )
    assert response.status_code == 413


# ---------------------------------------------------------------------------
# 7. Files are actually stored on disk
# ---------------------------------------------------------------------------

def test_files_stored_on_disk():
    """After a successful upload the files should exist in the upload dir."""
    ref_bytes = _minimal_png()
    tgt_bytes = _minimal_png()

    response = client.post(
        "/upload",
        files={
            "reference": ("ref.png", io.BytesIO(ref_bytes), "image/png"),
            "target": ("tgt.png", io.BytesIO(tgt_bytes), "image/png"),
        },
    )
    assert response.status_code == 200
    data = response.json()

    ref_path = UPLOAD_DIR / data["reference"]["stored_filename"]
    tgt_path = UPLOAD_DIR / data["target"]["stored_filename"]

    assert ref_path.exists(), f"Reference file not found: {ref_path}"
    assert tgt_path.exists(), f"Target file not found: {tgt_path}"
    assert ref_path.stat().st_size == data["reference"]["size_bytes"]
    assert tgt_path.stat().st_size == data["target"]["size_bytes"]


# ---------------------------------------------------------------------------
# 8. Returned metadata structure
# ---------------------------------------------------------------------------

def test_metadata_structure():
    """The response should include all required metadata fields."""
    ref_bytes = _minimal_png()
    tgt_bytes = _minimal_jpeg()

    response = client.post(
        "/upload",
        files={
            "reference": ("moon_ref.png", io.BytesIO(ref_bytes), "image/png"),
            "target": ("moon_tgt.jpg", io.BytesIO(tgt_bytes), "image/jpeg"),
        },
    )
    assert response.status_code == 200
    data = response.json()

    required_fields = {
        "original_filename", "stored_filename", "size_bytes",
        "content_type", "extension", "width", "height",
        "num_bands", "image_dtype",
    }
    for key in ("reference", "target"):
        assert required_fields.issubset(data[key].keys()), (
            f"Missing fields in {key}: {required_fields - data[key].keys()}"
        )

    # Original filenames should be preserved as metadata
    assert data["reference"]["original_filename"] == "moon_ref.png"
    assert data["target"]["original_filename"] == "moon_tgt.jpg"


# ---------------------------------------------------------------------------
# 9. Stored filename is not the original filename (security)
# ---------------------------------------------------------------------------

def test_stored_filename_is_not_original():
    """The stored filename must differ from the original (UUID-based)."""
    response = client.post(
        "/upload",
        files={
            "reference": ("ref.png", io.BytesIO(_minimal_png()), "image/png"),
            "target": ("tgt.png", io.BytesIO(_minimal_png()), "image/png"),
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["reference"]["stored_filename"] != "ref.png"
    assert data["target"]["stored_filename"] != "tgt.png"


# ---------------------------------------------------------------------------
# 10. Path traversal in filename is not used for storage
# ---------------------------------------------------------------------------

def test_path_traversal_filename_ignored():
    """A malicious filename with path traversal should be ignored for storage."""
    response = client.post(
        "/upload",
        files={
            "reference": ("../../etc/passwd.png", io.BytesIO(_minimal_png()), "image/png"),
            "target": ("tgt.png", io.BytesIO(_minimal_png()), "image/png"),
        },
    )
    assert response.status_code == 200
    data = response.json()
    # Stored file should be in the uploads directory, not escaped
    stored = data["reference"]["stored_filename"]
    assert ".." not in stored
    assert "/" not in stored
    assert "\\" not in stored
    assert (UPLOAD_DIR / stored).exists()


# ---------------------------------------------------------------------------
# 11. TIFF with octet-stream content type is accepted
# ---------------------------------------------------------------------------

def test_tiff_with_octet_stream_content_type():
    """Many clients send TIFF as application/octet-stream — we should accept it."""
    ref_bytes = _minimal_tiff()
    tgt_bytes = _minimal_tiff()

    response = client.post(
        "/upload",
        files={
            "reference": ("ref.tif", io.BytesIO(ref_bytes), "application/octet-stream"),
            "target": ("tgt.tiff", io.BytesIO(tgt_bytes), "application/octet-stream"),
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["reference"]["content_type"] == "image/tiff"
    assert data["target"]["content_type"] == "image/tiff"

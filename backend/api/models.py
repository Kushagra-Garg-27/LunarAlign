"""
SIH26166 — Pydantic response models for the upload endpoint.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ImageMetadata(BaseModel):
    """Metadata for a single uploaded image."""

    original_filename: str = Field(
        ..., description="Original filename provided by the client."
    )
    stored_filename: str = Field(
        ..., description="Safe, unique filename used for storage."
    )
    size_bytes: int = Field(..., description="File size in bytes.")
    content_type: str = Field(..., description="Validated MIME content type.")
    extension: str = Field(..., description="Lowercased file extension (e.g. '.tif').")
    width: int | None = Field(
        None,
        description="Image width in pixels.",
    )
    height: int | None = Field(
        None,
        description="Image height in pixels.",
    )
    num_bands: int | None = Field(
        None,
        description=(
            "Number of spectral bands / channels "
            "(1 for grayscale, 3 for RGB, arbitrary for multi-band raster)."
        ),
    )
    image_dtype: str | None = Field(
        None,
        description=(
            "NumPy-style dtype string for the pixel data "
            "(e.g. 'uint8', 'uint16', 'float32')."
        ),
    )


class UploadResponse(BaseModel):
    """Response returned by POST /upload on success."""

    status: str = Field(default="success")
    reference: ImageMetadata
    target: ImageMetadata

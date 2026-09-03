"""
SIH26166 — Registration API endpoint and response models.

POST /register
    Accepts reference + target images via multipart/form-data.
    Orchestrates the classical SIFT registration pipeline.
    Returns structured JSON with metrics and result_id.

GET /register/{result_id}/registered
GET /register/{result_id}/matches
GET /register/{result_id}/inliers
GET /register/{result_id}/overlay
    Return diagnostic visualization images as PNG.

Pipeline mode: ``classical_sift`` (same-modality baseline only).

This endpoint does NOT support:
- cross-modal registration (OHRC ↔ IIRS, etc.)
- illumination-invariant registration
- extreme-scale registration
- sub-pixel refinement

Those belong to later stages.
"""

from __future__ import annotations

import io
import logging
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.core.config import UPLOAD_DIR
from backend.core.registration_service import run_classical_registration
from backend.core.result_store import result_store
from backend.core.validation import validate_upload

logger = logging.getLogger("sih26166.api.register")

router = APIRouter()


# ===================================================================
# Response models
# ===================================================================

class CorrespondenceMetrics(BaseModel):
    """Match correspondence counts."""
    total_correspondences: int
    inlier_count: int
    outlier_count: int
    inlier_ratio: float


class AccuracyMetrics(BaseModel):
    """Reprojection error metrics."""
    inlier_rmse: float
    all_rmse: float
    inlier_median_error: float
    inlier_max_error: float


class SpatialMetrics(BaseModel):
    """Spatial distribution metrics."""
    spatial_entropy: float = Field(
        ..., description="Normalized spatial entropy [0,1]. -1 if not computed."
    )


class TransformInfo(BaseModel):
    """Transformation estimation metadata."""
    transform_model: str
    estimator_method: str
    transform_matrix: list[list[float]] | None


class OutputInfo(BaseModel):
    """Registered output image metadata."""
    width: int
    height: int


class TimingInfo(BaseModel):
    """Per-stage timing in seconds."""
    load: float = 0.0
    preprocess: float = 0.0
    sift: float = 0.0
    matching: float = 0.0
    ratio_test: float = 0.0
    geometry: float = 0.0
    warp: float = 0.0
    evaluation: float = 0.0
    visualization: float = 0.0
    total: float = 0.0


class RegisterResponse(BaseModel):
    """Complete response for POST /register."""
    success: bool
    result_id: str
    pipeline_mode: str = "classical_sift"

    correspondence: CorrespondenceMetrics | None = None
    accuracy: AccuracyMetrics | None = None
    spatial: SpatialMetrics | None = None
    transform: TransformInfo | None = None
    output: OutputInfo | None = None
    timing: TimingInfo | None = None

    failure_reason: str = ""
    failure_stage: str = ""


# ===================================================================
# Helper: save uploaded file to disk
# ===================================================================

async def _save_upload_to_disk(file: UploadFile, label: str) -> str:
    """Validate, save upload to disk, return path string."""
    import uuid
    import os

    ext, content_type, size_bytes = await validate_upload(file)

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    stored_name = f"{uuid.uuid4().hex}{ext}"
    dest = UPLOAD_DIR / stored_name

    try:
        with open(dest, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
    except Exception:
        if dest.exists():
            dest.unlink()
        raise

    logger.info("Stored %s: %s -> %s (%d bytes)", label, file.filename, stored_name, size_bytes)
    return str(dest)


# ===================================================================
# Helper: encode image as PNG bytes
# ===================================================================

def _encode_png(image: np.ndarray) -> bytes:
    """Encode a NumPy image array as PNG bytes."""
    if image is None:
        raise ValueError("Image is None")
    success, buf = cv2.imencode(".png", image)
    if not success:
        raise ValueError("Failed to encode image as PNG")
    return buf.tobytes()


# ===================================================================
# POST /register
# ===================================================================

@router.post(
    "/register",
    response_model=RegisterResponse,
    status_code=status.HTTP_200_OK,
    summary="Run classical SIFT registration pipeline",
    description=(
        "Accepts reference and target images via multipart/form-data. "
        "Runs the full classical pipeline: SIFT → FLANN → Lowe → MAGSAC++ → warp. "
        "Returns structured metrics and a result_id for retrieving artifacts. "
        "Pipeline mode: classical_sift (same-modality baseline only)."
    ),
)
async def register_images(
    reference: UploadFile = File(..., description="Reference lunar image"),
    target: UploadFile = File(..., description="Target lunar image"),
    transform_model: str = Form("affine", description="'affine' or 'homography'"),
    ratio_threshold: float = Form(0.75, description="Lowe ratio test threshold (0-1)"),
) -> RegisterResponse:
    """Execute the classical registration pipeline."""

    # --- Validate transform_model ---
    if transform_model not in ("affine", "homography"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported transform model: '{transform_model}'. Use 'affine' or 'homography'.",
        )

    # --- Validate ratio_threshold ---
    if not (0.0 < ratio_threshold <= 1.0):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"ratio_threshold must be in (0, 1], got {ratio_threshold}.",
        )

    # --- Save uploads ---
    try:
        ref_path = await _save_upload_to_disk(reference, "reference")
        tgt_path = await _save_upload_to_disk(target, "target")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to process uploaded files: {exc}",
        )

    # --- Run pipeline ---
    try:
        pipeline_result = run_classical_registration(
            ref_path,
            tgt_path,
            transform_model=transform_model,
            ratio_threshold=ratio_threshold,
        )
    except Exception as exc:
        logger.exception("Unexpected pipeline error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline error: {exc}",
        )

    # --- Store result for artifact retrieval ---
    result_store.put(pipeline_result.result_id, pipeline_result)

    # --- Build response ---
    if not pipeline_result.success:
        return RegisterResponse(
            success=False,
            result_id=pipeline_result.result_id,
            pipeline_mode=pipeline_result.pipeline_mode,
            failure_reason=pipeline_result.failure_reason,
            failure_stage=pipeline_result.failure_stage,
            timing=TimingInfo(**pipeline_result.timings) if pipeline_result.timings else None,
        )

    summary = pipeline_result.quality_summary
    reg_img = pipeline_result.registered_image

    return RegisterResponse(
        success=True,
        result_id=pipeline_result.result_id,
        pipeline_mode=pipeline_result.pipeline_mode,
        correspondence=CorrespondenceMetrics(
            total_correspondences=summary.total_correspondences,
            inlier_count=summary.inlier_count,
            outlier_count=summary.outlier_count,
            inlier_ratio=summary.inlier_ratio,
        ),
        accuracy=AccuracyMetrics(
            inlier_rmse=summary.inlier_rmse,
            all_rmse=summary.all_rmse,
            inlier_median_error=summary.inlier_median_error,
            inlier_max_error=summary.inlier_max_error,
        ),
        spatial=SpatialMetrics(
            spatial_entropy=summary.spatial_entropy,
        ),
        transform=TransformInfo(
            transform_model=summary.transform_model,
            estimator_method=summary.estimator_method,
            transform_matrix=summary.transform_matrix,
        ),
        output=OutputInfo(
            width=reg_img.shape[1] if reg_img is not None else 0,
            height=reg_img.shape[0] if reg_img is not None else 0,
        ),
        timing=TimingInfo(**pipeline_result.timings),
    )


# ===================================================================
# Artifact endpoints
# ===================================================================

def _get_stored_result(result_id: str):
    """Retrieve a pipeline result or raise 404."""
    result = result_store.get(result_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Result '{result_id}' not found. It may have been evicted.",
        )
    return result


def _image_response(image: np.ndarray | None, label: str) -> StreamingResponse:
    """Create a PNG streaming response from a NumPy image."""
    if image is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{label} not available for this result.",
        )
    png_bytes = _encode_png(image)
    return StreamingResponse(
        io.BytesIO(png_bytes),
        media_type="image/png",
        headers={"Content-Disposition": f"inline; filename={label}.png"},
    )


@router.get(
    "/register/{result_id}/registered",
    summary="Get registered (warped) image",
    responses={200: {"content": {"image/png": {}}}},
)
async def get_registered_image(result_id: str):
    """Return the registered image as PNG."""
    result = _get_stored_result(result_id)
    return _image_response(result.registered_image, "registered")


@router.get(
    "/register/{result_id}/matches",
    summary="Get match visualization",
    responses={200: {"content": {"image/png": {}}}},
)
async def get_match_visualization(result_id: str):
    """Return the filtered match visualization as PNG."""
    result = _get_stored_result(result_id)
    return _image_response(result.match_visualization, "matches")


@router.get(
    "/register/{result_id}/inliers",
    summary="Get inlier/outlier visualization",
    responses={200: {"content": {"image/png": {}}}},
)
async def get_inlier_visualization(result_id: str):
    """Return the inlier/outlier annotated match visualization as PNG."""
    result = _get_stored_result(result_id)
    return _image_response(result.inlier_visualization, "inliers")


@router.get(
    "/register/{result_id}/overlay",
    summary="Get registration overlay",
    responses={200: {"content": {"image/png": {}}}},
)
async def get_overlay_visualization(result_id: str):
    """Return the target/registered overlay as PNG."""
    result = _get_stored_result(result_id)
    return _image_response(result.overlay_visualization, "overlay")

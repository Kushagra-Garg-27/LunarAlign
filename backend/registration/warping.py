"""
SIH26166 — Image warping for geometric registration.

Applies an estimated affine or homography transformation to an image,
producing the registered (aligned) output.

Transform direction
-------------------
The geometry layer's ``estimate_transform()`` produces a matrix M that
maps **reference image coordinates → target image coordinates**::

    cv2.estimateAffine2D(ref_pts, tgt_pts)  →  M: ref → tgt

To produce the registered image:

1. **Source image**: the reference image.
2. **Matrix**: M (ref → tgt), passed directly — no inversion.
3. **Output dimensions**: target image dimensions (or explicitly set).
4. **Result**: the reference image warped into the target coordinate frame.

OpenCV's ``warpAffine(src, M, dsize)`` and ``warpPerspective(src, M, dsize)``
by default expect the forward mapping (src → dst) and internally compute
the inverse for pixel sampling.  Therefore we pass M directly without
explicit inversion.

This is the **geometric registration output** of the classical baseline.
Sub-pixel refinement is a later stage.

This module is part of the **CLASSICAL REGISTRATION BASELINE**.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from backend.geometry.models import TransformModel
from backend.registration.models import RegistrationResult, WarpConfig

logger = logging.getLogger("sih26166.registration.warping")


# ===================================================================
# Input validation
# ===================================================================

def _validate_matrix(
    matrix: np.ndarray,
    model: TransformModel,
) -> str | None:
    """Validate the transformation matrix.

    Returns
    -------
    str | None
        Error message if invalid, ``None`` if valid.
    """
    if matrix is None:
        return "Transformation matrix is None."

    if not isinstance(matrix, np.ndarray):
        return f"Matrix must be a numpy array, got {type(matrix).__name__}."

    expected_shape = (2, 3) if model == TransformModel.AFFINE else (3, 3)
    if matrix.shape != expected_shape:
        return (
            f"Expected matrix shape {expected_shape} for {model.value}, "
            f"got {matrix.shape}."
        )

    if not np.all(np.isfinite(matrix)):
        return "Matrix contains NaN or Inf values."

    return None


def _validate_image(image: np.ndarray) -> str | None:
    """Validate the source image for warping.

    Returns
    -------
    str | None
        Error message if invalid, ``None`` if valid.
    """
    if image is None:
        return "Source image is None."

    if not isinstance(image, np.ndarray):
        return f"Image must be a numpy array, got {type(image).__name__}."

    if image.ndim not in (2, 3):
        return f"Image must be 2-D (grayscale) or 3-D (multi-channel), got {image.ndim}-D."

    if image.size == 0:
        return "Source image is empty (zero pixels)."

    return None


def _validate_output_size(width: int, height: int) -> str | None:
    """Validate output dimensions.

    Returns
    -------
    str | None
        Error message if invalid, ``None`` if valid.
    """
    if width <= 0 or height <= 0:
        return f"Output dimensions must be positive, got ({width}, {height})."

    return None


# ===================================================================
# Core warping
# ===================================================================

def warp_image(
    image: np.ndarray,
    matrix: np.ndarray,
    model: TransformModel,
    *,
    config: WarpConfig | None = None,
    target_width: int | None = None,
    target_height: int | None = None,
) -> RegistrationResult:
    """Warp an image using the estimated transformation.

    This is the main entry point for image registration output.

    Transform direction
    -------------------
    The ``matrix`` is expected to map **source (reference) image
    coordinates → destination (target) image coordinates**.  This is
    the matrix produced directly by ``geometry.estimation.estimate_transform()``.

    **No matrix inversion is performed.**  The matrix is passed
    directly to ``cv2.warpAffine`` / ``cv2.warpPerspective``.

    Parameters
    ----------
    image : np.ndarray
        Source image to warp (the reference image).
        Can be 2-D (grayscale) or 3-D (multi-channel, channels-last).
    matrix : np.ndarray
        Transformation matrix.
        - Affine: shape ``(2, 3)``
        - Homography: shape ``(3, 3)``
    model : TransformModel
        Which transformation model the matrix represents.
    config : WarpConfig | None
        Warp configuration (interpolation, border, output size).
        Uses defaults if ``None``.
    target_width : int | None
        Width of the target/reference frame for output dimensions.
        Used when ``config.output_width`` is ``None``.
    target_height : int | None
        Height of the target/reference frame for output dimensions.
        Used when ``config.output_height`` is ``None``.

    Returns
    -------
    RegistrationResult
        Complete registration result with the warped image.
    """
    if config is None:
        config = WarpConfig()

    # --- Source image dimensions ---
    src_h, src_w = image.shape[:2] if isinstance(image, np.ndarray) and image.ndim >= 2 else (0, 0)

    # --- Validate inputs ---
    img_err = _validate_image(image)
    if img_err is not None:
        logger.error("Image validation failed: %s", img_err)
        return RegistrationResult(
            success=False,
            registered_image=None,
            transform_model=model,
            transform_matrix=matrix if isinstance(matrix, np.ndarray) else None,
            output_width=0,
            output_height=0,
            source_width=src_w,
            source_height=src_h,
            warp_config=config,
            failure_reason=img_err,
        )

    mat_err = _validate_matrix(matrix, model)
    if mat_err is not None:
        logger.error("Matrix validation failed: %s", mat_err)
        return RegistrationResult(
            success=False,
            registered_image=None,
            transform_model=model,
            transform_matrix=matrix if isinstance(matrix, np.ndarray) else None,
            output_width=0,
            output_height=0,
            source_width=src_w,
            source_height=src_h,
            warp_config=config,
            failure_reason=mat_err,
        )

    # --- Determine output dimensions ---
    # Priority: config explicit > target dimensions > source dimensions
    # Use `is not None` checks, not truthiness, so explicit 0 is caught
    # by validation rather than silently falling through.
    if config.output_width is not None:
        out_w = config.output_width
    elif target_width is not None:
        out_w = target_width
    else:
        out_w = src_w

    if config.output_height is not None:
        out_h = config.output_height
    elif target_height is not None:
        out_h = target_height
    else:
        out_h = src_h

    size_err = _validate_output_size(out_w, out_h)
    if size_err is not None:
        logger.error("Output size validation failed: %s", size_err)
        return RegistrationResult(
            success=False,
            registered_image=None,
            transform_model=model,
            transform_matrix=matrix,
            output_width=0,
            output_height=0,
            source_width=src_w,
            source_height=src_h,
            warp_config=config,
            failure_reason=size_err,
        )

    # --- Get OpenCV flags ---
    try:
        interp_flag = config.get_interpolation_flag()
        border_flag = config.get_border_flag()
    except ValueError as exc:
        logger.error("Configuration error: %s", exc)
        return RegistrationResult(
            success=False,
            registered_image=None,
            transform_model=model,
            transform_matrix=matrix,
            output_width=out_w,
            output_height=out_h,
            source_width=src_w,
            source_height=src_h,
            warp_config=config,
            failure_reason=str(exc),
        )

    # --- Prepare border value ---
    border_val = config.border_value
    if isinstance(border_val, (int, float)):
        border_val = (float(border_val),) * (image.shape[2] if image.ndim == 3 else 1)

    # --- Perform warp ---
    dsize = (out_w, out_h)  # OpenCV uses (width, height)

    try:
        if model == TransformModel.AFFINE:
            registered = cv2.warpAffine(
                image,
                matrix.astype(np.float64),
                dsize,
                flags=interp_flag,
                borderMode=border_flag,
                borderValue=border_val,
            )
        elif model == TransformModel.HOMOGRAPHY:
            registered = cv2.warpPerspective(
                image,
                matrix.astype(np.float64),
                dsize,
                flags=interp_flag,
                borderMode=border_flag,
                borderValue=border_val,
            )
        else:
            return RegistrationResult(
                success=False,
                registered_image=None,
                transform_model=model,
                transform_matrix=matrix,
                output_width=out_w,
                output_height=out_h,
                source_width=src_w,
                source_height=src_h,
                warp_config=config,
                failure_reason=f"Unsupported transform model: {model}",
            )
    except cv2.error as exc:
        logger.error("OpenCV warp failed: %s", exc)
        return RegistrationResult(
            success=False,
            registered_image=None,
            transform_model=model,
            transform_matrix=matrix,
            output_width=out_w,
            output_height=out_h,
            source_width=src_w,
            source_height=src_h,
            warp_config=config,
            failure_reason=f"OpenCV warp failed: {exc}",
        )

    logger.info(
        "Warp succeeded: model=%s, src=(%d×%d), out=(%d×%d), "
        "interp=%s, border=%s",
        model.value, src_w, src_h, out_w, out_h,
        config.interpolation, config.border_mode,
    )

    return RegistrationResult(
        success=True,
        registered_image=registered,
        transform_model=model,
        transform_matrix=matrix,
        output_width=out_w,
        output_height=out_h,
        source_width=src_w,
        source_height=src_h,
        warp_config=config,
        failure_reason="",
    )

"""
SIH26166 — Data models for image warping and registration output.

Typed structures for warp configuration and registration results.
These models are intentionally decoupled from FastAPI so they can be
used as internal data structures by any pipeline stage.

Transform direction convention
------------------------------
The geometry layer's ``estimate_transform()`` calls::

    cv2.estimateAffine2D(ref_pts, tgt_pts)
    cv2.findHomography(ref_pts, tgt_pts)

This produces a matrix **M** that maps:

    reference image coordinates → target image coordinates

To warp the **reference** image into the **target** coordinate frame
(i.e. align reference to target), we pass M directly to
``cv2.warpAffine`` / ``cv2.warpPerspective`` as the forward transform.

OpenCV's ``warpAffine(src, M, dsize)`` interprets M as the mapping
from **destination** to **source** (inverse mapping) *unless*
``cv2.WARP_INVERSE_MAP`` is used.  However, ``cv2.warpAffine`` and
``cv2.warpPerspective`` **by default** expect the *forward* matrix
(src → dst) and internally invert it.  Therefore:

- We pass the matrix from ``estimateAffine2D(ref, tgt)`` directly.
- **No explicit inversion is performed.**
- The source image for warping is the **reference** image.
- The output (registered image) is in the **target** coordinate frame.

This is documented explicitly to prevent silent convention errors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from backend.geometry.models import TransformModel


# ---------------------------------------------------------------------------
# OpenCV interpolation and border mode mappings
# ---------------------------------------------------------------------------
# Mapping of human-readable names → OpenCV constants for validation.

INTERPOLATION_MODES: dict[str, int] = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
    "lanczos4": cv2.INTER_LANCZOS4,
}

BORDER_MODES: dict[str, int] = {
    "constant": cv2.BORDER_CONSTANT,
    "replicate": cv2.BORDER_REPLICATE,
    "reflect": cv2.BORDER_REFLECT,
    "wrap": cv2.BORDER_WRAP,
    "reflect101": cv2.BORDER_REFLECT_101,
}


@dataclass
class WarpConfig:
    """Configuration for the image warping operation.

    Attributes
    ----------
    interpolation : str
        Interpolation method name.  One of: ``'nearest'``, ``'linear'``,
        ``'cubic'``, ``'area'``, ``'lanczos4'``.
        Default: ``'linear'``.
    border_mode : str
        Border extrapolation method.  One of: ``'constant'``,
        ``'replicate'``, ``'reflect'``, ``'wrap'``, ``'reflect101'``.
        Default: ``'constant'``.
    border_value : float | tuple[float, ...]
        Value used for ``BORDER_CONSTANT`` mode.
        Default: ``0.0`` (black).
    output_width : int | None
        Explicit output width.  If ``None``, uses the target image width
        (or reference image width if no target is provided).
    output_height : int | None
        Explicit output height.  If ``None``, uses the target image height
        (or reference image height if no target is provided).
    """

    interpolation: str = "linear"
    border_mode: str = "constant"
    border_value: float | tuple[float, ...] = 0.0
    output_width: int | None = None
    output_height: int | None = None

    def get_interpolation_flag(self) -> int:
        """Return the OpenCV interpolation flag.

        Raises
        ------
        ValueError
            If the interpolation name is not recognized.
        """
        if self.interpolation not in INTERPOLATION_MODES:
            raise ValueError(
                f"Unknown interpolation mode '{self.interpolation}'. "
                f"Supported: {list(INTERPOLATION_MODES.keys())}"
            )
        return INTERPOLATION_MODES[self.interpolation]

    def get_border_flag(self) -> int:
        """Return the OpenCV border mode flag.

        Raises
        ------
        ValueError
            If the border mode name is not recognized.
        """
        if self.border_mode not in BORDER_MODES:
            raise ValueError(
                f"Unknown border mode '{self.border_mode}'. "
                f"Supported: {list(BORDER_MODES.keys())}"
            )
        return BORDER_MODES[self.border_mode]


@dataclass
class RegistrationResult:
    """Complete result of image registration (warp).

    Attributes
    ----------
    success : bool
        Whether warping succeeded.
    registered_image : np.ndarray | None
        The warped (registered) image.  ``None`` on failure.
        Same dtype as the source image.
    transform_model : TransformModel
        The transformation model used (affine or homography).
    transform_matrix : np.ndarray | None
        The transformation matrix that was applied.
    output_width : int
        Width of the registered image in pixels.
    output_height : int
        Height of the registered image in pixels.
    source_width : int
        Width of the source (reference) image.
    source_height : int
        Height of the source (reference) image.
    warp_config : WarpConfig
        The warp configuration that was used.
    failure_reason : str
        Human-readable explanation if warping failed.
        Empty string on success.
    """

    success: bool
    registered_image: np.ndarray | None
    transform_model: TransformModel
    transform_matrix: np.ndarray | None
    output_width: int
    output_height: int
    source_width: int
    source_height: int
    warp_config: WarpConfig
    failure_reason: str = ""

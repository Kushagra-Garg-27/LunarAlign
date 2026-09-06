"""Image warping and overlay visualization for registered image pairs."""

from __future__ import annotations

import cv2
import numpy as np

_INTERPOLATION_MODES: dict[str, int] = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
}


def warp_image(
    image: np.ndarray,
    transform_matrix: np.ndarray,
    output_shape: tuple[int, int] | None = None,
    interpolation: str = "bicubic",
) -> np.ndarray:
    """Apply transform_matrix to warp image onto reference frame.

    Parameters
    ----------
    image : np.ndarray
        Input 2D or 3D image to be warped.
    transform_matrix : np.ndarray
        2x3 affine matrix or 3x3 homography matrix.
    output_shape : tuple[int, int] | None, default None
        Target (height, width). If None, uses image.shape[:2].
    interpolation : str, default "bicubic"
        Interpolation method: "nearest", "bilinear", or "bicubic".

    Returns
    -------
    np.ndarray
        Warped image with identical dtype to input image.
    """
    if image.size == 0:
        return image.copy()

    interp_flag = _INTERPOLATION_MODES.get(interpolation.lower(), cv2.INTER_CUBIC)

    if output_shape is None:
        target_h, target_w = image.shape[:2]
    else:
        target_h, target_w = output_shape[:2]

    dsize = (int(target_w), int(target_h))
    matrix = np.asarray(transform_matrix, dtype=np.float64)

    # Determine warp function based on matrix shape
    is_homography = matrix.shape == (3, 3)
    warp_fn = cv2.warpPerspective if is_homography else cv2.warpAffine

    if image.ndim == 2:
        warped = warp_fn(
            image,
            matrix,
            dsize,
            flags=interp_flag,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    elif image.ndim == 3:
        channels = image.shape[2]
        if channels <= 4:
            warped = warp_fn(
                image,
                matrix,
                dsize,
                flags=interp_flag,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        else:
            # Handle multi-band > 4 channels
            warped_bands = []
            for b in range(channels):
                band_warped = warp_fn(
                    image[:, :, b],
                    matrix,
                    dsize,
                    flags=interp_flag,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                warped_bands.append(band_warped)
            warped = np.stack(warped_bands, axis=-1)
    else:
        raise ValueError(f"Expected 2D or 3D array, got ndim={image.ndim}")

    return warped.astype(image.dtype)


def compute_overlap_mask(reference: np.ndarray, warped: np.ndarray) -> np.ndarray:
    """Compute binary mask of overlapping valid pixels between warped and reference.

    Parameters
    ----------
    reference : np.ndarray
        Reference image.
    warped : np.ndarray
        Warped image in reference frame.

    Returns
    -------
    np.ndarray
        uint8 binary mask (0 or 255) where both images contain non-zero pixels.
    """
    if reference.shape[:2] != warped.shape[:2]:
        raise ValueError(
            f"Shape mismatch: reference has shape {reference.shape[:2]}, "
            f"warped has shape {warped.shape[:2]}"
        )

    ref_valid = (
        np.any(reference != 0, axis=-1) if reference.ndim == 3 else (reference != 0)
    )
    warp_valid = (
        np.any(warped != 0, axis=-1) if warped.ndim == 3 else (warped != 0)
    )

    overlap = (ref_valid & warp_valid).astype(np.uint8) * 255
    return overlap


def _normalize_to_float32(img: np.ndarray) -> np.ndarray:
    """Normalize any numerical array to float32 in [0, 1]."""
    f = np.nan_to_num(img.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = float(f.min()), float(f.max())
    if lo == hi:
        return np.zeros_like(f, dtype=np.float32)
    return np.clip((f - lo) / (hi - lo), 0.0, 1.0)


def create_checkerboard_overlay(
    reference: np.ndarray,
    warped: np.ndarray,
    block_size: int = 64,
) -> np.ndarray:
    """Create checkerboard pattern alternating tiles from reference and warped.

    Parameters
    ----------
    reference : np.ndarray
        Reference image.
    warped : np.ndarray
        Warped image aligned to reference.
    block_size : int, default 64
        Size of square checkerboard blocks in pixels.

    Returns
    -------
    np.ndarray
        Float32 composite image in [0, 1].
    """
    h = min(reference.shape[0], warped.shape[0])
    w = min(reference.shape[1], warped.shape[1])

    ref_norm = _normalize_to_float32(reference[:h, :w])
    warp_norm = _normalize_to_float32(warped[:h, :w])

    # 2D coordinates for checkerboard pattern
    y, x = np.ogrid[:h, :w]
    mask = ((y // block_size) + (x // block_size)) % 2 == 0

    if ref_norm.ndim == 3 and mask.ndim == 2:
        mask = mask[:, :, np.newaxis]

    composite = np.where(mask, ref_norm, warp_norm)
    return composite.astype(np.float32)


def create_blend_overlay(
    reference: np.ndarray,
    warped: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Alpha-blend warped image onto reference.

    Parameters
    ----------
    reference : np.ndarray
        Reference image.
    warped : np.ndarray
        Warped image.
    alpha : float, default 0.5
        Blending weight for reference (1 - alpha for warped).

    Returns
    -------
    np.ndarray
        Float32 blended image in [0, 1].
    """
    h = min(reference.shape[0], warped.shape[0])
    w = min(reference.shape[1], warped.shape[1])

    ref_norm = _normalize_to_float32(reference[:h, :w])
    warp_norm = _normalize_to_float32(warped[:h, :w])

    blended = alpha * ref_norm + (1.0 - alpha) * warp_norm
    return np.clip(blended, 0.0, 1.0).astype(np.float32)

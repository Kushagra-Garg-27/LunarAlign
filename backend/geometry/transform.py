"""
SIH26166 — Point transformation utilities.

Applies estimated transformation matrices to point coordinates.
This is a building block used by:
- Reprojection error calculation
- Sub-pixel refinement (future)
- Image warping (future)

Does NOT implement image warping — only point-level transformation.

Coordinate convention
---------------------
All points are (x, y) with origin at top-left.
- x = column (rightward)
- y = row (downward)
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("sih26166.geometry.transform")


def transform_points(
    points: np.ndarray,
    matrix: np.ndarray,
) -> np.ndarray:
    """Apply a transformation matrix to a set of 2-D points.

    Supports both affine (2×3) and homography (3×3) matrices.

    Parameters
    ----------
    points : np.ndarray
        Array of shape ``(N, 2)`` with (x, y) coordinates.
    matrix : np.ndarray
        Transformation matrix.
        - Affine: shape ``(2, 3)``
        - Homography: shape ``(3, 3)``

    Returns
    -------
    np.ndarray
        Transformed points, shape ``(N, 2)`` with (x, y) coordinates.

    Raises
    ------
    ValueError
        If ``points`` is not 2-D with 2 columns, or ``matrix`` has
        an unsupported shape.
    """
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(
            f"Expected points of shape (N, 2), got {points.shape}"
        )

    if matrix.shape not in ((2, 3), (3, 3)):
        raise ValueError(
            f"Expected transformation matrix of shape (2,3) or (3,3), "
            f"got {matrix.shape}"
        )

    n = points.shape[0]
    if n == 0:
        return np.empty((0, 2), dtype=np.float64)

    pts = points.astype(np.float64)

    if matrix.shape == (2, 3):
        # Affine: [x', y'] = M @ [x, y, 1]^T
        ones = np.ones((n, 1), dtype=np.float64)
        homogeneous = np.hstack([pts, ones])  # (N, 3)
        M = matrix.astype(np.float64)
        transformed = homogeneous @ M.T  # (N, 2)
        return transformed

    else:
        # Homography: [wx', wy', w] = M @ [x, y, 1]^T
        ones = np.ones((n, 1), dtype=np.float64)
        homogeneous = np.hstack([pts, ones])  # (N, 3)
        M = matrix.astype(np.float64)
        projected = homogeneous @ M.T  # (N, 3)

        # Dehomogenize
        w = projected[:, 2:3]
        # Guard against division by zero
        w_safe = np.where(np.abs(w) < 1e-10, 1e-10, w)
        transformed = projected[:, :2] / w_safe  # (N, 2)
        return transformed

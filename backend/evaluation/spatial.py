"""
SIH26166 — Spatial distribution analysis for match correspondences.

Provides spatial bucketing and normalized spatial entropy to measure
how uniformly inlier correspondences are distributed across the image.

Grid definition
---------------
The image is divided into an ``rows × cols`` grid (default 8×8 = 64 cells).
Each cell covers a rectangular region of the image.

Cell assignment uses the **target** image coordinate of each inlier
match (``tgt_pt``), because the target coordinate frame is the
reference frame for the registered output.

For a point ``(x, y)`` in an image of size ``(width, height)``:

    col = min(int(x / width * cols), cols - 1)
    row = min(int(y / height * rows), rows - 1)

The ``min(..., N-1)`` clamp handles exact boundary coordinates
(e.g. ``x == width``).

Spatial entropy
---------------
Normalized Shannon entropy over the 64-cell grid:

    H = -Σ p_i log(p_i) / log(K)

where:
- ``p_i = count_i / total_inliers`` for cells with count > 0
- ``K = 64`` (total number of grid cells), used for normalization
- ``0 * log(0) = 0`` by convention (cells with zero inliers
  contribute 0 to the entropy sum)

Interpretation:
- H ≈ 0 → inliers are concentrated in very few cells
- H ≈ 1 → inliers are uniformly distributed across the grid

**Spatial entropy measures distribution, not registration correctness.**
A high entropy does not guarantee that the registration is accurate,
and a low entropy does not necessarily indicate failure.

This module does NOT implement match redistribution/reselection.
That belongs to a later spatial bucketing stage.

This module is part of the **CLASSICAL REGISTRATION BASELINE**.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from backend.features.models import FilteredMatch

logger = logging.getLogger("sih26166.evaluation.spatial")

# ---------------------------------------------------------------------------
# Default grid
# ---------------------------------------------------------------------------
DEFAULT_GRID_ROWS: int = 8
DEFAULT_GRID_COLS: int = 8


@dataclass
class SpatialDistribution:
    """Result of spatial distribution analysis.

    Attributes
    ----------
    grid_counts : np.ndarray
        2-D array of shape ``(rows, cols)`` with per-cell inlier counts.
    grid_rows : int
        Number of grid rows.
    grid_cols : int
        Number of grid columns.
    total_cells : int
        Total number of grid cells (``rows * cols``).
    occupied_cells : int
        Number of cells with at least one inlier.
    total_inliers : int
        Total number of inliers assigned to the grid.
    normalized_entropy : float
        Normalized spatial entropy in [0, 1].
        0.0 if there are 0 or 1 inliers.
    """

    grid_counts: np.ndarray
    grid_rows: int
    grid_cols: int
    total_cells: int
    occupied_cells: int
    total_inliers: int
    normalized_entropy: float


def compute_spatial_distribution(
    matches: list[FilteredMatch],
    inlier_mask: np.ndarray | None,
    image_width: int,
    image_height: int,
    *,
    grid_rows: int = DEFAULT_GRID_ROWS,
    grid_cols: int = DEFAULT_GRID_COLS,
) -> SpatialDistribution:
    """Compute spatial distribution of inlier matches over a grid.

    Parameters
    ----------
    matches : list[FilteredMatch]
        All filtered correspondences.
    inlier_mask : np.ndarray | None
        Boolean array of length ``len(matches)``.  If ``None``,
        all matches are treated as inliers.
    image_width : int
        Width of the target image in pixels.
    image_height : int
        Height of the target image in pixels.
    grid_rows : int
        Number of grid rows (default 8).
    grid_cols : int
        Number of grid columns (default 8).

    Returns
    -------
    SpatialDistribution
        Grid counts and normalized entropy.
    """
    total_cells = grid_rows * grid_cols
    grid_counts = np.zeros((grid_rows, grid_cols), dtype=np.int64)

    if len(matches) == 0 or image_width <= 0 or image_height <= 0:
        return SpatialDistribution(
            grid_counts=grid_counts,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            total_cells=total_cells,
            occupied_cells=0,
            total_inliers=0,
            normalized_entropy=0.0,
        )

    # Select inlier matches
    if inlier_mask is not None:
        inlier_matches = [
            m for m, is_inlier in zip(matches, inlier_mask) if is_inlier
        ]
    else:
        inlier_matches = list(matches)

    total_inliers = len(inlier_matches)

    if total_inliers == 0:
        return SpatialDistribution(
            grid_counts=grid_counts,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            total_cells=total_cells,
            occupied_cells=0,
            total_inliers=0,
            normalized_entropy=0.0,
        )

    # Assign each inlier to a grid cell using target-image coordinates
    for m in inlier_matches:
        x, y = m.tgt_pt
        col = min(int(x / image_width * grid_cols), grid_cols - 1)
        row = min(int(y / image_height * grid_rows), grid_rows - 1)
        # Clamp negative coordinates
        col = max(0, col)
        row = max(0, row)
        grid_counts[row, col] += 1

    occupied_cells = int(np.sum(grid_counts > 0))

    # Compute normalized entropy
    entropy = _compute_normalized_entropy(grid_counts, total_cells)

    logger.info(
        "Spatial distribution: %d inliers in %d/%d cells, entropy=%.3f",
        total_inliers, occupied_cells, total_cells, entropy,
    )

    return SpatialDistribution(
        grid_counts=grid_counts,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        total_cells=total_cells,
        occupied_cells=occupied_cells,
        total_inliers=total_inliers,
        normalized_entropy=entropy,
    )


def _compute_normalized_entropy(
    grid_counts: np.ndarray,
    total_cells: int,
) -> float:
    """Compute normalized Shannon entropy over the grid.

    Formula::

        H = -Σ p_i log(p_i) / log(K)

    where K = total_cells (64 for 8×8), and the sum is over all cells.
    Cells with zero count contribute 0 (since 0 * log(0) = 0).

    Parameters
    ----------
    grid_counts : np.ndarray
        Per-cell counts.
    total_cells : int
        Total number of grid cells (for normalization).

    Returns
    -------
    float
        Normalized entropy in [0, 1].
        0.0 if total count is 0 or 1, or total_cells <= 1.
    """
    total = grid_counts.sum()
    if total <= 0 or total_cells <= 1:
        return 0.0

    # Flatten and compute probabilities
    counts = grid_counts.ravel().astype(np.float64)
    probs = counts / total

    # Entropy: -Σ p_i log(p_i), with 0 log 0 = 0
    entropy = 0.0
    for p in probs:
        if p > 0:
            entropy -= p * math.log(p)

    # Normalize by log(K) where K = total_cells
    max_entropy = math.log(total_cells)
    if max_entropy <= 0:
        return 0.0

    return entropy / max_entropy

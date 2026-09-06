"""
SIH26166 — MIND Cross-Modal Descriptor (Module 07).

Implements the Modality Independent Neighbourhood Descriptor (MIND) for structural
feature representation across disparate sensors (e.g. IIRS spectral vs OHRC panchromatic).
Based on self-similarity patches to achieve contrast and modality invariance.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
from scipy.ndimage import uniform_filter

logger = logging.getLogger("sih26166.matching.mind")


def compute_mind_descriptor(
    image: np.ndarray,
    patch_radius: int = 3,
    search_radius: int = 1,
    sigma: float = 0.8,
) -> np.ndarray:
    """Compute dense Modality Independent Neighbourhood Descriptor (MIND) per pixel.

    Parameters
    ----------
    image : np.ndarray
        2D preprocessed grayscale image (float32, [0, 1]).
    patch_radius : int, default 3
        Spatial radius of local neighborhood patches (patch size is 2*r + 1).
    search_radius : int, default 1
        Neighborhood search radius for spatial displacements (1 -> 8 displacements).
    sigma : float, default 0.8
        Gaussian noise floor parameter for exponential SSD normalization.

    Returns
    -------
    np.ndarray
        3D array of shape (height, width, n_displacements) in float32.
    """
    if image.size == 0 or image.ndim != 2:
        n_disp = (2 * search_radius + 1) ** 2 - 1
        return np.zeros((0, 0, n_disp), dtype=np.float32)

    h, w = image.shape
    clean_img = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    patch_size = 2 * patch_radius + 1

    # Enumerate displacements around pixel (excluding (0, 0))
    displacements: list[tuple[int, int]] = []
    for dy in range(-search_radius, search_radius + 1):
        for dx in range(-search_radius, search_radius + 1):
            if dy == 0 and dx == 0:
                continue
            displacements.append((dy, dx))

    n_disp = len(displacements)
    ssd_maps = np.zeros((h, w, n_disp), dtype=np.float32)

    # Vectorized SSD computation across all displacements using moving box filters
    for i, (dy, dx) in enumerate(displacements):
        pad_y = abs(dy)
        pad_x = abs(dx)
        padded = np.pad(clean_img, ((pad_y, pad_y), (pad_x, pad_x)), mode="reflect")
        y_start = pad_y + dy
        x_start = pad_x + dx
        shifted_crop = padded[y_start:y_start + h, x_start:x_start + w]

        diff_sq = (clean_img - shifted_crop) ** 2
        # Patch-wise sum of squared differences via uniform filter
        ssd = uniform_filter(diff_sq, size=patch_size, mode="reflect")
        ssd_maps[:, :, i] = ssd

    # Local variance estimation (mean SSD across displacement neighborhood)
    local_var = np.mean(ssd_maps, axis=-1, keepdims=True)
    noise_var = float(max(sigma ** 2, 1e-4))
    denom = np.maximum(local_var, noise_var)

    # Compute exponential self-similarity descriptor
    mind = np.exp(-ssd_maps / denom)

    # Normalize across displacement vector so max response is 1.0
    max_val = np.max(mind, axis=-1, keepdims=True)
    mind = np.where(max_val > 1e-6, mind / max_val, mind)

    return np.clip(mind, 0.0, 1.0).astype(np.float32)


def match_mind_descriptors(
    mind1: np.ndarray,
    mind2: np.ndarray,
    block_size: int = 16,
    search_window: int = 64,
    stride: int = 16,
) -> list[tuple[tuple[int, int], tuple[int, int], float]]:
    """Block-based template matching between two dense MIND descriptor volumes.

    Parameters
    ----------
    mind1 : np.ndarray
        MIND descriptor volume for image 1 (H1, W1, C).
    mind2 : np.ndarray
        MIND descriptor volume for image 2 (H2, W2, C).
    block_size : int, default 16
        Size of local descriptor patches to correlate.
    search_window : int, default 64
        Window diameter around block center in which to search in image 2.
    stride : int, default 16
        Spatial sampling step between matched block centers.

    Returns
    -------
    list[tuple[tuple[int, int], tuple[int, int], float]]
        List of ((x1, y1), (x2, y2), distance) correspondences.
    """
    if mind1.size == 0 or mind2.size == 0:
        return []

    h1, w1, c1 = mind1.shape
    h2, w2, c2 = mind2.shape
    if c1 != c2:
        raise ValueError(
            f"MIND descriptor channel mismatch: mind1 has {c1}, mind2 has {c2} channels."
        )

    half_b = max(2, block_size // 2)
    half_s = max(half_b + 2, search_window // 2)
    step = max(1, stride)

    matches: list[tuple[tuple[int, int], tuple[int, int], float]] = []

    y_range = range(half_b + half_s, h1 - half_b - half_s, step)
    x_range = range(half_b + half_s, w1 - half_b - half_s, step)

    if not y_range or not x_range:
        # Fallback for small images: center crop matching
        y1, x1 = h1 // 2, w1 // 2
        y2, x2 = h2 // 2, w2 // 2
        matches.append(((x1, y1), (x2, y2), 0.0))
        return matches

    for y1 in y_range:
        for x1 in x_range:
            patch1 = mind1[y1 - half_b:y1 + half_b, x1 - half_b:x1 + half_b]

            y_min = max(half_b, y1 - half_s)
            y_max = min(h2 - half_b, y1 + half_s)
            x_min = max(half_b, x1 - half_s)
            x_max = min(w2 - half_b, x1 + half_s)

            best_dist = float("inf")
            best_pt2 = (x1, y1)

            # Search with sub-sampled step for speed
            for y2 in range(y_min, y_max, 2):
                for x2 in range(x_min, x_max, 2):
                    patch2 = mind2[y2 - half_b:y2 + half_b, x2 - half_b:x2 + half_b]
                    if patch2.shape != patch1.shape:
                        continue
                    dist = float(np.mean((patch1 - patch2) ** 2))
                    if dist < best_dist:
                        best_dist = dist
                        best_pt2 = (x2, y2)

            matches.append(((x1, y1), best_pt2, best_dist))

    return matches

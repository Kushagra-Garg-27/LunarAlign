"""
SIH26166 — IIRS Band Reduction (Module 02).

Reduces high-dimensional hyperspectral cubes (e.g. Chandrayaan-2 IIRS with 256 bands)
into representative 2D images or low-dimensional feature representations for spatial registration.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
from sklearn.decomposition import PCA

logger = logging.getLogger("sih26166.preprocessing.band_reduction")


def reduce_bands_pca(
    cube: np.ndarray,
    n_components: int = 3,
) -> np.ndarray:
    """Reduce spectral bands of a 3D data cube using Principal Component Analysis.

    Parameters
    ----------
    cube : np.ndarray
        Input array. Expected shape is (bands, height, width) [BSQ layout],
        or 2D (height, width) for single-band passthrough.
    n_components : int, default 3
        Number of principal components to extract.

    Returns
    -------
    np.ndarray
        Array of shape (n_components, height, width) containing the top
        principal component projections in float32.
    """
    if cube.size == 0:
        return np.zeros((n_components, 0, 0), dtype=np.float32)

    # Clean NaNs/Infs
    clean_cube = np.nan_to_num(cube, nan=0.0, posinf=0.0, neginf=0.0)

    if clean_cube.ndim == 2:
        # 2D single band input passthrough
        h, w = clean_cube.shape
        out = np.zeros((n_components, h, w), dtype=np.float32)
        out[0] = clean_cube.astype(np.float32)
        for i in range(1, n_components):
            out[i] = clean_cube.astype(np.float32)
        return out

    if clean_cube.ndim != 3:
        raise ValueError(
            f"Expected 2D or 3D cube array, got shape {cube.shape} ({cube.ndim}D)."
        )

    bands, height, width = clean_cube.shape
    if bands == 1:
        out = np.zeros((n_components, height, width), dtype=np.float32)
        for i in range(n_components):
            out[i] = clean_cube[0].astype(np.float32)
        return out

    # Reshape from (bands, height, width) to (pixels, bands)
    n_pixels = height * width
    reshaped = clean_cube.reshape(bands, n_pixels).T  # (n_pixels, bands)

    actual_components = min(n_components, bands, n_pixels)
    if actual_components < 1:
        return np.zeros((n_components, height, width), dtype=np.float32)

    pca = PCA(n_components=actual_components)
    transformed = pca.fit_transform(reshaped)  # (n_pixels, actual_components)

    # Output container (n_components, height, width)
    result = np.zeros((n_components, height, width), dtype=np.float32)
    for c in range(actual_components):
        result[c] = transformed[:, c].reshape(height, width).astype(np.float32)

    logger.debug(
        "PCA band reduction: input (%d, %d, %d) -> output (%d, %d, %d), "
        "PC1 explained variance: %.2f%%",
        bands, height, width, n_components, height, width,
        (pca.explained_variance_ratio_[0] * 100.0) if len(pca.explained_variance_ratio_) > 0 else 0.0,
    )

    return result


def reduce_bands_mean(cube: np.ndarray) -> np.ndarray:
    """Collapse spectral bands into a 2D image via mean averaging across bands.

    Parameters
    ----------
    cube : np.ndarray
        Input 3D array (bands, height, width) or 2D array (height, width).

    Returns
    -------
    np.ndarray
        2D float32 array of shape (height, width).
    """
    if cube.size == 0:
        if cube.ndim == 3:
            return np.zeros((cube.shape[1], cube.shape[2]), dtype=np.float32)
        return np.zeros(cube.shape, dtype=np.float32)

    clean_cube = np.nan_to_num(cube, nan=0.0, posinf=0.0, neginf=0.0)

    if clean_cube.ndim == 2:
        return clean_cube.astype(np.float32)

    if clean_cube.ndim == 3:
        return np.mean(clean_cube, axis=0, dtype=np.float32)

    raise ValueError(
        f"Expected 2D or 3D cube array, got shape {cube.shape} ({cube.ndim}D)."
    )


def select_bands(cube: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    """Select specific spectral bands from a 3D data cube.

    Parameters
    ----------
    cube : np.ndarray
        Input 3D array (bands, height, width).
    indices : Sequence[int]
        List of 0-based band indices to extract.

    Returns
    -------
    np.ndarray
        Subset 3D array of shape (len(indices), height, width) in float32.
    """
    if cube.ndim == 2:
        cube = cube[np.newaxis, ...]

    if cube.ndim != 3:
        raise ValueError(
            f"Expected 2D or 3D cube array, got shape {cube.shape} ({cube.ndim}D)."
        )

    bands, height, width = cube.shape
    for idx in indices:
        if idx < 0 or idx >= bands:
            raise IndexError(
                f"Band index {idx} out of range for cube with {bands} bands (0..{bands - 1})."
            )

    selected = cube[list(indices), :, :]
    return np.nan_to_num(selected, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def select_solar_reflective_bands(
    wavelengths_nm: Sequence[float | dict[str, Any]],
    cutoff_nm: float = 2500.0,
) -> list[int]:
    """Select spectral band indices below a solar-reflective cutoff wavelength.

    Returns the 0-based indices of all bands whose center wavelength is strictly
    below ``cutoff_nm``. These indices can be directly passed to :func:`select_bands`
    to subset a 3D hyperspectral cube.

    Methodological and Validation Notes
    -----------------------------------
    - This cutoff (~2500nm) is a heuristic based on where lunar thermal emission
      begins to dominate reflected solar radiance for IIRS, per general lunar
      remote-sensing literature — it is NOT independently validated against real
      Chandrayaan-2 thermal-band data, because no such data exists in this
      project's fixtures yet.
    - While the default cutoff (2500nm) naturally excludes wavelengths above
      2500nm, if ``cutoff_nm`` is configured above ~3000nm, note that this
      function does not specifically notch-filter or exclude known narrow
      absorption features (such as the ~2800-3000nm OH/H2O band) — that
      refinement is a possible future improvement, not implemented here.

    Parameters
    ----------
    wavelengths_nm : Sequence[float | dict[str, Any]]
        List or sequence of center wavelengths in nanometers (nm). Accepts either
        numeric values (float or int) or dictionary entries containing a
        ``"center_wavelength"`` key (such as returned by
        :func:`backend.preprocessing.pds4.extract_iirs_band_wavelengths`).
    cutoff_nm : float, default 2500.0
        Upper wavelength cutoff in nanometers. Bands with
        ``center_wavelength < cutoff_nm`` are retained.

    Returns
    -------
    list[int]
        0-based band indices for all bands with center wavelength strictly below
        ``cutoff_nm``, in their original order.
    """
    selected_indices: list[int] = []
    for idx, item in enumerate(wavelengths_nm):
        if isinstance(item, dict):
            wl = float(item.get("center_wavelength", 0.0))
        else:
            wl = float(item)
        if wl < cutoff_nm:
            selected_indices.append(idx)
    return selected_indices


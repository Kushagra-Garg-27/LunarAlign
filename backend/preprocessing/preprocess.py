"""
SIH26166 — Preprocessing Pipeline Orchestrator.

Orchestrates multi-sensor radiometric normalization, band reduction, scale handling,
and illumination-invariant phase congruency feature enhancement for Chandrayaan-2 imagery.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from backend.preprocessing.band_reduction import reduce_bands_pca
from backend.preprocessing.datamodel import PreprocessedImage
from backend.preprocessing.illumination import compute_phase_congruency
from backend.preprocessing.normalize import normalize_intensity
from backend.preprocessing.scale_handler import (
    compute_scale_ratio,
    downsample_to_match,
    get_resolution,
)

logger = logging.getLogger("sih26166.preprocessing.preprocess")


def preprocess_single(
    raster: np.ndarray,
    instrument: str,
    metadata: dict[str, Any] | None = None,
    normalize_method: str = "clahe",
    target_resolution: float | None = None,
) -> PreprocessedImage:
    """Execute full preprocessing pipeline on a single image raster.

    Steps applied:
    1. Spectral band reduction (if 3D hyperspectral cube, e.g. IIRS -> PC1 2D image)
    2. Radiometric intensity normalization (CLAHE, minmax, or histogram equalization)
    3. Multi-resolution scale matching (downsampling if target resolution specified)

    Parameters
    ----------
    raster : np.ndarray
        Input 2D or 3D raster array.
    instrument : str
        Instrument identifier ("OHRC", "TMC-2", "TMC2", "IIRS").
    metadata : dict, optional
        Associated PDS4 metadata dictionary.
    normalize_method : str, default "clahe"
        Intensity normalization method ("clahe", "histogram_eq", "minmax").
    target_resolution : float, optional
        Target spatial resolution in meters/pixel for downsampling.

    Returns
    -------
    PreprocessedImage
        Container with 2D float32 normalized image, resolution, and step audit trail.
    """
    orig_shape = raster.shape
    steps: list[str] = []
    processed = raster

    # 1. Band reduction if 3D (e.g. IIRS cube)
    if processed.ndim == 3:
        # Check if bands-first or bands-last
        if processed.shape[0] < min(processed.shape[1], processed.shape[2]):
            # BSQ: (bands, H, W)
            reduced = reduce_bands_pca(processed, n_components=1)
            processed = reduced[0]
        else:
            # Bands-last: (H, W, bands)
            bsq = np.moveaxis(processed, -1, 0)
            reduced = reduce_bands_pca(bsq, n_components=1)
            processed = reduced[0]
        steps.append("pca_band_reduction")

    # 2. Intensity normalization
    processed = normalize_intensity(processed, method=normalize_method)
    steps.append(f"normalize_{normalize_method}")

    # 3. Multi-resolution scale matching
    src_res = get_resolution(metadata, instrument)
    effective_res = src_res

    if target_resolution is not None and target_resolution > 0:
        ratio = compute_scale_ratio(src_res, target_resolution)
        if ratio < 0.9:  # Source is significantly higher-res than target
            processed = downsample_to_match(processed, ratio)
            steps.append(f"downsampled_{ratio:.3f}")
            effective_res = target_resolution

    return PreprocessedImage(
        data=processed.astype(np.float32),
        instrument=instrument,
        original_shape=orig_shape,
        pixel_resolution=effective_res,
        preprocessing_steps=steps,
    )


def preprocess_pair(
    source_raster: np.ndarray,
    source_instrument: str,
    source_meta: dict[str, Any] | None = None,
    ref_raster: np.ndarray | None = None,
    ref_instrument: str = "TMC-2",
    ref_meta: dict[str, Any] | None = None,
    apply_phase_congruency: bool = True,
    normalize_method: str = "clahe",
) -> tuple[PreprocessedImage, PreprocessedImage]:
    """Preprocess a source-reference pair for multi-modal registration.

    Harmonizes spatial resolution to the coarser of the two sensors, applies
    radiometric normalization, and optionally extracts phase congruency structural edges.

    Parameters
    ----------
    source_raster : np.ndarray
        Source image raster array.
    source_instrument : str
        Source instrument identifier (e.g. "OHRC").
    source_meta : dict, optional
        Metadata for source image.
    ref_raster : np.ndarray
        Reference image raster array.
    ref_instrument : str, default "TMC-2"
        Reference instrument identifier (e.g. "TMC-2").
    ref_meta : dict, optional
        Metadata for reference image.
    apply_phase_congruency : bool, default True
        Whether to transform images to phase congruency edge representations.
    normalize_method : str, default "clahe"
        Radiometric normalization method.

    Returns
    -------
    tuple[PreprocessedImage, PreprocessedImage]
        Preprocessed (source, reference) pair ready for feature detection.
    """
    if ref_raster is None:
        raise ValueError("ref_raster must be provided for pair preprocessing.")

    # Determine common target resolution (use the coarser resolution)
    src_res = get_resolution(source_meta, source_instrument)
    ref_res = get_resolution(ref_meta, ref_instrument)
    target_res = max(src_res, ref_res)

    # Preprocess both to common target resolution
    src_prep = preprocess_single(
        source_raster,
        instrument=source_instrument,
        metadata=source_meta,
        normalize_method=normalize_method,
        target_resolution=target_res,
    )

    ref_prep = preprocess_single(
        ref_raster,
        instrument=ref_instrument,
        metadata=ref_meta,
        normalize_method=normalize_method,
        target_resolution=target_res,
    )

    # Optionally compute phase congruency for illumination invariance
    if apply_phase_congruency:
        src_pc, _ = compute_phase_congruency(src_prep.data)
        src_prep.data = src_pc
        src_prep.preprocessing_steps.append("phase_congruency")

        ref_pc, _ = compute_phase_congruency(ref_prep.data)
        ref_prep.data = ref_pc
        ref_prep.preprocessing_steps.append("phase_congruency")

    return src_prep, ref_prep

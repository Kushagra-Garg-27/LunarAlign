"""
SIH26166 — Preprocessing package.

Provides image I/O, internal representation, normalization, band reduction,
scale handling, phase congruency, and pipeline orchestration for Chandrayaan-2 imagery.

Modules:
    io              — Image loading (PNG, JPEG, TIFF, multi-band raster)
    datamodel       — Data containers (RawImage, FeatureImage, PDS4Metadata, PreprocessedImage)
    pds4            — PDS4 label parser and raster loader (TMC-2, OHRC, IIRS)
    metadata        — Lightweight metadata extraction (standard images + PDS4)
    normalize       — Radiometric intensity normalization & histogram matching
    band_reduction  — Hyperspectral PCA / mean band reduction for IIRS
    scale_handler   — Multi-resolution scale matching and Gaussian pyramids
    illumination    — Phase congruency and contrast-invariant edge features
    preprocess      — High-level single and pair preprocessing pipeline
"""

from backend.preprocessing.band_reduction import (
    reduce_bands_mean,
    reduce_bands_pca,
    select_bands,
)
from backend.preprocessing.datamodel import (
    FeatureImage,
    PDS4Metadata,
    PreprocessedImage,
    RawImage,
)
from backend.preprocessing.illumination import (
    compute_edge_map,
    compute_phase_congruency,
)
from backend.preprocessing.normalize import (
    match_histograms,
    normalize_intensity,
    normalize_minmax,
    normalize_percentile,
)
from backend.preprocessing.preprocess import (
    preprocess_pair,
    preprocess_single,
)
from backend.preprocessing.scale_handler import (
    build_gaussian_pyramid,
    compute_scale_ratio,
    downsample_to_match,
    get_resolution,
)

__all__ = [
    "RawImage",
    "FeatureImage",
    "PDS4Metadata",
    "PreprocessedImage",
    "normalize_intensity",
    "normalize_minmax",
    "normalize_percentile",
    "match_histograms",
    "reduce_bands_pca",
    "reduce_bands_mean",
    "select_bands",
    "compute_scale_ratio",
    "downsample_to_match",
    "build_gaussian_pyramid",
    "get_resolution",
    "compute_phase_congruency",
    "compute_edge_map",
    "preprocess_single",
    "preprocess_pair",
]


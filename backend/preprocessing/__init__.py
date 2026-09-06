"""
SIH26166 — Preprocessing package.

Provides image I/O, internal representation, normalization, and optional
CLAHE preprocessing for the lunar image registration pipeline.

Modules:
    io          — Image loading (PNG, JPEG, TIFF, multi-band raster)
    datamodel   — Internal image representation (RawImage, FeatureImage, PDS4Metadata)
    pds4        — PDS4 label parser and raster loader (TMC-2, OHRC, IIRS)
    metadata    — Lightweight metadata extraction (standard images + PDS4)
    normalize   — Intensity normalization utilities
    clahe       — Optional CLAHE contrast enhancement
"""

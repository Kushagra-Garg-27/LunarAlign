"""
SIH26166 — Registration package.

Image warping and registered output generation.

Takes the transformation estimated by the geometry layer and applies it
to produce a registered (aligned) image.

Modules:
    models   — WarpConfig, RegistrationResult data structures
    warping  — Affine and perspective image warping via OpenCV
"""

"""Registration stage: transform estimation, warping, and evaluation."""

from backend.registration.datamodel import RegistrationResult
from backend.registration.transform import estimate_transform, decompose_transform
from backend.registration.subpixel import refine_subpixel
from backend.registration.warper import (
    warp_image,
    compute_overlap_mask,
    create_checkerboard_overlay,
    create_blend_overlay,
)
from backend.registration.metrics import (
    compute_rmse,
    compute_inlier_stats,
    compute_ncc_score,
    compute_ssim_score,
    generate_evaluation_report,
)
from backend.registration.pipeline import register_pair

__all__ = [
    "RegistrationResult",
    "estimate_transform",
    "decompose_transform",
    "refine_subpixel",
    "warp_image",
    "compute_overlap_mask",
    "create_checkerboard_overlay",
    "create_blend_overlay",
    "compute_rmse",
    "compute_inlier_stats",
    "compute_ncc_score",
    "compute_ssim_score",
    "generate_evaluation_report",
    "register_pair",
]

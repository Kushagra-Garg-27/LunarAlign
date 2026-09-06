#!/usr/bin/env python3
"""LunarAlign — Multi-Modal Lunar Image Registration CLI.

Smart India Hackathon 2026 • SIH26166 • Team Six Seven
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any

# Ensure stdout and stderr handle unicode safely across Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import cv2
import numpy as np

from backend.matching.match_pipeline import match_pair
from backend.preprocessing.pds4 import (
    extract_pds4_metadata,
    is_pds4_label,
    load_pds4_raster,
)
from backend.preprocessing.preprocess import preprocess_pair
from backend.registration.pipeline import register_pair

BANNER = """═══════════════════════════════════════════════════════
 LunarAlign — Multi-Modal Lunar Image Registration
 Smart India Hackathon 2026 • SIH26166 • Team Six Seven
═══════════════════════════════════════════════════════"""

logger = logging.getLogger("lunaralign.cli")


def normalize_for_display(img: np.ndarray) -> np.ndarray:
    """Convert any array to uint8 [0, 255]."""
    arr = img.astype(np.float32)
    lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    if hi > lo:
        arr = (arr - lo) / (hi - lo) * 255.0
    else:
        arr = np.zeros_like(arr)
    return np.clip(arr, 0, 255).astype(np.uint8)


def draw_matches_image(
    img1: np.ndarray,
    img2: np.ndarray,
    match_result: Any,
    max_draw: int = 100,
) -> np.ndarray:
    """Create side-by-side match visualization with lines."""
    disp1 = normalize_for_display(img1)
    disp2 = normalize_for_display(img2)

    pts1 = match_result.match_points1
    pts2 = match_result.match_points2
    n_inliers = min(len(pts1), len(pts2))

    if n_inliers == 0:
        h1, w1 = disp1.shape[:2]
        h2, w2 = disp2.shape[:2]
        max_h = max(h1, h2)
        canvas = np.zeros((max_h, w1 + w2, 3), dtype=np.uint8)
        c1 = cv2.cvtColor(disp1, cv2.COLOR_GRAY2BGR) if disp1.ndim == 2 else disp1
        c2 = cv2.cvtColor(disp2, cv2.COLOR_GRAY2BGR) if disp2.ndim == 2 else disp2
        canvas[:h1, :w1] = c1
        canvas[:h2, w1 : w1 + w2] = c2
        return canvas

    num_to_draw = min(n_inliers, max_draw)
    indices = np.linspace(0, n_inliers - 1, num_to_draw, dtype=int)

    kps1 = [cv2.KeyPoint(x=float(pts1[i, 0]), y=float(pts1[i, 1]), size=10.0) for i in indices]
    kps2 = [cv2.KeyPoint(x=float(pts2[i, 0]), y=float(pts2[i, 1]), size=10.0) for i in indices]
    dmatches = [cv2.DMatch(_queryIdx=k, _trainIdx=k, _distance=0.0) for k in range(num_to_draw)]

    out_img = cv2.drawMatches(
        disp1,
        kps1,
        disp2,
        kps2,
        dmatches,
        None,
        matchColor=(0, 235, 130),
        singlePointColor=(255, 140, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    return out_img


def load_image_input(
    path_str: str,
    instrument_override: str | None = None,
) -> tuple[np.ndarray, str, str]:
    """Load image from path, determining instrument and array."""
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path_str}")

    filename = path.name
    if is_pds4_label(path):
        raster_obj = load_pds4_raster(path)
        arr = raster_obj.data
        if instrument_override:
            instrument = instrument_override
        else:
            try:
                meta = extract_pds4_metadata(path)
                instrument = meta.instrument if hasattr(meta, "instrument") and meta.instrument else "TMC-2"
            except Exception:
                instrument = "TMC-2"
    else:
        arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if arr is None:
            from PIL import Image as PILImg
            arr = np.array(PILImg.open(str(path)))
        instrument = instrument_override if instrument_override else "TMC-2"

    return arr, instrument, filename


def format_filesize(num_bytes: int) -> str:
    """Format bytes into human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}" if unit != "B" else f"{num_bytes} B"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def main():
    parser = argparse.ArgumentParser(
        description="LunarAlign — Multi-Modal Lunar Image Registration CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("reference", help="Path to reference image (PDS4 .xml, .tif, .png, .jpg)")
    parser.add_argument("target", help="Path to target image")
    parser.add_argument("--output-dir", "-o", default="./output", help="Output directory")
    parser.add_argument("--method", default="sift", choices=["sift", "sift_bucketed", "mind", "combined"], help="Feature method")
    parser.add_argument("--model", default="auto", choices=["auto", "affine", "homography"], help="Transform model")
    parser.add_argument("--no-subpixel", action="store_true", help="Disable sub-pixel refinement")
    parser.add_argument("--interpolation", default="bicubic", choices=["nearest", "bilinear", "bicubic"], help="Warp interpolation")
    parser.add_argument("--ref-instrument", default=None, help="Override reference instrument: TMC-2, OHRC, IIRS (auto-detect from PDS4)")
    parser.add_argument("--tgt-instrument", default=None, help="Override target instrument (auto-detect from PDS4)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # 1. Logging setup
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # 2. Banner
    print(BANNER)
    print()

    # 3. Load images
    try:
        ref_arr, ref_inst, ref_name = load_image_input(args.reference, args.ref_instrument)
        tgt_arr, tgt_inst, tgt_name = load_image_input(args.target, args.tgt_instrument)
    except Exception as exc:
        print(f"Error loading inputs: {exc}", file=sys.stderr)
        sys.exit(1)

    ref_shape_str = f"{ref_arr.shape[0]}×{ref_arr.shape[1]}" if ref_arr.ndim >= 2 else str(ref_arr.shape)
    tgt_shape_str = f"{tgt_arr.shape[0]}×{tgt_arr.shape[1]}" if tgt_arr.ndim >= 2 else str(tgt_arr.shape)

    print(f"[1/4] Loaded reference: {ref_name} ({ref_shape_str}, {ref_arr.dtype})")
    print(f"      Loaded target:    {tgt_name} ({tgt_shape_str}, {tgt_arr.dtype})")

    # 4. Preprocessing
    print("[2/4] Preprocessing...")
    try:
        prep_tgt, prep_ref = preprocess_pair(
            source_raster=tgt_arr,
            source_instrument=tgt_inst,
            source_meta=None,
            ref_raster=ref_arr,
            ref_instrument=ref_inst,
            ref_meta=None,
            apply_phase_congruency=False,
            normalize_method="clahe",
        )
    except Exception as exc:
        print(f"Error during preprocessing: {exc}", file=sys.stderr)
        sys.exit(1)

    ref_steps_str = " → ".join(prep_ref.preprocessing_steps) if prep_ref.preprocessing_steps else "none"
    tgt_steps_str = " → ".join(prep_tgt.preprocessing_steps) if prep_tgt.preprocessing_steps else "none"
    print(f"      Reference: {ref_steps_str}")
    print(f"      Target: {tgt_steps_str}")

    # 5. Feature matching
    print(f"[3/4] Feature matching ({args.method})...")
    try:
        match_result = match_pair(
            prep_ref.data,
            prep_tgt.data,
            method=args.method,
            use_phase_congruency=False,
            transform_model=args.model,
        )
    except Exception as exc:
        print(f"Error during feature matching: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"      Raw matches: {match_result.raw_matches} → Good: {match_result.good_matches} → Inliers: {match_result.inlier_matches} ({match_result.inlier_ratio:.1%})")
    print(f"      Spatial distribution: {match_result.spatial_distribution:.2f}")

    if match_result.inlier_matches == 0:
        print("No inlier matches found. Images may not overlap.", file=sys.stderr)
        sys.exit(1)

    # 6. Registration
    print(f"[4/4] Registering ({args.model})...")
    try:
        enable_subpixel = not args.no_subpixel
        reg_result = register_pair(
            reference=prep_ref.data,
            target=prep_tgt.data,
            match_result=match_result,
            model=args.model,
            enable_subpixel=enable_subpixel,
            interpolation=args.interpolation,
        )
    except Exception as exc:
        print(f"Error during registration: {exc}", file=sys.stderr)
        sys.exit(1)

    eval_data = reg_result.evaluation or {}
    rmse_val = float(eval_data.get("rmse", 0.0))
    ncc_val = float(eval_data.get("ncc", 0.0))
    ssim_val = float(eval_data.get("ssim", 0.0))
    grade_val = str(eval_data.get("quality_grade", "unknown")).upper()
    inlier_ratio_val = float(match_result.inlier_ratio)

    mat = reg_result.transform_matrix
    mat_rows, mat_cols = mat.shape if isinstance(mat, np.ndarray) else (0, 0)

    print("───────────────────────────────────────")
    print(" Registration Quality Report")
    print("───────────────────────────────────────")
    print(f" Transform:    {reg_result.transform_type} ({mat_rows}×{mat_cols})")
    print(f" RMSE:         {rmse_val:.3f} px")
    print(f" NCC:          {ncc_val:.4f}")
    print(f" SSIM:         {ssim_val:.4f}")
    print(f" Inlier Ratio: {inlier_ratio_val:.1%}")
    print(f" Quality:      {grade_val}")
    print("───────────────────────────────────────")

    # 7. Save outputs
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    warped_u8 = normalize_for_display(reg_result.warped_image)
    checkerboard_u8 = normalize_for_display(reg_result.checkerboard)
    overlap_mask = reg_result.overlap_mask
    overlap_u8 = overlap_mask if (overlap_mask.ndim == 2 and overlap_mask.dtype == np.uint8) else normalize_for_display(overlap_mask)
    matches_img = draw_matches_image(prep_ref.data, prep_tgt.data, match_result, max_draw=100)

    warped_path = out_dir / "warped.png"
    checkerboard_path = out_dir / "checkerboard.png"
    overlap_path = out_dir / "overlap.png"
    matches_path = out_dir / "matches.png"
    report_path = out_dir / "report.json"

    cv2.imwrite(str(warped_path), warped_u8)
    cv2.imwrite(str(checkerboard_path), checkerboard_u8)
    cv2.imwrite(str(overlap_path), overlap_u8)
    cv2.imwrite(str(matches_path), matches_img)

    report = {
        "metadata": {
            "reference_file": str(args.reference),
            "target_file": str(args.target),
            "reference_shape": list(ref_arr.shape),
            "target_shape": list(tgt_arr.shape),
            "reference_instrument": ref_inst,
            "target_instrument": tgt_inst,
            "method": args.method,
            "model": args.model,
            "enable_subpixel": enable_subpixel,
            "interpolation": args.interpolation,
        },
        "matching": {
            "raw_matches": int(match_result.raw_matches),
            "good_matches": int(match_result.good_matches),
            "inlier_matches": int(match_result.inlier_matches),
            "inlier_ratio": float(match_result.inlier_ratio),
            "spatial_distribution": float(match_result.spatial_distribution),
        },
        "registration": {
            "transform_type": reg_result.transform_type,
            "transform_matrix": mat.tolist() if isinstance(mat, np.ndarray) else [],
            "rmse": rmse_val,
            "ncc": ncc_val,
            "ssim": ssim_val,
            "quality_grade": grade_val,
            "evaluation": eval_data,
        },
    }

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    saved_files = [
        ("warped.png", warped_path.stat().st_size),
        ("checkerboard.png", checkerboard_path.stat().st_size),
        ("overlap.png", overlap_path.stat().st_size),
        ("matches.png", matches_path.stat().st_size),
        ("report.json", report_path.stat().st_size),
    ]

    print(f"Results saved to {args.output_dir}/")
    for name, size in saved_files:
        print(f"  • {name} ({format_filesize(size)})")

    # 8. Final summary line
    print(f"✓ Registration complete: {grade_val} (RMSE={rmse_val:.3f}px, NCC={ncc_val:.4f})")


if __name__ == "__main__":
    main()

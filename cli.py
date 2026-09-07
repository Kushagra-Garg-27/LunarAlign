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
import tempfile
from typing import Any

# Ensure stdout and stderr handle unicode safely across Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import cv2
import numpy as np

from backend.core.registration_service import run_classical_registration
from backend.preprocessing.pds4 import (
    extract_pds4_metadata,
    is_pds4_label,
    load_pds4_raster,
)

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
    parser.add_argument("--method", default="sift", choices=["sift", "sift_bucketed", "mind", "combined"], help="Feature method (canonical pipeline uses sift only)")
    parser.add_argument("--model", default="auto", choices=["auto", "affine", "homography"], help="Transform model")
    parser.add_argument("--no-subpixel", action="store_true", help="Disable sub-pixel refinement (not yet supported in canonical pipeline)")
    parser.add_argument("--interpolation", default="bicubic", choices=["nearest", "bilinear", "bicubic"], help="Warp interpolation (not yet configurable in canonical pipeline)")
    parser.add_argument("--ratio-threshold", type=float, default=0.75, help="Lowe ratio test threshold (0-1)")
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

    # 3. Validate inputs exist
    ref_path = Path(args.reference).resolve()
    tgt_path = Path(args.target).resolve()

    if not ref_path.exists():
        print(f"Error: reference file not found: {args.reference}", file=sys.stderr)
        sys.exit(1)
    if not tgt_path.exists():
        print(f"Error: target file not found: {args.target}", file=sys.stderr)
        sys.exit(1)

    # Show basic info about inputs
    ref_arr, ref_inst, ref_name = load_image_input(str(ref_path), args.ref_instrument)
    tgt_arr, tgt_inst, tgt_name = load_image_input(str(tgt_path), args.tgt_instrument)

    ref_shape_str = f"{ref_arr.shape[0]}×{ref_arr.shape[1]}" if ref_arr.ndim >= 2 else str(ref_arr.shape)
    tgt_shape_str = f"{tgt_arr.shape[0]}×{tgt_arr.shape[1]}" if tgt_arr.ndim >= 2 else str(tgt_arr.shape)

    print(f"[1/2] Loaded reference: {ref_name} ({ref_shape_str}, {ref_arr.dtype})")
    print(f"      Loaded target:    {tgt_name} ({tgt_shape_str}, {tgt_arr.dtype})")

    # Map CLI model arg to canonical pipeline's expected values
    transform_model = args.model
    if transform_model == "auto":
        transform_model = "affine"  # canonical pipeline requires explicit model

    # 4. Run canonical pipeline
    print(f"[2/2] Running canonical registration pipeline (model={transform_model}, ratio={args.ratio_threshold})...")
    try:
        pipeline_result = run_classical_registration(
            str(ref_path),
            str(tgt_path),
            transform_model=transform_model,
            ratio_threshold=args.ratio_threshold,
        )
    except Exception as exc:
        print(f"Error during registration: {exc}", file=sys.stderr)
        sys.exit(1)

    if not pipeline_result.success:
        print(f"Registration failed at stage '{pipeline_result.failure_stage}': {pipeline_result.failure_reason}", file=sys.stderr)
        sys.exit(1)

    # 5. Extract results
    summary = pipeline_result.quality_summary
    timings = pipeline_result.timings

    print("───────────────────────────────────────")
    print(" Registration Quality Report")
    print("───────────────────────────────────────")
    print(f" Transform:    {summary.transform_model} ({summary.estimator_method})")
    print(f" RMSE:         {summary.inlier_rmse:.3f} px")
    print(f" Inlier Count: {summary.inlier_count} / {summary.total_correspondences}")
    print(f" Inlier Ratio: {summary.inlier_ratio:.1%}")
    if pipeline_result.spatial is not None:
        print(f" Spatial Ent:  {pipeline_result.spatial.normalized_entropy:.4f}")
    if timings:
        print(f" Total Time:   {timings.get('total', 0.0):.3f} s")
    print("───────────────────────────────────────")

    # 6. Save outputs
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved_files: list[tuple[str, int]] = []

    # Registered image
    if pipeline_result.registered_image is not None:
        warped_u8 = normalize_for_display(pipeline_result.registered_image)
        warped_path = out_dir / "warped.png"
        cv2.imwrite(str(warped_path), warped_u8)
        saved_files.append(("warped.png", warped_path.stat().st_size))

    # Match visualization
    if pipeline_result.match_visualization is not None:
        matches_path = out_dir / "matches.png"
        cv2.imwrite(str(matches_path), pipeline_result.match_visualization)
        saved_files.append(("matches.png", matches_path.stat().st_size))

    # Inlier visualization
    if pipeline_result.inlier_visualization is not None:
        inliers_path = out_dir / "inliers.png"
        cv2.imwrite(str(inliers_path), pipeline_result.inlier_visualization)
        saved_files.append(("inliers.png", inliers_path.stat().st_size))

    # Overlay visualization
    if pipeline_result.overlay_visualization is not None:
        overlay_path = out_dir / "overlay.png"
        cv2.imwrite(str(overlay_path), pipeline_result.overlay_visualization)
        saved_files.append(("overlay.png", overlay_path.stat().st_size))

    # JSON report
    report = {
        "metadata": {
            "reference_file": str(args.reference),
            "target_file": str(args.target),
            "reference_shape": list(ref_arr.shape),
            "target_shape": list(tgt_arr.shape),
            "reference_instrument": ref_inst,
            "target_instrument": tgt_inst,
            "transform_model": transform_model,
            "ratio_threshold": args.ratio_threshold,
            "pipeline_mode": pipeline_result.pipeline_mode,
        },
        "matching": {
            "total_correspondences": summary.total_correspondences,
            "inlier_count": summary.inlier_count,
            "outlier_count": summary.outlier_count,
            "inlier_ratio": summary.inlier_ratio,
        },
        "registration": {
            "transform_model": summary.transform_model,
            "estimator_method": summary.estimator_method,
            "inlier_rmse": summary.inlier_rmse,
            "all_rmse": summary.all_rmse,
            "inlier_median_error": summary.inlier_median_error,
            "inlier_max_error": summary.inlier_max_error,
            "transform_matrix": summary.transform_matrix,
        },
        "spatial": {
            "normalized_entropy": pipeline_result.spatial.normalized_entropy if pipeline_result.spatial else -1.0,
        },
        "timings": timings,
    }

    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    saved_files.append(("report.json", report_path.stat().st_size))

    print(f"Results saved to {args.output_dir}/")
    for name, size in saved_files:
        print(f"  • {name} ({format_filesize(size)})")

    # 7. Final summary line
    print(f"✓ Registration complete (RMSE={summary.inlier_rmse:.3f}px, inliers={summary.inlier_count})")


if __name__ == "__main__":
    main()

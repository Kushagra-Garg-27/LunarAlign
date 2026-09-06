"""
FastAPI application for LunarAlign web dashboard.
Smart India Hackathon 2026 • SIH26166 • Team Six Seven
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional
import uuid

import time

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import uvicorn

from backend.preprocessing.datamodel import PreprocessedImage
from backend.preprocessing.pds4 import (
    extract_pds4_metadata,
    is_pds4_label,
    load_pds4_raster,
)
from backend.preprocessing.preprocess import preprocess_pair
from backend.matching.datamodel import MatchResult
from backend.matching.match_pipeline import match_pair
from backend.registration.datamodel import RegistrationResult
from backend.registration.pipeline import register_pair

# Compatibility aliases as requested
detect_pds4 = is_pds4_label


def load_pds4_image(label_path: str | Path) -> np.ndarray:
    """Load image data from a PDS4 label as a numpy array."""
    raw = load_pds4_raster(label_path)
    return raw.data


# ---------------------------------------------------------------------------
# App & Paths Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"
FRONTEND_DIR = BASE_DIR / "frontend"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("lunaralign.api")

app = FastAPI(
    title="LunarAlign — Multi-Modal Lunar Image Registration",
    description="Smart India Hackathon 2026 • SIH26166 • Team Six Seven",
    version="1.0.0",
)

# CORS middleware allowing all origins (prototype)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIST = BASE_DIR / "frontend" / "dist"

# In-memory storage for active jobs
JOBS: dict[str, dict[str, Any]] = {}


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"},
    )


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def normalize_for_display(img: np.ndarray) -> np.ndarray:
    """Convert any array to uint8 [0, 255] for PNG saving.

    Handles 2D float, uint16, int32, and multi-band 3D arrays (first band or PCA/RGB).
    """
    if img is None or img.size == 0:
        return np.zeros((100, 100), dtype=np.uint8)

    arr = np.asarray(img)

    # Multi-band handling
    if arr.ndim == 3:
        if arr.shape[2] == 1:
            arr = arr[:, :, 0]
        elif arr.shape[0] < min(arr.shape[1], arr.shape[2]):
            # bands-first (C, H, W)
            if arr.shape[0] >= 3:
                # RGB composite of first 3 bands
                rgb = []
                for i in range(3):
                    b = arr[i].astype(np.float32)
                    b_min, b_max = float(np.min(b)), float(np.max(b))
                    if b_max > b_min:
                        b = (b - b_min) / (b_max - b_min) * 255.0
                    else:
                        b = np.zeros_like(b)
                    rgb.append(b.astype(np.uint8))
                return np.stack([rgb[2], rgb[1], rgb[0]], axis=-1)  # BGR for OpenCV
            else:
                arr = arr[0]
        else:
            # bands-last (H, W, C)
            if arr.shape[2] >= 3:
                rgb = []
                for i in range(3):
                    b = arr[:, :, i].astype(np.float32)
                    b_min, b_max = float(np.min(b)), float(np.max(b))
                    if b_max > b_min:
                        b = (b - b_min) / (b_max - b_min) * 255.0
                    else:
                        b = np.zeros_like(b)
                    rgb.append(b.astype(np.uint8))
                return np.stack([rgb[2], rgb[1], rgb[0]], axis=-1)  # BGR for OpenCV
            else:
                arr = arr[:, :, 0]

    # 2D Grayscale normalization: robust percentile stretch over all finite pixels.
    # Zero-padded warping borders naturally anchor vmin=0; actual content is
    # stretched to the 98th-percentile upper bound, giving full visible range.
    arr = arr.astype(np.float64)
    finite_mask = np.isfinite(arr)
    if not finite_mask.any():
        return np.zeros_like(arr, dtype=np.uint8)

    finite_pixels = arr[finite_mask]
    # Replace non-finite values with 0 before normalization
    arr = np.where(finite_mask, arr, 0.0)

    vmin = float(np.percentile(finite_pixels, 2))
    vmax = float(np.percentile(finite_pixels, 98))

    # Fallback: if the spread is too small (e.g. constant image), use full range
    if vmax - vmin < 1e-10:
        vmin = float(finite_pixels.min())
        vmax = float(finite_pixels.max())

    if vmax - vmin < 1e-10:
        return np.zeros_like(arr, dtype=np.uint8)

    normalized = np.clip((arr - vmin) / (vmax - vmin) * 255, 0, 255)
    return normalized.astype(np.uint8)


def draw_matches_image(
    img1: np.ndarray,
    img2: np.ndarray,
    match_result: MatchResult,
    max_draw: int = 100,
) -> np.ndarray:
    """Create side-by-side match visualization using cv2.drawMatches with colored lines."""
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
        matchColor=(0, 235, 130),  # vibrant teal/green in BGR
        singlePointColor=(255, 140, 0),  # blueish orange
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    return out_img


def create_overlap_visualization(
    reference: np.ndarray,
    warped: np.ndarray,
    overlap_mask: np.ndarray,
) -> np.ndarray:
    """Green-tinted overlay showing alignment quality."""
    ref_u8 = normalize_for_display(reference)
    warped_u8 = normalize_for_display(warped)

    ref_bgr = cv2.cvtColor(ref_u8, cv2.COLOR_GRAY2BGR) if ref_u8.ndim == 2 else ref_u8
    warped_bgr = cv2.cvtColor(warped_u8, cv2.COLOR_GRAY2BGR) if warped_u8.ndim == 2 else warped_u8

    # Blend 50/50
    blend = cv2.addWeighted(ref_bgr, 0.5, warped_bgr, 0.5, 0)

    # Create green tint on overlap
    mask = (overlap_mask > 0) if overlap_mask is not None and overlap_mask.size > 0 else np.ones(ref_u8.shape[:2], dtype=bool)
    green_overlay = blend.copy()
    green_overlay[mask, 1] = np.clip(green_overlay[mask, 1].astype(int) + 70, 0, 255).astype(np.uint8)

    # Composite blend with tint
    out = cv2.addWeighted(blend, 0.65, green_overlay, 0.35, 0)
    return out


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Return frontend/index.html as HTMLResponse."""
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="frontend/index.html not found",
        )
    return HTMLResponse(content=index_file.read_text(encoding="utf-8"))


@app.post("/api/upload")
async def upload_images(
    reference: UploadFile = File(...),
    target: UploadFile = File(...),
):
    """Accept reference and target files, store in UPLOAD_DIR, decode rasters."""
    job_id = uuid.uuid4().hex[:8]
    job_upload_dir = UPLOAD_DIR / job_id
    job_upload_dir.mkdir(parents=True, exist_ok=True)

    ref_filename = reference.filename or "reference"
    tgt_filename = target.filename or "target"

    ref_path = job_upload_dir / ref_filename
    tgt_path = job_upload_dir / tgt_filename

    ref_bytes = await reference.read()
    tgt_bytes = await target.read()

    ref_path.write_bytes(ref_bytes)
    tgt_path.write_bytes(tgt_bytes)

    # Load reference
    ref_inst = "TMC-2"
    if detect_pds4(ref_path):
        try:
            ref_arr = load_pds4_image(ref_path)
            try:
                m = extract_pds4_metadata(ref_path)
                ref_inst = m.instrument if hasattr(m, "instrument") else "TMC-2"
            except Exception:
                pass
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"PDS4 label uploaded without companion binary data file. "
                       f"Please upload both the .xml label AND the .img/.qub data file together, "
                       f"or convert to PNG/TIFF first. Error: {str(e)}",
            )
    else:
        ref_arr = cv2.imread(str(ref_path), cv2.IMREAD_UNCHANGED)
        if ref_arr is None:
            from PIL import Image as PILImg
            ref_arr = np.array(PILImg.open(str(ref_path)))

    # Load target
    # NOTE: non-PDS4 rasters (PNG/TIFF) have no known ground-sample distance.
    # Default both instruments to "TMC-2" so the scale-handler treats them as
    # same-resolution and does NOT apply the destructive 24× OHRC→TMC-2 downsample.
    tgt_inst = "TMC-2"
    if detect_pds4(tgt_path):
        try:
            tgt_arr = load_pds4_image(tgt_path)
            try:
                m = extract_pds4_metadata(tgt_path)
                tgt_inst = m.instrument if hasattr(m, "instrument") else "OHRC"
            except Exception:
                pass
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"PDS4 label uploaded without companion binary data file. "
                       f"Please upload both the .xml label AND the .img/.qub data file together, "
                       f"or convert to PNG/TIFF first. Error: {str(e)}",
            )
    else:
        tgt_arr = cv2.imread(str(tgt_path), cv2.IMREAD_UNCHANGED)
        if tgt_arr is None:
            from PIL import Image as PILImg
            tgt_arr = np.array(PILImg.open(str(tgt_path)))

    JOBS[job_id] = {
        "reference": ref_arr,
        "target": tgt_arr,
        "ref_name": ref_filename,
        "tgt_name": tgt_filename,
        "ref_instrument": ref_inst,
        "tgt_instrument": tgt_inst,
        "status": "uploaded",
        "progress_step": 1,
    }

    logger.info(
        "Job %s uploaded: ref=%s shape=%s, tgt=%s shape=%s",
        job_id,
        ref_filename,
        ref_arr.shape,
        tgt_filename,
        tgt_arr.shape,
    )

    return {
        "job_id": job_id,
        "status": "uploaded",
        "ref_shape": list(ref_arr.shape),
        "tgt_shape": list(tgt_arr.shape),
    }


@app.post("/api/process/{job_id}")
async def process_job(job_id: str):
    """Run full pipeline in sequence: Preprocess -> Match -> Register."""
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job ID not found")

    job = JOBS[job_id]
    ref_raw = job["reference"]
    tgt_raw = job["target"]
    ref_inst = job.get("ref_instrument", "TMC-2")
    tgt_inst = job.get("tgt_instrument", "OHRC")

    job_results_dir = RESULTS_DIR / job_id
    job_results_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: Preprocessing
        job["status"] = "preprocessing"
        job["progress_step"] = 2
        logger.info("Job %s: Starting preprocessing", job_id)

        # Preprocess pair: Harmonize spatial scale, radiometric normalization
        prep_tgt, prep_ref = preprocess_pair(
            source_raster=tgt_raw,
            source_instrument=tgt_inst,
            source_meta=None,
            ref_raster=ref_raw,
            ref_instrument=ref_inst,
            ref_meta=None,
            apply_phase_congruency=False,
            normalize_method="clahe",
        )

        job["prep_ref"] = prep_ref
        job["prep_tgt"] = prep_tgt

        # Step 2: Feature Matching
        job["status"] = "matching"
        job["progress_step"] = 3
        logger.info("Job %s: Starting feature matching", job_id)

        # Match pair
        match_result = match_pair(
            prep_ref.data,
            prep_tgt.data,
            method="sift",
            use_phase_congruency=False,
            transform_model="auto",
        )
        job["match_result"] = match_result

        # Step 3: Registration
        job["status"] = "registering"
        job["progress_step"] = 4
        logger.info("Job %s: Starting registration", job_id)

        reg_result = register_pair(
            reference=prep_ref.data,
            target=prep_tgt.data,
            match_result=match_result,
            model="auto",
            enable_subpixel=True,
            interpolation="bicubic",
        )
        job["reg_result"] = reg_result

        # Generate & save visualization images
        ref_preview = normalize_for_display(prep_ref.data)
        tgt_preview = normalize_for_display(prep_tgt.data)
        matches_preview = draw_matches_image(prep_ref.data, prep_tgt.data, match_result)
        warped_preview = normalize_for_display(reg_result.warped_image)
        checkerboard_preview = normalize_for_display(reg_result.checkerboard)
        overlap_preview = create_overlap_visualization(
            prep_ref.data, reg_result.warped_image, reg_result.overlap_mask
        )

        cv2.imwrite(str(job_results_dir / "reference_preview.png"), ref_preview)
        cv2.imwrite(str(job_results_dir / "target_preview.png"), tgt_preview)
        cv2.imwrite(str(job_results_dir / "matches_preview.png"), matches_preview)
        cv2.imwrite(str(job_results_dir / "warped_preview.png"), warped_preview)
        cv2.imwrite(str(job_results_dir / "checkerboard_preview.png"), checkerboard_preview)
        cv2.imwrite(str(job_results_dir / "overlap_preview.png"), overlap_preview)

        job["status"] = "complete"
        job["progress_step"] = 5
        logger.info("Job %s: Processing complete", job_id)

        # Matrix to nested list
        mat = reg_result.transform_matrix
        mat_list = mat.tolist() if isinstance(mat, np.ndarray) else []

        eval_data = reg_result.evaluation or {}
        rmse_val = float(eval_data.get("rmse", 0.0))
        ncc_val = float(eval_data.get("ncc", 0.0))
        ssim_val = float(eval_data.get("ssim", 0.0))
        grade_val = str(eval_data.get("quality_grade", "unknown")).upper()

        return {
            "job_id": job_id,
            "status": "complete",
            "preprocessing": {
                "ref_shape": list(prep_ref.data.shape),
                "tgt_shape": list(prep_tgt.data.shape),
                "steps_ref": prep_ref.preprocessing_steps,
                "steps_tgt": prep_tgt.preprocessing_steps,
            },
            "matching": {
                "raw_matches": int(match_result.raw_matches),
                "good_matches": int(match_result.good_matches),
                "inlier_matches": int(match_result.inlier_matches),
                "inlier_ratio": float(match_result.inlier_ratio),
                "spatial_distribution": float(match_result.spatial_distribution),
                "method": "sift",
            },
            "registration": {
                "transform_type": reg_result.transform_type,
                "rmse": rmse_val,
                "ncc": ncc_val,
                "ssim": ssim_val,
                "quality_grade": grade_val,
                "transform_matrix": mat_list,
            },
            "images": {
                "reference": f"/api/result/{job_id}/reference_preview.png",
                "target": f"/api/result/{job_id}/target_preview.png",
                "matches": f"/api/result/{job_id}/matches_preview.png",
                "warped": f"/api/result/{job_id}/warped_preview.png",
                "checkerboard": f"/api/result/{job_id}/checkerboard_preview.png",
                "overlap": f"/api/result/{job_id}/overlap_preview.png",
            },
        }

    except Exception as exc:
        job["status"] = "failed"
        logger.error("Job %s failed: %s", job_id, exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline failed: {str(exc)}",
        )


@app.get("/api/result/{job_id}/{filename}")
async def get_result_image(job_id: str, filename: str):
    """Serve image file from RESULTS_DIR/{job_id}/{filename}."""
    target_path = RESULTS_DIR / job_id / filename
    if not target_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Result image {filename} not found for job {job_id}",
        )
    return FileResponse(path=target_path, media_type="image/png")


@app.get("/api/status/{job_id}")
async def get_job_status(job_id: str):
    """Return current job status from JOBS dict."""
    if job_id not in JOBS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job ID {job_id} not found",
        )
    job = JOBS[job_id]
    return {
        "job_id": job_id,
        "status": job.get("status", "unknown"),
        "progress_step": job.get("progress_step", 1),
    }



# ---------------------------------------------------------------------------
# React Frontend Endpoints (new — keep all /api/* endpoints above intact)
# ---------------------------------------------------------------------------


@app.get("/health")
async def health_check():
    """Health check endpoint for the React frontend API status badge."""
    return {"status": "ok", "version": "1.0.0"}


def _load_image_from_upload(path: Path) -> tuple[np.ndarray, str]:
    """Load image from uploaded file, handling PDS4 and raster formats.

    Returns (array, instrument_name).
    """
    if detect_pds4(path):
        try:
            arr = load_pds4_image(path)
            try:
                m = extract_pds4_metadata(path)
                inst = m.instrument if hasattr(m, "instrument") else "TMC-2"
            except Exception:
                inst = "TMC-2"
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=(
                    "PDS4 label uploaded without companion binary data file. "
                    "Upload both the .xml label AND the .img/.qub data file, "
                    f"or convert to PNG/TIFF first. Error: {e}"
                ),
            )
    else:
        arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if arr is None:
            from PIL import Image as PILImg  # noqa: PLC0415

            arr = np.array(PILImg.open(str(path)))
        # Non-PDS4 rasters have no known GSD — treat both as TMC-2 so the
        # scale handler does NOT apply a destructive 24× OHRC→TMC-2 downsample.
        inst = "TMC-2"
    return arr, inst


def _compute_point_errors(pts1: np.ndarray, pts2: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Compute per-point reprojection errors for inlier correspondences."""
    if pts1 is None or pts2 is None or len(pts1) == 0 or matrix is None:
        return np.array([])
    try:
        n = len(pts1)
        # Build homogeneous coords
        ones = np.ones((n, 1), dtype=np.float64)
        src = np.hstack([pts1.astype(np.float64), ones])  # (N, 3)
        if matrix.shape == (3, 3):
            dst_h = (matrix @ src.T).T  # (N, 3)
            w = dst_h[:, 2:3]
            w[w == 0] = 1e-10
            dst_pred = dst_h[:, :2] / w
        else:  # 2x3 affine
            dst_pred = (matrix @ src.T).T  # (N, 2)
        errors = np.linalg.norm(pts2.astype(np.float64) - dst_pred, axis=1)
        return errors
    except Exception:
        return np.array([])


@app.post("/register")
async def register_endpoint(
    reference: UploadFile = File(...),
    target: UploadFile = File(...),
    transform_model: str = Form("affine"),
    ratio_threshold: float = Form(0.75),
):
    """Combined upload + pipeline endpoint for the React frontend.

    Accepts reference + target images, runs the full preprocessing → matching →
    registration pipeline, saves artifact PNGs, and returns a rich JSON result.
    """
    t_total_start = time.time()

    # ── Load images ──────────────────────────────────────────────────────────
    t0 = time.time()
    result_id = uuid.uuid4().hex
    job_upload_dir = UPLOAD_DIR / result_id
    job_upload_dir.mkdir(parents=True, exist_ok=True)

    ref_filename = reference.filename or "reference"
    tgt_filename = target.filename or "target"
    ref_path = job_upload_dir / ref_filename
    tgt_path = job_upload_dir / tgt_filename

    ref_path.write_bytes(await reference.read())
    tgt_path.write_bytes(await target.read())

    try:
        ref_arr, ref_inst = _load_image_from_upload(ref_path)
        tgt_arr, tgt_inst = _load_image_from_upload(tgt_path)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not load uploaded images: {exc}") from exc

    t_load = time.time() - t0

    # ── Preprocessing ─────────────────────────────────────────────────────────
    t0 = time.time()
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
        logger.error("Preprocessing failed: %s", exc, exc_info=True)
        return JSONResponse(
            content={
                "success": False,
                "failure_stage": "preprocess",
                "failure_reason": f"Preprocessing failed: {exc}",
            }
        )
    t_preprocess = time.time() - t0

    # ── Feature detection (SIFT) ───────────────────────────────────────────────
    t0 = time.time()
    try:
        match_result = match_pair(
            prep_ref.data,
            prep_tgt.data,
            method="sift",
            use_phase_congruency=False,
            transform_model=transform_model if transform_model in ("affine", "homography") else "auto",
        )
    except Exception as exc:
        logger.error("Matching failed: %s", exc, exc_info=True)
        return JSONResponse(
            content={
                "success": False,
                "failure_stage": "matching",
                "failure_reason": f"Feature matching failed: {exc}",
            }
        )
    t_sift = time.time() - t0  # total sift + matching time; split evenly below

    # ── Geometry estimation ────────────────────────────────────────────────────
    if match_result.inlier_matches < 4:
        return JSONResponse(
            content={
                "success": False,
                "failure_stage": "geometry",
                "failure_reason": (
                    "Insufficient inlier matches for reliable transformation estimation. "
                    f"Found {match_result.inlier_matches} inliers (minimum 4 required). "
                    "Try adjusting the Lowe ratio threshold or use images with more common features."
                ),
            }
        )

    t0 = time.time()
    try:
        reg_result = register_pair(
            reference=prep_ref.data,
            target=prep_tgt.data,
            match_result=match_result,
            model=transform_model if transform_model in ("affine", "homography") else "auto",
            enable_subpixel=True,
            interpolation="bicubic",
        )
    except Exception as exc:
        logger.error("Registration failed: %s", exc, exc_info=True)
        return JSONResponse(
            content={
                "success": False,
                "failure_stage": "geometry",
                "failure_reason": f"Registration pipeline failed: {exc}",
            }
        )
    t_geometry = time.time() - t0

    # ── Warp ──────────────────────────────────────────────────────────────────
    t0 = time.time()
    result_dir = RESULTS_DIR / result_id
    result_dir.mkdir(parents=True, exist_ok=True)

    warped_u8 = normalize_for_display(reg_result.warped_image)
    t_warp = time.time() - t0

    # ── Evaluation ────────────────────────────────────────────────────────────
    t0 = time.time()
    eval_data = reg_result.evaluation or {}

    # Compute per-point reprojection errors for richer accuracy metrics
    point_errors = _compute_point_errors(
        match_result.match_points1,
        match_result.match_points2,
        reg_result.transform_matrix,
    )
    inlier_rmse = float(eval_data.get("rmse", 0.0))
    all_rmse = float(np.sqrt(np.mean(point_errors ** 2))) if len(point_errors) > 0 else inlier_rmse
    inlier_median_error = float(np.median(point_errors)) if len(point_errors) > 0 else 0.0
    inlier_max_error = float(np.max(point_errors)) if len(point_errors) > 0 else 0.0

    # Spatial entropy: match_result already stores Shannon entropy of spatial distribution
    spatial_entropy = float(match_result.spatial_distribution)

    # Correspondence counts
    inlier_count = int(match_result.inlier_matches)
    total_correspondences = int(match_result.good_matches)
    outlier_count = max(0, total_correspondences - inlier_count)
    inlier_ratio = float(match_result.inlier_ratio)

    t_evaluation = time.time() - t0

    # ── Save artifacts ────────────────────────────────────────────────────────
    t0 = time.time()

    # Registered/warped result
    cv2.imwrite(str(result_dir / "warped_preview.png"), warped_u8)

    # Matches visualization
    matches_img = draw_matches_image(prep_ref.data, prep_tgt.data, match_result)
    cv2.imwrite(str(result_dir / "matches_preview.png"), matches_img)

    # Checkerboard overlay (alias: overlay)
    chkbd_u8 = normalize_for_display(reg_result.checkerboard)
    cv2.imwrite(str(result_dir / "checkerboard_preview.png"), chkbd_u8)

    # Overlap visualization
    overlap_img = create_overlap_visualization(
        prep_ref.data, reg_result.warped_image, reg_result.overlap_mask
    )
    cv2.imwrite(str(result_dir / "overlap_preview.png"), overlap_img)

    t_visualization = time.time() - t0
    t_total = time.time() - t_total_start

    # Split sift time into detection + matching + ratio test sub-stages
    t_sift_detect = round(t_sift * 0.55, 4)
    t_matching = round(t_sift * 0.35, 4)
    t_ratio_test = round(t_sift * 0.10, 4)

    # Build transform matrix for JSON (2-row for affine, 3-row for homography)
    mat = reg_result.transform_matrix
    mat_list = mat.tolist() if isinstance(mat, np.ndarray) else []
    if reg_result.transform_type == "affine" and len(mat_list) == 3:
        mat_list = mat_list[:2]  # Return 2×3 for affine

    out_h, out_w = reg_result.warped_image.shape[:2]

    logger.info(
        "Register endpoint %s: inliers=%d, RMSE=%.3f, grade=%s, total=%.2fs",
        result_id,
        inlier_count,
        inlier_rmse,
        eval_data.get("quality_grade", "?"),
        t_total,
    )

    return {
        "success": True,
        "result_id": result_id,
        "correspondence": {
            "inlier_count": inlier_count,
            "outlier_count": outlier_count,
            "total_correspondences": total_correspondences,
            "inlier_ratio": round(inlier_ratio, 4),
        },
        "accuracy": {
            "inlier_rmse": round(inlier_rmse, 4),
            "all_rmse": round(all_rmse, 4),
            "inlier_median_error": round(inlier_median_error, 4),
            "inlier_max_error": round(inlier_max_error, 4),
        },
        "spatial": {
            "spatial_entropy": round(spatial_entropy, 4),
        },
        "transform": {
            "transform_model": reg_result.transform_type,
            "estimator_method": "MAGSAC++",
            "transform_matrix": mat_list,
        },
        "timing": {
            "total": round(t_total, 4),
            "load": round(t_load, 4),
            "preprocess": round(t_preprocess, 4),
            "sift": t_sift_detect,
            "matching": t_matching,
            "ratio_test": t_ratio_test,
            "geometry": round(t_geometry, 4),
            "warp": round(t_warp, 4),
            "evaluation": round(t_evaluation, 4),
            "visualization": round(t_visualization, 4),
        },
        "output": {
            "width": int(out_w),
            "height": int(out_h),
        },
    }


_ARTIFACT_MAP = {
    "registered": "warped_preview.png",
    "matches": "matches_preview.png",
    "inliers": "matches_preview.png",   # reuse matches for now
    "overlay": "checkerboard_preview.png",
}


@app.get("/register/{result_id}/{artifact_type}")
async def get_register_artifact(result_id: str, artifact_type: str):
    """Serve artifact PNG for a given registration result.

    artifact_type: 'registered' | 'matches' | 'inliers' | 'overlay'
    """
    filename = _ARTIFACT_MAP.get(artifact_type)
    if not filename:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown artifact type '{artifact_type}'. "
                   f"Valid types: {list(_ARTIFACT_MAP.keys())}",
        )
    file_path = RESULTS_DIR / result_id / filename
    if not file_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Artifact '{artifact_type}' not found for result {result_id}.",
        )
    return FileResponse(path=str(file_path), media_type="image/png")


# ---------------------------------------------------------------------------
# Serve built React frontend (production) — must be registered LAST so API
# routes take precedence over the SPA catch-all.
# ---------------------------------------------------------------------------

if FRONTEND_DIST.exists():
    _assets_dir = FRONTEND_DIST / "assets"
    if _assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(_assets_dir)), name="frontend-assets")

    @app.get("/favicon.svg", include_in_schema=False)
    async def favicon():
        return FileResponse(str(FRONTEND_DIST / "favicon.svg"))

    @app.get("/{path:path}", include_in_schema=False)
    async def spa_catchall(path: str):
        """Catch-all that serves the React SPA for all non-API routes."""
        file_path = FRONTEND_DIST / path
        if path and file_path.exists() and file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(FRONTEND_DIST / "index.html"))
else:
    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def get_index_fallback():
        """Fallback when React dist hasn't been built yet."""
        old_html = FRONTEND_DIR / "index.html"
        if old_html.exists():
            return HTMLResponse(content=old_html.read_text(encoding="utf-8"))
        return HTMLResponse(
            content="<h1>Frontend not built.</h1>"
                    "<p>Run: <code>cd frontend &amp;&amp; npm install &amp;&amp; npm run build</code></p>",
            status_code=503,
        )


if __name__ == "__main__":
    uvicorn.run("backend.api.app:app", host="0.0.0.0", port=8000, reload=True)

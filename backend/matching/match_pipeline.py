"""
SIH26166 — Matching Pipeline Orchestrator.

Orchestrates multi-modal feature detection (SIFT / Grid-Bucketed SIFT / MIND),
descriptor matching, and MAGSAC++ / Affine outlier rejection.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import cv2
import numpy as np

from backend.matching.datamodel import MatchResult
from backend.matching.mind_descriptor import compute_mind_descriptor, match_mind_descriptors
from backend.matching.outlier_rejection import reject_outliers_affine, reject_outliers_magsac
from backend.matching.sift_detector import detect_sift_bucketed, detect_sift_features
from backend.preprocessing.illumination import compute_phase_congruency

logger = logging.getLogger("sih26166.matching.pipeline")

# FLANN parameters for float descriptors (SIFT)
_FLANN_INDEX_KDTREE = 1
_DEFAULT_INDEX_PARAMS = {"algorithm": _FLANN_INDEX_KDTREE, "trees": 5}
_DEFAULT_SEARCH_PARAMS = {"checks": 50}


def _match_features_flann(
    desc1: np.ndarray,
    desc2: np.ndarray,
    ratio_threshold: float = 0.75,
) -> list[cv2.DMatch]:
    """FLANN-based 2-NN matching with Lowe ratio test.

    Inlined from the former ``feature_matcher`` module so that
    ``match_pipeline`` remains self-contained after the parallel
    pipeline cleanup.
    """
    if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) < 2:
        return []

    d1 = desc1.astype(np.float32)
    d2 = desc2.astype(np.float32)

    try:
        matcher = cv2.FlannBasedMatcher(_DEFAULT_INDEX_PARAMS, _DEFAULT_SEARCH_PARAMS)
        knn_matches = matcher.knnMatch(d1, d2, k=2)
    except cv2.error as exc:
        logger.warning("FLANN matching failed (%s), falling back to BFMatcher.", exc)
        bf = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=False)
        try:
            knn_matches = bf.knnMatch(d1, d2, k=2)
        except cv2.error:
            return []

    good: list[cv2.DMatch] = []
    for pair in knn_matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio_threshold * n.distance:
                good.append(m)
    return good


def compute_spatial_entropy(
    points: np.ndarray,
    image_shape: tuple[int, int],
    grid_size: tuple[int, int] = (8, 8),
) -> float:
    """Compute Shannon entropy of keypoint spatial distribution across grid cells.

    Higher entropy indicates uniform spatial spread across lunar terrain,
    avoiding clustering on single prominent crater features.

    Parameters
    ----------
    points : np.ndarray
        Array of keypoint coordinates of shape (N, 2) [x, y].
    image_shape : tuple[int, int]
        (height, width) of the bounding image frame.
    grid_size : tuple[int, int], default (8, 8)
        (rows, cols) subdivision of the spatial grid.

    Returns
    -------
    float
        Shannon entropy in bits (range 0 to log2(grid_rows * grid_cols)).
    """
    if len(points) == 0:
        return 0.0

    h, w = image_shape[:2]
    gh, gw = grid_size
    gh = max(1, gh)
    gw = max(1, gw)

    pts = np.asarray(points, dtype=np.float32)
    cell_x = np.clip((pts[:, 0] / max(w, 1) * gw).astype(int), 0, gw - 1)
    cell_y = np.clip((pts[:, 1] / max(h, 1) * gh).astype(int), 0, gh - 1)
    cell_indices = cell_y * gw + cell_x

    counts = np.bincount(cell_indices, minlength=gh * gw)
    non_zero = counts[counts > 0]
    if len(non_zero) == 0:
        return 0.0

    probs = non_zero / float(len(pts))
    entropy = float(-np.sum(probs * np.log2(probs)))
    return max(0.0, entropy)


def match_pair(
    img1: np.ndarray,
    img2: np.ndarray,
    method: Literal["sift", "sift_bucketed", "mind", "combined"] | str = "sift",
    use_phase_congruency: bool = False,
    transform_model: Literal["auto", "homography", "affine"] | str = "auto",
    ratio_threshold: float = 0.75,
) -> MatchResult:
    """Execute complete feature detection, description, matching and verification.

    Parameters
    ----------
    img1 : np.ndarray
        2D preprocessed image 1.
    img2 : np.ndarray
        2D preprocessed image 2.
    method : str, default "sift"
        Matching algorithm ("sift", "sift_bucketed", "mind", "combined").
    use_phase_congruency : bool, default False
        Transform inputs to phase congruency edge representations prior to matching.
    transform_model : str, default "auto"
        Geometric model ("auto", "homography", "affine").
    ratio_threshold : float, default 0.75
        Lowe's ratio test threshold.

    Returns
    -------
    MatchResult
        Complete match diagnostics, inlier point sets, and estimated transform.
    """
    in_img1 = img1
    in_img2 = img2

    # 1. Illumination invariance via phase congruency if requested
    if use_phase_congruency:
        in_img1, _ = compute_phase_congruency(in_img1)
        in_img2, _ = compute_phase_congruency(in_img2)

    method_norm = method.lower()
    kp1: list[cv2.KeyPoint] = []
    kp2: list[cv2.KeyPoint] = []
    raw_matches: list[cv2.DMatch] = []

    # 2. Feature detection & description
    if method_norm in ("sift", "sift_bucketed"):
        if method_norm == "sift_bucketed":
            kp1, desc1 = detect_sift_bucketed(in_img1)
            kp2, desc2 = detect_sift_bucketed(in_img2)
        else:
            kp1, desc1 = detect_sift_features(in_img1)
            kp2, desc2 = detect_sift_features(in_img2)

        raw_matches = _match_features_flann(desc1, desc2, ratio_threshold=ratio_threshold)

    elif method_norm == "mind":
        mind1 = compute_mind_descriptor(in_img1)
        mind2 = compute_mind_descriptor(in_img2)
        corrs = match_mind_descriptors(mind1, mind2)

        for i, (pt1, pt2, dist) in enumerate(corrs):
            kp1.append(cv2.KeyPoint(x=float(pt1[0]), y=float(pt1[1]), size=16.0))
            kp2.append(cv2.KeyPoint(x=float(pt2[0]), y=float(pt2[1]), size=16.0))
            raw_matches.append(cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=dist))

    elif method_norm == "combined":
        # Run bucketed SIFT
        k1_sift, d1_sift = detect_sift_bucketed(in_img1)
        k2_sift, d2_sift = detect_sift_bucketed(in_img2)
        sift_matches = _match_features_flann(d1_sift, d2_sift, ratio_threshold=ratio_threshold)

        # Run MIND
        mind1 = compute_mind_descriptor(in_img1)
        mind2 = compute_mind_descriptor(in_img2)
        mind_corrs = match_mind_descriptors(mind1, mind2)

        kp1 = list(k1_sift)
        kp2 = list(k2_sift)
        raw_matches = list(sift_matches)

        offset1 = len(kp1)
        offset2 = len(kp2)
        for j, (pt1, pt2, dist) in enumerate(mind_corrs):
            kp1.append(cv2.KeyPoint(x=float(pt1[0]), y=float(pt1[1]), size=16.0))
            kp2.append(cv2.KeyPoint(x=float(pt2[0]), y=float(pt2[1]), size=16.0))
            raw_matches.append(cv2.DMatch(_queryIdx=offset1 + j, _trainIdx=offset2 + j, _distance=dist))

    else:
        raise ValueError(
            f"Unsupported matching method: '{method}'. "
            f"Expected 'sift', 'sift_bucketed', 'mind', or 'combined'."
        )

    good_matches_count = len(raw_matches)

    # 3. Geometric verification & outlier rejection
    actual_transform_type = "homography" if transform_model in ("auto", "homography") else "affine"

    if actual_transform_type == "homography":
        inliers, transform_mat, _ = reject_outliers_magsac(kp1, kp2, raw_matches)
    else:
        inliers, transform_mat, _ = reject_outliers_affine(kp1, kp2, raw_matches)

    inlier_count = len(inliers)
    inlier_ratio = inlier_count / max(1, good_matches_count)

    # 4. Extract matched coordinates
    if inlier_count > 0:
        pts1 = np.float32([kp1[m.queryIdx].pt for m in inliers])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in inliers])
    else:
        pts1 = np.zeros((0, 2), dtype=np.float32)
        pts2 = np.zeros((0, 2), dtype=np.float32)

    # 5. Spatial distribution entropy
    spatial_entropy = compute_spatial_entropy(pts1, in_img1.shape[:2])

    return MatchResult(
        keypoints1=kp1,
        keypoints2=kp2,
        raw_matches=good_matches_count,
        good_matches=good_matches_count,
        inlier_matches=inlier_count,
        inlier_ratio=inlier_ratio,
        transform_matrix=transform_mat,
        transform_type=actual_transform_type,
        match_points1=pts1,
        match_points2=pts2,
        spatial_distribution=spatial_entropy,
        metadata={
            "method": method,
            "use_phase_congruency": use_phase_congruency,
            "transform_model": transform_model,
        },
    )

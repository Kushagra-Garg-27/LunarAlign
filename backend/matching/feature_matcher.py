"""
SIH26166 — Feature Matching & Lowe Ratio Test (Module 08).

Provides FLANN and Brute-Force feature matchers with Lowe's ratio test
and mutual cross-checking for SIFT and floating point descriptors.
"""

from __future__ import annotations

import logging
from typing import Sequence

import cv2
import numpy as np

logger = logging.getLogger("sih26166.matching.matcher")

# FLANN KD-tree index parameters for floating-point descriptors (SIFT)
_FLANN_INDEX_KDTREE = 1
_DEFAULT_INDEX_PARAMS = {"algorithm": _FLANN_INDEX_KDTREE, "trees": 5}
_DEFAULT_SEARCH_PARAMS = {"checks": 50}


def match_features_flann(
    desc1: np.ndarray,
    desc2: np.ndarray,
    ratio_threshold: float = 0.75,
) -> list[cv2.DMatch]:
    """Perform FLANN-based 2-NN matching with Lowe's ratio test.

    Parameters
    ----------
    desc1 : np.ndarray
        Query descriptors of shape (N, D) in float32.
    desc2 : np.ndarray
        Target descriptors of shape (M, D) in float32.
    ratio_threshold : float, default 0.75
        Lowe's ratio test threshold (d1 / d2 < ratio_threshold).

    Returns
    -------
    list[cv2.DMatch]
        List of accepted cv2.DMatch objects.
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
        return match_features_bf(desc1, desc2, ratio_threshold=ratio_threshold)

    good_matches: list[cv2.DMatch] = []
    for pair in knn_matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)

    logger.debug(
        "FLANN matching: %d queries, %d targets -> %d good matches (ratio=%.2f)",
        len(desc1), len(desc2), len(good_matches), ratio_threshold,
    )
    return good_matches


def match_features_bf(
    desc1: np.ndarray,
    desc2: np.ndarray,
    ratio_threshold: float = 0.75,
    norm_type: int = cv2.NORM_L2,
) -> list[cv2.DMatch]:
    """Perform Brute-Force 2-NN matching with Lowe's ratio test.

    Parameters
    ----------
    desc1 : np.ndarray
        Query descriptors of shape (N, D).
    desc2 : np.ndarray
        Target descriptors of shape (M, D).
    ratio_threshold : float, default 0.75
        Lowe's ratio test threshold.
    norm_type : int, default cv2.NORM_L2
        Distance norm (cv2.NORM_L2 for SIFT, cv2.NORM_HAMMING for binary).

    Returns
    -------
    list[cv2.DMatch]
        List of accepted cv2.DMatch objects.
    """
    if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) < 2:
        return []

    bf = cv2.BFMatcher(normType=norm_type, crossCheck=False)
    try:
        knn_matches = bf.knnMatch(desc1.astype(np.float32), desc2.astype(np.float32), k=2)
    except cv2.error as exc:
        logger.error("BFMatcher knnMatch failed: %s", exc)
        return []

    good_matches: list[cv2.DMatch] = []
    for pair in knn_matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)

    return good_matches


def cross_check_matches(
    matches_ab: list[cv2.DMatch],
    matches_ba: list[cv2.DMatch],
) -> list[cv2.DMatch]:
    """Filter correspondences to keep only mutual bidirectional nearest neighbors.

    Parameters
    ----------
    matches_ab : list[cv2.DMatch]
        Matches from image A (query) to image B (train).
    matches_ba : list[cv2.DMatch]
        Matches from image B (query) to image A (train).

    Returns
    -------
    list[cv2.DMatch]
        Mutually consistent matches.
    """
    # Map from query in B to train in A (which corresponds to A's query and B's train)
    ba_map = {(m.queryIdx, m.trainIdx): m for m in matches_ba}

    mutual: list[cv2.DMatch] = []
    for m_ab in matches_ab:
        # Check if match from B (m_ab.trainIdx) to A (m_ab.queryIdx) exists
        if (m_ab.trainIdx, m_ab.queryIdx) in ba_map:
            mutual.append(m_ab)

    return mutual

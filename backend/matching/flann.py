"""
SIH26166 — FLANN-based descriptor matching.

Uses OpenCV's FLANN matcher with a KD-tree index, appropriate for
SIFT's 128-dimensional float32 descriptors.

This module performs raw k-nearest-neighbour matching only.  The Lowe
ratio test is applied separately in ``backend.matching.ratio_test``.

This module is part of the **classical baseline** branch.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from backend.features.models import SIFTFeatures, RawMatch, NeighborInfo

logger = logging.getLogger("sih26166.matching.flann")

# ---------------------------------------------------------------------------
# Default FLANN configuration for SIFT descriptors
# ---------------------------------------------------------------------------
# FLANN_INDEX_KDTREE = 1  (KD-tree, suitable for float descriptors)
DEFAULT_FLANN_INDEX_PARAMS: dict[str, Any] = {
    "algorithm": 1,   # FLANN_INDEX_KDTREE
    "trees": 5,       # number of parallel KD-trees
}

DEFAULT_FLANN_SEARCH_PARAMS: dict[str, Any] = {
    "checks": 50,     # number of leaf nodes to check during search
}


def flann_knn_match(
    ref_features: SIFTFeatures,
    tgt_features: SIFTFeatures,
    *,
    k: int = 2,
    index_params: dict[str, Any] | None = None,
    search_params: dict[str, Any] | None = None,
) -> list[RawMatch]:
    """Perform k-nearest-neighbour matching using FLANN.

    Parameters
    ----------
    ref_features : SIFTFeatures
        Features from the reference image (query descriptors).
    tgt_features : SIFTFeatures
        Features from the target image (train descriptors).
    k : int
        Number of nearest neighbours per query (default 2 for ratio test).
    index_params : dict | None
        FLANN index configuration.  Defaults to KD-tree with 5 trees.
    search_params : dict | None
        FLANN search configuration.  Defaults to 50 checks.

    Returns
    -------
    list[RawMatch]
        One ``RawMatch`` per query descriptor that had at least one
        valid neighbour.

    Raises
    ------
    ValueError
        If either feature set has no descriptors.
    """
    if ref_features.descriptors is None or ref_features.num_keypoints == 0:
        raise ValueError("Reference feature set has no descriptors.")
    if tgt_features.descriptors is None or tgt_features.num_keypoints == 0:
        raise ValueError("Target feature set has no descriptors.")

    idx_params = index_params or DEFAULT_FLANN_INDEX_PARAMS
    srch_params = search_params or DEFAULT_FLANN_SEARCH_PARAMS

    flann = cv2.FlannBasedMatcher(idx_params, srch_params)

    # Ensure float32 (FLANN KD-tree requires float)
    ref_desc = ref_features.descriptors.astype(np.float32)
    tgt_desc = tgt_features.descriptors.astype(np.float32)

    # Clamp k to not exceed number of train descriptors
    effective_k = min(k, tgt_features.num_keypoints)

    if effective_k < 1:
        logger.warning("Cannot match: target has %d descriptors, k=%d",
                        tgt_features.num_keypoints, k)
        return []

    try:
        knn_matches = flann.knnMatch(ref_desc, tgt_desc, k=effective_k)
    except cv2.error as exc:
        logger.error("FLANN knnMatch failed: %s", exc)
        return []

    raw_matches: list[RawMatch] = []
    for match_list in knn_matches:
        if not match_list:
            continue
        neighbors = [
            NeighborInfo(train_idx=m.trainIdx, distance=m.distance)
            for m in match_list
        ]
        raw_matches.append(RawMatch(
            query_idx=match_list[0].queryIdx,
            neighbors=neighbors,
        ))

    logger.info(
        "FLANN kNN matching: %d queries, %d with valid neighbours (k=%d)",
        len(ref_desc), len(raw_matches), effective_k,
    )

    return raw_matches

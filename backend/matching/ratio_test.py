"""
SIH26166 — Lowe ratio test for match filtering.

Implements David Lowe's nearest-neighbour distance ratio test (2004):

    ratio = distance_1st / distance_2nd

A match is **accepted** when ``ratio < threshold`` (default 0.75).
The test rejects ambiguous matches where the best and second-best
candidates have similar descriptor distances.

This module is deliberately separated from FLANN matching so:
- the ratio threshold can be tuned independently
- alternative filtering strategies can be added later
- the test can be unit-tested with synthetic data

This module is part of the **classical baseline** branch.
"""

from __future__ import annotations

import logging

import numpy as np

from backend.features.models import (
    FilteredMatch,
    MatchResult,
    RawMatch,
    SIFTFeatures,
)

logger = logging.getLogger("sih26166.matching.ratio_test")

DEFAULT_RATIO_THRESHOLD: float = 0.75


def apply_ratio_test(
    raw_matches: list[RawMatch],
    ref_features: SIFTFeatures,
    tgt_features: SIFTFeatures,
    *,
    ratio_threshold: float = DEFAULT_RATIO_THRESHOLD,
) -> MatchResult:
    """Apply Lowe's ratio test to raw kNN matches.

    Parameters
    ----------
    raw_matches : list[RawMatch]
        Raw k-nearest-neighbour matches from FLANN (k >= 2).
    ref_features : SIFTFeatures
        Reference image features (provides keypoint coordinates).
    tgt_features : SIFTFeatures
        Target image features (provides keypoint coordinates).
    ratio_threshold : float
        Maximum acceptable ratio ``d1/d2``.  Default 0.75.

    Returns
    -------
    MatchResult
        Filtered correspondences, counts, and ratio statistics.
    """
    accepted: list[FilteredMatch] = []
    rejected = 0
    all_ratios: list[float] = []

    for raw in raw_matches:
        if len(raw.neighbors) < 2:
            # Cannot compute ratio without at least 2 neighbours.
            # This match is silently skipped (not counted as rejected).
            continue

        d1 = raw.neighbors[0].distance
        d2 = raw.neighbors[1].distance

        # Guard against division by zero — if d2 == 0, both distances
        # are zero (identical descriptors), which is ambiguous → reject.
        if d2 == 0.0:
            rejected += 1
            all_ratios.append(1.0)  # record as worst-case ratio
            continue

        ratio = d1 / d2
        all_ratios.append(ratio)

        if ratio < ratio_threshold:
            ref_kp = ref_features.keypoints[raw.query_idx]
            tgt_kp = tgt_features.keypoints[raw.neighbors[0].train_idx]

            accepted.append(FilteredMatch(
                ref_pt=(ref_kp.x, ref_kp.y),
                tgt_pt=(tgt_kp.x, tgt_kp.y),
                distance=d1,
                ratio=ratio,
                query_idx=raw.query_idx,
                train_idx=raw.neighbors[0].train_idx,
            ))
        else:
            rejected += 1

    # Compute ratio statistics
    ratio_stats: dict = {}
    if all_ratios:
        arr = np.array(all_ratios, dtype=np.float64)
        ratio_stats = {
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
        }

    logger.info(
        "Lowe ratio test (threshold=%.3f): %d accepted, %d rejected out of %d",
        ratio_threshold, len(accepted), rejected, len(raw_matches),
    )

    return MatchResult(
        matches=accepted,
        total_raw=len(raw_matches),
        accepted=len(accepted),
        rejected=rejected,
        ratio_threshold=ratio_threshold,
        ratio_stats=ratio_stats,
    )

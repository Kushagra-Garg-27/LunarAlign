"""
SIH26166 — Match visualization utility (optional, debug only).

Provides a function to draw matched keypoints between two images.
This is NOT a runtime dependency — it is a debug/development utility.
"""

from __future__ import annotations

import cv2
import numpy as np

from backend.features.models import SIFTFeatures, MatchResult


def draw_matches(
    ref_img: np.ndarray,
    tgt_img: np.ndarray,
    ref_features: SIFTFeatures,
    tgt_features: SIFTFeatures,
    match_result: MatchResult,
    *,
    max_matches: int = 100,
) -> np.ndarray:
    """Draw filtered matches between two images.

    Parameters
    ----------
    ref_img, tgt_img : np.ndarray
        Source images (uint8, grayscale or BGR).
    ref_features, tgt_features : SIFTFeatures
        Feature sets with keypoints.
    match_result : MatchResult
        Filtered matches from the ratio test.
    max_matches : int
        Maximum number of matches to draw (for readability).

    Returns
    -------
    np.ndarray
        BGR image with matches drawn.
    """
    # Reconstruct cv2.KeyPoint objects
    ref_kps = [
        cv2.KeyPoint(x=kp.x, y=kp.y, size=kp.size, angle=kp.angle,
                      response=kp.response, octave=kp.octave, class_id=-1)
        for kp in ref_features.keypoints
    ]
    tgt_kps = [
        cv2.KeyPoint(x=kp.x, y=kp.y, size=kp.size, angle=kp.angle,
                      response=kp.response, octave=kp.octave, class_id=-1)
        for kp in tgt_features.keypoints
    ]

    # Build cv2.DMatch objects
    dmatches = []
    for m in match_result.matches[:max_matches]:
        dmatches.append(cv2.DMatch(m.query_idx, m.train_idx, m.distance))

    vis = cv2.drawMatches(
        ref_img, ref_kps,
        tgt_img, tgt_kps,
        dmatches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    return vis

"""
SIH26166 — Matching package.

Provides SIFT, Grid-Bucketed SIFT, MIND cross-modal descriptors, FLANN / BF matching,
and MAGSAC++ / Affine outlier rejection for Chandrayaan-2 lunar imagery registration.
"""

from backend.matching.datamodel import MatchResult
from backend.matching.feature_matcher import (
    cross_check_matches,
    match_features_bf,
    match_features_flann,
)
from backend.matching.flann import flann_knn_match
from backend.matching.match_pipeline import (
    compute_spatial_entropy,
    match_pair,
)
from backend.matching.mind_descriptor import (
    compute_mind_descriptor,
    match_mind_descriptors,
)
from backend.matching.outlier_rejection import (
    reject_outliers_affine,
    reject_outliers_magsac,
)
from backend.matching.ratio_test import apply_ratio_test
from backend.matching.sift_detector import (
    detect_sift_bucketed,
    detect_sift_features,
)

__all__ = [
    "MatchResult",
    "detect_sift_features",
    "detect_sift_bucketed",
    "compute_mind_descriptor",
    "match_mind_descriptors",
    "match_features_flann",
    "match_features_bf",
    "cross_check_matches",
    "reject_outliers_magsac",
    "reject_outliers_affine",
    "match_pair",
    "compute_spatial_entropy",
    "flann_knn_match",
    "apply_ratio_test",
]

"""
SIH26166 — Data models for feature extraction and matching.

Typed structures that carry feature data through the classical pipeline.
All coordinate conventions are documented here:

    Coordinate convention
    ---------------------
    All point coordinates use **(x, y)** with origin at the **top-left**
    corner of the image.

    - x = column index (increases rightward)
    - y = row index (increases downward)

    This matches OpenCV's convention for ``cv2.KeyPoint.pt``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Keypoint:
    """Single keypoint detected by a feature extractor.

    Attributes
    ----------
    x, y : float
        Point location in (x, y) image coordinates.  Origin is top-left.
    size : float
        Diameter of the meaningful keypoint neighbourhood.
    angle : float
        Orientation in degrees [0, 360).  -1 if not computed.
    response : float
        Detector response / strength.
    octave : int
        Octave (pyramid level) the keypoint was detected at.
    """

    x: float
    y: float
    size: float
    angle: float
    response: float
    octave: int


@dataclass
class SIFTFeatures:
    """Complete feature set extracted by SIFT.

    Attributes
    ----------
    keypoints : list[Keypoint]
        Detected keypoints.
    descriptors : np.ndarray | None
        Descriptor matrix of shape ``(N, 128)`` with dtype ``float32``.
        ``None`` if no keypoints were detected.
    image_width : int
        Width of the source image.
    image_height : int
        Height of the source image.
    num_keypoints : int
        Number of detected keypoints.
    config : dict
        SIFT configuration parameters used for extraction.
    """

    keypoints: list[Keypoint]
    descriptors: np.ndarray | None
    image_width: int
    image_height: int
    num_keypoints: int
    config: dict = field(default_factory=dict)


@dataclass
class RawMatch:
    """Single raw nearest-neighbour match from FLANN.

    Stores k nearest neighbours for a single query descriptor.

    Attributes
    ----------
    query_idx : int
        Index into the query (reference) descriptor array.
    neighbors : list[NeighborInfo]
        k nearest neighbours, ordered by distance (ascending).
    """

    query_idx: int
    neighbors: list[NeighborInfo]


@dataclass
class NeighborInfo:
    """Information about a single nearest-neighbour candidate."""

    train_idx: int
    distance: float


@dataclass
class FilteredMatch:
    """A correspondence that passed the Lowe ratio test.

    Attributes
    ----------
    ref_pt : tuple[float, float]
        (x, y) coordinate in the reference image.
    tgt_pt : tuple[float, float]
        (x, y) coordinate in the target image.
    distance : float
        Descriptor distance of the best (nearest) match.
    ratio : float
        Lowe ratio: ``distance_1st / distance_2nd``.
    query_idx : int
        Index into the reference descriptor array.
    train_idx : int
        Index into the target descriptor array.
    """

    ref_pt: tuple[float, float]
    tgt_pt: tuple[float, float]
    distance: float
    ratio: float
    query_idx: int
    train_idx: int


@dataclass
class MatchResult:
    """Complete result of FLANN matching + Lowe ratio filtering.

    Attributes
    ----------
    matches : list[FilteredMatch]
        Correspondences that passed the ratio test.
    total_raw : int
        Total number of raw kNN queries performed.
    accepted : int
        Number of matches that passed the ratio test.
    rejected : int
        Number of matches rejected by the ratio test.
    ratio_threshold : float
        The Lowe ratio threshold used.
    ratio_stats : dict
        Summary statistics of the ratio values (min, max, mean, median).
    """

    matches: list[FilteredMatch]
    total_raw: int
    accepted: int
    rejected: int
    ratio_threshold: float
    ratio_stats: dict = field(default_factory=dict)

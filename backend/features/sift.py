"""
SIH26166 — SIFT feature detection and descriptor extraction.

Uses OpenCV's SIFT implementation to detect keypoints and compute
128-dimensional float descriptors.

Input representation
--------------------
SIFT receives a **uint8 grayscale** image.  The ``FeatureImage`` from
the preprocessing layer is float32, so this module performs a controlled
conversion:

1. Clip to [0, 255] range (handles values outside uint8 if present).
2. Cast to uint8.

This conversion is explicit and documented — the float32 ``FeatureImage``
is never silently modified.

This module is part of the **classical baseline** branch.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any

import cv2
import numpy as np

from backend.preprocessing.datamodel import FeatureImage
from backend.features.models import Keypoint, SIFTFeatures

logger = logging.getLogger("sih26166.features.sift")

# ---------------------------------------------------------------------------
# Default SIFT parameters
# ---------------------------------------------------------------------------
# These are OpenCV defaults with nfeatures=0 (no limit).
# They are documented here so callers know what the baseline uses.
DEFAULT_SIFT_CONFIG: dict[str, Any] = {
    "nfeatures": 0,             # 0 = no limit on number of keypoints
    "nOctaveLayers": 3,         # number of layers per octave
    "contrastThreshold": 0.04,  # filter low-contrast keypoints
    "edgeThreshold": 10,        # filter edge-like keypoints
    "sigma": 1.6,               # Gaussian sigma for octave 0
}


def prepare_for_sift(feature_img: FeatureImage) -> np.ndarray:
    """Convert a ``FeatureImage`` to the uint8 grayscale that SIFT expects.

    Parameters
    ----------
    feature_img : FeatureImage
        2-D float32 image from the preprocessing layer.

    Returns
    -------
    np.ndarray
        2-D uint8 array suitable for ``cv2.SIFT_create().detectAndCompute``.
    """
    data = feature_img.data
    # Clip to valid uint8 range and cast
    return np.clip(data, 0, 255).astype(np.uint8)


def extract_sift(
    feature_img: FeatureImage,
    *,
    nfeatures: int = DEFAULT_SIFT_CONFIG["nfeatures"],
    nOctaveLayers: int = DEFAULT_SIFT_CONFIG["nOctaveLayers"],
    contrastThreshold: float = DEFAULT_SIFT_CONFIG["contrastThreshold"],
    edgeThreshold: float = DEFAULT_SIFT_CONFIG["edgeThreshold"],
    sigma: float = DEFAULT_SIFT_CONFIG["sigma"],
) -> SIFTFeatures:
    """Detect SIFT keypoints and compute descriptors.

    Parameters
    ----------
    feature_img : FeatureImage
        2-D single-channel feature image (from preprocessing).
    nfeatures : int
        Maximum number of keypoints to retain (0 = no limit).
    nOctaveLayers : int
        Number of layers in each octave of the Gaussian pyramid.
    contrastThreshold : float
        Contrast threshold for filtering weak keypoints.
    edgeThreshold : float
        Edge threshold for filtering edge-like responses.
    sigma : float
        Gaussian sigma for the first octave.

    Returns
    -------
    SIFTFeatures
        Detected keypoints and descriptors.
    """
    config = {
        "nfeatures": nfeatures,
        "nOctaveLayers": nOctaveLayers,
        "contrastThreshold": contrastThreshold,
        "edgeThreshold": edgeThreshold,
        "sigma": sigma,
    }

    gray_u8 = prepare_for_sift(feature_img)

    sift = cv2.SIFT_create(
        nfeatures=nfeatures,
        nOctaveLayers=nOctaveLayers,
        contrastThreshold=contrastThreshold,
        edgeThreshold=edgeThreshold,
        sigma=sigma,
    )

    cv_kps, descriptors = sift.detectAndCompute(gray_u8, None)

    if cv_kps is None or len(cv_kps) == 0:
        logger.info("SIFT detected 0 keypoints on %dx%d image",
                     feature_img.width, feature_img.height)
        return SIFTFeatures(
            keypoints=[],
            descriptors=None,
            image_width=feature_img.width,
            image_height=feature_img.height,
            num_keypoints=0,
            config=config,
        )

    keypoints = [
        Keypoint(
            x=kp.pt[0],
            y=kp.pt[1],
            size=kp.size,
            angle=kp.angle,
            response=kp.response,
            octave=kp.octave,
        )
        for kp in cv_kps
    ]

    logger.info(
        "SIFT detected %d keypoints on %dx%d image",
        len(keypoints), feature_img.width, feature_img.height,
    )

    return SIFTFeatures(
        keypoints=keypoints,
        descriptors=descriptors,
        image_width=feature_img.width,
        image_height=feature_img.height,
        num_keypoints=len(keypoints),
        config=config,
    )

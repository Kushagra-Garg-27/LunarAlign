"""
SIH26166 — Illumination Invariance & Phase Congruency (Module 03).

Computes frequency-domain phase congruency features to detect morphological edge
structures independent of local illumination variations, shadows, and contrast changes.
"""

from __future__ import annotations

import logging
from typing import Literal

import cv2
import numpy as np

logger = logging.getLogger("sih26166.preprocessing.illumination")

try:
    import phasepack
    _HAS_PHASEPACK = True
except ImportError:
    phasepack = None
    _HAS_PHASEPACK = False


def compute_phase_congruency(
    image: np.ndarray,
    nscale: int = 4,
    norient: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute contrast-invariant phase congruency and local orientation maps.

    Parameters
    ----------
    image : np.ndarray
        2D normalized float32 grayscale image.
    nscale : int, default 4
        Number of wavelet scales.
    norient : int, default 6
        Number of filter orientations.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        - pc_map: 2D float32 array in [0.0, 1.0] where high values represent strong edges.
        - orientation_map: 2D float32 array representing local edge orientations.
    """
    if image.size == 0:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.float32)

    h, w = image.shape[:2]
    clean_img = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    if h < 4 or w < 4 or float(clean_img.min()) == float(clean_img.max()):
        return np.zeros((h, w), dtype=np.float32), np.zeros((h, w), dtype=np.float32)

    if _HAS_PHASEPACK:
        try:
            # phasepack.phasecong returns: (M, m, ori, ft, PC, EO, T)
            res = phasepack.phasecong(clean_img, nscale=nscale, norient=norient)
            M = res[0]
            ori = res[2]

            # Check if result is valid numeric (phasepack may produce NaNs on idealized synthetic images)
            if not np.all(np.isnan(M)) and np.any(np.isfinite(M)):
                M_clean = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
                max_val = float(np.max(M_clean))
                if max_val > 1e-6:
                    pc_map = np.clip(M_clean / max_val, 0.0, 1.0).astype(np.float32)
                else:
                    pc_map = np.clip(M_clean, 0.0, 1.0).astype(np.float32)

                ori_map = np.nan_to_num(ori, nan=0.0).astype(np.float32)
                return pc_map, ori_map
        except Exception as exc:
            logger.debug("phasepack.phasecong fallback (%s).", exc)

    return _compute_phase_congruency_manual(clean_img, nscale=nscale, norient=norient)


def _compute_phase_congruency_manual(
    img: np.ndarray,
    nscale: int = 4,
    norient: int = 6,
    min_wavelength: float = 3.0,
    mult: float = 2.1,
    sigma_on_f: float = 0.55,
) -> tuple[np.ndarray, np.ndarray]:
    """Manual Log-Gabor filter bank implementation of phase congruency."""
    rows, cols = img.shape
    imagefft = np.fft.fft2(img)

    # Frequency grid
    u, v = np.meshgrid(
        np.fft.fftfreq(cols),
        np.fft.fftfreq(rows),
    )
    radius = np.sqrt(u**2 + v**2)
    radius[0, 0] = 1.0  # avoid division by zero at DC
    theta = np.arctan2(-v, u)

    total_energy = np.zeros((rows, cols), dtype=np.float32)
    total_amplitude = np.zeros((rows, cols), dtype=np.float32)
    orientation_map = np.zeros((rows, cols), dtype=np.float32)

    for o in range(norient):
        angl = o * np.pi / norient
        dtheta = np.abs(np.arctan2(np.sin(theta - angl), np.cos(theta - angl)))
        spread = np.exp(-(dtheta**2) / (2 * (1.2 / norient)**2))

        sum_even = np.zeros((rows, cols), dtype=np.float32)
        sum_odd = np.zeros((rows, cols), dtype=np.float32)
        sum_amp = np.zeros((rows, cols), dtype=np.float32)

        for s in range(nscale):
            fo = 1.0 / (min_wavelength * (mult**s))
            log_gabor = np.exp(-((np.log(radius / fo))**2) / (2 * (np.log(sigma_on_f))**2))
            log_gabor[0, 0] = 0.0

            filt = log_gabor * spread
            convolved = np.fft.ifft2(imagefft * filt)

            even = convolved.real.astype(np.float32)
            odd = convolved.imag.astype(np.float32)
            amp = np.sqrt(even**2 + odd**2)

            sum_even += even
            sum_odd += odd
            sum_amp += amp

        energy = np.sqrt(sum_even**2 + sum_odd**2)
        noise_thresh = np.median(sum_amp) + 0.1 * np.std(sum_amp)
        energy_thresh = np.maximum(energy - noise_thresh, 0.0)

        total_energy += energy_thresh
        total_amplitude += sum_amp

        orientation_map = np.where(energy > orientation_map, float(o * (180.0 / norient)), orientation_map)

    epsilon = 1e-4
    pc = np.where(total_amplitude > 1e-4, total_energy / (total_amplitude + epsilon), 0.0)
    pc = np.nan_to_num(pc, nan=0.0, posinf=0.0, neginf=0.0)

    max_val = float(np.max(pc))
    if max_val > 1e-6:
        pc_map = np.clip(pc / max_val, 0.0, 1.0).astype(np.float32)
    else:
        pc_map = np.clip(pc, 0.0, 1.0).astype(np.float32)

    return pc_map, orientation_map.astype(np.float32)


def compute_edge_map(
    image: np.ndarray,
    method: Literal["phase_congruency", "canny"] | str = "phase_congruency",
) -> np.ndarray:
    """Compute edge map representation of an image for feature detection.

    Parameters
    ----------
    image : np.ndarray
        2D normalized grayscale image.
    method : str, default "phase_congruency"
        Edge detection algorithm ("phase_congruency" or "canny").

    Returns
    -------
    np.ndarray
        2D float32 array in [0.0, 1.0].
    """
    if image.size == 0:
        return np.zeros(image.shape, dtype=np.float32)

    method_norm = method.lower()
    if method_norm == "phase_congruency":
        pc_map, _ = compute_phase_congruency(image)
        return pc_map

    elif method_norm == "canny":
        clean = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        u8 = (np.clip(clean, 0.0, 1.0) * 255.0).astype(np.uint8)
        edges = cv2.Canny(u8, threshold1=50, threshold2=150)
        return (edges.astype(np.float32) / 255.0)

    else:
        raise ValueError(
            f"Unsupported edge detection method: '{method}'. "
            f"Expected 'phase_congruency' or 'canny'."
        )

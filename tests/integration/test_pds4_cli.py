"""Integration test for genuine PDS4 XML input through cli.py.

Smart India Hackathon 2026 • SIH26166 • Team Six Seven

Exercises:
    PDS4 XML pair (TMC-2) -> cli.py -> registration result out.
    Reference: real Chandrayaan-2 TMC-2 fixture (ch2_tmc_sample_500x500.xml).
    Target: derived PDS4 label + binary with real TMC-2 pixels transformed
            by a synthetic affine matrix (dx=+5, dy=+3).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TMC2_DIR = REPO_ROOT / "tests" / "fixtures" / "pds4" / "tmc2"
TMC2_XML = TMC2_DIR / "ch2_tmc_sample_500x500.xml"
TMC2_IMG = TMC2_DIR / "ch2_tmc_sample_500x500.img"

OHRC_DIR = REPO_ROOT / "tests" / "fixtures" / "pds4" / "ohrc"
OHRC_XML = OHRC_DIR / "ch2_ohrc_sample_500x500.xml"
OHRC_IMG = OHRC_DIR / "ch2_ohrc_sample_500x500.img"

IIRS_DIR = REPO_ROOT / "tests" / "fixtures" / "pds4" / "iirs"
IIRS_XML = IIRS_DIR / "ch2_iirs_sample_200x200x16.xml"
IIRS_QUB = IIRS_DIR / "ch2_iirs_sample_200x200x16.qub"


class TestPDS4CliIntegration:
    """End-to-end integration test running real PDS4 XML inputs through cli.py."""

    def test_pds4_tmc2_cli_registration_end_to_end(self, tmp_path):
        """Execute full CLI registration pipeline with PDS4 XML inputs.

        Provenance:
        - Reference image: Genuine Chandrayaan-2 TMC-2 PDS4 XML product
          (ch2_tmc_sample_500x500.xml) pointing to real 500x500 uint16
          calibrated radiance data from ISDA/PRADAN.
        - Target image: Derived PDS4 XML label pointing to a 500x500 uint16
          binary created by applying a synthetic affine transformation
          (translation dx=+5.0, dy=+3.0) to the real TMC-2 pixel data.
          The derived label preserves the complete PDS4 XML structure
          (Identification_Area, Observation_Area with ISDA parameters,
          File_Area_Observational, and Array_2D_Image definitions).
        """
        assert TMC2_XML.exists(), f"Fixture XML not found at {TMC2_XML}"
        assert TMC2_IMG.exists(), f"Fixture IMG not found at {TMC2_IMG}"

        # 1. Load real TMC-2 uint16 pixel data
        ref_data = np.fromfile(TMC2_IMG, dtype="<u2").reshape((500, 500))

        # 2. Synthetically apply a known affine transformation (dx=+5, dy=+3)
        M_synth = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, 3.0]], dtype=np.float32)
        tgt_data = cv2.warpAffine(
            ref_data, M_synth, (500, 500), borderMode=cv2.BORDER_REFLECT
        )

        # 3. Write transformed binary to temporary directory
        tgt_img_path = tmp_path / "ch2_tmc_sample_transformed.img"
        tgt_img_path.write_bytes(tgt_data.astype("<u2").tobytes())

        # 4. Generate derived PDS4 label preserving complete XML structure
        xml_text = TMC2_XML.read_text(encoding="utf-8")
        tgt_xml_text = xml_text.replace(
            "ch2_tmc_sample_500x500.img", "ch2_tmc_sample_transformed.img"
        )
        tgt_xml_path = tmp_path / "ch2_tmc_sample_transformed.xml"
        tgt_xml_path.write_text(tgt_xml_text, encoding="utf-8")

        # 5. Run cli.py via subprocess
        out_dir = tmp_path / "cli_output"
        cmd = [
            sys.executable,
            str(REPO_ROOT / "cli.py"),
            str(TMC2_XML),
            str(tgt_xml_path),
            "--output-dir",
            str(out_dir),
            "--model",
            "affine",
        ]
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        assert proc.returncode == 0, (
            f"cli.py failed with returncode {proc.returncode}.\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}"
        )

        # 6. Verify report.json
        report_path = out_dir / "report.json"
        assert report_path.exists(), f"report.json was not generated at {report_path}"

        report = json.loads(report_path.read_text(encoding="utf-8"))

        # Assert inlier_count > 0 and RMSE is finite and non-negative
        inlier_count = report["matching"]["inlier_count"]
        inlier_rmse = report["registration"]["inlier_rmse"]

        assert inlier_count > 0, f"Expected inlier_count > 0, got {inlier_count}"
        assert math.isfinite(inlier_rmse), f"Expected finite RMSE, got {inlier_rmse}"
        assert inlier_rmse >= 0.0, f"Expected non-negative RMSE, got {inlier_rmse}"

        # 7. Verify all expected artifact files exist
        assert (out_dir / "warped.png").exists(), "warped.png not generated"
        assert (out_dir / "matches.png").exists(), "matches.png not generated"
        assert (out_dir / "inliers.png").exists(), "inliers.png not generated"
        assert (out_dir / "overlay.png").exists(), "overlay.png not generated"

    def test_pds4_ohrc_cli_registration_end_to_end(self, tmp_path):
        """Execute full CLI registration pipeline with OHRC PDS4 XML inputs.

        Provenance:
        - Reference image: Genuine Chandrayaan-2 OHRC PDS4 XML product
          (ch2_ohrc_sample_500x500.xml) pointing to real 500x500 uint8
          calibrated count data from ISDA/PRADAN.
        - Target image: Derived PDS4 XML label pointing to a 500x500 uint8
          binary created by applying a synthetic affine transformation
          (translation dx=+5.0, dy=+3.0) to the real OHRC pixel data.
          The derived label preserves the complete PDS4 XML structure
          (Identification_Area, Observation_Area with ISDA parameters,
          File_Area_Observational, and Array_2D_Image definitions).
        """
        assert OHRC_XML.exists(), f"Fixture XML not found at {OHRC_XML}"
        assert OHRC_IMG.exists(), f"Fixture IMG not found at {OHRC_IMG}"

        # 1. Load real OHRC uint8 pixel data
        ref_data = np.fromfile(OHRC_IMG, dtype=np.uint8).reshape((500, 500))

        # 2. Synthetically apply a known affine transformation (dx=+5, dy=+3)
        M_synth = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, 3.0]], dtype=np.float32)
        tgt_data = cv2.warpAffine(
            ref_data, M_synth, (500, 500), borderMode=cv2.BORDER_REFLECT
        )

        # 3. Write transformed binary to temporary directory
        tgt_img_path = tmp_path / "ch2_ohrc_sample_transformed.img"
        tgt_img_path.write_bytes(tgt_data.tobytes())

        # 4. Generate derived PDS4 label preserving complete XML structure
        xml_text = OHRC_XML.read_text(encoding="utf-8")
        tgt_xml_text = xml_text.replace(
            "ch2_ohrc_sample_500x500.img", "ch2_ohrc_sample_transformed.img"
        )
        tgt_xml_path = tmp_path / "ch2_ohrc_sample_transformed.xml"
        tgt_xml_path.write_text(tgt_xml_text, encoding="utf-8")

        # 5. Run cli.py via subprocess
        out_dir = tmp_path / "cli_output_ohrc"
        cmd = [
            sys.executable,
            str(REPO_ROOT / "cli.py"),
            str(OHRC_XML),
            str(tgt_xml_path),
            "--output-dir",
            str(out_dir),
            "--model",
            "affine",
        ]
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        assert proc.returncode == 0, (
            f"cli.py failed with returncode {proc.returncode}.\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}"
        )

        # 6. Verify report.json
        report_path = out_dir / "report.json"
        assert report_path.exists(), f"report.json was not generated at {report_path}"

        report = json.loads(report_path.read_text(encoding="utf-8"))

        # Assert inlier_count > 0 and RMSE is finite and non-negative
        inlier_count = report["matching"]["inlier_count"]
        inlier_rmse = report["registration"]["inlier_rmse"]

        assert inlier_count > 0, f"Expected inlier_count > 0, got {inlier_count}"
        assert math.isfinite(inlier_rmse), f"Expected finite RMSE, got {inlier_rmse}"
        assert inlier_rmse >= 0.0, f"Expected non-negative RMSE, got {inlier_rmse}"

        # 7. Verify all expected artifact files exist
        assert (out_dir / "warped.png").exists(), "warped.png not generated"
        assert (out_dir / "matches.png").exists(), "matches.png not generated"
        assert (out_dir / "inliers.png").exists(), "inliers.png not generated"
        assert (out_dir / "overlay.png").exists(), "overlay.png not generated"

    def test_pds4_ohrc_cli_registration_with_rotation_scale(self, tmp_path):
        """Execute full CLI registration pipeline with OHRC PDS4 XML inputs under rotation and scale.

        Validates that registration successfully recovers combined rotation, scale,
        and translation when applied to genuine Chandrayaan-2 OHRC PDS4 data.
        Synthetic transform parameters:
        - 8.0 degree rotation around center (250, 250)
        - 5% scale reduction (scale = 0.95)
        - Translation: tx = +20.0, ty = -12.0
        """
        assert OHRC_XML.exists(), f"Fixture XML not found at {OHRC_XML}"
        assert OHRC_IMG.exists(), f"Fixture IMG not found at {OHRC_IMG}"

        # 1. Load real OHRC uint8 pixel data (500x500)
        ref_data = np.fromfile(OHRC_IMG, dtype=np.uint8).reshape((500, 500))

        # 2. Synthetically apply affine transform: 8 deg rotation, scale 0.95, tx=20, ty=-12
        center = (250.0, 250.0)
        angle_deg = 8.0
        scale = 0.95
        tx = 20.0
        ty = -12.0

        M_synth = cv2.getRotationMatrix2D(center, angle_deg, scale)
        M_synth[0, 2] += tx
        M_synth[1, 2] += ty

        tgt_data = cv2.warpAffine(
            ref_data, M_synth, (500, 500), borderMode=cv2.BORDER_REFLECT
        )

        # 3. Write transformed binary to temporary directory
        tgt_img_path = tmp_path / "ch2_ohrc_sample_rot_scale.img"
        tgt_img_path.write_bytes(tgt_data.tobytes())

        # 4. Generate derived PDS4 label preserving complete XML structure
        xml_text = OHRC_XML.read_text(encoding="utf-8")
        tgt_xml_text = xml_text.replace(
            "ch2_ohrc_sample_500x500.img", "ch2_ohrc_sample_rot_scale.img"
        )
        tgt_xml_path = tmp_path / "ch2_ohrc_sample_rot_scale.xml"
        tgt_xml_path.write_text(tgt_xml_text, encoding="utf-8")

        # 5. Run cli.py via subprocess with --model affine
        out_dir = tmp_path / "cli_output_ohrc_rot_scale"
        cmd = [
            sys.executable,
            str(REPO_ROOT / "cli.py"),
            str(OHRC_XML),
            str(tgt_xml_path),
            "--output-dir",
            str(out_dir),
            "--model",
            "affine",
        ]
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        assert proc.returncode == 0, (
            f"cli.py failed with returncode {proc.returncode}.\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}"
        )

        # 6. Verify report.json
        report_path = out_dir / "report.json"
        assert report_path.exists(), f"report.json was not generated at {report_path}"

        report = json.loads(report_path.read_text(encoding="utf-8"))

        # Baseline assertions: inlier_count > 0, finite non-negative RMSE
        inlier_count = report["matching"]["inlier_count"]
        inlier_rmse = report["registration"]["inlier_rmse"]

        assert inlier_count > 0, f"Expected inlier_count > 0, got {inlier_count}"
        assert math.isfinite(inlier_rmse), f"Expected finite RMSE, got {inlier_rmse}"
        assert inlier_rmse >= 0.0, f"Expected non-negative RMSE, got {inlier_rmse}"

        # 7. Verify all expected artifact files exist
        assert (out_dir / "warped.png").exists(), "warped.png not generated"
        assert (out_dir / "matches.png").exists(), "matches.png not generated"
        assert (out_dir / "inliers.png").exists(), "inliers.png not generated"
        assert (out_dir / "overlay.png").exists(), "overlay.png not generated"

        # 8. Decompose recovered transform matrix and compare against ground truth
        M_rec = np.array(report["registration"]["transform_matrix"], dtype=np.float64)
        assert M_rec.shape == (2, 3), f"Expected 2x3 affine matrix, got shape {M_rec.shape}"

        # Decompose 2x3 affine matrix: [[a, b, tx], [c, d, ty]]
        # For M = S * R with translation:
        # scale_x = sqrt(a^2 + c^2), scale_y = sqrt(b^2 + d^2)
        # angle = atan2(b, a)
        a_synth, b_synth, tx_synth = M_synth[0, 0], M_synth[0, 1], M_synth[0, 2]
        c_synth, d_synth, ty_synth = M_synth[1, 0], M_synth[1, 1], M_synth[1, 2]
        true_scale = (math.hypot(a_synth, c_synth) + math.hypot(b_synth, d_synth)) / 2.0
        true_angle_deg = math.degrees(math.atan2(b_synth, a_synth))
        true_tx = float(tx_synth)
        true_ty = float(ty_synth)

        a_rec, b_rec, tx_rec = M_rec[0, 0], M_rec[0, 1], M_rec[0, 2]
        c_rec, d_rec, ty_rec = M_rec[1, 0], M_rec[1, 1], M_rec[1, 2]
        rec_scale = (math.hypot(a_rec, c_rec) + math.hypot(b_rec, d_rec)) / 2.0
        rec_angle_deg = math.degrees(math.atan2(b_rec, a_rec))
        rec_tx = float(tx_rec)
        rec_ty = float(ty_rec)

        # Tolerances explicitly defined for assertions
        tol_scale = 0.03          # Scale tolerance: within 0.03 of 0.95 (~3.1% error margin)
        tol_angle_deg = 1.0       # Rotation tolerance: within 1.0 degree of 8.0 degrees
        tol_translation_px = 3.0  # Translation tolerance: within 3.0 pixels of true translation

        diff_scale = abs(rec_scale - true_scale)
        diff_angle = abs(rec_angle_deg - true_angle_deg)
        diff_tx = abs(rec_tx - true_tx)
        diff_ty = abs(rec_ty - true_ty)

        assert diff_scale <= tol_scale, (
            f"Recovered scale {rec_scale:.5f} deviates from true scale {true_scale:.5f} "
            f"by {diff_scale:.5f} (tolerance: {tol_scale})"
        )
        assert diff_angle <= tol_angle_deg, (
            f"Recovered rotation {rec_angle_deg:.4f}° deviates from true rotation {true_angle_deg:.4f}° "
            f"by {diff_angle:.4f}° (tolerance: {tol_angle_deg}°)"
        )
        assert diff_tx <= tol_translation_px, (
            f"Recovered tx {rec_tx:.3f} px deviates from true tx {true_tx:.3f} px "
            f"by {diff_tx:.3f} px (tolerance: {tol_translation_px} px)"
        )
        assert diff_ty <= tol_translation_px, (
            f"Recovered ty {rec_ty:.3f} px deviates from true ty {true_ty:.3f} px "
            f"by {diff_ty:.3f} px (tolerance: {tol_translation_px} px)"
        )

    def test_pds4_iirs_cli_registration_with_rotation_scale(self, tmp_path):
        """Execute full CLI registration pipeline with IIRS PDS4 XML inputs under rotation and scale.

        (same-instrument synthetic transform only — not a real cross-modal or cross-pass registration test)

        Validates that registration successfully recovers combined rotation, scale,
        and translation when applied to genuine Chandrayaan-2 IIRS PDS4 hyperspectral data.
        Synthetic transform parameters:
        - 8.0 degree rotation around center (100, 100)
        - 5% scale reduction (scale = 0.95)
        - Translation: tx = +20.0, ty = -12.0
        """
        assert IIRS_XML.exists(), f"Fixture XML not found at {IIRS_XML}"
        assert IIRS_QUB.exists(), f"Fixture QUB not found at {IIRS_QUB}"

        # 1. Load real IIRS float32 16-band cube (200x200x16 BSQ layout)
        ref_cube = np.fromfile(IIRS_QUB, dtype="<f4").reshape((16, 200, 200))

        # 2. Synthetically apply affine transform: 8 deg rotation, scale 0.95, tx=20, ty=-12
        center = (100.0, 100.0)
        angle_deg = 8.0
        scale = 0.95
        tx = 20.0
        ty = -12.0

        M_synth = cv2.getRotationMatrix2D(center, angle_deg, scale)
        M_synth[0, 2] += tx
        M_synth[1, 2] += ty

        tgt_cube = np.zeros_like(ref_cube)
        for b in range(16):
            tgt_cube[b] = cv2.warpAffine(
                ref_cube[b], M_synth, (200, 200), borderMode=cv2.BORDER_REFLECT
            )

        # 3. Write transformed binary to temporary directory
        tgt_qub_path = tmp_path / "ch2_iirs_sample_rot_scale.qub"
        tgt_qub_path.write_bytes(tgt_cube.tobytes())

        # 4. Generate derived PDS4 label preserving complete XML structure
        xml_text = IIRS_XML.read_text(encoding="utf-8")
        tgt_xml_text = xml_text.replace(
            "ch2_iirs_sample_200x200x16.qub", "ch2_iirs_sample_rot_scale.qub"
        )
        tgt_xml_path = tmp_path / "ch2_iirs_sample_rot_scale.xml"
        tgt_xml_path.write_text(tgt_xml_text, encoding="utf-8")

        # 5. Run cli.py via subprocess with --model affine
        out_dir = tmp_path / "cli_output_iirs_rot_scale"
        cmd = [
            sys.executable,
            str(REPO_ROOT / "cli.py"),
            str(IIRS_XML),
            str(tgt_xml_path),
            "--output-dir",
            str(out_dir),
            "--model",
            "affine",
        ]
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        assert proc.returncode == 0, (
            f"cli.py failed with returncode {proc.returncode}.\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}"
        )

        # 6. Verify report.json
        report_path = out_dir / "report.json"
        assert report_path.exists(), f"report.json was not generated at {report_path}"

        report = json.loads(report_path.read_text(encoding="utf-8"))

        # Baseline assertions: inlier_count > 0, finite non-negative RMSE
        inlier_count = report["matching"]["inlier_count"]
        inlier_rmse = report["registration"]["inlier_rmse"]

        assert inlier_count > 0, f"Expected inlier_count > 0, got {inlier_count}"
        assert math.isfinite(inlier_rmse), f"Expected finite RMSE, got {inlier_rmse}"
        assert inlier_rmse >= 0.0, f"Expected non-negative RMSE, got {inlier_rmse}"

        # 7. Verify all expected artifact files exist
        assert (out_dir / "warped.png").exists(), "warped.png not generated"
        assert (out_dir / "matches.png").exists(), "matches.png not generated"
        assert (out_dir / "inliers.png").exists(), "inliers.png not generated"
        assert (out_dir / "overlay.png").exists(), "overlay.png not generated"

        # 8. Decompose recovered transform matrix and compare against ground truth
        M_rec = np.array(report["registration"]["transform_matrix"], dtype=np.float64)
        assert M_rec.shape == (2, 3), f"Expected 2x3 affine matrix, got shape {M_rec.shape}"

        a_synth, b_synth, tx_synth = M_synth[0, 0], M_synth[0, 1], M_synth[0, 2]
        c_synth, d_synth, ty_synth = M_synth[1, 0], M_synth[1, 1], M_synth[1, 2]
        true_scale = (math.hypot(a_synth, c_synth) + math.hypot(b_synth, d_synth)) / 2.0
        true_angle_deg = math.degrees(math.atan2(b_synth, a_synth))
        true_tx = float(tx_synth)
        true_ty = float(ty_synth)

        a_rec, b_rec, tx_rec = M_rec[0, 0], M_rec[0, 1], M_rec[0, 2]
        c_rec, d_rec, ty_rec = M_rec[1, 0], M_rec[1, 1], M_rec[1, 2]
        rec_scale = (math.hypot(a_rec, c_rec) + math.hypot(b_rec, d_rec)) / 2.0
        rec_angle_deg = math.degrees(math.atan2(b_rec, a_rec))
        rec_tx = float(tx_rec)
        rec_ty = float(ty_rec)

        # Tolerances explicitly defined for assertions
        tol_scale = 0.03          # Scale tolerance: within 0.03 of 0.95 (~3.1% error margin)
        tol_angle_deg = 1.0       # Rotation tolerance: within 1.0 degree of 8.0 degrees
        tol_translation_px = 3.0  # Translation tolerance: within 3.0 pixels of true translation

        diff_scale = abs(rec_scale - true_scale)
        diff_angle = abs(rec_angle_deg - true_angle_deg)
        diff_tx = abs(rec_tx - true_tx)
        diff_ty = abs(rec_ty - true_ty)

        assert diff_scale <= tol_scale, (
            f"Recovered scale {rec_scale:.5f} deviates from true scale {true_scale:.5f} "
            f"by {diff_scale:.5f} (tolerance: {tol_scale})"
        )
        assert diff_angle <= tol_angle_deg, (
            f"Recovered rotation {rec_angle_deg:.4f}° deviates from true rotation {true_angle_deg:.4f}° "
            f"by {diff_angle:.4f}° (tolerance: {tol_angle_deg}°)"
        )
        assert diff_tx <= tol_translation_px, (
            f"Recovered tx {rec_tx:.3f} px deviates from true tx {true_tx:.3f} px "
            f"by {diff_tx:.3f} px (tolerance: {tol_translation_px} px)"
        )
        assert diff_ty <= tol_translation_px, (
            f"Recovered ty {rec_ty:.3f} px deviates from true ty {true_ty:.3f} px "
            f"by {diff_ty:.3f} px (tolerance: {tol_translation_px} px)"
        )




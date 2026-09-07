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


"""
SIH26166 — Unit tests for band_reduction utilities (Module 02).

Tests:
1. Synthetic wavelength arrays covering:
   - bands entirely below cutoff
   - bands entirely above cutoff
   - bands straddling the cutoff
   - empty list
2. Real Chandrayaan-2 IIRS fixture wavelengths confirming safe-case behavior:
   - With default cutoff (2500nm), all 16 fixture bands (~712-965nm) are selected
     as a "safe case" (all within the solar-reflective range, not a thermal-cutoff test).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from backend.preprocessing.band_reduction import (
    reduce_bands_mean,
    reduce_bands_pca,
    select_bands,
    select_solar_reflective_bands,
)
from backend.preprocessing.pds4 import extract_iirs_band_wavelengths

FIXTURES_PDS4 = Path("tests/fixtures/pds4")
IIRS_XML = FIXTURES_PDS4 / "iirs" / "ch2_iirs_sample_200x200x16.xml"


class TestSelectSolarReflectiveBands:
    """Unit tests for select_solar_reflective_bands()."""

    def test_synthetic_bands_entirely_below_cutoff(self):
        """When all band wavelengths are strictly below cutoff, all indices are returned."""
        wavelengths = [500.0, 750.0, 1000.0, 1500.0, 2000.0, 2499.0]
        cutoff_nm = 2500.0

        indices = select_solar_reflective_bands(wavelengths, cutoff_nm=cutoff_nm)
        assert indices == [0, 1, 2, 3, 4, 5], (
            f"Expected all 6 band indices [0..5] below {cutoff_nm}nm, got {indices}"
        )

    def test_synthetic_bands_entirely_above_cutoff(self):
        """When all band wavelengths are at or above cutoff, an empty list is returned."""
        wavelengths = [2500.0, 2500.1, 3000.0, 3500.0, 4500.0]
        cutoff_nm = 2500.0

        indices = select_solar_reflective_bands(wavelengths, cutoff_nm=cutoff_nm)
        assert indices == [], (
            f"Expected no band indices at or above {cutoff_nm}nm, got {indices}"
        )

    def test_synthetic_bands_straddling_cutoff(self):
        """When bands straddle the cutoff, only strictly below indices are returned."""
        # Wavelengths: 0: 600nm, 1: 1200nm, 2: 2400nm, 3: 2500nm (boundary), 4: 2600nm, 5: 3200nm
        wavelengths = [600.0, 1200.0, 2400.0, 2500.0, 2600.0, 3200.0]
        cutoff_nm = 2500.0

        indices = select_solar_reflective_bands(wavelengths, cutoff_nm=cutoff_nm)
        # Note: 2500.0 is not strictly below 2500.0 cutoff, so only indices 0, 1, 2 are selected
        assert indices == [0, 1, 2], (
            f"Expected indices [0, 1, 2] strictly below {cutoff_nm}nm, got {indices}"
        )

    def test_synthetic_empty_list(self):
        """Empty input list returns an empty list."""
        assert select_solar_reflective_bands([], cutoff_nm=2500.0) == []

    def test_synthetic_dict_input(self):
        """Function accepts list of dicts with 'center_wavelength' key."""
        dict_bands = [
            {"band_number": 1, "center_wavelength": 750.0},
            {"band_number": 2, "center_wavelength": 1500.0},
            {"band_number": 3, "center_wavelength": 2600.0},
        ]
        indices = select_solar_reflective_bands(dict_bands, cutoff_nm=2500.0)
        assert indices == [0, 1]

    def test_real_iirs_fixture_all_16_bands_selected_safe_case(self):
        """Validate on genuine Chandrayaan-2 IIRS fixture data.

        IMPORTANT CONTEXT FOR FUTURE READERS:
        The real IIRS fixture (ch2_iirs_sample_200x200x16.xml) covers the
        spectral range ~712.3nm to ~965.1nm across its 16 bands. Because all
        16 bands are situated well within the reflected solar spectral range
        (and far below the heuristic 2500nm thermal-emission onset), ALL 16 bands
        must be selected under the default cutoff.

        This test is intentionally a "safe case" sanity check confirming that
        standard reflective-range IIRS imagery is fully retained; it does NOT
        exercise thermal-band rejection on real data, as no thermal-band fixtures
        (>2500nm) are present in the test suite yet.
        """
        assert IIRS_XML.exists(), f"IIRS XML fixture not found at {IIRS_XML}"

        extracted_bands = extract_iirs_band_wavelengths(IIRS_XML)
        assert len(extracted_bands) == 16, f"Expected 16 extracted bands, got {len(extracted_bands)}"

        # 1. Test passing list of extracted dicts directly
        indices_from_dicts = select_solar_reflective_bands(extracted_bands, cutoff_nm=2500.0)
        assert indices_from_dicts == list(range(16)), (
            f"Expected all 16 bands [0..15] to be selected under 2500nm cutoff, got {indices_from_dicts}"
        )
        assert len(indices_from_dicts) == 16

        # 2. Test passing list of extracted float wavelengths
        wavelengths_float = [b["center_wavelength"] for b in extracted_bands]
        # Verify min and max wavelengths in fixture
        min_wl = min(wavelengths_float)
        max_wl = max(wavelengths_float)
        assert min_wl == pytest.approx(712.3, abs=0.1)
        assert max_wl == pytest.approx(965.1, abs=0.1)
        assert max_wl < 2500.0, "Fixture bands must all be well below the 2500nm heuristic cutoff"

        indices_from_floats = select_solar_reflective_bands(wavelengths_float, cutoff_nm=2500.0)
        assert indices_from_floats == list(range(16)), (
            f"Expected all 16 bands [0..15] selected from float list, got {indices_from_floats}"
        )

        # 3. If an artificial tighter cutoff is applied (e.g. 800nm), only bands < 800nm are selected
        # Bands 1-6 are: 712.3, 729.2, 746.0, 762.9, 779.7, 796.6 (< 800nm)
        # Band 7 is: 813.4nm (>= 800nm)
        indices_sub = select_solar_reflective_bands(wavelengths_float, cutoff_nm=800.0)
        assert indices_sub == [0, 1, 2, 3, 4, 5], (
            f"Expected first 6 bands below 800nm, got {indices_sub}"
        )

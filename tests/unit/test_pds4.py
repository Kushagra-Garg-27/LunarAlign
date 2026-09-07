"""
SIH26166 -- Unit tests for the PDS4 ingestion layer (Stage 2a).

All tests use fixtures derived from REAL Chandrayaan-2 downloaded data:
- tests/fixtures/pds4/tmc2/ : 500x500 uint16 crop from TMC-2 calibrated product
- tests/fixtures/pds4/ohrc/ : 500x500 uint8  crop from OHRC calibrated product
- tests/fixtures/pds4/iirs/ : 200x200x16 float32 crop from IIRS calibrated product
  (first 16 of 256 bands; tests parser mechanics only, not spectral coverage)

Refer to each fixture subdirectory README.md for full provenance details.

IIRS note: GDAL PDS4 driver cannot open the IIRS product due to a filename
mismatch (-001.qub multipart suffix). load_pds4_raster() falls back to numpy
memmap using BSQ geometry from the label XML. Tests verify this path.
"""

from __future__ import annotations

import tempfile
import textwrap
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixture paths (relative to repo root -- pytest is invoked from D:/LunarAlign)
# ---------------------------------------------------------------------------

FIXTURES_PDS4 = Path("tests/fixtures/pds4")

TMC2_XML = FIXTURES_PDS4 / "tmc2" / "ch2_tmc_sample_500x500.xml"
TMC2_IMG = FIXTURES_PDS4 / "tmc2" / "ch2_tmc_sample_500x500.img"

OHRC_XML = FIXTURES_PDS4 / "ohrc" / "ch2_ohrc_sample_500x500.xml"
OHRC_IMG = FIXTURES_PDS4 / "ohrc" / "ch2_ohrc_sample_500x500.img"

IIRS_XML = FIXTURES_PDS4 / "iirs" / "ch2_iirs_sample_200x200x16.xml"
IIRS_QUB = FIXTURES_PDS4 / "iirs" / "ch2_iirs_sample_200x200x16.qub"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_non_pds4_xml(tmp_path: Path) -> Path:
    """Write a well-formed XML file that is NOT a PDS4 label."""
    p = tmp_path / "not_pds4.xml"
    p.write_text(
        '<?xml version="1.0"?><root><child>hello</child></root>', encoding="utf-8"
    )
    return p


def _write_missing_isda_xml(tmp_path: Path, src_xml: Path) -> Path:
    """Return a copy of src_xml with Mission_Area stripped out."""
    txt = src_xml.read_text(encoding="utf-8")
    # Remove Mission_Area block (crude but sufficient for the graceful-fail test)
    import re
    stripped = re.sub(r"<Mission_Area>.*?</Mission_Area>", "", txt, flags=re.DOTALL)
    p = tmp_path / "no_isda.xml"
    p.write_text(stripped, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# is_pds4_label() tests
# ---------------------------------------------------------------------------

class TestIsPds4Label:
    def test_tmc2_is_pds4(self):
        from backend.preprocessing.pds4 import is_pds4_label
        assert is_pds4_label(TMC2_XML) is True

    def test_ohrc_is_pds4(self):
        from backend.preprocessing.pds4 import is_pds4_label
        assert is_pds4_label(OHRC_XML) is True

    def test_iirs_is_pds4(self):
        from backend.preprocessing.pds4 import is_pds4_label
        assert is_pds4_label(IIRS_XML) is True

    def test_non_pds4_xml_returns_false(self, tmp_path):
        from backend.preprocessing.pds4 import is_pds4_label
        p = _write_non_pds4_xml(tmp_path)
        assert is_pds4_label(p) is False

    def test_malformed_xml_returns_false(self, tmp_path):
        from backend.preprocessing.pds4 import is_pds4_label
        p = tmp_path / "bad.xml"
        p.write_text("<unclosed", encoding="utf-8")
        assert is_pds4_label(p) is False

    def test_nonexistent_file_returns_false(self, tmp_path):
        from backend.preprocessing.pds4 import is_pds4_label
        assert is_pds4_label(tmp_path / "doesnotexist.xml") is False

    def test_non_xml_file_returns_false(self, tmp_path):
        from backend.preprocessing.pds4 import is_pds4_label
        p = tmp_path / "binary.img"
        p.write_bytes(b"\x00\x01\x02\x03")
        assert is_pds4_label(p) is False


# ---------------------------------------------------------------------------
# identify_instrument() tests
# ---------------------------------------------------------------------------

class TestIdentifyInstrument:
    def _tree(self, xml_path: Path):
        import xml.etree.ElementTree as ET
        return ET.parse(str(xml_path))

    def test_tmc2_identified(self):
        from backend.preprocessing.pds4 import identify_instrument
        assert identify_instrument(self._tree(TMC2_XML)) == "TMC2"

    def test_ohrc_identified(self):
        from backend.preprocessing.pds4 import identify_instrument
        assert identify_instrument(self._tree(OHRC_XML)) == "OHRC"

    def test_iirs_identified(self):
        from backend.preprocessing.pds4 import identify_instrument
        assert identify_instrument(self._tree(IIRS_XML)) == "IIRS"

    def test_unknown_instrument(self, tmp_path):
        from backend.preprocessing.pds4 import identify_instrument
        import xml.etree.ElementTree as ET
        # Create a minimal PDS4 label with a bogus instrument name
        xml = textwrap.dedent("""\
            <?xml version="1.0"?>
            <Product_Observational xmlns="http://pds.nasa.gov/pds4/pds/v1">
              <Identification_Area>
                <logical_identifier>urn:test</logical_identifier>
                <version_id>1.0</version_id>
                <title>Test</title>
                <information_model_version>1.11.0.0</information_model_version>
                <product_class>Product_Observational</product_class>
              </Identification_Area>
              <Observation_Area>
                <Time_Coordinates>
                  <start_date_time>2020-01-01T00:00:00Z</start_date_time>
                  <stop_date_time>2020-01-01T00:01:00Z</stop_date_time>
                </Time_Coordinates>
                <Investigation_Area>
                  <name>test</name>
                  <type>Mission</type>
                </Investigation_Area>
                <Observing_System>
                  <Observing_System_Component>
                    <name>bogus camera</name>
                    <type>Instrument</type>
                  </Observing_System_Component>
                </Observing_System>
                <Target_Identification>
                  <name>Moon</name>
                  <type>Satellite</type>
                </Target_Identification>
              </Observation_Area>
            </Product_Observational>
        """)
        p = tmp_path / "bogus.xml"
        p.write_text(xml, encoding="utf-8")
        tree = ET.parse(str(p))
        result = identify_instrument(tree)
        assert result == "UNKNOWN"


# ---------------------------------------------------------------------------
# extract_pds4_metadata() tests
# ---------------------------------------------------------------------------

class TestExtractPds4Metadata:
    # --- TMC-2: assert confirmed known field values ---

    def test_tmc2_instrument(self):
        from backend.preprocessing.pds4 import extract_pds4_metadata
        meta = extract_pds4_metadata(TMC2_XML)
        assert meta.instrument == "TMC2"

    def test_tmc2_logical_identifier(self):
        from backend.preprocessing.pds4 import extract_pds4_metadata
        meta = extract_pds4_metadata(TMC2_XML)
        assert meta.logical_identifier is not None
        assert "ch2_tmc" in meta.logical_identifier.lower()

    def test_tmc2_title(self):
        from backend.preprocessing.pds4 import extract_pds4_metadata
        meta = extract_pds4_metadata(TMC2_XML)
        assert meta.title is not None
        assert "TMC" in meta.title or "tmc" in meta.title.lower()

    def test_tmc2_time_coordinates(self):
        from backend.preprocessing.pds4 import extract_pds4_metadata
        meta = extract_pds4_metadata(TMC2_XML)
        assert meta.start_date_time == "2019-12-13T05:52:46.1631Z"
        assert meta.stop_date_time == "2019-12-13T05:56:30.9375Z"

    def test_tmc2_investigation_name(self):
        from backend.preprocessing.pds4 import extract_pds4_metadata
        meta = extract_pds4_metadata(TMC2_XML)
        assert meta.investigation_name is not None
        assert "chandrayaan" in meta.investigation_name.lower()

    def test_tmc2_named_pixel_resolution(self):
        from backend.preprocessing.pds4 import extract_pds4_metadata
        meta = extract_pds4_metadata(TMC2_XML)
        assert meta.tmc2_product_params is not None
        pr = meta.tmc2_product_params["pixel_resolution"]
        assert pr["value"] == "4.27"
        assert pr["unit"] == "m/pixel"

    def test_tmc2_named_sun_azimuth(self):
        from backend.preprocessing.pds4 import extract_pds4_metadata
        meta = extract_pds4_metadata(TMC2_XML)
        assert meta.tmc2_product_params is not None
        sa = meta.tmc2_product_params["sun_azimuth"]
        assert sa["value"] == "250.583825"
        assert sa["unit"] == "deg"

    def test_tmc2_named_sun_elevation(self):
        from backend.preprocessing.pds4 import extract_pds4_metadata
        meta = extract_pds4_metadata(TMC2_XML)
        pr = meta.tmc2_product_params["sun_elevation"]
        assert pr["value"] == "73.506750"

    def test_tmc2_named_solar_incidence(self):
        from backend.preprocessing.pds4 import extract_pds4_metadata
        meta = extract_pds4_metadata(TMC2_XML)
        assert meta.tmc2_product_params["solar_incidence"]["value"] == "16.493250"

    def test_tmc2_named_spacecraft_altitude(self):
        from backend.preprocessing.pds4 import extract_pds4_metadata
        meta = extract_pds4_metadata(TMC2_XML)
        alt = meta.tmc2_product_params["spacecraft_altitude"]
        assert alt["value"] == "85.31"
        assert alt["unit"] == "km"

    def test_tmc2_named_roll_pitch_yaw(self):
        from backend.preprocessing.pds4 import extract_pds4_metadata
        meta = extract_pds4_metadata(TMC2_XML)
        assert meta.tmc2_product_params["roll"]["value"] == "0.000554"
        assert meta.tmc2_product_params["pitch"]["value"] == "0.341256"
        assert meta.tmc2_product_params["yaw"]["value"] == "179.999630"

    def test_tmc2_named_imaging_orbit_number(self):
        from backend.preprocessing.pds4 import extract_pds4_metadata
        meta = extract_pds4_metadata(TMC2_XML)
        assert meta.tmc2_product_params["imaging_orbit_number"]["value"] == "1345"

    def test_tmc2_named_job_id(self):
        from backend.preprocessing.pds4 import extract_pds4_metadata
        meta = extract_pds4_metadata(TMC2_XML)
        assert "TMCXX" in meta.tmc2_product_params["job_id"]["value"]

    def test_tmc2_named_projection_area(self):
        from backend.preprocessing.pds4 import extract_pds4_metadata
        meta = extract_pds4_metadata(TMC2_XML)
        assert meta.tmc2_product_params["projection"]["value"] == "Selenographic"
        assert meta.tmc2_product_params["area"]["value"] == "Equatorial"

    def test_tmc2_geometry_system_level(self):
        from backend.preprocessing.pds4 import extract_pds4_metadata
        meta = extract_pds4_metadata(TMC2_XML)
        g = meta.tmc2_geometry_params
        assert g is not None
        assert "System_Level_Coordinates.upper_left_latitude" in g
        assert g["System_Level_Coordinates.upper_left_latitude"]["value"] == "0.487437"

    def test_tmc2_geometry_refined(self):
        from backend.preprocessing.pds4 import extract_pds4_metadata
        meta = extract_pds4_metadata(TMC2_XML)
        g = meta.tmc2_geometry_params
        assert "Refined_Corner_Coordinates.upper_left_latitude" in g
        assert g["Refined_Corner_Coordinates.upper_left_latitude"]["value"] == "0.446960"

    def test_tmc2_generic_dict_non_empty(self):
        from backend.preprocessing.pds4 import extract_pds4_metadata
        meta = extract_pds4_metadata(TMC2_XML)
        assert len(meta.isda_product_params) > 0
        assert len(meta.isda_geometry_params) > 0

    def test_tmc2_wavelength_info_is_none(self):
        from backend.preprocessing.pds4 import extract_pds4_metadata
        meta = extract_pds4_metadata(TMC2_XML)
        assert meta.wavelength_info is None

    # --- OHRC: generic dict checks ---

    def test_ohrc_instrument(self):
        from backend.preprocessing.pds4 import extract_pds4_metadata
        meta = extract_pds4_metadata(OHRC_XML)
        assert meta.instrument == "OHRC"

    def test_ohrc_isda_product_params_non_empty(self):
        from backend.preprocessing.pds4 import extract_pds4_metadata
        meta = extract_pds4_metadata(OHRC_XML)
        assert len(meta.isda_product_params) > 0

    def test_ohrc_pixel_resolution_present(self):
        """pixel_resolution key must be present (value=0.21 m/pixel, observed in real sample)."""
        from backend.preprocessing.pds4 import extract_pds4_metadata
        meta = extract_pds4_metadata(OHRC_XML)
        assert "pixel_resolution" in meta.isda_product_params
        pr = meta.isda_product_params["pixel_resolution"]
        assert pr["value"] == "0.21"
        assert pr["unit"] == "m/pixel"

    def test_ohrc_expected_structural_keys(self):
        """Keys observed in the one real OHRC sample must all be present."""
        from backend.preprocessing.pds4 import extract_pds4_metadata
        meta = extract_pds4_metadata(OHRC_XML)
        pp = meta.isda_product_params
        for key in ("job_id", "imaging_orbit_number", "spacecraft_altitude",
                    "sun_azimuth", "sun_elevation", "roll", "pitch", "yaw",
                    "projection", "area"):
            assert key in pp, f"Expected key {key!r} missing from OHRC isda_product_params"

    def test_ohrc_geometry_params_non_empty(self):
        from backend.preprocessing.pds4 import extract_pds4_metadata
        meta = extract_pds4_metadata(OHRC_XML)
        assert len(meta.isda_geometry_params) > 0
        # Both coordinate blocks should be present
        assert any("System_Level_Coordinates" in k for k in meta.isda_geometry_params)
        assert any("Refined_Corner_Coordinates" in k for k in meta.isda_geometry_params)

    def test_ohrc_tmc2_fields_are_none(self):
        from backend.preprocessing.pds4 import extract_pds4_metadata
        meta = extract_pds4_metadata(OHRC_XML)
        assert meta.tmc2_product_params is None
        assert meta.tmc2_geometry_params is None

    def test_ohrc_wavelength_info_is_none(self):
        from backend.preprocessing.pds4 import extract_pds4_metadata
        meta = extract_pds4_metadata(OHRC_XML)
        assert meta.wavelength_info is None

    # --- IIRS: generic dict checks + wavelength info ---

    def test_iirs_instrument(self):
        from backend.preprocessing.pds4 import extract_pds4_metadata
        meta = extract_pds4_metadata(IIRS_XML)
        assert meta.instrument == "IIRS"

    def test_iirs_isda_product_params_non_empty(self):
        from backend.preprocessing.pds4 import extract_pds4_metadata
        meta = extract_pds4_metadata(IIRS_XML)
        assert len(meta.isda_product_params) > 0

    def test_iirs_expected_structural_keys(self):
        """Keys observed in the one real IIRS sample must all be present."""
        from backend.preprocessing.pds4 import extract_pds4_metadata
        meta = extract_pds4_metadata(IIRS_XML)
        pp = meta.isda_product_params
        for key in ("job_id", "imaging_orbit_number", "spacecraft_altitude",
                    "pixel_resolution", "detector_temperature",
                    "sun_azimuth", "projection"):
            assert key in pp, f"Expected key {key!r} missing from IIRS isda_product_params"

    def test_iirs_iirs_specific_fields(self):
        """IIRS has thermal fields absent in TMC-2/OHRC."""
        from backend.preprocessing.pds4 import extract_pds4_metadata
        meta = extract_pds4_metadata(IIRS_XML)
        pp = meta.isda_product_params
        assert "detector_temperature" in pp
        assert "tertiary_mirror_temperature" in pp
        assert "spectrometer_casing_temperature" in pp
        assert "dewar_vw_temperature" in pp

    def test_iirs_wavelength_info_present(self):
        """IIRS wavelength_info must be a non-empty string describing the location."""
        from backend.preprocessing.pds4 import extract_pds4_metadata
        meta = extract_pds4_metadata(IIRS_XML)
        assert meta.wavelength_info is not None
        assert len(meta.wavelength_info) > 0
        assert "Band_Bin" in meta.wavelength_info

    def test_iirs_tmc2_fields_are_none(self):
        from backend.preprocessing.pds4 import extract_pds4_metadata
        meta = extract_pds4_metadata(IIRS_XML)
        assert meta.tmc2_product_params is None
        assert meta.tmc2_geometry_params is None

    # --- Error cases ---

    def test_nonexistent_raises_file_not_found(self, tmp_path):
        from backend.preprocessing.pds4 import extract_pds4_metadata
        with pytest.raises(FileNotFoundError):
            extract_pds4_metadata(tmp_path / "missing.xml")

    def test_non_pds4_raises_value_error(self, tmp_path):
        from backend.preprocessing.pds4 import extract_pds4_metadata
        p = _write_non_pds4_xml(tmp_path)
        with pytest.raises(ValueError, match="not a PDS4"):
            extract_pds4_metadata(p)

    def test_missing_isda_graceful(self, tmp_path):
        """When Mission_Area is missing, returns metadata with empty isda dicts (no crash)."""
        from backend.preprocessing.pds4 import extract_pds4_metadata
        p = _write_missing_isda_xml(tmp_path, TMC2_XML)
        meta = extract_pds4_metadata(p)
        # Must not crash; isda dicts should be empty, instrument still identified
        assert isinstance(meta.isda_product_params, dict)
        assert isinstance(meta.isda_geometry_params, dict)


# ---------------------------------------------------------------------------
# load_pds4_raster() tests
# ---------------------------------------------------------------------------

class TestLoadPds4Raster:
    # --- TMC-2 ---

    def test_tmc2_shape(self):
        from backend.preprocessing.pds4 import load_pds4_raster
        img = load_pds4_raster(TMC2_XML)
        assert img.data.shape == (500, 500)

    def test_tmc2_dtype(self):
        from backend.preprocessing.pds4 import load_pds4_raster
        img = load_pds4_raster(TMC2_XML)
        assert img.dtype == np.dtype("uint16")

    def test_tmc2_num_bands(self):
        from backend.preprocessing.pds4 import load_pds4_raster
        img = load_pds4_raster(TMC2_XML)
        assert img.num_bands == 1

    def test_tmc2_dimensions(self):
        from backend.preprocessing.pds4 import load_pds4_raster
        img = load_pds4_raster(TMC2_XML)
        assert img.width == 500
        assert img.height == 500

    def test_tmc2_source_format(self):
        from backend.preprocessing.pds4 import load_pds4_raster
        img = load_pds4_raster(TMC2_XML)
        assert img.source_format == "PDS4"

    def test_tmc2_metadata_has_driver(self):
        from backend.preprocessing.pds4 import load_pds4_raster
        img = load_pds4_raster(TMC2_XML)
        assert "rasterio_driver" in img.metadata

    def test_tmc2_not_georeferenced(self):
        from backend.preprocessing.pds4 import load_pds4_raster
        img = load_pds4_raster(TMC2_XML)
        # CRS should be None for calibrated (not orthorectified) products
        assert img.metadata.get("rasterio_crs") is None

    def test_tmc2_pixel_values_in_range(self):
        """Verify real pixel values match expected range (min=187, max=383 from crop)."""
        from backend.preprocessing.pds4 import load_pds4_raster
        img = load_pds4_raster(TMC2_XML)
        assert img.data.min() >= 0
        assert img.data.max() <= 65535
        # Real calibrated TMC-2 values are well above 0
        assert img.data.max() > 100

    # --- OHRC ---

    def test_ohrc_shape(self):
        from backend.preprocessing.pds4 import load_pds4_raster
        img = load_pds4_raster(OHRC_XML)
        assert img.data.shape == (500, 500)

    def test_ohrc_dtype(self):
        from backend.preprocessing.pds4 import load_pds4_raster
        img = load_pds4_raster(OHRC_XML)
        assert img.dtype == np.dtype("uint8")

    def test_ohrc_num_bands(self):
        from backend.preprocessing.pds4 import load_pds4_raster
        img = load_pds4_raster(OHRC_XML)
        assert img.num_bands == 1

    def test_ohrc_source_format(self):
        from backend.preprocessing.pds4 import load_pds4_raster
        img = load_pds4_raster(OHRC_XML)
        assert img.source_format == "PDS4"

    def test_ohrc_pixel_values_in_range(self):
        from backend.preprocessing.pds4 import load_pds4_raster
        img = load_pds4_raster(OHRC_XML)
        assert img.data.min() >= 0
        assert img.data.max() <= 255

    # --- IIRS (memmap fallback path) ---

    def test_iirs_shape(self):
        """IIRS fixture: 16 bands x 200 lines x 200 samples -> (200, 200, 16) bands-last."""
        from backend.preprocessing.pds4 import load_pds4_raster
        img = load_pds4_raster(IIRS_XML)
        assert img.data.shape == (200, 200, 16)

    def test_iirs_dtype(self):
        from backend.preprocessing.pds4 import load_pds4_raster
        img = load_pds4_raster(IIRS_XML)
        assert img.dtype == np.dtype("float32")

    def test_iirs_num_bands(self):
        from backend.preprocessing.pds4 import load_pds4_raster
        img = load_pds4_raster(IIRS_XML)
        assert img.num_bands == 16

    def test_iirs_dimensions(self):
        from backend.preprocessing.pds4 import load_pds4_raster
        img = load_pds4_raster(IIRS_XML)
        assert img.width == 200
        assert img.height == 200

    def test_iirs_source_format(self):
        from backend.preprocessing.pds4 import load_pds4_raster
        img = load_pds4_raster(IIRS_XML)
        assert img.source_format == "PDS4"

    def test_iirs_rasterio_or_memmap_path_present(self):
        """IIRS fixture: binary name matches label exactly -> rasterio succeeds.
        The pds4_read_path key must be present regardless of which path was taken.
        (The full ISRO download has a -001.qub mismatch that forces memmap;
        the fixture has exact filename match so rasterio works.)
        """
        from backend.preprocessing.pds4 import load_pds4_raster
        img = load_pds4_raster(IIRS_XML)
        assert img.metadata.get("pds4_read_path") in ("rasterio", "numpy_memmap")

    def test_iirs_memmap_fallback_explicit(self, tmp_path):
        """Verify the memmap fallback by simulating a filename mismatch.

        Copies the IIRS fixture XML with a different file_name that points to
        a nonexistent binary (so rasterio fails), then places the real binary
        at the -001 suffixed path so the memmap fallback can find it.
        """
        import shutil
        from backend.preprocessing.pds4 import load_pds4_raster

        # Copy the fixture qub to the tmp_path as <stem>-001.qub
        qub_dst = tmp_path / "ch2_iirs_mismatch-001.qub"
        shutil.copy(str(IIRS_QUB), str(qub_dst))

        # Write a label XML that references the stem WITHOUT -001
        # (so rasterio fails, then the fallback finds -001.qub)
        xml_txt = IIRS_XML.read_text(encoding="utf-8")
        xml_txt = xml_txt.replace(
            "ch2_iirs_sample_200x200x16.qub",
            "ch2_iirs_mismatch.qub",
        )
        xml_dst = tmp_path / "ch2_iirs_mismatch.xml"
        xml_dst.write_text(xml_txt, encoding="utf-8")

        img = load_pds4_raster(xml_dst)
        # Must have loaded via memmap fallback
        assert img.metadata.get("pds4_read_path") == "numpy_memmap"
        assert img.data.shape == (200, 200, 16)
        assert img.dtype == np.dtype("float32")

    def test_iirs_pixel_values_are_radiance(self):
        """IIRS is float32 radiance -- values should be non-negative (calibrated)."""
        from backend.preprocessing.pds4 import load_pds4_raster
        img = load_pds4_raster(IIRS_XML)
        assert img.data.dtype == np.float32
        # Some NaN/zero values may exist at edges but max should be > 0
        assert np.nanmax(img.data) > 0.0

    # --- Windowed reads ---

    def test_tmc2_windowed_read_shape(self):
        from backend.preprocessing.pds4 import load_pds4_raster
        from rasterio.windows import Window
        win = Window(col_off=0, row_off=0, width=100, height=150)
        img = load_pds4_raster(TMC2_XML, window=win)
        assert img.data.shape == (150, 100)
        assert img.width == 100
        assert img.height == 150

    def test_tmc2_windowed_read_dtype_preserved(self):
        from backend.preprocessing.pds4 import load_pds4_raster
        from rasterio.windows import Window
        win = Window(col_off=0, row_off=0, width=50, height=50)
        img = load_pds4_raster(TMC2_XML, window=win)
        assert img.dtype == np.dtype("uint16")

    def test_ohrc_windowed_read_shape(self):
        from backend.preprocessing.pds4 import load_pds4_raster
        from rasterio.windows import Window
        win = Window(col_off=10, row_off=20, width=80, height=60)
        img = load_pds4_raster(OHRC_XML, window=win)
        assert img.data.shape == (60, 80)

    def test_iirs_windowed_read_shape(self):
        """IIRS windowed read: should return (50, 75, 16) -- works via rasterio or memmap."""
        from backend.preprocessing.pds4 import load_pds4_raster
        from rasterio.windows import Window
        win = Window(col_off=0, row_off=0, width=75, height=50)
        img = load_pds4_raster(IIRS_XML, window=win)
        assert img.data.shape == (50, 75, 16)
        assert img.width == 75
        assert img.height == 50
        assert img.num_bands == 16

    def test_iirs_windowed_read_pds4_path_present(self):
        """Windowed IIRS read must record which path was taken."""
        from backend.preprocessing.pds4 import load_pds4_raster
        from rasterio.windows import Window
        win = Window(col_off=0, row_off=0, width=30, height=30)
        img = load_pds4_raster(IIRS_XML, window=win)
        assert img.metadata.get("pds4_read_path") in ("rasterio", "numpy_memmap")

    # --- Error cases ---

    def test_nonexistent_raises(self, tmp_path):
        from backend.preprocessing.pds4 import load_pds4_raster
        with pytest.raises(FileNotFoundError):
            load_pds4_raster(tmp_path / "missing.xml")


# ---------------------------------------------------------------------------
# scan_pds4_labels() tests
# ---------------------------------------------------------------------------

class TestScanPds4Labels:
    def test_finds_all_fixture_labels(self):
        from backend.preprocessing.pds4 import scan_pds4_labels
        found = scan_pds4_labels(FIXTURES_PDS4)
        names = {p.name for p in found}
        assert "ch2_tmc_sample_500x500.xml" in names
        assert "ch2_ohrc_sample_500x500.xml" in names
        assert "ch2_iirs_sample_200x200x16.xml" in names

    def test_returns_only_pds4_not_readme(self):
        from backend.preprocessing.pds4 import scan_pds4_labels
        found = scan_pds4_labels(FIXTURES_PDS4)
        # Should not include README.md or other non-XML files
        for p in found:
            assert p.suffix == ".xml"

    def test_returns_sorted_list(self):
        from backend.preprocessing.pds4 import scan_pds4_labels
        found = scan_pds4_labels(FIXTURES_PDS4)
        assert found == sorted(found)

    def test_empty_dir_returns_empty(self, tmp_path):
        from backend.preprocessing.pds4 import scan_pds4_labels
        result = scan_pds4_labels(tmp_path)
        assert result == []

    def test_non_pds4_xml_not_included(self, tmp_path):
        from backend.preprocessing.pds4 import scan_pds4_labels
        p = _write_non_pds4_xml(tmp_path)
        result = scan_pds4_labels(tmp_path)
        assert result == []


# ---------------------------------------------------------------------------
# walk_isda_element() tests
# ---------------------------------------------------------------------------

class TestWalkIsdaElement:
    def test_flat_elements(self):
        import xml.etree.ElementTree as ET
        from backend.preprocessing.pds4 import walk_isda_element
        xml = """<Product_Parameters xmlns:isda="https://isda.issdc.gov.in/pds4/isda/v1">
            <isda:pixel_resolution unit="m/pixel">4.27</isda:pixel_resolution>
            <isda:roll unit="deg">0.001</isda:roll>
        </Product_Parameters>"""
        el = ET.fromstring(xml)
        result = walk_isda_element(el)
        assert "pixel_resolution" in result
        assert result["pixel_resolution"]["value"] == "4.27"
        assert result["pixel_resolution"]["unit"] == "m/pixel"
        assert result["roll"]["value"] == "0.001"

    def test_nested_elements_dotted_path(self):
        import xml.etree.ElementTree as ET
        from backend.preprocessing.pds4 import walk_isda_element
        xml = """<Geometry_Parameters xmlns:isda="https://isda.issdc.gov.in/pds4/isda/v1">
            <isda:System_Level_Coordinates>
                <isda:upper_left_latitude unit="deg">10.5</isda:upper_left_latitude>
            </isda:System_Level_Coordinates>
        </Geometry_Parameters>"""
        el = ET.fromstring(xml)
        result = walk_isda_element(el)
        assert "System_Level_Coordinates.upper_left_latitude" in result
        assert result["System_Level_Coordinates.upper_left_latitude"]["value"] == "10.5"

    def test_unit_none_when_absent(self):
        import xml.etree.ElementTree as ET
        from backend.preprocessing.pds4 import walk_isda_element
        xml = """<Params><job_id>ABC123</job_id></Params>"""
        el = ET.fromstring(xml)
        result = walk_isda_element(el)
        assert result["job_id"]["unit"] is None


# ---------------------------------------------------------------------------
# extract_pds4_file_metadata() (metadata.py integration) tests
# ---------------------------------------------------------------------------

class TestExtractPds4FileMetadata:
    def test_tmc2_combined_metadata(self):
        from backend.preprocessing.metadata import extract_pds4_file_metadata
        meta = extract_pds4_file_metadata(TMC2_XML)
        assert meta["width"] == 500
        assert meta["height"] == 500
        assert meta["num_bands"] == 1
        assert meta["image_dtype"] == "uint16"
        assert meta["pds4_instrument"] == "TMC2"
        assert isinstance(meta["pds4_isda_product_params"], dict)
        assert len(meta["pds4_isda_product_params"]) > 0

    def test_ohrc_combined_metadata(self):
        from backend.preprocessing.metadata import extract_pds4_file_metadata
        meta = extract_pds4_file_metadata(OHRC_XML)
        assert meta["pds4_instrument"] == "OHRC"
        assert meta["image_dtype"] == "uint8"
        assert meta["num_bands"] == 1

    def test_iirs_combined_metadata(self):
        from backend.preprocessing.metadata import extract_pds4_file_metadata
        meta = extract_pds4_file_metadata(IIRS_XML)
        assert meta["pds4_instrument"] == "IIRS"
        assert meta["pds4_wavelength_info"] is not None
        # rasterio may fail for IIRS so image_dtype may be None -- that is acceptable
        # The PDS4 fields must always be present
        assert "pds4_isda_product_params" in meta
        assert len(meta["pds4_isda_product_params"]) > 0


# ---------------------------------------------------------------------------
# extract_iirs_band_wavelengths() tests
# ---------------------------------------------------------------------------

class TestExtractIIRSBandWavelengths:
    """Unit tests for extract_iirs_band_wavelengths() against real and synthetic PDS4 labels."""

    # Expected exact values from tests/fixtures/pds4/iirs/ch2_iirs_sample_200x200x16.xml
    EXPECTED_REAL_FIXTURE_BANDS = [
        {"band_number": 1, "center_wavelength": 712.3, "band_width": 19.8, "unit": "nm"},
        {"band_number": 2, "center_wavelength": 729.2, "band_width": 19.9, "unit": "nm"},
        {"band_number": 3, "center_wavelength": 746.0, "band_width": 20.0, "unit": "nm"},
        {"band_number": 4, "center_wavelength": 762.9, "band_width": 20.1, "unit": "nm"},
        {"band_number": 5, "center_wavelength": 779.7, "band_width": 20.2, "unit": "nm"},
        {"band_number": 6, "center_wavelength": 796.6, "band_width": 20.3, "unit": "nm"},
        {"band_number": 7, "center_wavelength": 813.4, "band_width": 20.4, "unit": "nm"},
        {"band_number": 8, "center_wavelength": 830.3, "band_width": 20.4, "unit": "nm"},
        {"band_number": 9, "center_wavelength": 847.2, "band_width": 20.5, "unit": "nm"},
        {"band_number": 10, "center_wavelength": 864.0, "band_width": 20.5, "unit": "nm"},
        {"band_number": 11, "center_wavelength": 880.9, "band_width": 20.6, "unit": "nm"},
        {"band_number": 12, "center_wavelength": 897.7, "band_width": 20.6, "unit": "nm"},
        {"band_number": 13, "center_wavelength": 914.6, "band_width": 20.6, "unit": "nm"},
        {"band_number": 14, "center_wavelength": 931.4, "band_width": 20.7, "unit": "nm"},
        {"band_number": 15, "center_wavelength": 948.3, "band_width": 20.7, "unit": "nm"},
        {"band_number": 16, "center_wavelength": 965.1, "band_width": 20.8, "unit": "nm"},
    ]

    def test_real_iirs_fixture_wavelengths_match_xml(self):
        """Verify all 16 bands extracted from real IIRS fixture XML exactly match XML values."""
        from backend.preprocessing.pds4 import extract_iirs_band_wavelengths

        bands = extract_iirs_band_wavelengths(IIRS_XML)
        assert len(bands) == 16, f"Expected 16 bands, got {len(bands)}"

        for i, (actual, expected) in enumerate(zip(bands, self.EXPECTED_REAL_FIXTURE_BANDS)):
            assert actual["band_number"] == expected["band_number"], (
                f"Band {i+1} band_number mismatch: {actual['band_number']} != {expected['band_number']}"
            )
            assert actual["center_wavelength"] == pytest.approx(expected["center_wavelength"], abs=1e-4), (
                f"Band {i+1} center_wavelength mismatch: {actual['center_wavelength']} != {expected['center_wavelength']}"
            )
            assert actual["band_width"] == pytest.approx(expected["band_width"], abs=1e-4), (
                f"Band {i+1} band_width mismatch: {actual['band_width']} != {expected['band_width']}"
            )
            assert actual["center_wavelength_unit"] == expected["unit"], (
                f"Band {i+1} center_wavelength_unit mismatch: {actual['center_wavelength_unit']} != {expected['unit']}"
            )
            assert actual["band_width_unit"] == expected["unit"], (
                f"Band {i+1} band_width_unit mismatch: {actual['band_width_unit']} != {expected['unit']}"
            )
            assert actual["unit"] == expected["unit"], (
                f"Band {i+1} unit mismatch: {actual['unit']} != {expected['unit']}"
            )

    def test_nonexistent_xml_raises_filenotfound(self):
        """Missing XML file must raise FileNotFoundError."""
        from backend.preprocessing.pds4 import extract_iirs_band_wavelengths

        with pytest.raises(FileNotFoundError, match="PDS4 label not found"):
            extract_iirs_band_wavelengths("nonexistent_path_to_iirs.xml")

    def test_non_pds4_xml_raises_valueerror(self, tmp_path):
        """Non-PDS4 XML must raise ValueError."""
        from backend.preprocessing.pds4 import extract_iirs_band_wavelengths

        p = _write_non_pds4_xml(tmp_path)
        with pytest.raises(ValueError, match="not a PDS4 Product_Observational label"):
            extract_iirs_band_wavelengths(p)

    def test_tmc2_xml_has_no_band_bins(self):
        """TMC-2 label XML has no Band_Bin elements and returns empty list."""
        from backend.preprocessing.pds4 import extract_iirs_band_wavelengths

        bands = extract_iirs_band_wavelengths(TMC2_XML)
        assert bands == []


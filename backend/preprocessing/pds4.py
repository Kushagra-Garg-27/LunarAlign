"""
SIH26166 -- PDS4 label parser and raster loader.

Handles Chandrayaan-2 PDS4 products for three instruments:

* **TMC-2** (Terrain Mapping Camera-2)  -- Array_2D_Image, uint16, single band
* **OHRC**  (Orbiter High Resolution Camera) -- Array_2D_Image, uint8, single band
* **IIRS**  (Imaging InfraRed Spectrometer)  -- Array_3D_Spectrum, float32, 256 bands

Design notes
------------
- PDS4 detection is namespace + root-tag based, never extension-based.
- Instrument identification reads Observing_System_Component[type=Instrument]/name
  from the label XML -- not the filename, because filename suffixes encode pipeline
  version tags, not instrument identity.
- TMC-2 metadata uses a confirmed hardcoded field list (verified on real sample).
- OHRC / IIRS metadata uses a generic isda: namespace walker that collects *all*
  child elements as {localname: {"value": str, "unit": str|None}}. Field names
  were observed in a single real sample; they are not validated against a schema.
- Raster loading:
  * TMC-2 / OHRC: rasterio.open(xml_path) via GDAL PDS4 driver. Works directly
    because the label file_name matches the binary on disk.
  * IIRS: the downloaded sample has a filename mismatch (label references .qub but
    the binary is named -001.qub, a multipart convention). GDAL fails with
    "Image file is too small". Fallback: parse the Array_3D_Spectrum axis elements
    from the label XML and read via numpy memmap (BSQ layout, float32). This
    fallback is triggered automatically when rasterio fails AND Array_3D_Spectrum
    is present in the label.
- Windowed reads:
  * For rasterio path: rasterio.Window is passed directly to src.read().
  * For memmap path (IIRS): window is interpreted as (row_off, col_off, height,
    width) compatible with rasterio.windows.Window attributes.
- Filesystem paths are never stored in returned metadata.

IIRS wavelength metadata
------------------------
256 Band_Bin entries with center_wavelength and band_width (nm) are embedded in
the label XML under File_Area_Observational/Array_3D_Spectrum/Axis_Array/
Band_Bin_Set/Band_Bin. Extraction is out of scope for Stage 2a.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np

from backend.preprocessing.datamodel import PDS4Metadata, RawImage

logger = logging.getLogger("sih26166.preprocessing.pds4")

# ---------------------------------------------------------------------------
# PDS4 namespace constants
# ---------------------------------------------------------------------------

_PDS4_NS = "http://pds.nasa.gov/pds4/pds/v1"
_ISDA_NS = "https://isda.issdc.gov.in/pds4/isda/v1"
_PDS4_ROOT_TAG = f"{{{_PDS4_NS}}}Product_Observational"

_NS = {
    "pds": _PDS4_NS,
    "isda": _ISDA_NS,
}

# Instrument name strings from Observing_System_Component[type=Instrument]/name
# (lowercased for matching)
_INSTRUMENT_NAME_MAP: dict[str, str] = {
    "terrain mapping camera": "TMC2",
    "orbiter high resolution camera": "OHRC",
    "imaging infrared spectrometer": "IIRS",
}

# TMC-2 confirmed field list under isda:Product_Parameters
# (verified on real sample ch2_tmc_ncn_20191213T0552461631_d_img_gds)
_TMC2_PRODUCT_PARAM_FIELDS = (
    "pixel_resolution",
    "sun_azimuth",
    "sun_elevation",
    "solar_incidence",
    "spacecraft_altitude",
    "roll",
    "pitch",
    "yaw",
    "imaging_orbit_number",
    "job_id",
    "projection",
    "area",
)

# PDS4 data_type -> numpy dtype
_PDS4_DTYPE_MAP: dict[str, str] = {
    "UnsignedByte": "uint8",
    "UnsignedLSB2": "uint16",
    "SignedLSB2": "int16",
    "UnsignedLSB4": "uint32",
    "SignedLSB4": "int32",
    "IEEE754LSBSingle": "float32",
    "IEEE754LSBDouble": "float64",
    "IEEE754MSBSingle": "float32",  # will need byteswap but handle gracefully
    "IEEE754MSBDouble": "float64",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def is_pds4_label(path: str | Path) -> bool:
    """Return True if *path* is a PDS4 Product_Observational label.

    Detection is namespace + root-tag based (never extension-based).
    Returns False on any parse error or IO error.
    """
    path = Path(path)
    try:
        context = ET.iterparse(str(path), events=("start",))
        _, root_elem = next(context)
        return root_elem.tag == _PDS4_ROOT_TAG
    except (ET.ParseError, StopIteration, OSError):
        return False


def scan_pds4_labels(root_dir: str | Path) -> list[Path]:
    """Recursively scan *root_dir* and return paths of all PDS4 labels found.

    Parameters
    ----------
    root_dir:
        Directory to search (recursive).

    Returns
    -------
    list[Path]
        Sorted list of absolute paths to PDS4 Product_Observational labels.
    """
    root_dir = Path(root_dir)
    results: list[Path] = []
    for xml_path in root_dir.rglob("*.xml"):
        if is_pds4_label(xml_path):
            results.append(xml_path)
    return sorted(results)


def identify_instrument(tree: ET.ElementTree) -> str:
    """Return the canonical instrument identifier from a parsed PDS4 label tree.

    Reads Observation_Area/Observing_System/Observing_System_Component entries,
    finds the one with type == Instrument, and maps its name text.

    Returns one of "TMC2", "OHRC", "IIRS", or "UNKNOWN".
    """
    root = tree.getroot()
    components = root.findall(
        "pds:Observation_Area/pds:Observing_System/pds:Observing_System_Component",
        _NS,
    )
    for comp in components:
        type_el = comp.find("pds:type", _NS)
        name_el = comp.find("pds:name", _NS)
        if type_el is not None and name_el is not None:
            if (type_el.text or "").strip().lower() == "instrument":
                raw_name = (name_el.text or "").strip().lower()
                canonical = _INSTRUMENT_NAME_MAP.get(raw_name, "UNKNOWN")
                if canonical == "UNKNOWN":
                    logger.warning(
                        "Unrecognised instrument name %r -- returning 'UNKNOWN'.",
                        raw_name,
                    )
                return canonical
    logger.warning("No Instrument-type Observing_System_Component found in label.")
    return "UNKNOWN"


def walk_isda_element(element: ET.Element) -> dict[str, Any]:
    """Flatten all children of an isda: element into a key-value dict.

    Keys are local tag names (or dotted paths for nested containers).
    Values are {"value": str, "unit": str | None} dicts.
    """
    result: dict[str, Any] = {}
    _walk_recursive(element, prefix="", result=result)
    return result


def _walk_recursive(
    element: ET.Element, prefix: str, result: dict[str, Any]
) -> None:
    for child in element:
        local_name = _local_tag(child.tag)
        key = local_name if not prefix else f"{prefix}.{local_name}"
        if len(child) > 0:
            _walk_recursive(child, prefix=key, result=result)
        else:
            text = (child.text or "").strip()
            unit = child.get("unit")
            result[key] = {"value": text, "unit": unit}


def _local_tag(tag: str) -> str:
    if tag.startswith("{"):
        return tag.split("}", 1)[1]
    return tag


def extract_pds4_metadata(xml_path: str | Path) -> PDS4Metadata:
    """Parse a PDS4 label and return a structured PDS4Metadata object.

    Parameters
    ----------
    xml_path:
        Path to the PDS4 .xml label file.

    Raises
    ------
    FileNotFoundError
        If xml_path does not exist.
    ValueError
        If xml_path is not a valid PDS4 Product_Observational label.
    """
    xml_path = Path(xml_path)
    if not xml_path.exists():
        raise FileNotFoundError(f"PDS4 label not found: {xml_path}")

    try:
        tree = ET.parse(str(xml_path))
    except ET.ParseError as exc:
        raise ValueError(f"Malformed XML in {xml_path.name}: {exc}") from exc

    root = tree.getroot()
    if root.tag != _PDS4_ROOT_TAG:
        raise ValueError(
            f"{xml_path.name} is not a PDS4 Product_Observational label "
            f"(root tag: {root.tag!r})"
        )

    instrument = identify_instrument(tree)

    id_area = root.find("pds:Identification_Area", _NS)
    logical_identifier = _text(id_area, "pds:logical_identifier")
    title = _text(id_area, "pds:title")

    obs_area = root.find("pds:Observation_Area", _NS)
    tc = obs_area.find("pds:Time_Coordinates", _NS) if obs_area is not None else None
    start_date_time = _text(tc, "pds:start_date_time")
    stop_date_time = _text(tc, "pds:stop_date_time")

    inv = (
        obs_area.find("pds:Investigation_Area", _NS) if obs_area is not None else None
    )
    investigation_name = _text(inv, "pds:name")

    mission_area = (
        obs_area.find("pds:Mission_Area", _NS) if obs_area is not None else None
    )

    isda_product_params: dict[str, Any] = {}
    isda_geometry_params: dict[str, Any] = {}

    if mission_area is not None:
        pp_el = mission_area.find("isda:Product_Parameters", _NS)
        gp_el = mission_area.find("isda:Geometry_Parameters", _NS)
        if pp_el is not None:
            isda_product_params = walk_isda_element(pp_el)
        else:
            logger.warning(
                "%s: isda:Product_Parameters not found in Mission_Area. "
                "Expected for Chandrayaan-2 ISRO PDS4 products.",
                xml_path.name,
            )
        if gp_el is not None:
            isda_geometry_params = walk_isda_element(gp_el)
        else:
            logger.warning(
                "%s: isda:Geometry_Parameters not found in Mission_Area.",
                xml_path.name,
            )
    else:
        logger.warning(
            "%s: Mission_Area not found -- isda: metadata will be empty.",
            xml_path.name,
        )

    tmc2_product_params: dict[str, Any] | None = None
    tmc2_geometry_params: dict[str, Any] | None = None
    if instrument == "TMC2":
        tmc2_product_params = _extract_tmc2_product_params(isda_product_params)
        tmc2_geometry_params = _extract_tmc2_geometry_params(isda_geometry_params)

    wavelength_info: str | None = None
    if instrument == "IIRS":
        wavelength_info = (
            "Wavelength per-band metadata (center_wavelength, band_width in nm) "
            "is embedded in this PDS4 label under "
            "File_Area_Observational/Array_3D_Spectrum/Axis_Array/Band_Bin_Set/Band_Bin. "
            "256 Band_Bin entries present (in full product); "
            "extraction is out of scope for Stage 2a."
        )

    logger.info(
        "Parsed PDS4 label %s: instrument=%s, isda_product_fields=%d, "
        "isda_geometry_fields=%d",
        xml_path.name,
        instrument,
        len(isda_product_params),
        len(isda_geometry_params),
    )

    return PDS4Metadata(
        logical_identifier=logical_identifier,
        title=title,
        start_date_time=start_date_time,
        stop_date_time=stop_date_time,
        investigation_name=investigation_name,
        instrument=instrument,
        isda_product_params=isda_product_params,
        isda_geometry_params=isda_geometry_params,
        tmc2_product_params=tmc2_product_params,
        tmc2_geometry_params=tmc2_geometry_params,
        wavelength_info=wavelength_info,
    )


def load_pds4_raster(
    xml_path: str | Path,
    window: Any = None,
) -> RawImage:
    """Load a PDS4 raster product from its label file.

    Routing strategy
    ----------------
    1. If the label contains Array_3D_Spectrum (e.g. IIRS), attempt rasterio
       first (it works if the binary filename matches the label exactly).
       If rasterio fails, fall back to numpy memmap using the BSQ geometry
       parsed from the label XML. The fallback is needed for IIRS downloads
       where the binary is named ``*-001.qub`` (multipart convention) but the
       label references ``*.qub``.
    2. For all other products (Array_2D_Image), use rasterio exclusively.

    Parameters
    ----------
    xml_path:
        Path to the PDS4 .xml label file.
    window:
        Optional rasterio.windows.Window for spatial subsetting.
        For the rasterio path: passed directly to src.read(window=...).
        For the memmap path (IIRS): Window.col_off/row_off/width/height
        are used to slice the spatial dimensions.

    Returns
    -------
    RawImage
        data shape: (H, W) for single-band or (H, W, C) bands-last for multi-band.

    Raises
    ------
    FileNotFoundError
        If xml_path does not exist.
    ValueError
        If all read attempts fail.
    """
    xml_path = Path(xml_path)
    if not xml_path.exists():
        raise FileNotFoundError(f"PDS4 label not found: {xml_path}")

    # Parse label to detect array structure
    try:
        tree = ET.parse(str(xml_path))
    except ET.ParseError as exc:
        raise ValueError(f"Malformed XML in {xml_path.name}: {exc}") from exc

    root = tree.getroot()
    fao = root.find("pds:File_Area_Observational", _NS)
    is_3d_spectrum = (
        fao is not None and fao.find("pds:Array_3D_Spectrum", _NS) is not None
    )

    # --- Try rasterio first (works for TMC-2, OHRC, and IIRS when filenames match) ---
    try:
        import rasterio
        with rasterio.open(str(xml_path)) as src:
            num_bands = src.count
            dtype = np.dtype(src.dtypes[0])
            if window is not None:
                data_bhw = src.read(window=window)
            else:
                data_bhw = src.read()
            height, width = data_bhw.shape[1], data_bhw.shape[2]
            if num_bands == 1:
                data = data_bhw[0]
            else:
                data = np.moveaxis(data_bhw, 0, -1)
            raster_meta: dict[str, Any] = {
                "rasterio_driver": src.driver,
                "rasterio_crs": str(src.crs) if src.crs else None,
                "rasterio_transform": list(src.transform) if src.transform else None,
                "rasterio_band_count": num_bands,
                "rasterio_dtypes": list(src.dtypes),
                "pds4_read_path": "rasterio",
            }
            if src.descriptions and any(src.descriptions):
                raster_meta["rasterio_band_descriptions"] = list(src.descriptions)

        logger.info(
            "Loaded PDS4 raster %s via rasterio: %dx%d, %d band(s), dtype=%s%s",
            xml_path.name, width, height, num_bands, dtype,
            " [windowed]" if window is not None else "",
        )
        return RawImage(
            data=data, width=width, height=height,
            num_bands=num_bands, dtype=dtype,
            source_format="PDS4", metadata=raster_meta,
        )

    except Exception as rio_exc:
        if not is_3d_spectrum:
            # For 2D products there is no memmap fallback -- re-raise
            raise ValueError(
                f"Failed to open PDS4 raster via rasterio for {xml_path.name}: "
                f"{rio_exc}"
            ) from rio_exc
        logger.warning(
            "%s: rasterio failed (%s) -- falling back to numpy memmap "
            "(Array_3D_Spectrum / BSQ layout).",
            xml_path.name, rio_exc,
        )

    # --- Memmap fallback for Array_3D_Spectrum (IIRS) ---
    return _load_3d_spectrum_memmap(xml_path, root, fao, window)


def _load_3d_spectrum_memmap(
    xml_path: Path,
    root: ET.Element,
    fao: ET.Element,
    window: Any,
) -> RawImage:
    """Load an Array_3D_Spectrum product via numpy memmap (BSQ).

    The label XML is parsed to extract:
    - The binary file name from File/file_name
    - Axis dimensions (BAND, LINE, SAMPLE) from Axis_Array elements
    - Element dtype from Element_Array/data_type

    The binary is located in the same directory as the label, with fallbacks
    for the -001.qub multipart naming convention used by ISRO downloads.
    """
    # Parse File/file_name
    file_el = fao.find("pds:File/pds:file_name", _NS)
    if file_el is None or not file_el.text:
        raise ValueError(f"{xml_path.name}: cannot find File/file_name in label")
    ref_filename = file_el.text.strip()

    # Resolve binary path with fallback for -001 multipart suffix
    binary_path = xml_path.parent / ref_filename
    if not binary_path.exists():
        # Try -001 suffix (ISRO multipart convention)
        stem = binary_path.stem
        suffix = binary_path.suffix
        candidate = xml_path.parent / f"{stem}-001{suffix}"
        if candidate.exists():
            logger.info(
                "%s: binary '%s' not found; using multipart '%s'.",
                xml_path.name, ref_filename, candidate.name,
            )
            binary_path = candidate
        else:
            raise FileNotFoundError(
                f"{xml_path.name}: referenced binary '{ref_filename}' not found in "
                f"{xml_path.parent} (also tried '{candidate.name}')"
            )

    # Parse Array_3D_Spectrum element
    arr3d = fao.find("pds:Array_3D_Spectrum", _NS)
    if arr3d is None:
        raise ValueError(f"{xml_path.name}: Array_3D_Spectrum element not found")

    # data_type -> numpy dtype
    dt_el = arr3d.find("pds:Element_Array/pds:data_type", _NS)
    if dt_el is None or not dt_el.text:
        raise ValueError(f"{xml_path.name}: Element_Array/data_type not found")
    pds4_dtype_str = dt_el.text.strip()
    np_dtype_str = _PDS4_DTYPE_MAP.get(pds4_dtype_str)
    if np_dtype_str is None:
        raise ValueError(
            f"{xml_path.name}: unknown PDS4 data_type '{pds4_dtype_str}'"
        )
    dtype = np.dtype(np_dtype_str)

    # Axis dimensions: find BAND, Line, Sample by axis_name
    axes: dict[str, int] = {}
    for axis_el in arr3d.findall("pds:Axis_Array", _NS):
        name_el = axis_el.find("pds:axis_name", _NS)
        elem_el = axis_el.find("pds:elements", _NS)
        if name_el is not None and elem_el is not None:
            axes[name_el.text.strip().upper()] = int(elem_el.text.strip())

    n_bands = axes.get("BAND", 0)
    n_lines = axes.get("LINE", 0)
    n_samples = axes.get("SAMPLE", 0)

    if n_bands == 0 or n_lines == 0 or n_samples == 0:
        raise ValueError(
            f"{xml_path.name}: could not parse BAND/LINE/SAMPLE axes, got {axes}"
        )

    logger.info(
        "%s: memmap read -- binary=%s shape=(%d, %d, %d) BSQ dtype=%s",
        xml_path.name, binary_path.name, n_bands, n_lines, n_samples, dtype,
    )

    # Memory-map the full cube (BSQ: bands x lines x samples)
    data_3d = np.memmap(
        str(binary_path),
        dtype=dtype,
        mode="r",
        shape=(n_bands, n_lines, n_samples),
    )

    # Apply spatial window if requested
    if window is not None:
        row_off = int(window.row_off)
        col_off = int(window.col_off)
        h = int(window.height)
        w = int(window.width)
        # Clamp to valid range
        row_off = max(0, min(row_off, n_lines))
        col_off = max(0, min(col_off, n_samples))
        h = min(h, n_lines - row_off)
        w = min(w, n_samples - col_off)
        data_bhw = np.array(
            data_3d[:, row_off: row_off + h, col_off: col_off + w]
        )  # copy out of memmap
    else:
        data_bhw = np.array(data_3d)

    del data_3d  # release memmap

    num_bands, height, width = data_bhw.shape
    # Convert BSQ (bands, H, W) -> bands-last (H, W, C) or (H, W) if single band
    if num_bands == 1:
        data = data_bhw[0]
    else:
        data = np.moveaxis(data_bhw, 0, -1)  # (H, W, C)

    raster_meta: dict[str, Any] = {
        "rasterio_driver": "numpy_memmap_bsq",
        "rasterio_crs": None,
        "rasterio_transform": None,
        "rasterio_band_count": num_bands,
        "rasterio_dtypes": [str(dtype)] * num_bands,
        "pds4_read_path": "numpy_memmap",
        "pds4_binary_file": binary_path.name,
    }

    logger.info(
        "Loaded PDS4 Array_3D_Spectrum %s via memmap: %dx%d, %d band(s), dtype=%s%s",
        xml_path.name, width, height, num_bands, dtype,
        " [windowed]" if window is not None else "",
    )

    return RawImage(
        data=data, width=width, height=height,
        num_bands=num_bands, dtype=dtype,
        source_format="PDS4", metadata=raster_meta,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _text(element: ET.Element | None, path: str) -> str | None:
    if element is None:
        return None
    found = element.find(path, _NS)
    if found is None or found.text is None:
        return None
    return found.text.strip()


def _extract_tmc2_product_params(generic: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in _TMC2_PRODUCT_PARAM_FIELDS:
        if field in generic:
            result[field] = generic[field]
        else:
            logger.debug("TMC-2 expected field %r not found in label.", field)
    return result


def _extract_tmc2_geometry_params(generic: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, val in generic.items():
        if key.startswith("System_Level_Coordinates.") or key.startswith(
            "Refined_Corner_Coordinates."
        ):
            result[key] = val
    return result

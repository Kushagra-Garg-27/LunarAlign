# IIRS Test Fixture

**Real Chandrayaan-2 IIRS (Imaging InfraRed Spectrometer) data, cropped from a
downloaded calibrated sample. NOT synthetic.**

## Source

- Instrument: IIRS (Imaging InfraRed Spectrometer)
- Original product: `ch2_iir_nci_20240616T1338294007_d_img_d18`
- Observation date: 2024-06-16T13:38:29Z
- Processing level: Calibrated (radiance, units: uW/cm^2/sr/um)
- Downloaded from ISDA/PRADAN portal

## Verified raster properties (from real data)

- Full dimensions: 250 samples x 13342 lines x 256 bands
- Data type: IEEE754LSBSingle (float32)
- Array type: Array_3D_Spectrum (BSQ interleave -- BAND is axis 1)
- Pixel resolution: 82.70 m/pixel
- Spectral range: 712.3 nm to ~5000 nm (256 bands)
- Band_Bin_Set with center_wavelength/band_width per band embedded in PDS4 label XML

## GDAL/rasterio note

GDAL's PDS4 driver CANNOT open this product because the downloaded binary is
named `*-001.qub` (ISRO multipart convention) while the label's <file_name>
references `*.qub`. The loader falls back to numpy memmap using BSQ geometry
parsed from the label XML.

## Crop / reduction parameters

- Spatial crop: lines 0-200, samples 0-200 of the full 250x13342 grid
- Band reduction: first 16 of 256 bands
- Result: 16 bands x 200 lines x 200 samples, float32, BSQ layout
- **This fixture is NOT representative of full IIRS spectral coverage.**
  **It tests parser mechanics only (loading, metadata extraction, windowed read).**

## Fixture files

- `ch2_iirs_sample_200x200x16.qub` — 2,560,000 bytes (16×200×200×4 bytes), real pixel data
- `ch2_iirs_sample_200x200x16.xml` — PDS4 label with real isda: metadata, axis elements
  reduced to 16/200/200, Band_Bin_Set trimmed to 16 entries, md5_checksum omitted

## IIRS isda:Product_Parameters fields observed in real sample

(Observed in one real sample -- not a validated schema across all IIRS products)

job_id, level0_dir_name, imaging_orbit_number, dumping_orbit_number,
line_exposure_duration, gain, exposure, exposure_duration, detector_temperature,
tertiary_mirror_temperature, spectrometer_casing_temperature, dewar_vw_temperature,
detector_pixel_width, focal_length, reference_data_used, orbit_limb_direction,
spacecraft_yaw_direction, spacecraft_altitude, pixel_resolution, roll, pitch, yaw,
sun_azimuth, sun_elevation, solar_incidence, projection, area

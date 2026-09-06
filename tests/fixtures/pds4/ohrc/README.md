# OHRC Test Fixture

**Real Chandrayaan-2 OHRC (Orbiter High Resolution Camera) data, cropped from a
downloaded calibrated sample. NOT synthetic.**

## Source

- Instrument: OHRC (Orbiter High Resolution Camera)
- Original product: `ch2_ohr_ncp_20230226T2137368347_d_img_n18`
- Observation date: 2023-02-26T21:37:36Z
- Processing level: Calibrated
- Downloaded from ISDA/PRADAN portal

## Verified raster properties (from real data)

- Full dimensions: 12000 samples x 101057 lines, 1 band, uint8 (UnsignedByte)
- Pixel resolution: 0.21 m/pixel (OBSERVED from isda:Product_Parameters --
  the instrument specification quotes ~0.25 m/pixel)
- Area: South Pole (Polar stereographic projection)

## Crop parameters

- Spatial crop: rows 200–700, columns 0–500 of the full 12000×101057 strip
- Result: 500×500 pixels, 1 band, uint8
- Binary: written as raw uint8, row-major

## Fixture files

- `ch2_ohrc_sample_500x500.img` — 250,000 bytes (500×500×1 byte), real pixel data
- `ch2_ohrc_sample_500x500.xml` — PDS4 label with real isda: metadata, axis elements
  reduced to 500×500, md5_checksum omitted

## OHRC isda:Product_Parameters fields observed in real sample

(Observed in one real sample -- not a validated schema across all OHRC products)

job_id, level0_dir_name, imaging_orbit_number, dumping_orbit_number,
line_exposure_duration, bits_selection, tdi_stages, detector_pixel_width,
focal_length, reference_data_used, orbit_limb_direction, spacecraft_yaw_direction,
spacecraft_altitude, pixel_resolution, roll, pitch, yaw, sun_azimuth, sun_elevation,
solar_incidence, projection, area

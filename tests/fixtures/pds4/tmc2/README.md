# TMC-2 Test Fixture

**Real Chandrayaan-2 TMC-2 (Terrain Mapping Camera-2) data, cropped from a
downloaded calibrated sample. NOT synthetic.**

## Source

- Instrument: TMC-2 (Terrain Mapping Camera-2)
- Original product: `ch2_tmc_ncn_20191213T0552461631_d_img_gds`
- Observation date: 2019-12-13T05:52:46Z
- Processing level: Calibrated (radiometric correction applied)
- Downloaded from ISDA/PRADAN portal

## Crop parameters

- Spatial crop: rows 100–600, columns 0–500 of the full 4000×69461 strip
- Result: 500×500 pixels, 1 band, uint16 (UnsignedLSB2)
- Binary: written as raw little-endian uint16, row-major (last index fastest)

## Fixture files

- `ch2_tmc_sample_500x500.img` — 500,000 bytes (500×500×2 bytes), real pixel data
- `ch2_tmc_sample_500x500.xml` — PDS4 label with real isda: metadata, axis elements
  reduced to 500×500, md5_checksum omitted

## Provenance

Pixel values in this file are real Chandrayaan-2 TMC-2 radiance counts.
Min value observed in crop: 187. Max value: 383.
The full strip shows a known horizontal band of dropped/invalid data (KNOWN DATA
QUALITY ISSUE per ISRO) — this crop falls in a valid region.

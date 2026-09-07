# Architecture

## Preprocessing Layer

The preprocessing layer (`backend/preprocessing/`) provides the
foundational image I/O and processing bridge between the upload/input
pipeline and the downstream feature-extraction / registration algorithms.

### Module Structure

```text
backend/preprocessing/
├── __init__.py       — package docstring and symbol exports
├── datamodel.py      — RawImage, FeatureImage, PDS4Metadata data structures
├── io.py             — image loading (PNG, JPEG, TIFF, PDS4 XML)
├── pds4.py           — PDS4 label parser and raster loader (TMC-2, OHRC, IIRS)
├── grayscale.py      — conversion to 2-D feature image with dynamic range normalization
├── normalize.py      — intensity normalization utilities
├── metadata.py       — lightweight header-only metadata extraction
└── clahe.py          — optional CLAHE contrast enhancement
```

### Internal Image Representation

Two distinct data structures are maintained:

| Structure      | Purpose                        | Shape            | dtype      |
|---------------|-------------------------------|------------------|------------|
| `RawImage`     | Full-fidelity loaded data      | `(H, W)` or `(H, W, C)` | preserved  |
| `FeatureImage` | 2-D single-channel for feature extraction | `(H, W)` | `float32` |

**RawImage** preserves all original bands, dtype, and source metadata.
No silent normalisation or dtype conversion is performed during loading.

**FeatureImage** is derived from a RawImage via an explicit conversion
step.  It is always 2-D float32 and is the representation consumed by
classical CV algorithms (SIFT, Phase Congruency, etc.).

### Image I/O

| Format     | Loader    | Notes                                    |
|-----------|-----------|------------------------------------------|
| PNG       | Pillow    | Standard RGB/RGBA/grayscale (`io.py:L71`) |
| JPEG      | Pillow    | Lossy; RGB uint8 (`io.py:L71`)           |
| TIFF      | rasterio → Pillow fallback | Supports multi-band GeoTIFF (`io.py:L69, L141-161`) |
| PDS4 XML  | rasterio GDAL PDS4 driver | TMC-2 and OHRC (`io.py:L73-74, L84-126`) |

For TIFF files, rasterio is attempted first (accurate multi-band and
GeoTIFF support via `_load_rasterio` at `backend/preprocessing/io.py:L164-210`).
If rasterio fails, Pillow is used as a fallback for simple TIFF files.

For PDS4 `.xml` label inputs, `load_image()` (`backend/preprocessing/io.py:L42-76`)
delegates to `_load_pds4()` (`backend/preprocessing/io.py:L84-126`). The loader
validates that the file is a genuine PDS4 `Product_Observational` XML label via
`is_pds4_label()` (`backend/preprocessing/pds4.py:L112-125`), verifies the instrument
identity using `identify_instrument()` (`backend/preprocessing/pds4.py:L148-175`),
and accepts **TMC-2** and **OHRC** products (`io.py:L118-122`),
loading the pixel raster via `load_pds4_raster()` (`backend/preprocessing/pds4.py:L340-416`).
**IIRS** products remain explicitly unsupported at this stage (`io.py:L118-122` raises
`ValueError("Unsupported PDS4 instrument: 'IIRS'. Only TMC-2 and OHRC products are supported in this pipeline.")`)
pending architectural decisions on hyperspectral band reduction (simple band-averaging vs. PCA).
Any non-PDS4 XML or unsupported instrument product raises a descriptive `ValueError`.

Multi-band rasters retain **all** bands in the RawImage.  No bands are
silently discarded.

### Grayscale / Feature Image Conversion

Conversion from `RawImage` to `FeatureImage` is performed by `to_feature_image()`
(`backend/preprocessing/grayscale.py:L33-90`):

| Input                  | Strategy                          | Method label           |
|-----------------------|----------------------------------|----------------------|
| Single-band (grayscale)| Passthrough to float32           | `passthrough` (`grayscale.py:L56-62`) |
| RGB (3-band)          | BT.709 luminance                 | `luminance_bt709` (`grayscale.py:L63-66`) |
| RGBA (4-band)         | Drop alpha → BT.709              | `luminance_bt709_alpha_dropped` (`grayscale.py:L67-71`) |
| Multi-band (>4)       | Per-pixel mean across bands      | `band_mean_Nbands` (`grayscale.py:L72-76`) |

#### Dynamic Range Normalization

Following channel reduction, `to_feature_image()` executes `_normalize_dynamic_range()`
(`backend/preprocessing/grayscale.py:L92-117`).
- When input pixel values exceed standard uint8 bounds (`lo < 0.0 or hi > 255.0`),
  such as raw uint16 TMC-2 radiance counts in `[187, 383]` (or up to 40,000+),
  a linear min-max stretch to `[0, 255]` is automatically applied (`grayscale.py:L108-113`).
  This ensures contrast is preserved and prevents OpenCV's downstream `prepare_for_sift()`
  (`backend/features/sift.py:L50-66`) from clipping all high-range values to uniform 255
  (which would result in near-zero keypoints).
- If pixel values already reside within `[0, 255]`, the array is preserved as-is without
  modification (`grayscale.py:L115-116`).

The multi-band mean strategy is a documented generic approach.  It will
be replaced by PCA, specific band selection, or domain-aware spectral
reduction when the IIRS pipeline is implemented.

### Normalization

Two opt-in normalization functions:

- **`normalize_minmax`** — linear stretch to `[0, 1]` (or custom range).
- **`normalize_percentile`** — robust stretch with percentile clipping
  (default 2nd–98th percentile) to reduce influence of outlier pixels.

Both operate on copies; the original data is never modified.
Normalisation is never automatic — the caller explicitly chooses when
and how to normalise.

### CLAHE

CLAHE (Contrast Limited Adaptive Histogram Equalisation) is available as
an **optional** preprocessing step via `apply_clahe()`.

- Operates only on `FeatureImage` (2-D float), never on raw data.
- Returns a new `FeatureImage`; the input is not modified.
- Default parameters: `clip_limit=2.0`, `tile_grid_size=(8, 8)`.
- Useful for lunar imagery with strong sun-angle / illumination
  variations.
- The conversion method string is appended with `+clahe` for
  provenance tracking.

### Upload → Preprocessing Boundary

The `/upload` endpoint remains responsible for:
- file validation
- storage
- **lightweight** metadata extraction (dimensions, bands, dtype via
  header-only reading)

Full pixel-data preprocessing (loading into RawImage, conversion,
normalisation, CLAHE) is **not** performed during upload.  It will be
triggered by the processing/registration pipeline in later stages.

### Metadata in API Response

The upload response now includes:

| Field         | Type       | Description                            |
|--------------|-----------|----------------------------------------|
| `width`       | `int?`    | Image width in pixels                  |
| `height`      | `int?`    | Image height in pixels                 |
| `num_bands`   | `int?`    | Number of spectral bands / channels    |
| `image_dtype` | `str?`    | NumPy dtype string (e.g. `"uint8"`)    |

These are populated via header-only reading at upload time.  Full
raster metadata (CRS, transform, band descriptions) is available in
the `RawImage.metadata` dict when the image is loaded for processing.

---

## Feature Extraction Layer

The feature extraction layer (`backend/features/`) provides keypoint
detection and descriptor computation.

### Module Structure

```text
backend/features/
├── __init__.py         — package docstring
├── models.py           — Keypoint, SIFTFeatures, match data structures
├── sift.py             — SIFT detection + descriptor extraction
└── visualization.py    — debug match drawing utility (optional)
```

### Data Flow

```text
FeatureImage (float32)
    → prepare_for_sift() → uint8 grayscale
    → cv2.SIFT_create().detectAndCompute()
    → SIFTFeatures (keypoints + descriptors)
```

The `FeatureImage` is explicitly converted to uint8 for OpenCV.
The original float32 data is never modified.

---

## Matching Layer

The matching layer (`backend/matching/`) provides descriptor matching
and correspondence filtering.

### Module Structure

```text
backend/matching/
├── __init__.py       — package docstring
├── flann.py          — FLANN kNN matching (KD-tree)
└── ratio_test.py     — Lowe distance ratio test
```

### Classical Baseline Pipeline

```text
SIFTFeatures (ref)  ─┐
                      ├─→ FLANN kNN match ─→ list[RawMatch]
SIFTFeatures (tgt)  ─┘                          │
                                                 ▼
                                    Lowe ratio test (d1/d2 < 0.75)
                                                 │
                                                 ▼
                                           MatchResult
                                    (FilteredMatch with ref_pt, tgt_pt)
```

### Coordinate Convention

All point coordinates throughout the project use **(x, y)** with origin
at the **top-left** corner of the image:

- `x` = column index (increases rightward)
- `y` = row index (increases downward)

This matches OpenCV's `cv2.KeyPoint.pt` convention.

### API Boundary

The feature/matching/geometry/registration/evaluation modules are
**internal services** orchestrated by the registration service.
The `/register` endpoint exposes the classical pipeline.
The `/upload` endpoint remains unchanged.

---

## Geometry Layer

The geometry layer (`backend/geometry/`) provides geometric verification,
transformation estimation, and point transformation utilities.

### Module Structure

```text
backend/geometry/
├── __init__.py       — package docstring
├── models.py          — GeometricResult, ErrorMetrics, TransformModel, etc.
├── estimation.py      — robust estimation (MAGSAC++ / USAC / RANSAC)
└── transform.py       — point transformation utilities
```

### Classical Registration Baseline Pipeline

```text
SIFTFeatures (ref)  ─┐
                      ├─→ FLANN kNN match ─→ Lowe ratio test
SIFTFeatures (tgt)  ─┘                          │
                                                 ▼
                                          MatchResult
                                    (list[FilteredMatch])
                                                 │
                                                 ▼
                                     estimate_transform()
                                    MAGSAC++ / RANSAC fallback
                                                 │
                                                 ▼
                                        GeometricResult
                              (transform_matrix, inlier_mask,
                               error_metrics, estimator_method)
```

### Estimator Selection

The module detects OpenCV's USAC capabilities at runtime:
- If `cv2.USAC_MAGSAC` is available → uses MAGSAC++
- Otherwise → falls back to `cv2.RANSAC`

The actual method used is always recorded in `GeometricResult.estimator_method`.

### Data Separation

Geometry is kept separate from feature extraction and matching:
- Input: `list[FilteredMatch]` (from matching layer)
- Output: `GeometricResult` (self-contained result)
- No direct dependency on SIFT internals or FLANN

---

## Registration Layer

The registration layer (`backend/registration/`) produces the registered
(warped) output image from the estimated transformation.

### Module Structure

```text
backend/registration/
├── __init__.py       — package docstring
├── models.py          — WarpConfig, RegistrationResult data structures
└── warping.py         — affine and perspective image warping
```

### Data Flow

```text
GeometricResult
   (transform_matrix, transform_model)
         │
         ▼
    warp_image()
   cv2.warpAffine / cv2.warpPerspective
         │
         ▼
  RegistrationResult
   (registered_image, metadata)
```

### Transform Direction

The matrix from `estimate_transform()` maps **reference → target**.
It is passed directly to `cv2.warpAffine` / `cv2.warpPerspective`
with **no inversion**.

- Source image: reference image
- Output: reference image warped into target coordinate frame

### Separation of Concerns

- Registration/warping does not re-estimate the transformation
- No dependency on SIFT, FLANN, or matching internals
- Input: `np.ndarray` (image) + `np.ndarray` (matrix) + `TransformModel`
- Output: `RegistrationResult` (self-contained)

---

## Evaluation Layer

The evaluation layer (`backend/evaluation/`) provides metrics
extraction, spatial distribution analysis, and visualization utilities.

### Module Structure

```text
backend/evaluation/
├── __init__.py         — package docstring
├── metrics.py           — MatchQualitySummary from GeometricResult
├── spatial.py           — 8×8 grid bucketing, normalized entropy
└── visualization.py     — match drawing, inlier/outlier, overlay
```

### Data Flow

```text
GeometricResult ──────┐
FilteredMatches ──────┤
RegistrationResult ───┤
                      ▼
               evaluation layer
              ┌────────────────┐
              │  metrics.py    │ → MatchQualitySummary
              │  spatial.py    │ → SpatialDistribution
              │  visualization │ → BGR uint8 images
              └────────────────┘
```

### Separation of Concerns

- Metrics **reuse** existing `ErrorMetrics` from geometry — no duplicate RMSE
- Spatial analysis consumes `FilteredMatch` + `inlier_mask` only
- Visualization returns NumPy arrays, never saves files
- No dependency on FastAPI, upload API, or frontend

---

## Registration API

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/register` | Run classical SIFT registration pipeline |
| GET | `/register/{id}/registered` | Retrieve registered image (PNG) |
| GET | `/register/{id}/matches` | Retrieve match visualization (PNG) |
| GET | `/register/{id}/inliers` | Retrieve inlier/outlier visualization (PNG) |
| GET | `/register/{id}/overlay` | Retrieve registration overlay (PNG) |

### Service Layer

```text
POST /register
    │
    ▼
backend/api/register.py        (thin route handler)
    │
    ▼
backend/core/registration_service.py   (orchestrator)
    │
    ├─→ preprocessing.io.load_image()
    ├─→ preprocessing.grayscale.to_feature_image()
    ├─→ features.sift.extract_sift()
    ├─→ matching.flann.flann_knn_match()
    ├─→ matching.ratio_test.apply_ratio_test()
    ├─→ geometry.estimation.estimate_transform()
    ├─→ registration.warping.warp_image()
    ├─→ evaluation.spatial.compute_spatial_distribution()
    ├─→ evaluation.metrics.build_quality_summary()
    └─→ evaluation.visualization.draw_*()
    │
    ▼
backend/core/result_store.py   (in-memory, FIFO eviction)
```

### Result Storage

- In-memory `OrderedDict` with FIFO eviction (max 50 results)
- No database, no persistence, no background workers
- Results keyed by UUID-based `result_id`
- Thread-safe access

### Pipeline Mode

Currently: `classical_sift` only (same-modality baseline).

Not yet supported:
- Cross-modal (OHRC ↔ IIRS, OHRC ↔ TMC-2, TMC-2 ↔ IIRS)
- Illumination-invariant registration
- Extreme-scale registration
- Sub-pixel refinement

# Algorithms

See `PROJECT_SPEC.md` §6 for algorithm specifications.

## Implemented

### Grayscale Conversion

Converts multi-channel images to a single-channel 2-D representation
suitable for feature extraction.

| Input type        | Method                                     |
|------------------|-------------------------------------------|
| Single-band       | Direct passthrough (cast to float32)       |
| RGB (3-band)      | ITU-R BT.709 luminance: `0.2126R + 0.7152G + 0.0722B` |
| RGBA (4-band)     | Alpha discarded, then BT.709 luminance    |
| Multi-band (>4)   | Per-pixel mean across all bands            |

The multi-band mean is a generic fallback.  It does not claim to be
appropriate for IIRS hyperspectral data — that will require PCA or
domain-specific band selection (see §6 of PROJECT_SPEC.md).

### Intensity Normalization

Two opt-in methods:

1. **Min-max normalization** — linear stretch:

   `output = (pixel - min) / (max - min) * (hi - lo) + lo`

   Default output range: `[0.0, 1.0]`.

2. **Percentile normalization** — robust linear stretch with clipping:

   - Compute low/high percentiles (default 2%/98%)
   - Clip values outside the range
   - Apply linear stretch

   Reduces influence of hot pixels, saturated values, or extreme
   outliers common in raw lunar imagery.

Both return new arrays; original data is preserved.

### CLAHE

Contrast Limited Adaptive Histogram Equalisation (OpenCV implementation).

- Applied to the 2-D feature-processing representation (FeatureImage)
- Default: `clip_limit=2.0`, `tile_grid_size=(8, 8)`
- **Optional** — not applied automatically during loading
- Useful for illumination / sun-angle invariance in lunar imagery

CLAHE is documented as an optional illumination normalisation step,
not a universal preprocessing rule.  The pipeline router will decide
whether to apply it based on pair type and illumination metadata.

---

## Classical Baseline: SIFT → FLANN → Lowe Ratio Test

> **Status**: CLASSICAL BASELINE
>
> This implementation provides the benchmark correspondence pipeline
> against which later approaches (SuperPoint+LightGlue, Phase Congruency,
> MIND) will be compared.
>
> It does **not** claim to solve: cross-modal matching, extreme scale
> differences, illumination invariance, or sub-pixel registration.

### SIFT Feature Detection

**Purpose**: Detect repeatable keypoints and compute 128-dimensional
float32 descriptors for correspondence matching.

**Input representation**: SIFT receives a **uint8 grayscale** image.
The pipeline's `FeatureImage` (float32) is explicitly converted by
clipping to `[0, 255]` and casting to uint8.  The original `FeatureImage`
is never modified.

**Configurable parameters**:

| Parameter           | Default | Description                                  |
|--------------------|---------|----------------------------------------------|
| `nfeatures`         | 0       | Max keypoints (0 = unlimited)                |
| `nOctaveLayers`     | 3       | Layers per octave in the Gaussian pyramid    |
| `contrastThreshold` | 0.04    | Filter low-contrast keypoints                |
| `edgeThreshold`     | 10      | Filter edge-like responses (Harris)          |
| `sigma`             | 1.6     | Gaussian sigma for the first octave          |

**Output** (`SIFTFeatures`):

- List of `Keypoint` objects: `(x, y, size, angle, response, octave)`
- Descriptor matrix: `(N, 128)` float32
- Image dimensions and configuration record

**Coordinate convention**: All coordinates are **(x, y)** with origin at
the **top-left** corner.  x = column (rightward), y = row (downward).

### FLANN Matching

**Purpose**: Find the k nearest descriptor neighbours between two
feature sets using an approximate nearest-neighbour search.

**Descriptor type**: SIFT float32 descriptors (128-D).

**Matcher configuration**: FLANN with **KD-tree** index.

| Parameter        | Default | Description                              |
|-----------------|---------|------------------------------------------|
| `algorithm`      | 1       | FLANN_INDEX_KDTREE                       |
| `trees`          | 5       | Number of parallel KD-trees              |
| `checks`         | 50      | Leaf nodes to examine during search      |
| `k`              | 2       | Nearest neighbours per query             |

This is **not** brute-force matching.  FLANN does not silently fall back
to brute-force.

**Output** (`list[RawMatch]`):

For each query descriptor, returns up to k `NeighborInfo` objects
containing `(train_idx, distance)`.

### Lowe Ratio Test

**Purpose**: Filter ambiguous matches by comparing the best and
second-best candidate distances.

**Formula**:

```
ratio = distance_1st / distance_2nd
accept if ratio < threshold
```

**Default threshold**: `0.75`

This is **strict less-than** (`<`), not less-than-or-equal.

**Rationale**: The ratio test was introduced by David Lowe (2004) to
reject matches where the nearest and second-nearest descriptors have
similar distances — indicating that the match is ambiguous and likely
incorrect.  A threshold of 0.75 is a standard choice that balances
recall (keeping true matches) against precision (rejecting false ones).

**Edge cases**:

| Condition                    | Behaviour                      |
|-----------------------------|-------------------------------|
| `d2 == 0` (both distances zero) | Rejected (ambiguous)       |
| Fewer than 2 neighbours      | Skipped (not counted as rejected) |
| Ratio exactly at threshold   | Rejected (strict `<`)       |

**Output** (`MatchResult`):

- `matches`: List of `FilteredMatch` with `(ref_pt, tgt_pt, distance, ratio)`
- `accepted` / `rejected` counts
- `ratio_stats`: min, max, mean, median of all computed ratios

---

## Geometric Verification and Transformation Estimation

> **Status**: CLASSICAL REGISTRATION BASELINE
>
> This stage establishes the first complete registration baseline by
> adding geometric verification to the existing SIFT → FLANN → Lowe
> pipeline.
>
> It does **not** claim: sub-pixel accuracy, cross-modal robustness,
> extreme-scale robustness, or illumination invariance.  Those require
> later stages.

### Purpose

Filter geometrically inconsistent correspondences and estimate a spatial
transformation from the filtered SIFT matches.  The output is the first
complete classical registration result.

### Pipeline Position

```
FeatureImage → SIFT → FLANN → Lowe ratio test
    → geometric verification (this stage)
    → transformation estimation
    → registration metrics (inlier RMSE, inlier ratio)
```

### Robust Estimator

**Preferred**: MAGSAC++ through OpenCV's USAC framework
(`cv2.USAC_MAGSAC`).  Available in OpenCV ≥ 4.5.

**Fallback**: `cv2.RANSAC`.  Used automatically if USAC_MAGSAC is not
available in the installed OpenCV version.

The estimator actually used at runtime is recorded in the result —
the code never silently claims MAGSAC++ if RANSAC was used instead.

**Runtime capability detection**: At estimation time, the code queries
OpenCV for available USAC flags and records the OpenCV version, whether
MAGSAC++ was available, and what method was actually used.

### Configurable Parameters

| Parameter           | Default | Description                                      |
|--------------------|---------|--------------------------------------------------|
| `reproj_threshold`  | 3.0     | Reprojection error threshold in pixels           |
| `confidence`        | 0.999   | Desired estimation confidence                    |
| `max_iters`         | 2000    | Maximum robust estimation iterations             |
| `model`             | affine  | Transformation model (`affine` or `homography`)  |

### Transformation Models

**Affine** (2×3 matrix, 6 DoF):
- Preserves parallelism
- Requires ≥ 3 non-collinear point correspondences
- Default for the classical baseline
- Appropriate for near-nadir views of approximately planar lunar terrain

**Homography** (3×3 matrix, 8 DoF):
- Preserves collinearity only
- Requires ≥ 4 non-collinear point correspondences
- Configurable option for cases with significant perspective distortion

**Geometric assumption**: For lunar orbital imagery, a single affine
transform may not perfectly represent all geometric distortions present
— terrain relief and oblique viewing angles can introduce perspective
effects not captured by affine.  This limitation is kept explicit.

### Minimum Correspondences

| Model      | Minimum | Behaviour if insufficient |
|-----------|---------|--------------------------|
| Affine     | 3       | Returns structured failure result |
| Homography | 4       | Returns structured failure result |

Zero, one, or insufficient correspondences are handled safely —
the function returns a failure result with a clear reason string,
never crashes.

### Metrics

#### Inlier Count and Ratio

- **inlier_count**: Number of correspondences consistent with the
  estimated transformation.
- **outlier_count**: Number rejected by the robust estimator.
- **total_correspondences**: inlier_count + outlier_count.
- **inlier_ratio**: `inlier_count / total_correspondences`.
  Returns 0.0 if there are zero correspondences.

#### Reprojection Error

For each correspondence (ref_pt, tgt_pt):
1. Transform ref_pt using the estimated transformation
2. Compute Euclidean pixel distance to tgt_pt

#### RMSE

For N correspondences with per-match errors e_i:

```
RMSE = sqrt( sum(e_i^2) / N )
```

Two RMSE values are reported:

| Metric         | Computed over          | Primary use                |
|---------------|----------------------|---------------------------|
| `inlier_rmse`  | Inlier correspondences | **Primary** accuracy metric |
| `all_rmse`     | All correspondences    | Completeness               |

**`inlier_rmse`** is the primary geometric accuracy metric for
registration quality.

#### Additional Error Statistics

- **inlier_median_error**: Median reprojection error over inliers
- **inlier_max_error**: Maximum reprojection error over inliers
- **all_median_error**: Median error over all correspondences
- **per_match_errors**: Full array of per-correspondence errors

### Output Structure (`GeometricResult`)

| Field                    | Type              | Description                           |
|-------------------------|-------------------|---------------------------------------|
| `success`                | bool              | Whether estimation succeeded          |
| `transform_matrix`       | ndarray or None   | (2,3) affine or (3,3) homography      |
| `transform_model`        | TransformModel    | Model that was used                   |
| `estimator_method`       | EstimatorMethod   | Method actually used at runtime       |
| `inlier_mask`            | bool array        | Per-correspondence inlier flag        |
| `inlier_count`           | int               | Geometrically consistent matches      |
| `outlier_count`          | int               | Rejected matches                      |
| `total_correspondences`  | int               | Total input matches                   |
| `inlier_ratio`           | float             | inlier_count / total                  |
| `error_metrics`          | ErrorMetrics      | Reprojection error statistics         |
| `failure_reason`         | str               | Explanation if failed                 |
| `capability`             | EstimatorCapability | Runtime OpenCV capability record    |

### Point Transformation Utility

A utility function `transform_points(points, matrix)` applies the
estimated transformation to point coordinates.  It supports both
affine (2×3) and homography (3×3) matrices.

This utility is reused by:
- Reprojection error calculation
- Sub-pixel refinement (future)
- Image warping (future)

It does **not** implement image warping — only point-level
transformation.

---

## Image Warping and Registration Output

> **Status**: CLASSICAL REGISTRATION BASELINE
>
> This stage completes the classical registration pipeline by warping
> the reference image into the target coordinate frame using the
> estimated transformation.
>
> This is **geometric registration output**, not sub-pixel refined
> registration.

### Purpose

Apply the estimated affine or homography transformation to the
reference image, producing a registered (aligned) output image.

### Pipeline Position

```
FeatureImage → SIFT → FLANN → Lowe ratio test
    → geometric verification (MAGSAC++)
    → transformation estimation
    → **image warping** (this stage)
    → registered image output
```

### Transform Direction

The geometry layer's `estimate_transform()` produces a matrix **M** via:

```
cv2.estimateAffine2D(ref_pts, tgt_pts)
cv2.findHomography(ref_pts, tgt_pts)
```

This matrix maps: **reference image coordinates → target image coordinates**.

For warping:
- **Source image**: the reference image
- **Matrix**: M (passed directly — **no inversion performed**)
- **Output dimensions**: target image dimensions (or explicitly set)
- **Result**: reference image warped into the target coordinate frame

OpenCV's `warpAffine` and `warpPerspective` by default expect the
forward mapping (src → dst) and internally compute the inverse for
pixel sampling.

### Warping Operations

| Transform Model | OpenCV Function       | Matrix Shape |
|----------------|----------------------|-------------|
| Affine          | `cv2.warpAffine`     | (2, 3)      |
| Homography      | `cv2.warpPerspective` | (3, 3)      |

### Interpolation and Border Handling

**Interpolation modes** (configurable, default: `linear`):

| Mode       | OpenCV Flag          | Description              |
|-----------|---------------------|--------------------------|
| `nearest`  | `INTER_NEAREST`     | Nearest-neighbor          |
| `linear`   | `INTER_LINEAR`      | Bilinear interpolation    |
| `cubic`    | `INTER_CUBIC`       | Bicubic interpolation     |
| `area`     | `INTER_AREA`        | Area-based resampling     |
| `lanczos4` | `INTER_LANCZOS4`    | Lanczos interpolation     |

**Border modes** (configurable, default: `constant` with value 0):

| Mode         | OpenCV Flag           |
|-------------|----------------------|
| `constant`   | `BORDER_CONSTANT`    |
| `replicate`  | `BORDER_REPLICATE`   |
| `reflect`    | `BORDER_REFLECT`     |
| `wrap`       | `BORDER_WRAP`        |
| `reflect101` | `BORDER_REFLECT_101` |

### Current Limitations

- This is geometric registration output based on the classical SIFT
  baseline.
- No sub-pixel refinement is applied.
- No cross-modal or illumination-invariant registration.
- No extreme-scale robustness.
- Image quality in the output depends on interpolation method and
  the accuracy of the estimated transformation.

---

## Evaluation Metrics and Spatial Distribution

> **Status**: CLASSICAL REGISTRATION BASELINE
>
> Provides structured metrics extraction, spatial distribution
> analysis, and match/registration visualization.

### Metrics Extraction

The evaluation metrics layer **reuses** the error metrics already
computed by `backend.geometry.estimation.compute_reprojection_errors()`.
No duplicate RMSE formulas exist.

The `MatchQualitySummary` collects:
- Correspondence counts (total, inlier, outlier)
- Inlier ratio
- Inlier RMSE (primary accuracy metric)
- All-match RMSE
- Inlier median and maximum error
- Spatial entropy
- Transform model and estimator method
- Transform matrix (JSON-serializable)

### Spatial Bucketing

Default 8×8 grid (64 cells) over the target image.

Cell assignment for each inlier at target coordinate (x, y):

```
col = min(int(x / width * 8), 7)
row = min(int(y / height * 8), 7)
```

- Uses **target-image** coordinates (the output reference frame)
- Boundary coordinates are clamped to the last cell

### Normalized Spatial Entropy

```
H = -Σ p_i log(p_i) / log(64)
```

where:
- `p_i = count_i / total_inliers` for each cell
- `0 × log(0) = 0` by convention
- Normalized by `log(64)` (total cells), NOT `log(occupied_cells)`

| Value | Interpretation |
|-------|---------------|
| H ≈ 0 | Concentrated in few cells |
| H ≈ 1 | Uniformly distributed |

### Limitations

- RMSE is an evaluation metric, not proof of sub-pixel accuracy.
- Spatial entropy measures distribution, not registration correctness.
- Synthetic test results do not establish lunar-data performance.
- Match redistribution/reselection is not yet implemented.

---

## Not Yet Implemented

The following algorithms are specified in PROJECT_SPEC.md but are
not implemented at this stage:

- Phase Congruency / Log-Gabor
- SuperPoint + LightGlue
- MIND cross-modal descriptor
- Hough Circle crater detection
- NCC + parabolic sub-pixel refinement
- PCA / spectral reduction for IIRS
- DEM relighting
- Confidence heatmap generation
- Spatial bucketing / match selection (redistribution)



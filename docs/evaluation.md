# Evaluation

> **Status**: CLASSICAL REGISTRATION BASELINE
>
> This document describes the evaluation metrics, spatial distribution
> analysis, and visualization utilities for the classical registration
> pipeline.

---

## Evaluation Pipeline

```
GeometricResult + RegistrationResult + FilteredMatches
    → metrics extraction (reuses geometry error metrics)
    → spatial distribution analysis (8×8 grid bucketing + entropy)
    → match visualization (filtered, inlier/outlier annotated)
    → registration overlay (alpha blend, side-by-side)
    → MatchQualitySummary (JSON-serializable)
```

---

## Metrics

The evaluation metrics layer **reuses** the error metrics already
computed by `backend.geometry.estimation`.  No duplicate RMSE
calculations exist.

### Exposed Metrics

| Metric                  | Source                        | Description                           |
|------------------------|-------------------------------|---------------------------------------|
| `total_correspondences` | `GeometricResult`             | Filtered matches (pre-geometry)       |
| `inlier_count`          | `GeometricResult`             | Geometrically consistent matches      |
| `outlier_count`         | `GeometricResult`             | Rejected matches                      |
| `inlier_ratio`          | `GeometricResult`             | `inlier_count / total`                |
| `inlier_rmse`           | `ErrorMetrics`                | RMSE over inliers (primary metric)    |
| `all_rmse`              | `ErrorMetrics`                | RMSE over all matches                 |
| `inlier_median_error`   | `ErrorMetrics`                | Median error among inliers            |
| `inlier_max_error`      | `ErrorMetrics`                | Maximum error among inliers           |
| `spatial_entropy`       | `SpatialDistribution`         | Normalized spatial entropy [0, 1]     |
| `transform_model`       | `GeometricResult`             | `'affine'` or `'homography'`          |
| `estimator_method`      | `GeometricResult`             | `'USAC_MAGSAC'` or `'RANSAC'`        |
| `transform_matrix`      | `GeometricResult`             | Nested list (JSON-serializable)       |

### `MatchQualitySummary`

A compact, serializable dataclass containing all metrics above.
All fields are plain Python types (no NumPy arrays) so the structure
can be directly converted to JSON.

Built via `build_quality_summary(geo_result, spatial_entropy=...)`.

### Interpretation

- **RMSE is an evaluation metric**, not proof of sub-pixel accuracy.
- **Inlier RMSE** is the primary accuracy metric — it reflects the
  geometric consistency of correspondences that passed robust estimation.
- **All-match RMSE** includes outliers and is provided for completeness.
- Metrics are computed on synthetic test data — **synthetic tests do not
  establish lunar-data performance**.

---

## Spatial Distribution

### Grid Bucketing

The target image is divided into a configurable grid (default **8×8 = 64 cells**).

Each **inlier** match is assigned to a cell using its **target-image
coordinate** (`tgt_pt`):

```
col = min(int(x / width * cols), cols - 1)
row = min(int(y / height * rows), rows - 1)
```

- Target coordinates are used because the target coordinate frame is the
  reference frame for the registered output.
- `min(..., N-1)` clamps boundary coordinates (e.g. `x == width`).
- Negative coordinates are clamped to 0.

### Normalized Spatial Entropy

Shannon entropy normalized by `log(K)` where K = total grid cells:

```
H = -Σ p_i log(p_i) / log(K)
```

where:
- `p_i = count_i / total_inliers` for each cell
- K = 64 (for the default 8×8 grid)
- `0 * log(0) = 0` by convention (cells with zero inliers contribute 0)

Normalization uses `log(64)` (total cells), **NOT** `log(occupied cells)`.

| Value | Interpretation |
|-------|---------------|
| H ≈ 0 | Inliers concentrated in very few cells |
| H ≈ 1 | Inliers uniformly distributed across the grid |

### Limitations

- **Spatial entropy measures distribution, not registration correctness.**
  High entropy does not guarantee accurate registration.
- Match redistribution/reselection is not implemented — that belongs
  to a later spatial bucketing stage.
- Entropy alone does not guarantee uniformity — it is one signal among
  many for evaluating match quality.

---

## Visualization

### Match Visualization

`draw_filtered_matches(ref_img, tgt_img, matches)`:
- Side-by-side canvas with reference (left) and target (right)
- Correspondence lines connecting matched points
- Configurable max matches for readability

### Inlier/Outlier Visualization

`draw_inlier_outlier_matches(ref_img, tgt_img, matches, inlier_mask)`:
- Inliers drawn in **green**, outliers in **red**
- Outliers drawn first (inliers on top for visibility)
- Both point markers and correspondence lines

### Registration Overlay

`draw_registration_overlay(target_img, registered_img, alpha=0.5)`:
- Alpha-blended overlay of target and registered images
- Useful for visual inspection of geometric alignment
- Handles mismatched dimensions (uses common area)

`draw_side_by_side(img_a, img_b, label_a="", label_b="")`:
- Horizontal concatenation with optional text labels
- Vertically centers images of different heights

### Output Format

- All visualization functions return **BGR uint8 NumPy arrays**
- No files are saved automatically
- Grayscale inputs are converted to BGR for drawing
- Float images are clipped to [0, 255] and cast to uint8
- Empty matches, None images, and invalid inputs are handled cleanly

### Interpretation

**Visualization is diagnostic, not a substitute for quantitative
evaluation.**  Visual overlap in an overlay does not prove sub-pixel
accuracy — always consult RMSE and spatial metrics for quantitative
assessment.

---

## Current Limitations

- This is evaluation of the **classical SIFT baseline** only.
- No cross-modal evaluation (IIRS vs. OHRC, DEM vs. optical).
- No uncertainty estimation or confidence heatmaps.
- No change detection or temporal analysis.
- Synthetic test results do not establish performance on real
  Chandrayaan-2/3 data.

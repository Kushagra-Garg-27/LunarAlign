# SIH26166 --- AI-Agent Development Master Specification

## 1. Project

**Problem Statement:** SIH26166 --- Multi-Modal, Sun Angle & Scale
Invariant Image Correspondence Using Chandrayaan-2 Optical Images (OHRC,
TMC-2, IIRS).

Build a software system that accepts two lunar images, finds reliable
correspondences despite illumination, viewpoint, scale and modality
differences, estimates the geometric transformation, and produces a
registered image with measurable quality.

Core challenges: - Illumination / sun-angle variation - Extreme scale
variation - Cross-modal mismatch, especially IIRS hyperspectral vs
OHRC/TMC-2 panchromatic imagery - Uniform spatial distribution of
matches - Sub-pixel registration target

## 2. Supported Pair Types

  Pair              Approx. Scale Main Challenge
  --------------- --------------- -----------------------------
  OHRC ↔ TMC-2                20× Scale
  OHRC ↔ IIRS                320× Extreme scale + cross-modal
  TMC-2 ↔ IIRS                16× Cross-modal
  OHRC ↔ OHRC                  1× Sun-angle / illumination
  TMC-2 ↔ TMC-2                1× Sun-angle / illumination
  IIRS ↔ IIRS                  1× Spectral + illumination

For OHRC ↔ IIRS, use a coarse-to-fine TMC-mediated chain where
appropriate: **OHRC ↔ TMC-2 → TMC-2 ↔ IIRS**, then compose
transformations. Do not attempt to solve the entire 320× gap with one
resize.

## 3. Expected Outputs

-   Registered/aligned image
-   Match-point and inlier/outlier visualization
-   Confidence heatmap
-   Metrics dashboard
-   Optional uncertainty estimate
-   Optional change detection
-   Exportable matches and transformation

Required metrics: - RMSE in pixels - Inlier count - Inlier ratio -
Reprojection-error distribution - Spatial distribution entropy /
uniformity

**Target:** RMSE \< 1.0 pixel. This is a target, never a fabricated
claim.

## 4. High-Level Workflow

``` text
Browser / User
      ↓
React Dashboard
      ↓
FastAPI API
      ↓
Input Validation + Metadata / Pair Router
      ↓
Preprocessing
      ↓
 ┌───────────────┬────────────────┐
 ↓               ↓                ↓
Classical      Learned       Cross-Modal
Phase Cong.    SuperPoint    MIND
SIFT→FLANN     →LightGlue
 ↓               ↓                ↓
 └───────────────┴────────────────┘
                ↓
           Match Fusion
                ↓
       Spatial Distribution
                ↓
        Crater Anchoring
       (when applicable)
                ↓
             MAGSAC++
                ↓
       Transform Estimation
       Affine / Projective
                ↓
       Sub-pixel Refinement
          NCC + Parabolic
                ↓
            Image Warp
                ↓
     ┌──────────┼──────────┐
     ↓          ↓          ↓
  Metrics    Confidence  Uncertainty
             Heatmap
                ↓
      Optional Change Detection
```

The classical baseline must always remain available. The learned branch
is an enhancement/parallel branch, not a replacement.

## 5. Architecture Layers

### Frontend

React, HTML5, CSS3, JavaScript, Canvas/WebGL. Features: upload,
metadata, match visualization, before/after overlay, opacity control,
confidence heatmap, metrics, export.

### Backend

Python 3.10+, FastAPI, validation, orchestration, routing, pipeline
execution, serialization, logging and error handling.

### Computer Vision

OpenCV SIFT, FLANN, Lowe ratio test, Phase Congruency/Log-Gabor,
SuperPoint, LightGlue, MIND, MAGSAC++, NCC/parabolic refinement,
affine/projective transforms, Hough Circles.

### Scientific/Data Processing

NumPy, SciPy, scikit-image, rasterio, GDAL, spectral Python, Matplotlib.

### Integration

ISRO PRADAN, Chandrayaan Data Explorer/MapBrowse, NASA LRO NAC, JAXA
SELENE/Kaguya, Docker.

## 6. Core Concepts

### Preprocessing

Use normalization such as CLAHE where appropriate. For IIRS, reduce
hyperspectral data to a spatial representation suitable for
correspondence using PCA or another documented reduction method. Phase
congruency provides illumination/contrast-robust structural information.

### Classical Branch

**Phase Congruency → SIFT → FLANN → Lowe Ratio Test**

SIFT remains the fallback baseline.

### Learned Branch

**SuperPoint → LightGlue**

Use pretrained weights only. No training/fine-tuning is planned. If
unavailable or unsuitable, fall back gracefully to classical matching.

### Cross-Modal Branch

For OHRC/TMC-2 ↔ IIRS, use structural/modality-independent
representations. MIND is the primary proposed descriptor. Mutual
information may be experimental.

### Uniform Distribution

Use spatial bucketing (proposed 8×8 grid) so matches do not cluster in
one distinctive region. Measure spatial entropy.

### Crater Anchoring

Hough Circle detection can provide coarse domain-specific
initialization. It is optional and must not block registration if
unreliable.

### Robust Estimation

Use MAGSAC++ / OpenCV USAC-compatible implementation where available.
Reject geometrically inconsistent matches and estimate an appropriate
affine/projective model.

### Sub-Pixel Refinement

After robust matching: **NCC → correlation peak → parabolic
interpolation → refined correspondence → optional transform
re-estimation.**

### Confidence

Build a spatial confidence visualization from match locations, error and
inlier information. Do not infer confidence merely from visual
smoothness.

### Uncertainty

Optional bootstrap/repeated robust estimation can estimate
transformation spread and uncertainty intervals.

### Change Detection

Optional downstream module for suitably comparable registered images:
**difference → adaptive threshold → morphology → candidate change map.**
Do not interpret every intensity difference as physical change.

## 7. Deterministic Router

Use metadata/rules, not a trained classifier:

``` text
if cross_modal:
    enable MIND / structural representation

if scale_ratio is large:
    enable coarse_to_fine

if illumination difference is significant:
    prioritize illumination-robust representation

if DEM + sun metadata are available:
    enable DEM relighting

classical branch:
    always available

learned branch:
    run when supported
```

Log the route and all fallbacks.

## 8. Repository Structure

``` text
sih26166/
├── README.md
├── PROJECT_SPEC.md
├── requirements.txt
├── .env.example
├── Dockerfile
├── docker-compose.yml
├── backend/
│   ├── main.py
│   ├── api/
│   ├── core/
│   ├── router/
│   ├── preprocessing/
│   ├── features/
│   ├── matching/
│   ├── geometry/
│   ├── registration/
│   ├── evaluation/
│   └── optional/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── services/
│   │   └── utils/
│   └── package.json
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── data/
│   ├── sample/
│   └── README.md
├── outputs/
└── docs/
    ├── architecture.md
    ├── algorithms.md
    └── evaluation.md
```

Agents may improve organization, but must explain structural changes and
preserve the architecture.

## 9. Development Order

1.  Environment + repository skeleton
2.  FastAPI + React shell + upload
3.  Classical baseline end-to-end
4.  Metrics + visualization
5.  IIRS reduction + MIND
6.  Extreme-scale coarse-to-fine
7.  SuperPoint + LightGlue
8.  Sub-pixel refinement
9.  DEM relighting
10. Crater anchoring
11. Confidence + uncertainty
12. Dashboard polish
13. Six-pair validation
14. Docker/README/demo hardening

**Principle:** a reliable, measurable baseline is more valuable than
many partially working novelties.

## 10. Evaluation Result Contract

Use a structured result such as:

``` json
{
  "success": true,
  "transform_type": "affine",
  "rmse_px": 0.0,
  "inlier_count": 0,
  "total_matches": 0,
  "inlier_ratio": 0.0,
  "spatial_entropy": 0.0,
  "reprojection_error": {},
  "confidence_summary": {},
  "pipeline_route": [],
  "fallbacks": []
}
```

These are placeholders only. Never fabricate values.

## 11. AI-Agent Engineering Rules

1.  Read `PROJECT_SPEC.md` before coding.
2.  Inspect the current repository before changing files.
3.  Work incrementally.
4.  Never rewrite unrelated working code.
5.  Do not invent APIs, packages, model weights or dataset paths.
6.  Verify dependencies and compatibility.
7.  Prefer established libraries.
8.  Keep modules independently testable.
9.  Add tests for non-trivial mathematical code.
10. Handle missing metadata gracefully.
11. Never fabricate scientific metrics.
12. Never claim RMSE \< 1 px without measurement.
13. Keep CPU compatibility as the baseline.
14. No GPU training or fine-tuning.
15. Log routing and fallbacks.
16. If blocked, report the blocker and smallest viable fallback.
17. Do not silently replace agreed algorithms with unrelated approaches.

## 12. Agent Roles

### Claude Opus 4.6

Lead architecture, complex debugging, integration planning, difficult
algorithms and final code review.

### Claude Sonnet 4.6

Primary implementation, refactoring, module development and tests.

### Gemini 3.8 Flash

Fast implementation, UI, documentation, debugging and test generation.

### Gemini 3.7 Flash

Fast implementation, UI, documentation, debugging and test generation.

### Gemini 3.6 Flash

Small fixes, code inspection, boilerplate, tests and UI polish.

### Gemini 3.1 Pro Low

Independent architecture/algorithm review and consistency checking.

### GPT-OSS 120B

Independent code review, edge cases, alternative implementations and
validation.

No model is automatically authoritative. Test agent-generated code.

## 13. Standard Implementation Prompt

``` text
You are working on SIH26166.

FIRST:
1. Read PROJECT_SPEC.md completely.
2. Inspect the current repository structure.
3. Inspect all relevant existing files before modifying anything.
4. Identify current dependencies and implementation state.

TASK:
[PASTE ONE CLEAR TASK HERE]

CONSTRAINTS:
- Follow PROJECT_SPEC.md.
- Do not rewrite unrelated modules.
- Preserve working functionality.
- CPU-compatible baseline.
- No training/fine-tuning.
- Do not fabricate scientific results.
- Add/update tests.
- Run relevant tests before finishing.

IMPLEMENT the task in the existing repository.

FINAL REPORT:
1. Files created/modified
2. What was implemented
3. Dependencies added/changed
4. Commands/tests executed
5. Results
6. Known limitations/blockers
7. Recommended next step

Do not merely describe code. Implement and test it.
```

## 14. Standard Code Review Prompt

``` text
You are the independent code-review agent for SIH26166.

Read PROJECT_SPEC.md first.

Inspect the current repository and review:
[MODULE / FEATURE]

Check:
1. Scientific/algorithmic correctness
2. Coordinate and transformation conventions
3. Image dimensions/resolution handling
4. Failure/fallback handling
5. Metric correctness
6. Dependency compatibility
7. CPU compatibility
8. Hidden assumptions or fabricated values
9. Test coverage
10. Cross-pair regressions

Do not rewrite code yet.

Return:
- CRITICAL issues
- MAJOR issues
- MINOR issues
- What is correct
- Exact fixes recommended
- Tests to add
```

## 15. Standard Debugging Prompt

``` text
You are debugging SIH26166.

Read PROJECT_SPEC.md first.

PROBLEM:
[PASTE ERROR / LOG / BEHAVIOR]

Inspect the relevant files and reproduce the issue if possible.

Determine whether the root cause is:
- algorithmic
- dependency-related
- data-related
- API/backend
- frontend/UI
- integration

Implement the smallest safe fix.
Do not rewrite unrelated code.
Run targeted tests.

Return:
1. Root cause
2. Files changed
3. Fix
4. Tests run
5. Results
6. Remaining risks
```

## 16. Architecture Decision Prompt

``` text
Act as the lead architect for SIH26166.

Read PROJECT_SPEC.md and inspect the current repository.

DECISION:
[DESCRIBE THE DECISION]

Compare options by:
- scientific correctness
- SIH requirements
- CPU feasibility
- complexity
- robustness
- maintainability
- explainability
- compatibility with existing pipeline

Do not code yet.

Return:
1. Recommendation
2. Reasoning
3. Trade-offs
4. Impact on existing modules
5. Exact implementation plan
6. Tests required

Do not introduce a new approach merely because it is fashionable.
```

## 17. Integration Prompt

``` text
Act as the senior integration engineer for SIH26166.

Read PROJECT_SPEC.md completely.
Inspect the entire repository.

Goal:
Make the current implementation run end-to-end from browser upload to registered output and metrics.

Verify:
- React → FastAPI communication
- file handling
- metadata
- pair routing
- classical branch
- learned branch fallback
- cross-modal branch
- geometry
- sub-pixel refinement
- warping
- metrics
- confidence heatmap
- serialization
- error handling

Do not add major new algorithms.
Fix integration issues only.

Run unit, integration and minimal end-to-end tests.

Return:
- working components
- failing components
- fixes
- tests/results
- remaining blockers
```

## 18. Data Sources

Primary: - Chandrayaan-2 OHRC, TMC-2, IIRS - ISRO PRADAN - Chandrayaan
Data Explorer / MapBrowse

Reference: - NASA LRO NAC - JAXA SELENE/Kaguya

Crater validation: - Robbins lunar crater database

Do not hard-code remote datasets unless explicitly required. Prefer
documented ingestion adapters or user-provided files.

## 19. Definition of Done

-   Repository installs reproducibly
-   FastAPI starts
-   React starts
-   Two supported images can be uploaded
-   Metadata/pair type is handled
-   Classical baseline works
-   At least one real Chandrayaan-2 pair registers successfully
-   Matches and inliers are visualized
-   Registered output is produced
-   RMSE/inlier metrics are displayed
-   Spatial distribution is measured
-   Cross-modal path exists
-   Extreme-scale routing exists
-   Learned branch has fallback
-   Confidence heatmap works
-   Six pair types are represented in validation
-   No fabricated metrics
-   Errors are understandable
-   README/setup/demo instructions exist
-   Docker is tested if included

## 20. Operating Principle

The human operator coordinates agents rather than manually coding.

``` text
Human gives focused task
        ↓
Agent reads PROJECT_SPEC.md
        ↓
Agent inspects repository
        ↓
Agent implements + tests
        ↓
Agent reports files / tests / blockers
        ↓
Human gives report + next task to coordinator/reviewer
        ↓
Next focused task
```

Never blindly paste generated code between agents. Agents should work in
the shared repository.

**Single source of truth:** `PROJECT_SPEC.md`.

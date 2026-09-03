import './TransformPanel.css';

/**
 * TransformPanel component for setting geometric transform model and Lowe's ratio test threshold.
 *
 * @param {object} props
 * @param {'affine' | 'homography'} props.transformModel
 * @param {(model: string) => void} props.onTransformModelChange
 * @param {number} props.ratioThreshold
 * @param {(threshold: number) => void} props.onRatioThresholdChange
 * @param {boolean} [props.disabled=false]
 */
export default function TransformPanel({
  transformModel = 'affine',
  onTransformModelChange,
  ratioThreshold = 0.75,
  onRatioThresholdChange,
  disabled = false,
}) {
  const MIN_RATIO = 0.5;
  const MAX_RATIO = 0.95;

  const handleRatioSliderChange = (e) => {
    const val = parseFloat(e.target.value);
    if (!isNaN(val)) {
      const clamped = Math.min(MAX_RATIO, Math.max(MIN_RATIO, val));
      onRatioThresholdChange(clamped);
    }
  };

  const handleRatioNumberChange = (e) => {
    const val = parseFloat(e.target.value);
    if (isNaN(val)) return;
    // Clamp in UI to suggested range [0.5, 0.95]
    const clamped = Math.min(MAX_RATIO, Math.max(MIN_RATIO, val));
    onRatioThresholdChange(clamped);
  };

  return (
    <div className={`transform-panel ${disabled ? 'disabled' : ''}`}>
      <div className="transform-panel-header">
        <h3 className="transform-panel-title">Transformation & Matching Parameters</h3>
        <span className="transform-panel-badge">Classical Settings</span>
      </div>

      <div className="transform-grid">
        {/* Transform Model Selection */}
        <div className="param-group">
          <label htmlFor="transform-model-select" className="param-label">
            Geometric Model
            <span className="param-tooltip-hint" title="Affine (6-DoF) vs Homography (8-DoF planar projective)">
              (Degrees of Freedom)
            </span>
          </label>
          <div className="select-wrapper">
            <select
              id="transform-model-select"
              className="param-select"
              value={transformModel}
              onChange={(e) => onTransformModelChange(e.target.value)}
              disabled={disabled}
            >
              <option value="affine">Affine (2×3 matrix — rotation, scale, shear, translation)</option>
              <option value="homography">Homography (3×3 matrix — planar projective)</option>
            </select>
          </div>
          <p className="param-desc">
            {transformModel === 'affine'
              ? 'Recommended for narrow-angle or high-altitude orbital imagery with moderate tilt.'
              : 'Full perspective transformation. Recommended for oblique angles or wide baselines.'}
          </p>
        </div>

        {/* Lowe's Ratio Threshold Slider & Number */}
        <div className="param-group">
          <div className="param-label-row">
            <label htmlFor="ratio-threshold-input" className="param-label">
              Lowe Ratio Threshold
            </label>
            <span className="param-current-value">{ratioThreshold.toFixed(2)}</span>
          </div>

          <div className="slider-control-row">
            <input
              id="ratio-threshold-slider"
              type="range"
              min={MIN_RATIO}
              max={MAX_RATIO}
              step="0.01"
              value={ratioThreshold}
              onChange={handleRatioSliderChange}
              disabled={disabled}
              className="param-slider"
              aria-label="Lowe ratio threshold slider"
            />
            <input
              id="ratio-threshold-input"
              type="number"
              min={MIN_RATIO}
              max={MAX_RATIO}
              step="0.01"
              value={ratioThreshold}
              onChange={handleRatioNumberChange}
              disabled={disabled}
              className="param-num-input"
              aria-label="Lowe ratio threshold value"
            />
          </div>

          <div className="slider-ticks">
            <span>0.50 (Strict)</span>
            <span>0.75 (Default)</span>
            <span>0.95 (Permissive)</span>
          </div>

          <p className="param-desc">
            Distance ratio test threshold for FLANN 2-NN descriptor matching. Lower values discard ambiguous matches.
          </p>
        </div>
      </div>
    </div>
  );
}

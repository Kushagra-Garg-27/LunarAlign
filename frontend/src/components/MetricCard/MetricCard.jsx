import './MetricCard.css';

/**
 * MetricCard component for displaying key quantitative and qualitative registration results.
 *
 * @param {object} props
 * @param {string} props.label - Metric label (e.g. "Inlier Ratio").
 * @param {string | number} props.value - Metric value.
 * @param {string} [props.unit] - Optional unit string (e.g. "px", "%", "s").
 * @param {string} [props.subtext] - Context or interpretation hint.
 * @param {string} [props.badge] - Optional badge/status tag (e.g. "Optimal", "MAGSAC++").
 * @param {'default' | 'accent' | 'success' | 'highlight' | 'secondary'} [props.variant='default']
 */
export default function MetricCard({
  label,
  value,
  unit = '',
  subtext = '',
  badge = '',
  variant = 'default',
}) {
  const displayValue = value !== undefined && value !== null ? value : '—';

  return (
    <div className={`metric-card metric-${variant}`}>
      <div className="metric-header">
        <span className="metric-label">{label}</span>
        {badge && <span className="metric-badge">{badge}</span>}
      </div>

      <div className="metric-value-row">
        <span className="metric-value">{displayValue}</span>
        {unit && <span className="metric-unit">{unit}</span>}
      </div>

      {subtext && <p className="metric-subtext">{subtext}</p>}
    </div>
  );
}

import { useState } from 'react';
import ImageUploader from '../../components/ImageUploader/ImageUploader';
import TransformPanel from '../../components/TransformPanel/TransformPanel';
import StatusBanner from '../../components/StatusBanner/StatusBanner';
import MetricCard from '../../components/MetricCard/MetricCard';
import ResultViewer from '../../components/ResultViewer/ResultViewer';
import { registerImages, RegistrationApiError } from '../../services/registrationApi';
import './RegistrationDashboard.css';

/**
 * Format documented HTTP status codes into clear, human-readable user guidance.
 *
 * @param {Error} err
 * @returns {{ type: string, status: number | null, message: string }}
 */
function formatRegistrationError(err) {
  if (!(err instanceof RegistrationApiError)) {
    return {
      type: 'unknown',
      status: null,
      message: err?.message || 'An unexpected error occurred during registration.',
    };
  }

  const status = err.status;
  let humanMessage = err.message;

  switch (status) {
    case 400:
      humanMessage =
        err.message && !err.message.startsWith('Request failed')
          ? `Bad Request: ${err.message}`
          : 'Bad Request (400): The uploaded image files could not be processed or are empty. Please check the file contents.';
      break;
    case 413:
      humanMessage =
        'File Size Exceeded (413): Uploaded file exceeds the maximum allowed size of 50 MB. Please crop or compress the image.';
      break;
    case 415:
      humanMessage =
        err.message && err.message.toLowerCase().includes('unsupported')
          ? `Unsupported Media Type (415): ${err.message}`
          : 'Unsupported Media Type (415): The uploaded file format is not supported. Please select valid PNG, JPG, or TIFF lunar images.';
      break;
    case 422:
      humanMessage =
        err.message && !err.message.startsWith('Request failed')
          ? `Parameter Validation Error (422): ${err.message}`
          : 'Validation Error (422): Invalid registration parameters. Transform model must be affine or homography and ratio threshold must be in (0, 1].';
      break;
    case 404:
      humanMessage =
        'Resource Not Found (404): The requested registration result or artifact could not be found or has expired from cache.';
      break;
    case 500:
      humanMessage =
        err.message && !err.message.startsWith('Request failed')
          ? `Server Pipeline Failure (500): ${err.message}`
          : 'Internal Server Error (500): The registration pipeline failed during feature extraction or geometric estimation.';
      break;
    default:
      if (status >= 500) {
        humanMessage = `Server Error (${status}): The registration service is currently unavailable. Please verify the backend logs.`;
      }
      break;
  }

  return {
    type: err.type,
    status: err.status,
    message: humanMessage,
  };
}

/**
 * RegistrationDashboard — Classical SIFT Baseline (Stage A & Stage B).
 *
 * Implements the core dashboard for image uploads, classical parameter
 * tuning, execution trigger, quantitative metric presentation, and
 * Stage B diagnostic artifact visualization.
 */
export default function RegistrationDashboard() {
  // Input states
  const [referenceFile, setReferenceFile] = useState(null);
  const [targetFile, setTargetFile] = useState(null);
  const [transformModel, setTransformModel] = useState('affine');
  const [ratioThreshold, setRatioThreshold] = useState(0.75);

  // Execution states: 'idle' | 'processing' | 'success' | 'error'
  const [status, setStatus] = useState('idle');
  const [statusMessage, setStatusMessage] = useState('');
  const [errorType, setErrorType] = useState('unknown');
  const [statusCode, setStatusCode] = useState(null);
  const [failureStage, setFailureStage] = useState('');

  // Results state from POST /register
  const [registrationResult, setRegistrationResult] = useState(null);

  const isFormValid = Boolean(referenceFile && targetFile);
  const isBusy = status === 'processing';

  const handleRegister = async () => {
    if (!isFormValid || isBusy) return;

    setStatus('processing');
    setStatusMessage('Uploading images and executing classical SIFT + MAGSAC++ registration…');
    setErrorType('unknown');
    setStatusCode(null);
    setFailureStage('');
    setRegistrationResult(null);

    try {
      const response = await registerImages(referenceFile, targetFile, {
        transformModel,
        ratioThreshold,
      });

      if (!response.success) {
        // Backend returned 200 OK but pipeline reported registration failure (e.g. insufficient inliers)
        setStatus('error');
        setErrorType('registration_failed');
        setFailureStage(response.failure_stage || 'geometry');
        setStatusMessage(
          response.failure_reason ||
            `Registration failed during stage '${response.failure_stage || 'unknown'}'. Try adjusting the Lowe ratio threshold or using a different transform model.`
        );
        setRegistrationResult(null);
        return;
      }

      // Success
      setRegistrationResult(response);
      setStatus('success');
      setStatusMessage(
        `Registration succeeded in ${(response.timing?.total || 0).toFixed(3)}s. Model: ${
          response.transform?.transform_model?.toUpperCase() || transformModel.toUpperCase()
        } via ${response.transform?.estimator_method || 'MAGSAC++'}.`
      );
    } catch (err) {
      setStatus('error');
      setRegistrationResult(null);
      const parsed = formatRegistrationError(err);
      setErrorType(parsed.type);
      setStatusCode(parsed.status);
      setStatusMessage(parsed.message);
    }
  };

  const handleReset = () => {
    setReferenceFile(null);
    setTargetFile(null);
    setTransformModel('affine');
    setRatioThreshold(0.75);
    setStatus('idle');
    setStatusMessage('');
    setErrorType('unknown');
    setStatusCode(null);
    setFailureStage('');
    setRegistrationResult(null);
  };

  const handleLoadSample = async () => {
    try {
      setStatus('idle');
      setStatusMessage('Loading sample lunar image pair…');
      const [refRes, tgtRes] = await Promise.all([
        fetch('/ref_lunar.png'),
        fetch('/tgt_lunar.png'),
      ]);
      if (!refRes.ok || !tgtRes.ok) {
        throw new Error('Sample images could not be loaded from public storage.');
      }
      const [refBlob, tgtBlob] = await Promise.all([refRes.blob(), tgtRes.blob()]);
      const refFile = new File([refBlob], 'ref_lunar.png', { type: 'image/png' });
      const tgtFile = new File([tgtBlob], 'tgt_lunar.png', { type: 'image/png' });
      setReferenceFile(refFile);
      setTargetFile(tgtFile);
      setStatus('idle');
      setStatusMessage('Sample lunar imagery pair loaded. Ready for registration.');
    } catch (err) {
      setStatus('error');
      setStatusMessage(`Unable to load sample pair: ${err.message}`);
    }
  };

  const renderMatrix = (matrix) => {
    if (!matrix || !Array.isArray(matrix) || matrix.length === 0) {
      return <p className="matrix-empty">No transformation matrix available</p>;
    }

    const rows = matrix.length;
    const cols = matrix[0] ? matrix[0].length : 0;

    return (
      <div className="matrix-wrapper">
        <div className="matrix-header">
          <span className="matrix-type">
            {rows}×{cols} Estimated Transformation Matrix
          </span>
          <span className="matrix-sub">Floating-point coefficients</span>
        </div>
        <div className="matrix-bracket-box">
          <div className="matrix-bracket bracket-left" />
          <table className="matrix-table" aria-label="Transformation Matrix Grid">
            <tbody>
              {matrix.map((row, rIdx) => (
                <tr key={rIdx}>
                  {row.map((cell, cIdx) => (
                    <td key={cIdx} className="matrix-cell">
                      {typeof cell === 'number'
                        ? cell >= 0
                          ? ` ${cell.toFixed(5)}`
                          : cell.toFixed(5)
                        : cell}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          <div className="matrix-bracket bracket-right" />
        </div>
      </div>
    );
  };

  return (
    <div className="registration-dashboard">
      {/* Fixed Non-Removable Baseline Badge */}
      <div className="baseline-banner" role="complementary" aria-label="Baseline System Mode">
        <div className="baseline-badge-container">
          <span className="baseline-pill-fixed">CLASSICAL SIFT BASELINE</span>
          <span className="baseline-notice">
            Same-modality reference mode only · Deep / multi-modal pipelines disabled in Stage A
          </span>
        </div>
      </div>

      <header className="dashboard-intro">
        <div className="intro-badge">ISRO / Chandrayaan-2 Lunar Surface Matching</div>
        <h2 className="dashboard-title">Classical Registration Dashboard</h2>
        <p className="dashboard-subtitle">
          Align pair-wise lunar orbital imagery using SIFT feature extraction, FLANN nearest-neighbor ratio matching,
          and MAGSAC++ robust geometric estimator.
        </p>
      </header>

      {/* Main Status Banner */}
      <StatusBanner
        status={status}
        message={statusMessage}
        errorType={errorType}
        statusCode={statusCode}
        failureStage={failureStage}
        onDismiss={status === 'error' ? () => setStatus('idle') : undefined}
      />

      {/* Inputs Section */}
      <section className="dashboard-section uploader-section">
        <h3 className="section-title">Input Lunar Imagery</h3>
        <div className="uploaders-grid">
          <ImageUploader
            id="reference-uploader"
            label="Reference Image"
            subtitle="Base mosaic or reference frame"
            selectedFile={referenceFile}
            onFileSelect={setReferenceFile}
            disabled={isBusy}
          />
          <ImageUploader
            id="target-uploader"
            label="Target Image"
            subtitle="Unaligned or floating frame"
            selectedFile={targetFile}
            onFileSelect={setTargetFile}
            disabled={isBusy}
          />
        </div>
      </section>

      {/* Parameters Section */}
      <section className="dashboard-section">
        <TransformPanel
          transformModel={transformModel}
          onTransformModelChange={setTransformModel}
          ratioThreshold={ratioThreshold}
          onRatioThresholdChange={setRatioThreshold}
          disabled={isBusy}
        />
      </section>

      {/* Primary Action Button Bar */}
      <div className="action-bar">
        <button
          id="btn-register-images"
          type="button"
          className="btn-register"
          onClick={handleRegister}
          disabled={!isFormValid || isBusy}
        >
          {isBusy ? (
            <span className="btn-loading-content">
              <span className="btn-spinner" />
              REGISTERING IMAGES…
            </span>
          ) : (
            <span className="btn-label-content">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                <polygon points="5 3 19 12 5 21 5 3" />
              </svg>
              REGISTER IMAGES
            </span>
          )}
        </button>

        <button
          id="btn-load-sample"
          type="button"
          className="btn-sample"
          onClick={handleLoadSample}
          disabled={isBusy}
          title="Load synthetic test sample (384x384 px) for testing"
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <rect width="18" height="18" x="3" y="3" rx="2" ry="2" />
            <circle cx="9" cy="9" r="2" />
            <path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21" />
          </svg>
          Load Synthetic Test Sample
        </button>

        {(referenceFile || targetFile || registrationResult || status !== 'idle') && (
          <button
            type="button"
            className="btn-reset"
            onClick={handleReset}
            disabled={isBusy}
            title="Clear images and reset parameters"
          >
            Reset Form
          </button>
        )}
      </div>

      {/* Metrics & Results View on Successful Registration */}
      {registrationResult && registrationResult.success && (
        <section className="dashboard-section results-section" aria-label="Registration Results">
          <div className="results-header">
            <div>
              <h3 className="section-title">Registration Performance Metrics</h3>
              <p className="results-sub">
                Result ID: <code className="result-id-code">{registrationResult.result_id}</code>
              </p>
            </div>
            <span className="result-status-pill">EVALUATION PASSED</span>
          </div>

          {/* Metric Cards Grid */}
          <div className="metrics-grid">
            {/* Correspondence Metrics */}
            <MetricCard
              label="Inlier Count"
              value={registrationResult.correspondence?.inlier_count}
              unit="pts"
              subtext="MAGSAC++ geometrically consistent matches"
              badge="Inliers"
              variant="success"
            />
            <MetricCard
              label="Outlier Count"
              value={registrationResult.correspondence?.outlier_count}
              unit="pts"
              subtext="Rejected candidate correspondences"
              badge="Rejected"
              variant="default"
            />
            <MetricCard
              label="Inlier Ratio"
              value={
                registrationResult.correspondence?.inlier_ratio !== undefined
                  ? `${(registrationResult.correspondence.inlier_ratio * 100).toFixed(1)}%`
                  : '—'
              }
              subtext={`From ${registrationResult.correspondence?.total_correspondences || 0} total putative pairs`}
              badge="Ratio"
              variant="accent"
            />

            {/* Accuracy Metrics */}
            <MetricCard
              label="Inlier RMSE"
              value={
                registrationResult.accuracy?.inlier_rmse !== undefined
                  ? registrationResult.accuracy.inlier_rmse.toFixed(3)
                  : '—'
              }
              unit="px"
              subtext="Root mean square reprojection error of inliers"
              badge="Quality"
              variant="success"
            />
            <MetricCard
              label="All RMSE"
              value={
                registrationResult.accuracy?.all_rmse !== undefined
                  ? registrationResult.accuracy.all_rmse.toFixed(3)
                  : '—'
              }
              unit="px"
              subtext="Reprojection error over all putative matches"
              badge="Global"
              variant="default"
            />
            <MetricCard
              label="Inlier Median Error"
              value={
                registrationResult.accuracy?.inlier_median_error !== undefined
                  ? registrationResult.accuracy.inlier_median_error.toFixed(3)
                  : '—'
              }
              unit="px"
              subtext="Median error of verified correspondence points"
              badge="Robust"
              variant="secondary"
            />
            <MetricCard
              label="Inlier Max Error"
              value={
                registrationResult.accuracy?.inlier_max_error !== undefined
                  ? registrationResult.accuracy.inlier_max_error.toFixed(3)
                  : '—'
              }
              unit="px"
              subtext="Maximum residual inlier distance"
              badge="Peak"
              variant="default"
            />

            {/* Spatial & Geometric Metrics */}
            <MetricCard
              label="Spatial Entropy"
              value={
                registrationResult.spatial?.spatial_entropy !== undefined &&
                registrationResult.spatial.spatial_entropy >= 0
                  ? registrationResult.spatial.spatial_entropy.toFixed(3)
                  : 'N/A'
              }
              subtext="Normalized 2D spatial distribution entropy [0, 1]"
              badge="Distribution"
              variant="highlight"
            />
            <MetricCard
              label="Transform Model"
              value={registrationResult.transform?.transform_model?.toUpperCase() || '—'}
              subtext={`Estimated using ${registrationResult.transform?.estimator_method || 'MAGSAC++'}`}
              badge="Model"
              variant="accent"
            />
            <MetricCard
              label="Estimator Method"
              value={registrationResult.transform?.estimator_method || 'MAGSAC++'}
              subtext="Adaptive robust threshold estimator"
              badge="Robust"
              variant="default"
            />
            <MetricCard
              label="Output Dimensions"
              value={
                registrationResult.output?.width && registrationResult.output?.height
                  ? `${registrationResult.output.width} × ${registrationResult.output.height}`
                  : '—'
              }
              unit="px"
              subtext="Canvas dimensions of registered warped output"
              badge="Warp"
              variant="secondary"
            />
            <MetricCard
              label="Total Processing Time"
              value={
                registrationResult.timing?.total !== undefined
                  ? registrationResult.timing.total.toFixed(3)
                  : '—'
              }
              unit="s"
              subtext="End-to-end server orchestration latency"
              badge="Latency"
              variant="accent"
            />
          </div>

          {/* Detailed Stage Timing Breakdown */}
          {registrationResult.timing && (
            <div className="timing-breakdown-card">
              <h4 className="timing-breakdown-title">Pipeline Stage Latency Breakdown (seconds)</h4>
              <div className="timing-bars">
                {[
                  { key: 'load', label: 'Load' },
                  { key: 'preprocess', label: 'Preprocess' },
                  { key: 'sift', label: 'SIFT Detect/Extract' },
                  { key: 'matching', label: 'FLANN Match' },
                  { key: 'ratio_test', label: 'Lowe Ratio' },
                  { key: 'geometry', label: 'MAGSAC++ Geometry' },
                  { key: 'warp', label: 'Image Warp' },
                  { key: 'evaluation', label: 'Metric Eval' },
                  { key: 'visualization', label: 'Visual Diagnostics' },
                ].map(({ key, label }) => {
                  const stageTime = registrationResult.timing[key] || 0;
                  return (
                    <div key={key} className="timing-row">
                      <span className="timing-label">{label}</span>
                      <span className="timing-val">{stageTime.toFixed(4)} s</span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Transformation Matrix (2x3 or 3x3 Table) */}
          {registrationResult.transform?.transform_matrix && (
            <div className="matrix-section">
              {renderMatrix(registrationResult.transform.transform_matrix)}
            </div>
          )}

          {/* Stage B Diagnostic Visualizer */}
          {registrationResult.result_id && (
            <ResultViewer
              resultId={registrationResult.result_id}
              metrics={registrationResult}
            />
          )}
        </section>
      )}
    </div>
  );
}

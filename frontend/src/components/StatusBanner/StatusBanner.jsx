import './StatusBanner.css';

/**
 * StatusBanner component indicating dashboard status:
 * - 'idle': Ready for inputs
 * - 'processing': Uploading and executing registration pipeline
 * - 'success': Registration completed successfully
 * - 'error': Error with distinguished error type
 *
 * @param {object} props
 * @param {'idle' | 'processing' | 'success' | 'error'} props.status - Current operational state.
 * @param {string} [props.message] - Custom message text.
 * @param {string} [props.errorType] - 'validation' | 'server' | 'network' | 'not_found' | 'registration_failed' | 'unknown'
 * @param {number | null} [props.statusCode] - HTTP status code if applicable.
 * @param {string} [props.failureStage] - Pipeline stage that failed, if applicable.
 * @param {() => void} [props.onDismiss] - Optional dismiss callback.
 */
export default function StatusBanner({
  status = 'idle',
  message = '',
  errorType = 'unknown',
  statusCode = null,
  failureStage = '',
  onDismiss,
}) {
  if (status === 'idle' && !message) {
    return (
      <div className="status-banner banner-idle" role="status">
        <div className="banner-icon idle-icon">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="10" />
            <line x1="12" y1="16" x2="12" y2="12" />
            <line x1="12" y1="8" x2="12.01" y2="8" />
          </svg>
        </div>
        <div className="banner-body">
          <p className="banner-title">Classical Pipeline Ready</p>
          <p className="banner-text">Select both reference and target images to initiate classical SIFT registration.</p>
        </div>
      </div>
    );
  }

  if (status === 'processing') {
    return (
      <div className="status-banner banner-processing" role="status" aria-live="polite">
        <div className="banner-spinner">
          <div className="spinner-ring" />
        </div>
        <div className="banner-body">
          <div className="banner-header-row">
            <p className="banner-title">Processing Classical Registration…</p>
            <span className="banner-badge processing-badge">SIFT + MAGSAC++</span>
          </div>
          <p className="banner-text">
            {message || 'Uploading images, detecting SIFT keypoints, matching descriptors, and estimating geometric transform.'}
          </p>
        </div>
      </div>
    );
  }

  if (status === 'success') {
    return (
      <div className="status-banner banner-success" role="status">
        <div className="banner-icon success-icon">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
            <polyline points="22 4 12 14.01 9 11.01" />
          </svg>
        </div>
        <div className="banner-body">
          <div className="banner-header-row">
            <p className="banner-title">Registration Succeeded</p>
            <span className="banner-badge success-badge">200 OK</span>
          </div>
          <p className="banner-text">
            {message || 'Geometric transformation estimated and target image aligned successfully.'}
          </p>
        </div>
        {onDismiss && (
          <button type="button" className="banner-close" onClick={onDismiss} aria-label="Dismiss banner">
            ×
          </button>
        )}
      </div>
    );
  }

  if (status === 'error') {
    const errorTypeLabels = {
      validation: 'Validation Error',
      server: 'Pipeline Server Error',
      network: 'Network Error',
      not_found: 'Not Found',
      registration_failed: 'Registration Failed',
      unknown: 'Error',
    };

    const typeLabel = errorTypeLabels[errorType] || 'Error';

    return (
      <div className="status-banner banner-error" role="alert">
        <div className="banner-icon error-icon">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="10" />
            <line x1="12" y1="8" x2="12" y2="12" />
            <line x1="12" y1="16" x2="12.01" y2="16" />
          </svg>
        </div>
        <div className="banner-body">
          <div className="banner-header-row">
            <p className="banner-title">{typeLabel}</p>
            <div className="banner-badges">
              {statusCode && <span className="banner-badge status-code-badge">HTTP {statusCode}</span>}
              <span className={`banner-badge error-type-badge ${errorType}`}>
                {errorType.toUpperCase().replace('_', ' ')}
              </span>
              {failureStage && <span className="banner-badge failure-stage-badge">STAGE: {failureStage}</span>}
            </div>
          </div>
          <p className="banner-text error-message-text">{message || 'An unexpected error occurred.'}</p>
          {errorType === 'network' && (
            <p className="banner-hint">
              Check if backend is listening at <code>http://localhost:8000</code>. Run:{' '}
              <code>uvicorn backend.main:app --port 8000</code>
            </p>
          )}
        </div>
        {onDismiss && (
          <button type="button" className="banner-close" onClick={onDismiss} aria-label="Dismiss banner">
            ×
          </button>
        )}
      </div>
    );
  }

  return null;
}

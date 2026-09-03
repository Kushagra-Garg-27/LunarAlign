/**
 * SIH26166 — Registration API Service (Stage A).
 *
 * Provides HTTP client functions for the classical SIFT registration pipeline:
 * - POST /register: Uploads reference + target images with transform parameters.
 * - getArtifactUrl / getArtifactUrls: Generates URL strings for Stage B visualizers.
 * - Categorized error handling (validation, not_found, server, network).
 */

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

/**
 * Custom error class for registration API failures with categorised error types.
 */
export class RegistrationApiError extends Error {
  constructor(message, { type = 'unknown', status = null, details = null } = {}) {
    super(message);
    this.name = 'RegistrationApiError';
    this.type = type; // 'validation' | 'not_found' | 'server' | 'network' | 'unknown'
    this.status = status;
    this.details = details;
  }
}

/**
 * Helper to classify HTTP status codes into user-friendly error types.
 *
 * @param {number} status
 * @returns {'validation' | 'not_found' | 'server' | 'unknown'}
 */
function classifyHttpStatus(status) {
  if ([400, 413, 415, 422].includes(status)) {
    return 'validation';
  }
  if (status === 404) {
    return 'not_found';
  }
  if (status >= 500) {
    return 'server';
  }
  return 'unknown';
}

/**
 * Extract human-readable error detail from a failed HTTP response.
 *
 * @param {Response} response
 * @returns {Promise<{ message: string, raw: any }>}
 */
async function extractErrorDetail(response) {
  try {
    const data = await response.json();
    if (typeof data.detail === 'string') {
      return { message: data.detail, raw: data };
    }
    if (Array.isArray(data.detail)) {
      // Pydantic validation error list
      const msg = data.detail
        .map((err) => {
          const loc = err.loc ? err.loc.join('.') : '';
          return loc ? `${loc}: ${err.msg}` : err.msg;
        })
        .join('; ');
      return { message: msg || `Validation error (${response.status})`, raw: data };
    }
    if (data.message) {
      return { message: data.message, raw: data };
    }
  } catch {
    // Response body not JSON
  }
  return { message: `Request failed with HTTP ${response.status} (${response.statusText})`, raw: null };
}

/**
 * Execute classical registration on reference and target images.
 *
 * @param {File} referenceFile - Reference lunar image file.
 * @param {File} targetFile - Target lunar image file to be registered.
 * @param {object} [options]
 * @param {'affine' | 'homography'} [options.transformModel='affine'] - Transformation model.
 * @param {number} [options.ratioThreshold=0.75] - Lowe's ratio test threshold (0, 1].
 * @returns {Promise<object>} Parsed RegisterResponse payload.
 * @throws {RegistrationApiError} On network or HTTP error.
 */
export async function registerImages(
  referenceFile,
  targetFile,
  { transformModel = 'affine', ratioThreshold = 0.75 } = {}
) {
  if (!referenceFile) {
    throw new RegistrationApiError('Reference image is required', { type: 'validation' });
  }
  if (!targetFile) {
    throw new RegistrationApiError('Target image is required', { type: 'validation' });
  }

  const formData = new FormData();
  formData.append('reference', referenceFile);
  formData.append('target', targetFile);
  formData.append('transform_model', transformModel);
  formData.append('ratio_threshold', String(ratioThreshold));

  let res;
  try {
    res = await fetch(`${API_BASE}/register`, {
      method: 'POST',
      body: formData,
      // Do NOT set Content-Type header manually; fetch handles multipart boundary automatically.
    });
  } catch (networkErr) {
    throw new RegistrationApiError(
      `Network error: unable to reach the backend at ${API_BASE}. Please ensure the backend server is running. (${networkErr.message})`,
      { type: 'network', details: networkErr }
    );
  }

  if (!res.ok) {
    const errorType = classifyHttpStatus(res.status);
    const { message, raw } = await extractErrorDetail(res);
    throw new RegistrationApiError(message, {
      type: errorType,
      status: res.status,
      details: raw,
    });
  }

  return res.json();
}

/**
 * Build artifact URL string for a specific registration artifact.
 * Does NOT fetch the image — returns the URL string for Stage B consumption.
 *
 * @param {string} resultId - UUID of the registration result.
 * @param {'registered' | 'matches' | 'inliers' | 'overlay'} type - Artifact type.
 * @returns {string} URL to retrieve the artifact PNG.
 */
export function getArtifactUrl(resultId, type) {
  if (!resultId) return '';
  return `${API_BASE}/register/${encodeURIComponent(resultId)}/${type}`;
}

/**
 * Return an object containing all 4 artifact URL strings for a given result_id.
 *
 * @param {string} resultId
 * @returns {{ registered: string, matches: string, inliers: string, overlay: string }}
 */
export function getArtifactUrls(resultId) {
  return {
    registered: getArtifactUrl(resultId, 'registered'),
    matches: getArtifactUrl(resultId, 'matches'),
    inliers: getArtifactUrl(resultId, 'inliers'),
    overlay: getArtifactUrl(resultId, 'overlay'),
  };
}

export function getRegisteredImageUrl(resultId) {
  return getArtifactUrl(resultId, 'registered');
}

export function getMatchesImageUrl(resultId) {
  return getArtifactUrl(resultId, 'matches');
}

export function getInliersImageUrl(resultId) {
  return getArtifactUrl(resultId, 'inliers');
}

export function getOverlayImageUrl(resultId) {
  return getArtifactUrl(resultId, 'overlay');
}
